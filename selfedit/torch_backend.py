import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


def _find(model):
    """Locate the decoder layer list and final norm inside a (possibly multimodal) HF model."""
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and hasattr(mod[0], "mlp") \
                and "visual" not in name and "vision" not in name:
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            return mod, parent.norm
    raise RuntimeError("decoder layers not found")


class TorchModel:
    """CUDA/transformers twin of selfedit.model.Model with the same methods."""

    def __init__(self, model_id: str, thinking: bool = False, dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="cuda")
        except ValueError:
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, device_map="cuda")
        self.model.eval()
        self.thinking = thinking
        self.layers, self.final_norm = _find(self.model)
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) | {self.tok.eos_token_id}
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else next(iter(self.eos_ids))
        self.batch = 64

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        """Tokenize a chat history into the generation prompt's token ids."""
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                            enable_thinking=self.thinking)
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        """Decode generated token ids to text."""
        return self.tok.decode(ids)

    def seed(self, s: int):
        """Seed the sampler."""
        torch.manual_seed(s)

    @torch.no_grad()
    def generate(self, prompts: list[list[int]], temp: float, max_tokens: int = 1024) -> list[list[int]]:
        """Batch-generate one continuation per prompt (left-padded), trimmed after the first EOS."""
        out = []
        for i in range(0, len(prompts), self.batch):
            chunk = prompts[i:i + self.batch]
            L = max(map(len, chunk))
            ids = torch.tensor([[self.pad_id] * (L - len(p)) + p for p in chunk], device="cuda")
            mask = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in chunk], device="cuda")
            kw = {"do_sample": True, "temperature": temp, "top_p": 1.0, "top_k": 0} if temp > 0 else {"do_sample": False}
            gen = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_tokens,
                                      pad_token_id=self.pad_id, eos_token_id=list(self.eos_ids), **kw)
            for row in gen[:, L:].tolist():
                cut = next((j for j, t in enumerate(row) if t in self.eos_ids), len(row))
                out.append(row[:cut])
        return out

    def sample_ids(self, prompt: list[int], n: int, temp: float, max_tokens: int = 1024) -> list[list[int]]:
        """Generate n continuations of one prompt."""
        if temp <= 0 and n == 1:
            return [self._greedy_single(prompt, max_tokens)]
        return self.generate([prompt] * n, temp, max_tokens)

    @torch.no_grad()
    def _greedy_single(self, prompt: list[int], max_tokens: int) -> list[int]:
        """Unpadded greedy decode of a single prompt."""
        ids = torch.tensor([prompt], device="cuda")
        gen = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_tokens,
                                  do_sample=False, pad_token_id=self.pad_id, eos_token_id=list(self.eos_ids))
        row = gen[0, len(prompt):].tolist()
        cut = next((j for j, t in enumerate(row) if t in self.eos_ids), len(row))
        return row[:cut]

    @torch.no_grad()
    def argmax_next(self, ids: list[int]) -> list[int]:
        """Greedy next-token prediction at every position of a sequence."""
        logits = self.model(input_ids=torch.tensor([ids], device="cuda")).logits[0]
        return logits.argmax(-1).tolist()


class TorchForkEditor:
    """Rank-1 down_proj edits on the torch backend; same interface as selfedit.forks.ForkEditor."""

    def __init__(self, m: TorchModel):
        self.m = m
        self.layers = m.layers
        self._orig: dict[int, torch.Tensor] = {}
        self._committed: dict[int, torch.Tensor] = {}
        self._gain = self._effective_gain()

    @torch.no_grad()
    def _effective_gain(self) -> torch.Tensor:
        """Per-channel multiplier of the final RMSNorm, measured (handles 1+w zero-centred norms)."""
        norm = self.m.final_norm
        w = norm.weight
        x = torch.ones(1, w.shape[0], device=w.device, dtype=w.dtype)
        return norm(x)[0].float()

    def _W(self, layer: int) -> torch.nn.Parameter:
        """The down_proj weight parameter of a layer."""
        return self.layers[layer].mlp.down_proj.weight

    @torch.no_grad()
    def state(self, ctx: list[int], layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """down_proj input at the last context position and next-token logits (float32)."""
        cap = {}
        h = self.layers[layer].mlp.down_proj.register_forward_pre_hook(lambda mod, inp: cap.__setitem__("a", inp[0]))
        try:
            logits = self.m.model(input_ids=torch.tensor([ctx], device="cuda")).logits[0, -1].float()
        finally:
            h.remove()
        return cap["a"][0, -1].float(), logits

    def direction(self, right: int, wrong: int) -> torch.Tensor:
        """Unit residual-space direction that raises the right token's logit over the wrong one."""
        E = self.m.model.get_output_embeddings().weight
        u = (E[right].float() - E[wrong].float()) * self._gain
        return u / u.norm()

    def delta(self, u, a, alpha: float) -> torch.Tensor:
        """Rank-1 update alpha * u a^T / |a|^2."""
        return alpha * torch.outer(u, a) / (a * a).sum()

    def _base(self, layer: int) -> torch.Tensor:
        """Current committed weight (original if never edited)."""
        if layer not in self._orig:
            self._orig[layer] = self._W(layer).detach().clone()
            self._committed[layer] = self._orig[layer]
        return self._committed[layer]

    @torch.no_grad()
    def _set(self, layer: int, W: torch.Tensor):
        """Write a weight into a layer in place."""
        self._W(layer).copy_(W)

    @torch.no_grad()
    def apply_delta(self, layer: int, d, commit: bool = True):
        """Add a float32 delta (torch or numpy) on top of the committed weight."""
        if isinstance(d, np.ndarray):
            d = torch.from_numpy(d)
        base = self._base(layer)
        W = (base.float() + d.to(base.device).float()).to(base.dtype)
        self._set(layer, W)
        if commit:
            self._committed[layer] = W

    def revert_trial(self, layer: int):
        """Drop an uncommitted trial edit."""
        self._set(layer, self._base(layer))

    def rollback(self):
        """Restore every edited layer to its original weights."""
        for L, W in self._orig.items():
            self._set(L, W)
        self._orig.clear()
        self._committed.clear()

    def flip(self, ctx: list[int], right: int, wrong: int, layer: int, margin: float = 1.0,
             random_dir: bool = False, key=None):
        """Smallest doubling alpha whose edit makes `right` win by `margin`; commits it. u/a returned as numpy."""
        a, _ = self.state(ctx, layer)
        u = self.direction(right, wrong)
        if random_dir:
            g = torch.Generator(device="cuda").manual_seed(int(key) if key is not None else 0)
            u = torch.randn(u.shape, device="cuda", generator=g)
            u = u / u.norm()
        alpha = 1.0
        while alpha <= 4096:
            self.apply_delta(layer, self.delta(u, a, alpha), commit=False)
            _, logits = self.state(ctx, layer)
            others = torch.cat([logits[:right], logits[right + 1:]])
            if (logits[right] - others.max()).item() >= margin:
                self._committed[layer] = self._W(layer).detach().clone()
                return {"layer": layer, "alpha": alpha, "u": u.cpu().numpy(), "a": a.cpu().numpy(),
                        "dnorm": (alpha / a.norm()).item()}
            self.revert_trial(layer)
            alpha *= 2
        return None