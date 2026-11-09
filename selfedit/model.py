import os
import sys

MODEL_ID = "mlx-community/Qwen3.5-0.8B-bf16"

try:
    import mlx.core as mx
    from mlx_lm import batch_generate, generate, load
    from mlx_lm.sample_utils import make_sampler
except ImportError:
    mx = None


class Model:
    def __init__(self, model_id: str = MODEL_ID, thinking: bool = False):
        self.model, self.tok = load(model_id)
        self.thinking = thinking

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        """Tokenize a chat history into the generation prompt's token ids."""
        text = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.thinking
        )
        return self.tok.encode(text, add_special_tokens=False)

    def sample_ids(self, prompt: list[int], n: int, temp: float, max_tokens: int = 1024) -> list[list[int]]:
        """Generate n continuations of a token prompt in one batch, returning their token ids."""
        sampler = make_sampler(temp=temp) if temp > 0 else None
        res = batch_generate(self.model, self.tok, [prompt] * n, max_tokens=max_tokens,
                             return_token_ids=True, sampler=sampler)
        return res.token_ids

    def generate(self, prompts: list[list[int]], temp: float, max_tokens: int = 1024) -> list[list[int]]:
        """Batch-generate one continuation per prompt at the given temperature."""
        sampler = make_sampler(temp=temp) if temp > 0 else None
        return batch_generate(self.model, self.tok, prompts, max_tokens=max_tokens, return_token_ids=True,
                              sampler=sampler).token_ids

    def seed(self, s: int):
        """Seed the sampler."""
        mx.random.seed(s)

    def argmax_next(self, ids: list[int]) -> list[int]:
        """Greedy next-token prediction at every position of a sequence."""
        return mx.argmax(self.model(mx.array([ids]))[0], axis=-1).tolist()

    def decode(self, ids: list[int]) -> str:
        """Decode generated token ids to text."""
        return self.tok.decode(ids)

    def chat(self, messages: list[dict], temp: float = 0.0, max_tokens: int = 1024) -> str:
        """Generate a reply for a chat history at the given sampling temperature."""
        prompt = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.thinking
        )
        return generate(
            self.model, self.tok, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temp)
        )


def load_backend(model_id: str | None = None):
    """Return (model, fork editor): MLX on Apple Silicon, torch/CUDA elsewhere (SELFEDIT_BACKEND overrides)."""
    backend = os.environ.get("SELFEDIT_BACKEND") or ("mlx" if sys.platform == "darwin" and mx is not None else "torch")
    if backend == "mlx":
        from selfedit.forks import ForkEditor
        m = Model(model_id or MODEL_ID)
        return m, ForkEditor(m)
    from selfedit.torch_backend import TorchForkEditor, TorchModel
    m = TorchModel(model_id or "Qwen/Qwen3.5-0.8B")
    return m, TorchForkEditor(m)
