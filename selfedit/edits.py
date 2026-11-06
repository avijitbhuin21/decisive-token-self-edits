from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu


@dataclass
class NeuronEdit:
    layer: int
    neuron: int
    scale: float


class _CaptureMLP(nn.Module):
    """Wraps a SwiGLU MLP and records its hidden (post-activation) values."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.acts = None

    def __call__(self, x):
        a = swiglu(self.inner.gate_proj(x), self.inner.up_proj(x))
        self.acts = a
        return self.inner.down_proj(a)


class Editor:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.lm = model.language_model
        self.layers = self.lm.model.layers
        self._backup: dict[tuple[int, int], mx.array] = {}

    def _unembed(self) -> mx.array:
        """Return the (vocab, hidden) output projection matrix."""
        if getattr(self.lm, "lm_head", None) is not None:
            return self.lm.lm_head.weight
        return self.lm.model.embed_tokens.weight

    def neuron_activity(self, prompt: str, reply: str, layers: list[int]) -> dict[int, mx.array]:
        """Mean |activation| per MLP neuron over the reply tokens, for each requested layer."""
        p_ids = self.tok.encode(prompt)
        ids = p_ids + self.tok.encode(reply, add_special_tokens=False)
        orig = {L: self.layers[L].mlp for L in layers}
        for L in layers:
            self.layers[L].mlp = _CaptureMLP(orig[L])
        try:
            self.model(mx.array(ids)[None])
            out = {}
            for L in layers:
                a = self.layers[L].mlp.acts[0, len(p_ids):].astype(mx.float32)
                out[L] = mx.abs(a).mean(axis=0)
            mx.eval(out)
        finally:
            for L in layers:
                self.layers[L].mlp = orig[L]
        return out

    def neuron_tokens(self, layer: int, neuron: int, k: int = 6) -> tuple[list[str], list[str]]:
        """Logit-lens label: tokens a neuron most promotes and most suppresses when it fires."""
        col = self.layers[layer].mlp.down_proj.weight[:, neuron].astype(mx.float32)
        col = col * self.lm.model.norm.weight.astype(mx.float32)
        logits = self._unembed().astype(mx.float32) @ col
        order = mx.argsort(logits)
        top = order[-k:][::-1].tolist()
        bot = order[:k].tolist()
        dec = lambda ids: [self.tok.decode([i]) for i in ids]
        return dec(top), dec(bot)

    def apply(self, edits: list[NeuronEdit]):
        """Scale each neuron's output column in down_proj, backing up originals for rollback."""
        for e in edits:
            W = self.layers[e.layer].mlp.down_proj.weight
            key = (e.layer, e.neuron)
            if key not in self._backup:
                self._backup[key] = W[:, e.neuron]
            col = self._backup[key] * e.scale
            W[:, e.neuron] = col.astype(W.dtype)
            self.layers[e.layer].mlp.down_proj.weight = W
        mx.eval(self.model.parameters())

    def rollback(self):
        """Restore every edited neuron to its original weights."""
        for (L, n), col in self._backup.items():
            W = self.layers[L].mlp.down_proj.weight
            W[:, n] = col
            self.layers[L].mlp.down_proj.weight = W
        self._backup.clear()
        mx.eval(self.model.parameters())
