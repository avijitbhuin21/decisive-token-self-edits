from __future__ import annotations

import ast
import io
import tokenize

try:
    import mlx.core as mx

    from selfedit.edits import _CaptureMLP
except ImportError:
    mx = None

from selfedit.tasks import extract_code, run_hidden, run_visible


def first_divergence(a: list[int], b: list[int]) -> int:
    """Index of the first position where two token sequences differ."""
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k


def passes(m, task, ids: list[int]) -> bool:
    """True if the decoded answer passes both visible and hidden tests."""
    code = extract_code(m.decode(ids))
    return run_visible(task, code).passed and run_hidden(task, code).passed


def greedy_passes(m, task, prompt: list[int], prefix: list[int]) -> tuple[bool, list[int]]:
    """Force a prefix, continue greedily, and report whether the full answer passes; also returns the continuation."""
    cont = m.sample_ids(prompt + prefix, 1, 0.0)[0]
    return passes(m, task, prefix + cont), cont


def decisive_token(m, task, prompt: list[int], success: list[int]):
    """Binary-search the passing answer for the token whose forcing flips greedy from fail to pass.

    Returns {"pos", "right", "wrong"} with context success[:pos], or None if the endpoints don't bracket a flip.
    """
    if greedy_passes(m, task, prompt, [])[0]:
        return None
    if not greedy_passes(m, task, prompt, success)[0]:
        return None
    lo, hi = 0, len(success)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if greedy_passes(m, task, prompt, success[:mid])[0]:
            hi = mid
        else:
            lo = mid
    _, cont = greedy_passes(m, task, prompt, success[:lo])
    eos = m.eos
    return {"pos": lo, "right": success[lo], "wrong": cont[0] if cont else eos}


def strip_cosmetic(code: str) -> str:
    """Remove comments and docstrings from Python source, keeping all other lines verbatim."""
    tree = ast.parse(code)
    drop = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and len(node.body) > 1:
            b = node.body[0]
            if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str):
                drop.update(range(b.lineno, b.end_lineno + 1))
    comments = {t.start[0]: t.start[1] for t in tokenize.generate_tokens(io.StringIO(code).readline)
                if t.type == tokenize.COMMENT}
    out = []
    for i, line in enumerate(code.splitlines(), 1):
        if i in drop:
            continue
        if i in comments:
            line = line[:comments[i]].rstrip()
            if not line.strip():
                continue
        out.append(line)
    return "\n".join(out) + "\n"


def clean_success(m, task, ids: list[int]) -> list[int] | None:
    """Re-tokenize a passing answer with comments/docstrings stripped; None if it no longer parses or passes."""
    text = m.decode(ids)
    code = extract_code(text)
    try:
        clean = strip_cosmetic(code)
    except (SyntaxError, tokenize.TokenError, IndentationError):
        return None
    new = m.tok.encode(text.replace(code, clean, 1), add_special_tokens=False)
    return new if passes(m, task, new) else None


def is_logic(ctx: str, right: str, wrong: str) -> bool:
    """True if a token decision is about code, not comments, docstrings or whitespace."""
    if any(s in t for t in (right, wrong) for s in ("#", '"""', "'''")):
        return False
    if not right.strip() and not wrong.strip():
        return False
    if "#" in ctx.rsplit("\n", 1)[-1] or ctx.count('"""') % 2 or ctx.count("'''") % 2:
        return False
    return "```" not in ctx or ctx.count("```") % 2 == 1


def success_rate(m, task, prompt: list[int], prefix: list[int], n: int, temp: float = 0.7) -> float:
    """Fraction of n sampled continuations from prompt+prefix that pass visible and hidden tests."""
    conts = m.sample_ids(prompt + prefix, n, temp)
    return sum(passes(m, task, prefix + c) for c in conts) / n


def pivotal_token(m, task, prompt: list[int], path: list[int], n: int = 8, p0: float | None = None):
    """Binary-search the failing path for the token after which success probability collapses.

    Success is "alive" while the sampled pass rate stays >= half the starting rate (and >= 1/n).
    Returns (k, p0) where path[k] is the culprit token, or None if the task never succeeds.
    """
    if p0 is None:
        p0 = success_rate(m, task, prompt, [], n)
    if p0 <= 0:
        return None
    thr = max(p0 / 2, 1 / n)
    lo, hi = 0, len(path)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if success_rate(m, task, prompt, path[:mid], n) >= thr:
            lo = mid
        else:
            hi = mid
    return lo, p0


def best_alternative(m, fe, task, prompt: list[int], path: list[int], k: int, layer: int,
                     n: int = 8, top: int = 4):
    """Among the top next tokens at the pivot (excluding the culprit), pick the one with highest success rate."""
    _, logits = fe.state(prompt + path[:k], layer)
    cands = [t for t in mx.argsort(logits)[::-1][: top + 1].tolist() if t != path[k]][:top]
    scored = [(success_rate(m, task, prompt, path[:k] + [t], n), t) for t in cands]
    return max(scored)


def find_forks(m, task, prompt: list[int], greedy: list[int], success: list[int], max_forks: int = 5):
    """Force the success path at each divergence from greedy until greedy continuation passes."""
    eos = m.eos
    cur, forks = greedy, []
    for _ in range(max_forks):
        k = first_divergence(cur, success)
        if k >= len(cur) and k >= len(success):
            break
        wrong = cur[k] if k < len(cur) else eos
        right = success[k] if k < len(success) else eos
        forks.append({"pos": k, "wrong": wrong, "right": right})
        if right == eos:
            cur = success
        else:
            prefix = success[: k + 1]
            cur = prefix + m.sample_ids(prompt + prefix, 1, 0.0)[0]
        if passes(m, task, cur):
            return forks, True
    return forks, False


class ForkEditor:
    """Rank-1 edits to MLP down_proj that flip a single next-token decision."""

    def __init__(self, m):
        self.m = m
        self.lm = m.model.language_model
        self.layers = self.lm.model.layers
        self._orig: dict[int, mx.array] = {}
        self._committed: dict[int, mx.array] = {}

    def _unembed(self) -> mx.array:
        """Return the (vocab, hidden) output projection."""
        head = getattr(self.lm, "lm_head", None)
        return head.weight if head is not None else self.lm.model.embed_tokens.weight

    def _set(self, layer: int, W: mx.array):
        """Install a down_proj weight for a layer and materialize it."""
        self.layers[layer].mlp.down_proj.weight = W
        mx.eval(W)

    def state(self, ctx: list[int], layer: int) -> tuple[mx.array, mx.array]:
        """MLP hidden activation at the last context position and the next-token logits."""
        orig = self.layers[layer].mlp
        self.layers[layer].mlp = _CaptureMLP(orig)
        try:
            logits = self.m.model(mx.array(ctx)[None])[0, -1].astype(mx.float32)
            a = self.layers[layer].mlp.acts[0, -1].astype(mx.float32)
            mx.eval(logits, a)
        finally:
            self.layers[layer].mlp = orig
        return a, logits

    def direction(self, right: int, wrong: int) -> mx.array:
        """Unit residual-space direction that raises the right token's logit over the wrong one."""
        E = self._unembed()
        u = (E[right].astype(mx.float32) - E[wrong].astype(mx.float32)) * self.lm.model.norm.weight.astype(mx.float32)
        return u / mx.linalg.norm(u)

    def delta(self, u: mx.array, a: mx.array, alpha: float) -> mx.array:
        """Rank-1 update alpha * u a^T / |a|^2 (outputs exactly alpha*u for input a)."""
        return alpha * mx.outer(u, a) / mx.sum(a * a)

    def _base(self, layer: int) -> mx.array:
        """Current committed weight for a layer (original if never edited)."""
        if layer not in self._orig:
            self._orig[layer] = self.layers[layer].mlp.down_proj.weight
            self._committed[layer] = self._orig[layer]
        return self._committed[layer]

    def apply_delta(self, layer: int, d: mx.array, commit: bool = True):
        """Add a float32 delta on top of the committed weight; optionally commit it."""
        if not isinstance(d, mx.array):
            d = mx.array(d)
        base = self._base(layer)
        W = (base.astype(mx.float32) + d).astype(base.dtype)
        self._set(layer, W)
        if commit:
            self._committed[layer] = W

    def revert_trial(self, layer: int):
        """Drop an uncommitted trial edit, returning the layer to its committed weight."""
        self._set(layer, self._base(layer))

    def rollback(self):
        """Restore every edited layer to its original weights."""
        for L, W in self._orig.items():
            self._set(L, W)
        self._orig.clear()
        self._committed.clear()

    def flip(self, ctx: list[int], right: int, wrong: int, layer: int, margin: float = 1.0,
             random_dir: bool = False, key=None):
        """Find the smallest doubling alpha whose edit makes `right` win by `margin`; commits and returns it."""
        a, _ = self.state(ctx, layer)
        u = self.direction(right, wrong)
        if random_dir:
            u = mx.random.normal(u.shape, key=key)
            u = u / mx.linalg.norm(u)
        alpha = 1.0
        while alpha <= 4096:
            d = self.delta(u, a, alpha)
            self.apply_delta(layer, d, commit=False)
            _, logits = self.state(ctx, layer)
            gap = (logits[right] - mx.max(mx.concatenate([logits[:right], logits[right + 1:]]))).item()
            if gap >= margin:
                self._committed[layer] = self.layers[layer].mlp.down_proj.weight
                return {"layer": layer, "alpha": alpha, "u": u, "a": a,
                        "dnorm": (alpha / mx.linalg.norm(a)).item()}
            self.revert_trial(layer)
            alpha *= 2
        return None

