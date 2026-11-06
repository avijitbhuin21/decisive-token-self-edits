import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import generate, load

MODEL = "mlx-community/Qwen3.5-0.8B-bf16"


def main():
    """Load the model, print its structure, generate, and test a reversible weight edit."""
    model, tok = load(MODEL)
    params = tree_flatten(model.parameters())
    print("n_tensors", len(params), "n_params", sum(v.size for _, v in params))
    for k, v in params[:40]:
        print(k, v.shape, v.dtype)
    print(type(model), [n for n, _ in model.children().items()])

    msgs = [{"role": "user", "content": "Write a Python function is_even(n). Only code."}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    t = time.time()
    out = generate(model, tok, prompt, max_tokens=200)
    print(f"--- gen {time.time() - t:.1f}s\n{out}")

    name, w = next((k, v) for k, v in params if "mlp" in k and "down_proj" in k)
    orig = mx.array(w)
    model.load_weights([(name, w * 0)], strict=False)
    edited = dict(tree_flatten(model.parameters()))[name]
    print("edit applied:", bool(mx.all(edited == 0)))
    model.load_weights([(name, orig)], strict=False)
    restored = dict(tree_flatten(model.parameters()))[name]
    print("restored:", bool(mx.array_equal(restored, orig)))


if __name__ == "__main__":
    main()
