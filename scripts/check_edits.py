import mlx.core as mx
from mlx.utils import tree_flatten

from selfedit.edits import Editor, NeuronEdit
from selfedit.model import Model


def main():
    """Smoke-test activity capture, logit-lens labels, edit effect, and exact rollback."""
    m = Model()
    ed = Editor(m.model, m.tok)
    before = {k: mx.array(v) for k, v in tree_flatten(m.model.parameters())}
    msgs = [{"role": "user", "content": "Write a Python function add(a, b). One ```python block."}]
    prompt = m.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    reply = m.chat(msgs)
    print("reply:", reply)

    act = ed.neuron_activity(prompt, reply, [20, 22])
    top = mx.argsort(act[22])[-5:][::-1].tolist()
    for n in top:
        pro, sup = ed.neuron_tokens(22, n)
        print(f"L22 n{n} act={act[22][n].item():.3f} +{pro} -{sup}")

    ed.apply([NeuronEdit(22, n, 0.0) for n in top] + [NeuronEdit(20, n, 3.0) for n in range(200)])
    print("edited reply:", m.chat(msgs))
    ed.rollback()
    after = dict(tree_flatten(m.model.parameters()))
    same = all(mx.array_equal(before[k], after[k]).item() for k in before)
    print("exact restore:", same, "| reply again identical:", m.chat(msgs) == reply)


if __name__ == "__main__":
    main()
