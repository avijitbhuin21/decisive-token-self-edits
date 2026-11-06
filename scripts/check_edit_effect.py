import mlx.core as mx

from selfedit.edits import Editor, NeuronEdit
from selfedit.model import Model


def main():
    """Verify an edit actually changes forward-pass logits and that rollback restores them."""
    m = Model()
    ed = Editor(m.model, m.tok)
    ids = mx.array(m.tok.encode("def add(a, b):\n    return"))[None]
    base = m.model(ids)[0, -1].astype(mx.float32)
    for edits in ([NeuronEdit(22, 11, 0.0)], [NeuronEdit(20, n, 3.0) for n in range(200)]):
        ed.apply(edits)
        new = m.model(ids)[0, -1].astype(mx.float32)
        print(len(edits), "edits -> max|dlogit| =", mx.abs(new - base).max().item(),
              "| top token", m.tok.decode([mx.argmax(base).item()]), "->", m.tok.decode([mx.argmax(new).item()]))
        ed.rollback()
    print("restored max|dlogit| =", mx.abs(m.model(ids)[0, -1].astype(mx.float32) - base).max().item())


if __name__ == "__main__":
    main()
