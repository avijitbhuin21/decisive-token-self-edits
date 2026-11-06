import time

from selfedit.model import Model
from selfedit.tasks import extract_code, load_tasks, run_visible


def main():
    """Check batched sampling, token ids, greedy consistency, and forced-prefix continuation."""
    m = Model()
    task = load_tasks(3)[2]
    msgs = [{"role": "user", "content": task.user_message()}]
    p = m.prompt_ids(msgs)
    g_text = m.chat(msgs)
    g_ids = m.sample_ids(p, 1, 0.0)[0]
    print("greedy batch==single:", m.decode(g_ids).strip() == g_text.strip(), "| last ids", g_ids[-3:],
          "| eos", m.tok.eos_token_ids)
    t = time.time()
    samples = m.sample_ids(p, 10, 0.7)
    print(f"10 samples in {time.time() - t:.1f}s, lens", [len(s) for s in samples])
    print("visible pass:", [run_visible(task, extract_code(m.decode(s))).passed for s in samples])
    k = 5
    cont = m.sample_ids(p + g_ids[:k], 1, 0.0)[0]
    print("forced-prefix greedy reproduces greedy:", g_ids[:k] + cont == g_ids)


if __name__ == "__main__":
    main()
