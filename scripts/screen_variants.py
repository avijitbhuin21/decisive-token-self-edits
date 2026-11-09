import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from selfedit.model import load_backend
from selfedit.tasks import extract_code, run_hidden, run_visible
from selfedit.variants import load_variants


def score(m, task, ids: list[int]) -> dict:
    """Decode a generation and score it on visible tests, and on hidden tests if visible pass."""
    code = extract_code(m.decode(ids))
    vis = run_visible(task, code).passed
    hid = run_hidden(task, code).passed if vis else False
    return {"ids": ids, "visible": vis, "hidden": hid}


def generate(m, prompts: list[list[int]], temp: float) -> list[list[int]]:
    """Batch-generate one continuation per prompt at the given temperature."""
    return m.generate(prompts, temp)


def main():
    """Record greedy + N sampled answers for every variant of every template."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--templates", nargs="*")
    ap.add_argument("--model")
    ap.add_argument("--out", default="runs/variants/screen.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = {json.loads(l)["template"] for l in open(args.out)} if os.path.exists(args.out) else set()
    m, _ = load_backend(args.model)
    groups = {k: v for k, v in load_variants(args.templates).items() if k not in done}
    t0 = time.time()
    with open(args.out, "a") as f, ThreadPoolExecutor(8) as ex:
        for gi, (name, tasks) in enumerate(groups.items()):
            prompts = [m.prompt_ids([{"role": "user", "content": t.user_message()}]) for t in tasks]
            greedy = generate(m, prompts, 0.0)
            samples = generate(m, [p for p in prompts for _ in range(args.n)], args.temp)
            g_sc = list(ex.map(lambda x: score(m, *x), zip(tasks, greedy)))
            s_sc = list(ex.map(lambda x: score(m, *x),
                               [(t, samples[i * args.n + j]) for i, t in enumerate(tasks) for j in range(args.n)]))
            for i, t in enumerate(tasks):
                row = {"template": name, "task_id": t.task_id, "greedy": g_sc[i],
                       "samples": s_sc[i * args.n:(i + 1) * args.n]}
                f.write(json.dumps(row) + "\n")
            f.flush()
            g = sum(s["hidden"] for s in g_sc)
            s = sum(x["hidden"] for x in s_sc)
            pairs = sum((not g_sc[i]["hidden"]) and any(x["hidden"] for x in s_sc[i * args.n:(i + 1) * args.n])
                        for i in range(len(tasks)))
            print(f"[{gi + 1}/{len(groups)} {time.time() - t0:.0f}s] {name:16s} greedy {g}/{len(tasks)} "
                  f"samples {s}/{len(s_sc)} pairs {pairs}", flush=True)


if __name__ == "__main__":
    main()
