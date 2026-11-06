import argparse
import json
import os
import time

from selfedit.model import Model
from selfedit.tasks import extract_code, load_tasks, run_hidden, run_visible


def score(m: Model, task, ids: list[int]) -> dict:
    """Decode a generation and score it on visible tests, and on hidden tests if visible pass."""
    code = extract_code(m.decode(ids))
    vis = run_visible(task, code).passed
    hid = run_hidden(task, code).passed if vis else False
    return {"ids": ids, "visible": vis, "hidden": hid}


def main():
    """Record greedy + N sampled answers (token ids and pass/fail) for every task."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="runs/samples.jsonl")
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["task_id"] for l in open(args.out)}
    m = Model()
    tasks = [t for t in load_tasks(args.limit) if t.task_id not in done]
    t0 = time.time()
    with open(args.out, "a") as f:
        for i, task in enumerate(tasks):
            p = m.prompt_ids([{"role": "user", "content": task.user_message()}])
            row = {"task_id": task.task_id, "greedy": score(m, task, m.sample_ids(p, 1, 0.0)[0]),
                   "samples": [score(m, task, s) for s in m.sample_ids(p, args.n, args.temp)]}
            f.write(json.dumps(row) + "\n")
            f.flush()
            ok = sum(s["hidden"] for s in row["samples"])
            print(f"[{i + 1}/{len(tasks)} {time.time() - t0:.0f}s] task {task.task_id} "
                  f"greedy={row['greedy']['hidden']} samples_ok={ok}/{args.n}", flush=True)


if __name__ == "__main__":
    main()
