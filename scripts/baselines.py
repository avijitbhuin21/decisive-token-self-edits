import argparse
import json
import os
import time

from selfedit.model import Model
from selfedit.tasks import extract_code, load_tasks, run_hidden, run_visible


def attempt(m: Model, task, temp: float, history=None) -> tuple[str, str, list]:
    """Ask the model to solve a task, returning (reply, code, messages)."""
    msgs = history or [{"role": "user", "content": task.user_message()}]
    reply = m.chat(msgs, temp=temp)
    return reply, extract_code(reply), msgs + [{"role": "assistant", "content": reply}]


def retry(m: Model, task, rounds: int, feedback: bool, first_code: str, first_msgs: list):
    """Resample up to `rounds` times (optionally with error feedback) until visible tests pass."""
    code, msgs = first_code, first_msgs
    for r in range(1, rounds + 1):
        vis = run_visible(task, code)
        if vis.passed:
            return code, r - 1
        if feedback:
            msgs = msgs + [{
                "role": "user",
                "content": f"Your code failed a test:\n{vis.error}\nFix it. Reply with one ```python block.",
            }]
            _, code, msgs = attempt(m, task, 0.7, msgs)
        else:
            _, code, _ = attempt(m, task, 0.7)
    return code, rounds


def main():
    """Run single-attempt and retry baselines, writing per-task JSONL results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tag = f"{'think' if args.thinking else 'nothink'}_{args.limit or 'all'}"
    m = Model(thinking=args.thinking)
    tasks = load_tasks(args.limit)
    path = os.path.join(args.out, f"baselines_{tag}.jsonl")
    totals = {"attempt1": 0, "retry": 0, "feedback": 0}
    t0 = time.time()
    with open(path, "w") as f:
        for i, task in enumerate(tasks):
            reply, code, msgs = attempt(m, task, 0.0)
            row = {
                "task_id": task.task_id,
                "attempt1_visible": run_visible(task, code).passed,
                "attempt1_hidden": run_hidden(task, code).passed,
                "attempt1_code": code,
                "attempt1_len": len(reply),
            }
            totals["attempt1"] += row["attempt1_hidden"]
            for name, fb in (("retry", False), ("feedback", True)):
                if row["attempt1_visible"]:
                    row[f"{name}_hidden"], row[f"{name}_rounds"] = row["attempt1_hidden"], 0
                else:
                    c, used = retry(m, task, args.rounds, fb, code, msgs)
                    row[f"{name}_hidden"], row[f"{name}_rounds"] = run_hidden(task, c).passed, used
                totals[name] += row[f"{name}_hidden"]
            f.write(json.dumps(row) + "\n")
            f.flush()
            n = i + 1
            print(
                f"[{n}/{len(tasks)} {time.time() - t0:.0f}s] "
                + " ".join(f"{k}={v / n:.3f}" for k, v in totals.items()),
                flush=True,
            )
    print("wrote", path)


if __name__ == "__main__":
    main()
