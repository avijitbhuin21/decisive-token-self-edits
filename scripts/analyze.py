import argparse
import json


def load(path: str) -> dict[int, dict]:
    """Load a JSONL run file keyed by task_id."""
    return {r["task_id"]: r for r in map(json.loads, open(path))}


def main():
    """Compare baselines, self-edit and random-edit on the same tasks, overall and on first-attempt failures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", default="100")
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    base = load(f"{args.runs}/baselines_nothink_{args.limit}.jsonl")
    se = load(f"{args.runs}/selfedit_self_{args.limit}_s0.jsonl")
    rnd = load(f"{args.runs}/selfedit_random_{args.limit}_s0.jsonl")
    ids = sorted(set(base) & set(se) & set(rnd))
    mismatch = [i for i in ids if base[i]["attempt1_visible"] != se[i]["attempt1_visible"]]
    failed = [i for i in ids if not base[i]["attempt1_visible"]]

    rows = {
        "attempt1 (greedy)": lambda i: base[i]["attempt1_hidden"],
        "retry x3 (resample)": lambda i: base[i]["retry_hidden"],
        "retry x3 + error feedback": lambda i: base[i]["feedback_hidden"],
        "self-edit x3": lambda i: se[i]["final_hidden"],
        "random-edit x3": lambda i: rnd[i]["final_hidden"],
    }
    print(f"tasks={len(ids)} failed_attempt1={len(failed)} attempt1_mismatch={mismatch}")
    print(f"{'method':28} {'hidden pass (all)':>18} {'hidden pass (failed1)':>22}")
    for name, f in rows.items():
        a = sum(f(i) for i in ids)
        b = sum(f(i) for i in failed)
        print(f"{name:28} {a:>8}/{len(ids)} ({a / len(ids):.1%}) {b:>10}/{len(failed)} ({b / max(len(failed), 1):.1%})")

    for label, run in (("self", se), ("random", rnd)):
        rounds = [r for i in failed for r in run[i]["rounds"]]
        if not rounds:
            continue
        vis_fix = sum(any(r["visible"] for r in run[i]["rounds"]) for i in failed)
        print(f"[{label}] rounds={len(rounds)} parse_ok={sum(r['parse_ok'] for r in rounds) / len(rounds):.0%} "
              f"code_changed={sum(r['changed'] for r in rounds) / len(rounds):.0%} visible_fixed_tasks={vis_fix}")


if __name__ == "__main__":
    main()
