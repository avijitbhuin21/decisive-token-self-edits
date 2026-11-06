import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np

from selfedit.forks import (ForkEditor, best_alternative, clean_success, decisive_token, first_divergence,
                            greedy_passes, is_logic, passes, pivotal_token)
from selfedit.model import Model
from selfedit.tasks import load_tasks


def pick_success(row: dict, m: Model | None = None, task=None) -> list[int] | None:
    """Choose the passing sample that shares the longest prefix with the greedy answer (cleaned if m is given)."""
    g = row["greedy"]["ids"]
    ok = [s["ids"] for s in row["samples"] if s["hidden"]]
    if m is not None:
        ok = [c for c in (clean_success(m, task, s) for s in ok) if c is not None]
    return max(ok, key=lambda s: first_divergence(g, s)) if ok else None


def random_position(m: Model, prompt: list[int], success: list[int], rng, logic: bool = False):
    """Pick a random position on the passing answer where greedy's next token disagrees with it."""
    logits = m.model(mx.array([prompt + success]))[0, len(prompt) - 1:-1]
    pred = np.array(mx.argmax(logits, axis=-1)).tolist()
    bad = [j for j, (p, s) in enumerate(zip(pred, success)) if p != s]
    if logic:
        bad = [j for j in bad if is_logic(m.decode(success[:j]), m.decode([success[j]]), m.decode([pred[j]]))]
    if not bad:
        return None, None, None
    j = bad[int(rng.integers(len(bad)))]
    return j, success[j], pred[j]


def decisive_episode(m: Model, fe: ForkEditor, task, prompt: list[int], success: list[int], layer: int,
                     max_edits: int, random_dir: bool, key, random_pos: bool = False, rng=None,
                     logic: bool = False) -> tuple[dict, list]:
    """Edit at the decisive token of a passing answer; check whether greedy-from-prompt now passes; repeat."""
    log, deltas = [], []
    if success is None:
        return {"edits": [{"stop": "no_clean_success"}], "fixed": False}, deltas
    for _ in range(max_edits):
        d = {"pos": None, "right": None, "wrong": None} if random_pos else decisive_token(m, task, prompt, success)
        if d is None:
            log.append({"stop": "no_bracket"})
            break
        k, right, wrong = d["pos"], d["right"], d["wrong"]
        if random_pos:
            k, right, wrong = random_position(m, prompt, success, rng, logic)
            if k is None:
                log.append({"stop": "no_disagreement"})
                break
        entry = {"pos": k, "wrong": m.decode([wrong]), "right": m.decode([right]),
                 "context": m.decode(success[max(0, k - 16):k])}
        if entry["wrong"] == entry["right"]:
            log.append({**entry, "stop": "same_text"})
            break
        if logic and not is_logic(m.decode(success[:k]), entry["right"], entry["wrong"]):
            log.append({**entry, "stop": "cosmetic"})
            break
        e = fe.flip(prompt + success[:k], right, wrong, layer, random_dir=random_dir, key=key)
        if e is None:
            log.append({**entry, "flipped": False})
            break
        log.append({**entry, "flipped": True, "alpha": e["alpha"], "dnorm": e["dnorm"]})
        deltas.append((e["u"], e["a"], e["alpha"]))
        if greedy_passes(m, task, prompt, [])[0]:
            return {"edits": log, "fixed": True}, deltas
    return {"edits": log, "fixed": False}, deltas


def diverge_episode(m: Model, fe: ForkEditor, task, prompt: list[int], success: list[int], layer: int,
                    max_edits: int, random_dir: bool, key) -> tuple[dict, list]:
    """Flip greedy's first divergence from a passing sample, re-generate, repeat until greedy passes."""
    eos = next(iter(m.tok.eos_token_ids))
    log, deltas = [], []
    cur = m.sample_ids(prompt, 1, 0.0)[0]
    for _ in range(max_edits):
        k = first_divergence(cur, success)
        wrong = cur[k] if k < len(cur) else eos
        right = success[k] if k < len(success) else eos
        entry = {"pos": k, "wrong": m.decode([wrong]), "right": m.decode([right]),
                 "context": m.decode(success[max(0, k - 12):k])}
        e = fe.flip(prompt + success[:k], right, wrong, layer, random_dir=random_dir, key=key)
        if e is None:
            log.append({**entry, "flipped": False})
            break
        log.append({**entry, "flipped": True, "alpha": e["alpha"], "dnorm": e["dnorm"]})
        deltas.append((e["u"], e["a"], e["alpha"]))
        cur = m.sample_ids(prompt, 1, 0.0)[0]
        if passes(m, task, cur):
            return {"edits": log, "fixed": True}, deltas
    return {"edits": log, "fixed": False}, deltas


def edit_episode(m: Model, fe: ForkEditor, task, prompt: list[int], layer: int, max_edits: int,
                 random_dir: bool, key, n: int, p0_hint: float) -> tuple[dict, list]:
    """Find the pivotal culprit token on greedy's own path, flip it with a rank-1 edit, repeat until greedy passes."""
    log, deltas = [], []
    cur = m.sample_ids(prompt, 1, 0.0)[0]
    for step in range(max_edits):
        piv = pivotal_token(m, task, prompt, cur, n=n, p0=p0_hint if step == 0 else None)
        if piv is None:
            log.append({"stop": "low_success_prob"})
            break
        k, p0 = piv
        p_alt, right = best_alternative(m, fe, task, prompt, cur, k, layer, n=n)
        wrong = cur[k]
        e = fe.flip(prompt + cur[:k], right, wrong, layer, random_dir=random_dir, key=key)
        entry = {"pos": k, "p0": p0, "p_alt": p_alt, "wrong": m.decode([wrong]), "right": m.decode([right]),
                 "context": m.decode(cur[max(0, k - 12):k])}
        if e is None:
            log.append({**entry, "flipped": False})
            break
        log.append({**entry, "flipped": True, "alpha": e["alpha"], "dnorm": e["dnorm"]})
        deltas.append((e["u"], e["a"], e["alpha"]))
        cur = m.sample_ids(prompt, 1, 0.0)[0]
        if passes(m, task, cur):
            return {"edits": log, "fixed": True}, deltas
    return {"edits": log, "fixed": False}, deltas


def main():
    """Fork sufficiency + per-task rank-1 fork edits (and random-direction control) on failed tasks."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="runs/samples.jsonl")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--max-edits", type=int, default=3)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--min-sample-ok", type=int, default=1)
    ap.add_argument("--method", choices=["decisive", "diverge", "pivotal"], default="decisive")
    ap.add_argument("--random-dir", action="store_true")
    ap.add_argument("--random-pos", action="store_true")
    ap.add_argument("--logic", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="runs/forks")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.method}_L{args.layer}{'_rand' if args.random_dir else ''}{'_rpos' if args.random_pos else ''}{'_logic' if args.logic else ''}"
    m = Model()
    fe = ForkEditor(m)
    tasks = {t.task_id: t for t in load_tasks()}
    rows = [json.loads(l) for l in open(args.samples)]
    cands = [r for r in rows if not r["greedy"]["hidden"]
             and sum(s["hidden"] for s in r["samples"]) >= args.min_sample_ok]
    if args.limit:
        cands = cands[: args.limit]
    key = mx.random.key(0)
    t0 = time.time()
    n_fixed = 0
    with open(f"{args.out}/episodes_{tag}.jsonl", "w") as f:
        for i, r in enumerate(cands):
            task = tasks[r["task_id"]]
            prompt = m.prompt_ids([{"role": "user", "content": task.user_message()}])
            key, sub = mx.random.split(key)
            if args.method == "decisive":
                succ = pick_success(r, m, task) if args.logic else pick_success(r)
                ep, deltas = decisive_episode(m, fe, task, prompt, succ, args.layer,
                                              args.max_edits, args.random_dir, sub, args.random_pos,
                                              np.random.default_rng(task.task_id), args.logic)
            elif args.method == "diverge":
                ep, deltas = diverge_episode(m, fe, task, prompt, pick_success(r), args.layer,
                                             args.max_edits, args.random_dir, sub)
            else:
                ep, deltas = edit_episode(m, fe, task, prompt, args.layer, args.max_edits,
                                          args.random_dir, sub, args.n,
                                          sum(s["hidden"] for s in r["samples"]) / len(r["samples"]))
            fe.rollback()
            if deltas:
                np.savez(f"{args.out}/delta_{tag}_{task.task_id}.npz",
                         u=np.stack([np.array(u) for u, _, _ in deltas]),
                         a=np.stack([np.array(a) for _, a, _ in deltas]),
                         alpha=np.array([al for _, _, al in deltas]))
            n_fixed += ep["fixed"]
            f.write(json.dumps({"task_id": task.task_id, **ep}) + "\n")
            f.flush()
            last = ep["edits"][-1] if ep["edits"] else {}
            print(f"[{i + 1}/{len(cands)} {time.time() - t0:.0f}s] task {task.task_id} edits={len(ep['edits'])} "
                  f"fixed={ep['fixed']} last={last.get('context', '')!r}[{last.get('wrong', '')!r}->{last.get('right', '')!r}] "
                  f"| fixed={n_fixed}/{i + 1}", flush=True)


if __name__ == "__main__":
    main()
