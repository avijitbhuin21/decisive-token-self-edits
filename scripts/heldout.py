import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
from mlx_lm import batch_generate

from selfedit.forks import ForkEditor, passes
from selfedit.model import Model
from selfedit.tasks import load_tasks


def greedy_all(m: Model, tasks: list, batch: int) -> dict[int, list[int]]:
    """Greedy answers for many tasks, batched across different prompts."""
    out = {}
    for i in range(0, len(tasks), batch):
        chunk = tasks[i:i + batch]
        prompts = [m.prompt_ids([{"role": "user", "content": t.user_message()}]) for t in chunk]
        res = batch_generate(m.model, m.tok, prompts, max_tokens=1024, return_token_ids=True)
        out.update({t.task_id: ids for t, ids in zip(chunk, res.token_ids)})
    return out


def score(m: Model, tasks: list, answers: dict[int, list[int]]) -> dict[int, bool]:
    """Run hidden+visible tests for every answer in parallel."""
    with ThreadPoolExecutor(8) as ex:
        res = ex.map(lambda t: passes(m, t, answers[t.task_id]), tasks)
    return {t.task_id: ok for t, ok in zip(tasks, res)}


def load_delta(path: str, noise_rng=None) -> mx.array:
    """Sum the rank-1 edits stored in an npz; with noise_rng, replace u/a by norm-matched random vectors."""
    z = np.load(path)
    d = 0
    for u, a, al in zip(z["u"], z["a"], z["alpha"]):
        if noise_rng is not None:
            u = noise_rng.standard_normal(u.shape).astype(np.float32)
            u /= np.linalg.norm(u)
            a = noise_rng.standard_normal(a.shape).astype(np.float32) * np.linalg.norm(a) / np.sqrt(a.size)
        d = d + al * np.outer(u, a) / np.sum(a * a)
    return mx.array(d.astype(np.float32))


def main():
    """Sum per-task fix edits from a train split, apply them, and measure fixes/breaks on held-out tasks."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--forks", default="runs/forks")
    ap.add_argument("--samples", default="runs/samples.jsonl")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--configs", nargs="+", default=["decisive", "rpos", "noise"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--variant", default="", help="episode/delta tag suffix, e.g. _logic")
    ap.add_argument("--out", default="runs/heldout")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    m = Model()
    fe = ForkEditor(m)
    tasks = load_tasks()
    ids = sorted(t.task_id for t in tasks)
    t0 = time.time()

    base_path = f"{args.out}/base.json"
    if os.path.exists(base_path):
        base = {int(k): v for k, v in json.load(open(base_path)).items()}
    else:
        base = score(m, tasks, greedy_all(m, tasks, args.batch))
        json.dump(base, open(base_path, "w"))
    rec = {r["task_id"]: r["greedy"]["hidden"] for r in map(json.loads, open(args.samples))}
    agree = sum(base[i] == rec[i] for i in ids)
    print(f"base pass {sum(base.values())}/{len(ids)} | agrees with recorded greedy on {agree}/{len(ids)} "
          f"[{time.time() - t0:.0f}s]", flush=True)

    eps = {c: {r["task_id"]: r for r in map(json.loads, open(f"{args.forks}/episodes_decisive_L{args.layer}{s}{args.variant}.jsonl"))}
           for c, s in [("decisive", ""), ("rpos", "_rpos")]}
    with open(f"{args.out}/results.jsonl", "a") as f:
        for seed in args.seeds:
            perm = np.random.default_rng(seed).permutation(ids)
            train = set(perm[: len(ids) // 2].tolist())
            for cfg in args.configs:
                src, suffix = ("decisive", "") if cfg == "noise" else (cfg, "_rpos" if cfg == "rpos" else "")
                used = [i for i, r in eps[src].items() if i in train and r["fixed"]]
                rng = np.random.default_rng(1000 + seed) if cfg == "noise" else None
                d = sum(load_delta(f"{args.forks}/delta_decisive_L{args.layer}{suffix}{args.variant}_{i}.npz", rng)
                        for i in used)
                fe.apply_delta(args.layer, args.scale * d)
                after = score(m, tasks, greedy_all(m, tasks, args.batch))
                fe.rollback()
                row = {"seed": seed, "cfg": cfg, "variant": args.variant, "scale": args.scale,
                       "n_edits_tasks": len(used),
                       "dnorm": float(mx.linalg.norm(args.scale * d).item())}
                for name, part in [("train", train), ("test", set(ids) - train)]:
                    row[f"{name}_fixed"] = sum((not base[i]) and after[i] for i in part)
                    row[f"{name}_broke"] = sum(base[i] and not after[i] for i in part)
                    row[f"{name}_base_fail"] = sum(not base[i] for i in part)
                    row[f"{name}_base_pass"] = sum(base[i] for i in part)
                row["train_target_fixed"] = sum(after[i] for i in used)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"[{time.time() - t0:.0f}s] {row}", flush=True)


if __name__ == "__main__":
    main()
