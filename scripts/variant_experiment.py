import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fork_experiment import decisive_episode, pick_success  # noqa: E402
from heldout import load_delta  # noqa: E402
from screen_variants import generate  # noqa: E402

from selfedit.forks import passes  # noqa: E402
from selfedit.model import load_backend  # noqa: E402
from selfedit.variants import load_variants  # noqa: E402


def run_episodes(m, fe, rows, tasks, method, args, path, ddir):
    """Run (or resume) one edit episode per pair variant; returns {task_id: episode}."""
    done = {r["task_id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}
    todo = [r for r in rows if r["task_id"] not in done and not r["greedy"]["hidden"]
            and any(s["hidden"] for s in r["samples"])]
    t0 = time.time()
    with open(path, "a") as f:
        for i, r in enumerate(todo):
            task = tasks[r["task_id"]]
            prompt = m.prompt_ids([{"role": "user", "content": task.user_message()}])
            succ = pick_success(r, m, task) if args.logic else pick_success(r)
            ep, deltas = decisive_episode(m, fe, task, prompt, succ, args.layer, args.max_edits, False, None,
                                          method == "rpos", np.random.default_rng(task.task_id), args.logic)
            fe.rollback()
            if deltas:
                np.savez(f"{ddir}/{method}_{task.task_id}.npz",
                         u=np.stack([np.array(u) for u, _, _ in deltas]),
                         a=np.stack([np.array(a) for _, a, _ in deltas]),
                         alpha=np.array([al for _, _, al in deltas]))
            row = {"task_id": task.task_id, "template": r["template"], **ep}
            done[task.task_id] = row
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{method} {i + 1}/{len(todo)} {time.time() - t0:.0f}s] {r['template']} {task.task_id} "
                  f"edits={len(ep['edits'])} fixed={ep['fixed']}", flush=True)
    return done


def evaluate(m, ex, tasks: list, n: int, seed: int) -> dict[int, dict]:
    """Greedy pass and sampled pass-rate for each task under the current weights."""
    prompts = [m.prompt_ids([{"role": "user", "content": t.user_message()}]) for t in tasks]
    greedy = generate(m, prompts, 0.0)
    m.seed(seed)
    samples = generate(m, [p for p in prompts for _ in range(n)], 0.7)
    g = list(ex.map(lambda x: passes(m, *x), zip(tasks, greedy)))
    s = list(ex.map(lambda x: passes(m, *x), [(t, samples[i * n + j]) for i, t in enumerate(tasks) for j in range(n)]))
    return {t.task_id: {"greedy": g[i], "rate": sum(s[i * n:(i + 1) * n]) / n} for i, t in enumerate(tasks)}


def main():
    """Edit on failing variants of a template; measure the effect on its held-out variants and controls."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="runs/variants/screen.jsonl")
    ap.add_argument("--templates", nargs="*")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--max-edits", type=int, default=3)
    ap.add_argument("--n-edit", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--configs", nargs="+", default=["decisive", "rpos", "noise", "cross"])
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--logic", action="store_true")
    ap.add_argument("--min-pairs", type=int, default=0)
    ap.add_argument("--model")
    ap.add_argument("--out", default="runs/variants")
    args = ap.parse_args()

    tag = "_logic" if args.logic else ""
    ddir = f"{args.out}/deltas{tag}"
    os.makedirs(ddir, exist_ok=True)
    m, fe = load_backend(args.model)
    if args.layer is None:
        args.layer = round(len(fe.layers) * 2 / 3)
    print(f"model {args.model} | {len(fe.layers)} layers | editing layer {args.layer}", flush=True)
    groups = load_variants(args.templates)
    rows = [r for r in map(json.loads, open(args.screen)) if r["template"] in groups]
    npairs = {n: sum(not r["greedy"]["hidden"] and any(s["hidden"] for s in r["samples"])
                     for r in rows if r["template"] == n) for n in groups}
    groups = {n: ts for n, ts in groups.items() if npairs[n] >= args.min_pairs}
    rows = [r for r in rows if r["template"] in groups]
    print(f"{len(groups)} templates: {sorted(groups)}", flush=True)
    tasks = {t.task_id: t for ts in groups.values() for t in ts}
    meths = ["decisive"] + (["rpos"] if "rpos" in args.configs else [])
    eps = {meth: run_episodes(m, fe, rows, tasks, meth, args, f"{args.out}/episodes_{meth}{tag}.jsonl", ddir)
           for meth in meths}

    t0 = time.time()
    names = sorted(groups)
    ex = ThreadPoolExecutor(8)
    base_path = f"{args.out}/eval_base.json"
    base = {int(k): v for k, v in json.load(open(base_path)).items()} if os.path.exists(base_path) else {}
    for name in names:
        if groups[name][0].task_id not in base:
            base.update(evaluate(m, ex, groups[name], args.n_samples, 0))
            json.dump(base, open(base_path, "w"))
    print(f"base eval done [{time.time() - t0:.0f}s]", flush=True)

    res_path = f"{args.out}/results{tag}.jsonl"
    finished = {(r["seed"], r["template"], r["cfg"], r["scale"]) for r in map(json.loads, open(res_path))} \
        if os.path.exists(res_path) else set()
    with open(res_path, "a") as f:
        for seed in args.seeds:
            split = {}
            for name in names:
                perm = np.random.default_rng([seed, names.index(name)]).permutation(len(groups[name]))
                split[name] = {groups[name][i].task_id for i in perm[:args.n_edit]}

            def deltas(name, meth, rng=None):
                """Summed fixing edits from a template's edit split (None if there are none)."""
                used = [i for i in split[name] if eps[meth].get(i, {}).get("fixed")]
                d = [load_delta(f"{ddir}/{meth}_{i}.npz", rng) for i in used]
                return (sum(d), used) if d else (None, used)

            have = [n for n in names if deltas(n, "decisive")[0] is not None]
            for name in names:
                for cfg in args.configs:
                    if all((seed, name, cfg, s) in finished for s in args.scales):
                        continue
                    if cfg == "cross":
                        if name not in have or len(have) < 2:
                            continue
                        src = have[(have.index(name) + 1) % len(have)]
                        d, used = deltas(src, "decisive")
                    else:
                        src = name
                        rng = np.random.default_rng([seed, names.index(name), 7]) if cfg == "noise" else None
                        d, used = deltas(name, "decisive" if cfg == "noise" else cfg, rng)
                    if d is None:
                        continue
                    for scale in args.scales:
                        if (seed, name, cfg, scale) in finished:
                            continue
                        fe.apply_delta(args.layer, scale * d)
                        after = evaluate(m, ex, groups[name], args.n_samples, 0)
                        fe.rollback()
                        res = {"seed": seed, "template": name, "cfg": cfg, "src": src, "scale": scale,
                               "logic": args.logic, "n_used": len(used),
                               "dnorm": float(np.linalg.norm(scale * d))}
                        for part, ids in [("edit", split[name]), ("test", {t.task_id for t in groups[name]} - split[name])]:
                            res[f"{part}_n"] = len(ids)
                            res[f"{part}_greedy_base"] = sum(base[i]["greedy"] for i in ids)
                            res[f"{part}_greedy_after"] = sum(after[i]["greedy"] for i in ids)
                            res[f"{part}_rate_base"] = float(np.mean([base[i]["rate"] for i in ids]))
                            res[f"{part}_rate_after"] = float(np.mean([after[i]["rate"] for i in ids]))
                            res[f"{part}_fixed"] = sum((not base[i]["greedy"]) and after[i]["greedy"] for i in ids)
                            res[f"{part}_broke"] = sum(base[i]["greedy"] and not after[i]["greedy"] for i in ids)
                        f.write(json.dumps(res) + "\n")
                        f.flush()
                        print(f"[{time.time() - t0:.0f}s] s{seed} {name:16s} {cfg:8s} x{scale} used={len(used)} "
                              f"test greedy {res['test_greedy_base']}->{res['test_greedy_after']}/{res['test_n']} "
                              f"rate {res['test_rate_base']:.2f}->{res['test_rate_after']:.2f} | "
                              f"edit greedy {res['edit_greedy_base']}->{res['edit_greedy_after']}", flush=True)


if __name__ == "__main__":
    main()
