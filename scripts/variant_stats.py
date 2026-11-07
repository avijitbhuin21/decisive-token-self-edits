import argparse
import json
from collections import defaultdict

import numpy as np


def signflip_p(diffs, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation p-value for mean(diffs) != 0."""
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0 or not d.any():
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    return float(((np.abs((flips * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n_perm + 1))


def key_cosine(episodes_dir: str) -> dict:
    """Cosine similarity of first-edit keys (MLP inputs) within vs across templates."""
    import glob
    import itertools
    import re
    eps = [json.loads(l) for l in open(f"{episodes_dir}/episodes_decisive_logic.jsonl")]
    tmpl = {e["task_id"]: e["template"] for e in eps}
    first = {e["task_id"]: e["edits"][0]["context"].rstrip(" ").endswith("):\n")
             for e in eps if e["edits"] and e["edits"][0].get("flipped")}
    keys = {}
    for p in glob.glob(f"{episodes_dir}/deltas_logic/decisive_*.npz"):
        a = np.load(p)["a"][0].astype(np.float64)
        keys[int(re.findall(r"_(\d+)\.npz", p)[0])] = a / np.linalg.norm(a)
    within, across, wfirst = [], [], []
    for i, j in itertools.combinations(keys, 2):
        c = float(keys[i] @ keys[j])
        if tmpl[i] == tmpl[j]:
            within.append(c)
            if first.get(i) and first.get(j):
                wfirst.append(c)
        else:
            across.append(c)
    return {"n_keys": len(keys), "within_median": float(np.median(within)), "across_median": float(np.median(across)),
            "within_first_mean": float(np.mean(wfirst)), "n_within_first": len(wfirst)}


def write_tex(rows, unit, out_dir: str, screen: str, episodes_dir: str):
    """Write the paper's variant tables and headline numbers from the results."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    names = {"decisive": "Decisive token", "rpos": "Random position", "noise": "Norm-matched noise",
             "cross": "Other template's edits"}
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Scale & Edits & Units & Greedy pass (before$\to$after) & Fixed / broke & $\Delta$ sample rate \\",
             r"\midrule"]
    nums = {}
    for scale in (1.0, 0.5):
        for cfg in ("decisive", "rpos", "noise", "cross"):
            rs = [r for r in rows if r["cfg"] == cfg and r["scale"] == scale]
            n = sum(r["test_n"] for r in rs)
            gb, ga = sum(r["test_greedy_base"] for r in rs), sum(r["test_greedy_after"] for r in rs)
            fx, br = sum(r["test_fixed"] for r in rs), sum(r["test_broke"] for r in rs)
            dr = float(np.mean([r["test_rate_after"] - r["test_rate_base"] for r in rs]))
            nums[f"{cfg}_{scale}"] = {"units": len(rs), "variants": n, "greedy_base": gb, "greedy_after": ga,
                                      "fixed": fx, "broke": br, "d_rate": dr}
            lab = rf"$\times${scale:g}" if cfg == "decisive" else ""
            lines.append(rf"{lab} & {names[cfg]} & {len(rs)} & {gb}/{n} $\to$ {ga}/{n} & {fx} / {br} & {dr:+.2f} \\")
        if scale == 1.0:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out_dir}/tab_variants.tex", "w").write("\n".join(lines) + "\n")

    tests = {}
    for scale in (1.0, 0.5):
        for a, b in [("decisive", "noise"), ("decisive", "cross"), ("rpos", "noise"), ("decisive", "rpos"),
                     ("cross", "noise")]:
            for metric in ("net", "rate"):
                f = (lambda r: r["test_fixed"] - r["test_broke"]) if metric == "net" else \
                    (lambda r: r["test_rate_after"] - r["test_rate_base"])
                d = [f(c[a]) - f(c[b]) for (s, t, sc), c in unit.items() if sc == scale and a in c and b in c]
                tests[f"{a}_vs_{b}_{metric}_{scale}"] = {"n": len(d), "mean": float(np.mean(d)),
                                                        "wins": int(sum(x > 0 for x in d)),
                                                        "losses": int(sum(x < 0 for x in d)), "p": signflip_p(d)}
    per = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["scale"] == 1.0:
            per[r["template"]][r["cfg"]] += r["test_fixed"] - r["test_broke"]
    pt = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Template & Decisive & Random pos. & Noise & Other template \\",
          r"\midrule"]
    for t in sorted(per):
        p = per[t]
        pt.append(rf"\texttt{{{t.replace('_', chr(92) + '_')}}} & {p['decisive']:+d} & {p['rpos']:+d} & {p['noise']:+d} & {p['cross']:+d} \\")
    pt += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out_dir}/tab_variants_template.tex", "w").write("\n".join(pt) + "\n")

    scr = [json.loads(l) for l in open(screen)]
    by_t = defaultdict(list)
    for r in scr:
        by_t[r["template"]].append(r)
    pairs = {t: sum((not r["greedy"]["hidden"]) and any(s["hidden"] for s in r["samples"]) for r in v)
             for t, v in by_t.items()}
    used = {r["template"] for r in rows}
    eps = {}
    for m in ("decisive", "rpos"):
        E = [json.loads(l) for l in open(f"{episodes_dir}/episodes_{m}_logic.jsonl")]
        fl = [x for e in E if e["fixed"] for x in e["edits"] if x.get("flipped")]
        eps[m] = {"episodes": len(E), "fixed": sum(e["fixed"] for e in E), "flips": len(fl),
                  "first_body": sum(x["context"].rstrip(" ").endswith("):\n") for x in fl)}
    nums.update({"tests": tests, "n_templates": len(by_t), "n_variants": len(scr),
                 "n_used_templates": len(used), "n_pairs_used": sum(pairs[t] for t in used),
                 "greedy_pass_all": sum(r["greedy"]["hidden"] for r in scr), "episodes": eps,
                 "key_cosine": key_cosine(episodes_dir)})
    json.dump(nums, open(f"{out_dir}/variants.json", "w"), indent=1)
    print(f"wrote {out_dir}/tab_variants.tex, tab_variants_template.tex, variants.json")


def main():
    """Summarize the variant transfer runs: totals per config and paired tests between configs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/variants/results_logic.jsonl")
    ap.add_argument("--tex", help="directory to write paper tables into, e.g. paper/generated")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.results)]
    unit = defaultdict(dict)
    for r in rows:
        unit[(r["seed"], r["template"], r["scale"])][r["cfg"]] = r

    print("== totals on held-out variants ==")
    for scale in sorted({r["scale"] for r in rows}, reverse=True):
        for cfg in ["decisive", "rpos", "noise", "cross"]:
            rs = [r for r in rows if r["cfg"] == cfg and r["scale"] == scale]
            n = sum(r["test_n"] for r in rs)
            gb, ga = sum(r["test_greedy_base"] for r in rs), sum(r["test_greedy_after"] for r in rs)
            fx, br = sum(r["test_fixed"] for r in rs), sum(r["test_broke"] for r in rs)
            dr = np.mean([r["test_rate_after"] - r["test_rate_base"] for r in rs])
            print(f"x{scale} {cfg:8s} units={len(rs):2d} variants={n:3d} greedy {gb:3d}->{ga:3d} "
                  f"fixed {fx:3d} broke {br:2d} mean d_rate {dr:+.3f}")

    print("\n== paired tests (units = template x split with both configs) ==")
    for scale in sorted({r["scale"] for r in rows}, reverse=True):
        for a, b in [("decisive", "noise"), ("decisive", "cross"), ("rpos", "noise"), ("decisive", "rpos")]:
            for metric in ["net", "rate"]:
                diffs = []
                for (s, t, sc), cfgs in unit.items():
                    if sc != scale or a not in cfgs or b not in cfgs:
                        continue
                    f = (lambda r: r["test_fixed"] - r["test_broke"]) if metric == "net" else \
                        (lambda r: r["test_rate_after"] - r["test_rate_base"])
                    diffs.append(f(cfgs[a]) - f(cfgs[b]))
                wins = sum(d > 0 for d in diffs)
                losses = sum(d < 0 for d in diffs)
                print(f"x{scale} {a:8s} vs {b:8s} {metric:4s}: n={len(diffs):2d} mean diff {np.mean(diffs):+.3f} "
                      f"wins/losses {wins}/{losses} p={signflip_p(diffs):.4f}")

    print("\n== per template (x1.0, summed over splits): net greedy gain decisive / rpos / noise / cross ==")
    per = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["scale"] == 1.0:
            per[r["template"]][r["cfg"]] += r["test_fixed"] - r["test_broke"]
    for t in sorted(per):
        p = per[t]
        print(f"{t:16s} {p['decisive']:+4d} {p['rpos']:+4d} {p['noise']:+4d} {p['cross']:+4d}")

    print("\n== largest cross-template gains (x1.0) ==")
    cr = sorted((r for r in rows if r["cfg"] == "cross" and r["scale"] == 1.0),
                key=lambda r: -(r["test_fixed"] - r["test_broke"]))
    for r in cr[:6]:
        print(f"s{r['seed']} target {r['template']:16s} <- src {r['src']:16s} net {r['test_fixed'] - r['test_broke']:+d} "
              f"rate {r['test_rate_base']:.2f}->{r['test_rate_after']:.2f}")

    if args.tex:
        write_tex(rows, unit, args.tex, "runs/variants/screen.jsonl", "runs/variants")


if __name__ == "__main__":
    main()
