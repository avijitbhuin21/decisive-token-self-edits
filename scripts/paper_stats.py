"""Recompute every number in the paper from run artifacts and emit LaTeX tables + a JSON summary."""
import collections
import json
import math
import statistics
from pathlib import Path

RUNS = Path("runs")
OUT = Path("paper/generated")


def jl(p):
    """Read a JSONL file into a list of dicts."""
    return [json.loads(l) for l in open(p)]


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired booleans keyed by task id."""
    ids = set(a) & set(b)
    x = sum(a[i] and not b[i] for i in ids)
    y = sum(b[i] and not a[i] for i in ids)
    n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n)
    return x, y, p


def pilot():
    """Pilot: model-chosen neuron scaling vs random scaling vs resampling vs error feedback (first 100 tasks)."""
    base = {r["task_id"]: r for r in jl(RUNS / "prompt_v1/baselines_nothink_100.jsonl")}
    fails = [i for i, r in base.items() if not r["attempt1_visible"]]
    res = {"n_tasks": len(base), "n_fail": len(fails),
           "resample3": sum(base[i]["retry_hidden"] for i in fails),
           "feedback3": sum(base[i]["feedback_hidden"] for i in fails)}
    for mode in ["self", "random"]:
        ep = {r["task_id"]: r for r in jl(RUNS / f"prompt_v1/selfedit_{mode}_100_s0.jsonl")}
        res[f"edit_{mode}"] = sum(ep[i]["final_hidden"] for i in fails if i in ep)
    return res


def sampling():
    """Greedy pass rate and the fork-pair population."""
    rows = jl(RUNS / "samples.jsonl")
    pairs = [r for r in rows if not r["greedy"]["hidden"] and any(s["hidden"] for s in r["samples"])]
    rate = [sum(s["hidden"] for s in r["samples"]) / len(r["samples"]) for r in pairs]
    return {"n_tasks": len(rows), "greedy_pass": sum(r["greedy"]["hidden"] for r in rows),
            "greedy_fail_no_sample": sum(not r["greedy"]["hidden"] and not any(s["hidden"] for s in r["samples"]) for r in rows),
            "n_pairs": len(pairs), "pair_sample_rate_mean": statistics.mean(rate),
            "pair_sample_rate_median": statistics.median(rate)}, {r["task_id"]: r for r in rows}


def per_task(samples):
    """Per-task fix rates, single-edit fixes, stop reasons, McNemar and re-roll null for each fork variant."""
    variants = {"decisive": "", "rpos": "_rpos", "decisive_logic": "_logic", "rpos_logic": "_rpos_logic"}
    eps = {k: {r["task_id"]: r for r in jl(RUNS / f"forks/episodes_decisive_L16{v}.jsonl")} for k, v in variants.items()}
    out = {}
    for k, E in eps.items():
        nflip = {i: sum(e.get("flipped", False) for e in r["edits"]) for i, r in E.items()}
        null = sum(1 - (1 - sum(s["hidden"] for s in samples[i]["samples"]) / 10) ** max(1, nflip[i]) for i in E)
        stops = collections.Counter(e.get("stop", "flip" if e.get("flipped") else "noflip") for r in E.values() for e in r["edits"])
        out[k] = {"n": len(E), "fixed": sum(r["fixed"] for r in E.values()),
                  "fixed_1edit": sum(r["fixed"] and nflip[i] == 1 for i, r in E.items()),
                  "flips": sum(nflip.values()), "reroll_null": null, "stops": dict(stops)}
    for a, b in [("decisive", "rpos"), ("decisive_logic", "rpos_logic")]:
        fx = lambda E: {i: r["fixed"] for i, r in E.items()}
        one = lambda E: {i: r["fixed"] and sum(e.get("flipped", False) for e in r["edits"]) == 1 for i, r in E.items()}
        out[f"{a}_vs_{b}"] = {"any": mcnemar(fx(eps[a]), fx(eps[b])), "one": mcnemar(one(eps[a]), one(eps[b]))}
    return out


def heldout():
    """Held-out fixed/broke/net per (variant, cfg, scale) across split seeds."""
    rows = jl(RUNS / "heldout/results.jsonl") + jl(RUNS / "heldout_logic/results.jsonl")
    g = collections.defaultdict(list)
    for r in rows:
        g[(r.get("variant", ""), r["cfg"], r["scale"])].append(r)
    out = {}
    for (v, c, s), rs in sorted(g.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        net = [r["test_fixed"] - r["test_broke"] for r in rs]
        out[f"{v or 'all'}|{c}|{s}"] = {
            "seeds": [r["seed"] for r in rs], "net": net, "fixed": [r["test_fixed"] for r in rs],
            "broke": [r["test_broke"] for r in rs], "mean_net": statistics.mean(net),
            "sd_net": statistics.stdev(net) if len(net) > 1 else None,
            "targets": [f"{r['train_target_fixed']}/{r['n_edits_tasks']}" for r in rs],
            "test_base_fail": rs[0]["test_base_fail"], "test_base_pass": rs[0]["test_base_pass"]}
    base = {int(k): v for k, v in json.load(open(RUNS / "heldout/base.json")).items()}
    rec = {r["task_id"]: r["greedy"]["hidden"] for r in jl(RUNS / "samples.jsonl")}
    out["batch_flip"] = sum(base[i] != rec[i] for i in base)
    return out


def tex_tables(S):
    """Write the three LaTeX tables used by main.tex."""
    p, t, h = S["pilot"], S["per_task"], S["heldout"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tab_pilot.tex").write_text(
        "\\begin{tabular}{lc}\n\\toprule\nMethod (3 attempts) & Fixed / %d failures \\\\\n\\midrule\n" % p["n_fail"]
        + "Resample ($T{=}0.7$) & %d \\\\\nError feedback in context & %d \\\\\n" % (p["resample3"], p["feedback3"])
        + "Model-chosen neuron scaling & %d \\\\\nRandom neuron scaling & %d \\\\\n" % (p["edit_self"], p["edit_random"])
        + "\\bottomrule\n\\end{tabular}\n")
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Edit location & Fixed / %d & Fixed by 1 edit & Re-roll null & McNemar $p$ \\\\" % t["decisive"]["n"], "\\midrule"]
    for a, b, lab in [("decisive", "rpos", "all tokens"), ("decisive_logic", "rpos_logic", "code tokens only")]:
        m = t[f"{a}_vs_{b}"]
        lines.append("\\multicolumn{5}{l}{\\emph{%s}} \\\\" % lab)
        lines.append("\\quad Decisive token & %d & %d & %.1f & %.2f \\\\" % (t[a]["fixed"], t[a]["fixed_1edit"], t[a]["reroll_null"], m["any"][2]))
        lines.append("\\quad Random position & %d & %d & %.1f & -- \\\\" % (t[b]["fixed"], t[b]["fixed_1edit"], t[b]["reroll_null"]))
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_pertask.tex").write_text("\n".join(lines) + "\n")
    fmt = lambda k: "--" if k not in h else " / ".join("%+d" % x for x in h[k]["net"])
    fb = lambda k: "--" if k not in h else " / ".join("%d/%d" % fb_ for fb_ in zip(h[k]["fixed"], h[k]["broke"]))
    lines = ["\\begin{tabular}{llccc}", "\\toprule",
             "Tokens & Edit & $\\times1.0$ fixed/broke & $\\times0.25$ net (seeds 0/1/2) & $\\times0.5$ net \\\\", "\\midrule"]
    for v, lab in [("all", "all"), ("_logic", "code only")]:
        for c, cl in [("decisive", "Decisive"), ("rpos", "Random pos."), ("noise", "Noise")]:
            lines.append("%s & %s & %s & %s & %s \\\\" % (lab, cl, fb(f"{v}|{c}|1.0"), fmt(f"{v}|{c}|0.25"), fmt(f"{v}|{c}|0.5")))
        lines.append("\\midrule" if v == "all" else "\\bottomrule")
    lines.append("\\end{tabular}")
    (OUT / "tab_heldout.tex").write_text("\n".join(lines) + "\n")


def main():
    """Compute all statistics and write paper/generated/{numbers.json,*.tex}."""
    samp, samples = sampling()
    S = {"pilot": pilot(), "sampling": samp, "per_task": per_task(samples), "heldout": heldout()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.json").write_text(json.dumps(S, indent=1))
    tex_tables(S)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
