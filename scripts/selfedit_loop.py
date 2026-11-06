import argparse
import json
import os
import random
import re
import time

import mlx.core as mx

from selfedit.edits import Editor, NeuronEdit
from selfedit.model import Model
from selfedit.tasks import extract_code, load_tasks, run_hidden, run_visible

LAYERS = list(range(16, 24))
PER_LAYER = 4
SCALES = [0.0, 0.5, 2.0, 3.0]

PROPOSE = """You are editing your own neural network to fix a mistake you made.

TASK:
{task}

YOUR FAILING CODE:
```python
{code}
```
TEST ERROR:
{error}

These are neurons inside you that were most active while you wrote that code. "promotes" are tokens the neuron pushes you to write; "suppresses" are tokens it pushes you away from.
{table}
{tried}
Choose 1 to 5 neurons to change so that you would write correct code. Scale 0 turns a neuron off, 0.5 weakens it, 2 or 3 strengthens it.
Reply with only JSON: [{{"id": <id>, "scale": <number>}}, ...]"""


def candidates(ed: Editor, prompt: str, reply: str) -> list[dict]:
    """Pick the most active late-layer neurons on the failing reply and label them."""
    act = ed.neuron_activity(prompt, reply, LAYERS)
    out = []
    for L in LAYERS:
        for n in mx.argsort(act[L])[-PER_LAYER:][::-1].tolist():
            pro, sup = ed.neuron_tokens(L, n)
            out.append({"id": len(out), "layer": L, "neuron": n,
                        "act": round(act[L][n].item(), 2), "promotes": pro, "suppresses": sup})
    return out


def render_table(cands: list[dict]) -> str:
    """Format candidate neurons as compact text lines for the proposer prompt."""
    return "\n".join(
        f"id={c['id']} layer={c['layer']} activity={c['act']} promotes={c['promotes']} suppresses={c['suppresses']}"
        for c in cands
    )


def parse_edits(text: str, cands: list[dict]) -> list[NeuronEdit] | None:
    """Parse the model's JSON edit list, clamping scales and dropping invalid ids."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return None
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    edits = []
    for it in items[:5]:
        try:
            c = cands[int(it["id"])]
            s = min(max(float(it["scale"]), 0.0), 3.0)
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        edits.append(NeuronEdit(c["layer"], c["neuron"], s))
    return edits or None


def random_edits(cands: list[dict], rng: random.Random) -> list[NeuronEdit]:
    """Control condition: random neurons from the same candidate set with random scales."""
    picks = rng.sample(cands, rng.randint(1, 5))
    return [NeuronEdit(c["layer"], c["neuron"], rng.choice(SCALES)) for c in picks]


def episode(m: Model, ed: Editor, task, mode: str, rounds: int, rng: random.Random) -> dict:
    """Run one fail-then-self-edit episode on a task and return its log row."""
    msgs = [{"role": "user", "content": task.user_message()}]
    prompt = m.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                       enable_thinking=m.thinking)
    reply = m.chat(msgs)
    code = extract_code(reply)
    vis = run_visible(task, code)
    row = {"task_id": task.task_id, "mode": mode, "attempt1_visible": vis.passed, "rounds": []}
    if vis.passed:
        row["final_hidden"] = run_hidden(task, code).passed
        return row
    cands = candidates(ed, prompt, reply)
    tried = []
    final_code = code
    for r in range(rounds):
        parse_ok = True
        if mode == "self":
            tried_txt = ("\nThese edits were already tried and did NOT fix it:\n" + "\n".join(tried) + "\n") if tried else ""
            ask = PROPOSE.format(task=task.user_message(), code=code, error=vis.error,
                                 table=render_table(cands), tried=tried_txt)
            edits = parse_edits(m.chat([{"role": "user", "content": ask}], temp=0.7, max_tokens=300), cands)
            if edits is None:
                parse_ok = False
                edits = random_edits(cands, rng)
        else:
            edits = random_edits(cands, rng)
        ed.apply(edits)
        new_code = extract_code(m.chat(msgs))
        nv = run_visible(task, new_code)
        log = {"edits": [e.__dict__ for e in edits], "parse_ok": parse_ok,
               "changed": new_code != code, "visible": nv.passed}
        row["rounds"].append(log)
        final_code = new_code
        if nv.passed:
            row["final_hidden"] = run_hidden(task, new_code).passed
            row["accepted_edits"] = log["edits"]
            ed.rollback()
            return row
        tried.append(json.dumps([{"layer": e.layer, "neuron": e.neuron, "scale": e.scale} for e in edits]))
        ed.rollback()
    row["final_hidden"] = run_hidden(task, final_code).passed
    return row


def main():
    """Run the self-edit loop (or random-edit control) over tasks and write JSONL results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["self", "random"], default="self")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    m = Model()
    ed = Editor(m.model, m.tok)
    tasks = load_tasks(args.limit)
    path = os.path.join(args.out, f"selfedit_{args.mode}_{args.limit or 'all'}_s{args.seed}.jsonl")
    n_fail = fixed_vis = solved = 0
    t0 = time.time()
    with open(path, "w") as f:
        for i, task in enumerate(tasks):
            row = episode(m, ed, task, args.mode, args.rounds, rng)
            f.write(json.dumps(row) + "\n")
            f.flush()
            solved += row["final_hidden"]
            if not row["attempt1_visible"]:
                n_fail += 1
                fixed_vis += any(r["visible"] for r in row["rounds"])
            print(f"[{i + 1}/{len(tasks)} {time.time() - t0:.0f}s] hidden={solved / (i + 1):.3f} "
                  f"failed1={n_fail} fixed_by_edit={fixed_vis}", flush=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
