# selfedit

Paper: **Finding the Decisive Token Is Not Enough: Rank-One Self-Edits Repair Code Locally but Do Not Transfer** — Avijit Bhuin ([`paper/main.pdf`](paper/main.pdf)).

Regenerate every number and table in the paper from the logs in `runs/`, then build:
```
uv run python scripts/paper_stats.py   # writes paper/generated/
cd paper && tectonic -X compile main.tex
```

Can a small LLM fix its own failures by directly editing its own weights (no gradients), and does that beat simply retrying?

Model: `mlx-community/Qwen3.5-0.8B-bf16` (MLX, Apple Silicon). Tasks: MBPP+ — the model sees the 3 original asserts ("visible"); scoring uses the hidden MBPP+ suite. Task 590 is excluded (its reference fails its own hidden suite).

## Loop (`scripts/selfedit_loop.py`)
1. Greedy attempt. If visible tests pass, done.
2. Otherwise capture MLP neuron activity on the failing reply (layers 16-23, top 4 per layer), label each neuron with logit-lens tokens it promotes/suppresses.
3. The model itself proposes 1-5 neuron scalings (0 / 0.5 / 2 / 3) as JSON.
4. Apply edit, regenerate greedily, re-check visible tests. Keep on pass, else roll back and retry (max 3 rounds).
5. Weights are restored after every task (episodic edits).

`--mode random` is the control: same candidates and schedule, random neurons and scales.

## Baselines (`scripts/baselines.py`)
Greedy attempt; resample x3 (temp 0.7); retry x3 with the test error in context.

## Run
```
source /Volumes/AcasisSSDDrive/tools/env.sh
export PYTHONPATH=.
uv run python scripts/baselines.py --limit 100
uv run python scripts/selfedit_loop.py --mode self --limit 100
uv run python scripts/selfedit_loop.py --mode random --limit 100
uv run python scripts/analyze.py --limit 100
```
Sanity checks: `scripts/check_tasks.py`, `check_model.py`, `check_edits.py`, `check_edit_effect.py`.
## Fork experiment (`scripts/fork_experiment.py`, `scripts/heldout.py`)
Code-only prompt; `scripts/sample.py` records greedy + 10 samples per task. Greedy passes 130/377; 91 "pairs" (greedy fails, some sample passes).

Decisive token: force a passing sample's prefix, continue greedily, binary-search for the fail→pass flip. A rank-1 `down_proj` edit (layer 16, direction = unembed difference × final norm) flips that token. Controls: `--random-pos` (same edit at a random disagreement position), noise (norm-matched random rank-1). `--logic` strips comments/docstrings from the passing sample and accepts only code-token flips.

| | Per-task fixed (91) | Held-out net, ×0.25, seeds 0/1/2 | Held-out net, ×0.5 |
|---|---|---|---|
| Decisive | 50 | +6 / +7 / −3 | – |
| Random-position | 47 | −5 / +1 / +5 | – |
| Decisive, logic-only | 49 | −3 / +13 / +1 | −10 / −13 / −21 |
| Random-position, logic-only | 47 | −7 / +8 / +8 | −15 / −5 / −12 |
| Noise | – | ≈0 | – |

Held-out = sum the fixing edits from a random half of tasks, score greedy on the other half (66 pass / 123 fail at baseline). At ×1.0, edits break 42–47 of 66 passing tasks. Batched greedy alone flips 14/377 tasks.

Conclusion: locating the decisive token doesn't beat random positions, and the edits don't transfer to new tasks. Per-task fixes mostly re-roll the trajectory.

```
uv run python scripts/sample.py
uv run python scripts/fork_experiment.py [--random-pos] [--logic] --out runs/forks
uv run python scripts/heldout.py [--variant _logic] --scale 0.25 --out runs/heldout
```