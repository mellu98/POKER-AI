# Experiment ledger

Every line changed one variable against the v17 baseline, then had to clear the promotion gate from the
[README](../../README.md#results): a CI-clear gain on the standard pool, non-negative on the held-out
pool, no losing segment, and an exact match to the 51-feature / 5-action runtime contract. Numbers are
best-checkpoint bb/100 from the local harness with 95% confidence intervals.

## Representation and action-space lines

| Line | Change | Best bb/100 | Verdict |
|---|---|---|---|
| v17 | 51-feature / 5-action baseline | +22.41 (held-out +19.89) | Shipped |
| v18 | strength-percentile feature | +12.39 | Refuted: clean run, no gain |
| v19 | opponent-mixture sampling | no durable gain | Refuted |
| v20 | paired evaluator + exploitability probe | improved measurement, not policy | Tooling, not a promotion |
| v21 | multiway-equity feature (52-dim) | +22.66 | Parked: feature-shape mismatch |
| v22 | 8-action abstraction | +10.48 | Refuted: policy heads starved |
| v23 | side-pot representation (56-dim) | +23.73 | Parked: strongest, same mismatch |
| v24 | side-pot + 8-action | +12.49 best, -0.64 final | Refuted: negative overall |

## Later lines (archived, not promoted)

| Line | Change | Why it exists | Tag |
|---|---|---|---|
| v25 | mixed {2,3,4,5,6}-player training, 51 features | cover short-handed play without changing runtime shape | `archive/v25` |
| v30 | six-action pressure encoding + Modal launcher | a bigger structural bet | `archive/v30-moonshot` |
| v31 | six-action 6-max follow-on | continue the action-space search | `archive/v31-6act-6max` |
| finals | v29 model + board-aware runtime overlay | last submission line before finals | `archive/v29-r4-finals` |

## What the table doesn't show

Three problem classes came up repeatedly, and they did not mix in one branch:

1. **representation quality** (v21, v23): the only changes that beat baseline,
2. **table-size coverage** (v25): short-handed play mattered more than the early 6-max-only runs assumed,
3. **runtime adaptation** under the 2-second budget (the finals overlays).

The most expensive habit was changing more than one of these at once. When a branch carried a new
feature *and* a new action menu *and* new runtime logic, neither a good nor a bad result could be
attributed to anything; the experiment taught nothing, even when the code was worth keeping. The runs
that moved the project were the boring ones that changed exactly one variable.
