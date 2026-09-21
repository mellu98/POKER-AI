# The runtime contract

Training and serving share one feature encoder, one action menu, and one set of weights. That parity is
the single constraint everything else bent around, and the reason the strongest experiments never
shipped.

## The fixed shape

- **51 input features** from `bot/features.py`,
- **5 actions** (`FOLD`, `CHECK_CALL`, `BET_50`, `BET_POT`, `ALL_IN`), assumed by both training and runtime,
- **one `deep_cfr_model*.npz`** packaged into the submission.

A checkpoint that changes any of these cannot be served by the runtime as-is. Changing bet sizing or the
action count is a retrain-and-re-encode, not a config edit.

## Why the best numbers stayed parked

v21 emits 52 features, v23 emits 56. The runtime emits 51, so loading either weight file fails at
inference. Serving them meant retraining against the 51-feature contract from scratch; on a one-shot
submission, that is a different project, not an upgrade. So the +23.73 and +22.66 lines stayed on their
tags while +22.41 shipped.

v22 and v24 grew the action menu to 8. Both regressed, and they are worth keeping for the reason: more
policy heads on the same traversal and buffer budget means each head sees fewer samples and trains
worse. Action-space growth has to be paid for in sampling budget.

## Blueprint versus overlay

The finals branches added hand-written runtime logic on top of the learned policy. That is a separate
layer, and the repo keeps it labelled as one:

- **blueprint**: what the net learns from self-play (pure Deep CFR, no heuristics),
- **overlay**: hand-coded adaptation that runs at decision time, under the latency budget.

The short-stack push/fold chart in `bot/bot.py` is the only overlay that shipped. The rest stayed on
`archive/runtime-overlay` and the finals tags. Keeping the two layers separate is what keeps the
training loop honest: a learned-policy regression cannot hide behind a runtime patch.
