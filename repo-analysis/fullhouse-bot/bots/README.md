# Opponent pool

Every benchmark number in this project was measured against these bots, not against self-play. Self-play
scores don't transfer, so the harness ([`tooling/harness/bot_registry.py`](../tooling/harness/bot_registry.py))
scores each checkpoint against a fixed, varied pool and reports bb/100 with 95% normal-approximation CIs
(±1.96 × stderr). The submitted agent is in [`bot/`](../bot); these are only opponents.

## Standard pool (scored by default)

- **`competitors/`**: the strongest synthetic opponents. Equity-table players, a Bayesian exploiter
  that profiles opponents from the action log, a CFR-equity bot, a k-means router, and `solver_hybrid`.
- **`adversarial/`**: archetypes that each probe one leak rather than play well overall. `nit` (too
  tight), `station` (calls too much), `maniac` (over-aggressive), `solver_like`, `tournament_specialist`,
  and `random_bot` as a floor. A model can post a good average and still bleed against one archetype;
  these surface that.
- **`strong/`**: a single strong baseline reference.

## Held-out pool (opt-in, `--include-extra`)

`extra/` is never scored by default and was never used to tune the model: `mc_equity`,
`cfr_kmeans_router`, `cfr_opp_profiler`, `cfr_variance_guard`. It is the generalization gate. Requiring a
non-negative result here is what caught lines that beat the standard pool but overfit to it: the +22.41
standard against +19.89 held-out split on the shipped model is the gap it measures.

## Solver-derived data

`competitors/solver_hybrid/data/` is generated from `b-inary/postflop-solver` and `wasm-postflop` (see
`solver_hybrid/data/OPEN_SOURCE_NOTES.txt` for attribution). It is a benchmark opponent, not part of the
submission, and the digests are aggregated strategy data, not solver source.
