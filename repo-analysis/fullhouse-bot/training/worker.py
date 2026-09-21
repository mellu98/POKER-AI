"""Multiprocessing wrapper around the Python traversal implementation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from bot.features import FEATURE_DIM, N_ACTIONS
from . import traverse_fast
from .game import new_hand
from .traverse_fast import NumpyPolicy


def _policy_from_arrays(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
) -> NumpyPolicy:
    policy = NumpyPolicy.__new__(NumpyPolicy)
    (
        policy.trunk_w0,
        policy.trunk_ln0_g,
        policy.trunk_w1,
        policy.trunk_ln1_g,
        policy.val_w0,
        policy.val_w1,
        policy.adv_w0,
        policy.adv_w1,
    ) = weights
    (
        policy.trunk_b0,
        policy.trunk_ln0_b,
        policy.trunk_b1,
        policy.trunk_ln1_b,
        policy.val_b0,
        policy.val_b1,
        policy.adv_b0,
        policy.adv_b1,
    ) = biases
    return policy


def worker_run(
    args: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run traversal batches in a subprocess."""
    worker_id = int(args["worker_id"])
    n_traversals = int(args["n_traversals"])
    traverser = int(args["traverser"])
    n_players = int(args["n_players"])
    weights = args["weights"]
    biases = args["biases"]
    equity_tables = args["equity_tables"]
    opponent_exploration = float(args["opponent_exploration"])
    randomize_stacks = bool(args.get("randomize_stacks", True))

    if equity_tables is not None:
        traverse_fast._EQUITY_TABLES = equity_tables

    policy = _policy_from_arrays(weights, biases)
    rng = np.random.default_rng(worker_id)
    feat_list: list[np.ndarray] = []
    adv_list: list[np.ndarray] = []
    strat_feat_list: list[np.ndarray] = []
    strat_list: list[np.ndarray] = []
    strat_weight_list: list[np.ndarray] = []

    for _ in range(n_traversals):
        dealer = int(rng.integers(0, n_players))
        stacks = (
            traverse_fast.sample_match_stacks(rng, n_players)
            if randomize_stacks
            else None
        )
        state, deck, hole_cards = new_hand(n_players, dealer, rng, stacks)
        traverse_fast.traverse_external(
            state,
            deck,
            hole_cards,
            traverser,
            policy,
            feat_list,
            adv_list,
            strat_feat_list,
            strat_list,
            strat_weight_list,
            rng,
            opponent_exploration,
        )

    if feat_list:
        return (
            np.asarray(feat_list, dtype=np.float32),
            np.asarray(adv_list, dtype=np.float32),
            np.asarray(strat_feat_list, dtype=np.float32),
            np.asarray(strat_list, dtype=np.float32),
            np.asarray(strat_weight_list, dtype=np.float32),
        )
    return (
        np.zeros((0, FEATURE_DIM), dtype=np.float32),
        np.zeros((0, N_ACTIONS), dtype=np.float32),
        np.zeros((0, FEATURE_DIM), dtype=np.float32),
        np.zeros((0, N_ACTIONS), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )
