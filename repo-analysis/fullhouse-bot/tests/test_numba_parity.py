"""Parity tests for the Python and Numba training paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training import game as game_py
from training import game_numba as game_nb
from training._hand_eval_lut import evaluate_lut, load_or_generate
from training.features import FEATURE_DIM, load_preflop_equity
from training.game import N_ACTIONS
from training.traverse_numba import _strategy

pytestmark = pytest.mark.training

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _lut_tables():
    return load_or_generate(DATA_DIR / "hand_eval_lut.npz")


def _preflop_tables():
    return load_preflop_equity()


def _step_numba(
    state: np.ndarray,
    deck: np.ndarray,
    hole_cards: np.ndarray,
    action: int,
    lut_tables: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    flush_lut, nf_keys, nf_vals = lut_tables
    game_nb.step(state, deck, hole_cards, action, flush_lut, nf_keys, nf_vals)


def _assert_behavioral_alignment(state_nb: np.ndarray, state_py: np.ndarray) -> None:
    n_players = int(state_py[game_py.S_NPLAYERS])

    assert state_nb[game_nb.S_TOACT] == state_py[game_py.S_TOACT]
    assert state_nb[game_nb.S_STREET] == state_py[game_py.S_STREET]
    assert state_nb[game_nb.S_TERMINAL] == state_py[game_py.S_TERMINAL]
    assert state_nb[game_nb.S_NRAISES] == state_py[game_py.S_NRAISES]
    assert state_nb[game_nb.S_LASTRAISER] == state_py[game_py.S_LASTRAISER]
    assert state_nb[game_nb.S_DECKIDX] == state_py[game_py.S_DECKIDX]
    assert state_nb[game_nb.S_DEALER] == state_py[game_py.S_DEALER]

    np.testing.assert_array_equal(
        state_nb[game_nb.S_FOLDED : game_nb.S_FOLDED + n_players],
        state_py[game_py.S_FOLDED : game_py.S_FOLDED + n_players],
    )
    np.testing.assert_array_equal(
        state_nb[game_nb.S_ALLIN : game_nb.S_ALLIN + n_players],
        state_py[game_py.S_ALLIN : game_py.S_ALLIN + n_players],
    )
    np.testing.assert_array_equal(
        state_nb[game_nb.S_PENDING : game_nb.S_PENDING + n_players],
        state_py[game_py.S_PENDING : game_py.S_PENDING + n_players],
    )

    total_nb = int(state_nb[game_nb.S_POT]) + int(
        state_nb[game_nb.S_STACKS : game_nb.S_STACKS + n_players].sum()
    )
    total_py = int(state_py[game_py.S_POT]) + int(
        state_py[game_py.S_STACKS : game_py.S_STACKS + n_players].sum()
    )
    assert total_nb == total_py


def test_card_conversion() -> None:
    """Game encoding should map cleanly into evaluator encoding."""
    for game_card in range(52):
        suit = game_card // 13
        rank = game_card % 13
        eval_card = game_nb._game_to_eval_card(np.int8(game_card))
        assert eval_card == rank * 4 + suit


def test_hand_evaluation_parity() -> None:
    """The LUT evaluator should preserve eval7 hand ordering."""
    eval7 = pytest.importorskip("eval7")
    flush_lut, nf_keys, nf_vals = _lut_tables()

    card_strs = [
        "23456789TJQKA"[rank] + "cdhs"[suit] for suit in range(4) for rank in range(13)
    ]
    rng = np.random.default_rng(42)

    hands: list[tuple[int, int]] = []
    for _ in range(200):
        cards = rng.choice(52, 7, replace=False)
        eval_cards = np.array([(c % 13) * 4 + (c // 13) for c in cards], dtype=np.int8)
        eval7_cards = [eval7.Card(card_strs[c]) for c in cards]
        hands.append(
            (
                eval7.evaluate(eval7_cards),
                evaluate_lut(eval_cards, 7, flush_lut, nf_keys, nf_vals),
            )
        )

    misordered_pairs = 0
    for i in range(len(hands)):
        for j in range(i + 1, min(i + 20, len(hands))):
            eval7_cmp = (hands[i][0] > hands[j][0]) - (hands[i][0] < hands[j][0])
            lut_cmp = (hands[i][1] > hands[j][1]) - (hands[i][1] < hands[j][1])
            if eval7_cmp != lut_cmp:
                misordered_pairs += 1

    assert misordered_pairs == 0


@pytest.mark.slow
def test_game_engine_state_parity() -> None:
    """Python and Numba engines should stay behaviorally aligned during play."""
    lut_tables = _lut_tables()
    rng = np.random.default_rng(123)

    for _ in range(50):
        n_players = 6
        dealer = int(rng.integers(0, n_players))
        state_py, deck, hole_cards = game_py.new_hand(n_players, dealer, rng)
        state_nb = state_py.copy()

        for seat in range(n_players):
            np.testing.assert_array_equal(
                game_py.legal_actions(state_py, seat),
                game_nb.legal_actions(state_nb, seat),
            )

        for _ in range(50):
            _assert_behavioral_alignment(state_nb, state_py)
            if state_py[game_py.S_TERMINAL] == 1:
                break

            seat = int(state_py[game_py.S_TOACT])
            legal_py = game_py.legal_actions(state_py, seat)
            legal_nb = game_nb.legal_actions(state_nb, seat)
            np.testing.assert_array_equal(legal_nb, legal_py)

            actions = np.flatnonzero(legal_py)
            if len(actions) == 0:
                break
            action = int(rng.choice(actions))
            game_py.step(state_py, deck, hole_cards, action)
            _step_numba(state_nb, deck, hole_cards, action, lut_tables)

        _assert_behavioral_alignment(state_nb, state_py)


def test_short_all_in_does_not_reopen_action() -> None:
    lut_tables = _lut_tables()

    def build_state(module):
        state = np.zeros(module.STATE_SIZE, dtype=np.int32)
        state[module.S_NPLAYERS] = 3
        state[module.S_DEALER] = 0
        state[module.S_TOACT] = 2
        state[module.S_CURBET] = 1000
        state[module.S_LAST_FULL_RAISE] = 800
        state[module.S_NRAISES] = 2
        state[module.S_LASTRAISER] = 0
        state[module.S_POT] = 2900
        for seat, stack, bet in [(0, 5000, 1000), (1, 5000, 1000), (2, 300, 900)]:
            state[module.S_STACKS + seat] = stack
            state[module.S_INITIAL_STACKS + seat] = stack + bet
            state[module.S_BETS + seat] = bet
            state[module.S_INVESTED + seat] = bet
        state[module.S_PENDING + 1] = 1
        state[module.S_PENDING + 2] = 1
        return state

    deck = np.arange(52, dtype=np.int8)
    hole_cards = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int8)

    state_py = build_state(game_py)
    game_py.step(state_py, deck, hole_cards, game_py.ALL_IN)

    state_nb = build_state(game_nb)
    _step_numba(state_nb, deck, hole_cards, game_nb.ALL_IN, lut_tables)

    assert state_py[game_py.S_NRAISES] == 2
    assert state_py[game_py.S_LASTRAISER] == 0
    assert state_py[game_py.S_PENDING + 0] == 0
    assert state_py[game_py.S_PENDING + 1] == 1
    np.testing.assert_array_equal(state_nb, state_py)


def test_raises_remain_legal_after_four_full_raises() -> None:
    def build_state(module):
        state = np.zeros(module.STATE_SIZE, dtype=np.int32)
        state[module.S_NPLAYERS] = 3
        state[module.S_TOACT] = 1
        state[module.S_POT] = 600
        state[module.S_CURBET] = 300
        state[module.S_NRAISES] = 4
        state[module.S_LAST_FULL_RAISE] = 100
        state[module.S_STACKS + 0] = 3000
        state[module.S_STACKS + 1] = 2000
        state[module.S_STACKS + 2] = 2500
        state[module.S_BETS + 0] = 300
        return state

    state_py = build_state(game_py)
    legal_py = game_py.legal_actions(state_py, 1)
    state_nb = build_state(game_nb)
    legal_nb = game_nb.legal_actions(state_nb, 1)

    assert legal_py[game_py.BET_50]
    assert legal_py[game_py.BET_POT]
    np.testing.assert_array_equal(legal_nb, legal_py)


def test_single_remaining_actor_keeps_acting_across_streets() -> None:
    lut_tables = _lut_tables()

    def build_state(module):
        state = np.zeros(module.STATE_SIZE, dtype=np.int32)
        state[module.S_NPLAYERS] = 2
        state[module.S_DEALER] = 1
        state[module.S_TOACT] = 0
        state[module.S_POT] = 2000
        state[module.S_LAST_FULL_RAISE] = module.BIG_BLIND
        state[module.S_STACKS + 0] = 5000
        state[module.S_STACKS + 1] = 0
        state[module.S_INITIAL_STACKS + 0] = 6000
        state[module.S_INITIAL_STACKS + 1] = 1000
        state[module.S_INVESTED + 0] = 1000
        state[module.S_INVESTED + 1] = 1000
        state[module.S_ALLIN + 1] = 1
        state[module.S_PENDING + 0] = 1
        return state

    def assert_progression(module, state, step_fn):
        deck = np.arange(52, dtype=np.int8)
        hole_cards = np.array([[0, 1], [2, 3]], dtype=np.int8)

        for expected_street in (1, 2, 3):
            legal = module.legal_actions(state, 0)
            assert legal[module.CHECK_CALL]
            step_fn(state, deck, hole_cards, module.CHECK_CALL)
            assert state[module.S_TERMINAL] == 0
            assert state[module.S_STREET] == expected_street
            assert state[module.S_TOACT] == 0
            assert state[module.S_PENDING + 0] == 1
            assert state[module.S_PENDING + 1] == 0

        step_fn(state, deck, hole_cards, module.CHECK_CALL)
        assert state[module.S_TERMINAL] == 1
        assert state[module.S_STREET] == 3

    state_py = build_state(game_py)
    assert_progression(game_py, state_py, game_py.step)

    state_nb = build_state(game_nb)
    assert_progression(
        game_nb,
        state_nb,
        lambda state, deck, hole_cards, action: _step_numba(
            state, deck, hole_cards, action, lut_tables
        ),
    )
    np.testing.assert_array_equal(state_nb, state_py)


@pytest.mark.slow
def test_feature_encoding_parity() -> None:
    """Python and Numba feature encoders should agree on reachable states."""
    pytest.importorskip("torch")
    from training.traverse_fast import _encode_fast as encode_py
    from training.traverse_numba import _encode_fast as encode_nb

    lut_tables = _lut_tables()
    flush_lut, nf_keys, nf_vals = lut_tables
    eq_paired, eq_suited, eq_offsuit = _preflop_tables()

    rng = np.random.default_rng(456)
    tolerance = 1e-6
    hand_strength_tolerance = 1e-3

    for _ in range(200):
        n_players = 6
        dealer = int(rng.integers(0, n_players))
        state, deck, hole_cards = game_py.new_hand(n_players, dealer, rng)

        for _ in range(int(rng.integers(0, 15))):
            if state[game_py.S_TERMINAL] == 1:
                break
            seat = int(state[game_py.S_TOACT])
            legal = game_py.legal_actions(state, seat)
            actions = np.flatnonzero(legal)
            if len(actions) == 0:
                break
            game_py.step(state, deck, hole_cards, int(rng.choice(actions)))

        if state[game_py.S_TERMINAL] == 1:
            continue

        for seat in range(n_players):
            if state[game_py.S_FOLDED + seat] != 0:
                continue
            legal = game_py.legal_actions(state, seat)
            feat_py = encode_py(state, hole_cards, deck, seat, legal)
            feat_nb = encode_nb(
                state,
                hole_cards,
                deck,
                seat,
                legal,
                flush_lut,
                nf_keys,
                nf_vals,
                eq_paired,
                eq_suited,
                eq_offsuit,
            )

            assert feat_py.shape[0] == feat_nb.shape[0]
            tolerances = np.full(FEATURE_DIM, tolerance, dtype=np.float32)
            tolerances[1] = hand_strength_tolerance
            diffs = np.abs(feat_nb - feat_py)
            assert np.all(diffs <= tolerances), (
                f"feature parity mismatch at seat={seat}: "
                f"max_diff={float(diffs.max()):.6g} "
                f"max_tol={float(tolerances.max()):.6g}"
            )


@pytest.mark.slow
def test_forward_pass_parity() -> None:
    """Python and Numba policy forward passes should agree."""
    pytest.importorskip("torch")
    from training.network import DuelingAdvantageNet
    from training.traverse_fast import NumpyPolicy

    net = DuelingAdvantageNet(FEATURE_DIM, 256, 128)
    policy = NumpyPolicy(net)
    rng = np.random.default_rng(789)

    for _ in range(100):
        features = rng.random(FEATURE_DIM).astype(np.float32)
        legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
        legal_idx = rng.choice(N_ACTIONS, rng.integers(1, N_ACTIONS + 1), replace=False)
        legal_mask[legal_idx] = 1.0

        sigma_py = policy.strategy(features.copy(), legal_mask.copy())
        sigma_nb = _strategy(
            features.copy(),
            legal_mask.copy(),
            policy.trunk_w0,
            policy.trunk_b0,
            policy.trunk_ln0_g,
            policy.trunk_ln0_b,
            policy.trunk_w1,
            policy.trunk_b1,
            policy.trunk_ln1_g,
            policy.trunk_ln1_b,
            policy.val_w0,
            policy.val_b0,
            policy.val_w1,
            policy.val_b1,
            policy.adv_w0,
            policy.adv_b0,
            policy.adv_w1,
            policy.adv_b1,
        )

        np.testing.assert_allclose(sigma_nb, sigma_py, atol=1e-5, rtol=0.0)
