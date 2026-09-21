"""Parity checks for the current training/runtime feature encoders."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from bot.features import (
    FEATURE_DIM,
    _compute_legal_mask,
    _hand_invested_from_log,
    encode_state,
    encode_state_dict,
)
from training.game import (
    ALL_IN,
    BET_50,
    BET_POT,
    BIG_BLIND,
    CHECK_CALL,
    FOLD,
    S_ALLIN,
    S_BETS,
    S_CURBET,
    S_DEALER,
    S_FOLDED,
    S_LAST_FULL_RAISE,
    S_NRAISES,
    S_NPLAYERS,
    S_POT,
    S_STACKS,
    S_STREET,
    S_TERMINAL,
    S_TOACT,
    legal_actions,
    new_hand,
    step,
)

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_NON_PARITY_INDICES = set()
_PARITY_INDICES = [i for i in range(FEATURE_DIM) if i not in _NON_PARITY_INDICES]


def _card_to_idx(card: str) -> int:
    return _SUITS.index(card[1]) * 13 + _RANKS.index(card[0])


def _idx_to_card(idx: int) -> str:
    return _RANKS[idx % 13] + _SUITS[idx // 13]


def _state_to_engine_dict(
    state: np.ndarray,
    deck: np.ndarray,
    hole_cards: np.ndarray,
    action_log: list[dict],
    include_dealer: bool = True,
) -> dict:
    n_players = int(state[S_NPLAYERS])
    seat = int(state[S_TOACT])
    street_idx = int(state[S_STREET])
    street_name = ("preflop", "flop", "turn", "river")[street_idx]
    board_start = 2 * n_players
    board_count = (0, 3, 4, 5)[street_idx]

    players = []
    for player_seat in range(n_players):
        players.append(
            {
                "seat": player_seat,
                "stack": int(state[S_STACKS + player_seat]),
                "bet_this_street": int(state[S_BETS + player_seat]),
                "is_folded": bool(state[S_FOLDED + player_seat]),
                "is_all_in": bool(state[S_ALLIN + player_seat]),
            }
        )

    current_bet = int(state[S_CURBET])
    your_bet = int(state[S_BETS + seat])
    amount_owed = max(0, current_bet - your_bet)
    min_raise_to = (
        current_bet + max(int(state[S_LAST_FULL_RAISE]), BIG_BLIND)
        if int(state[S_STACKS + seat]) > amount_owed
        else 0
    )

    state_dict = {
        "your_cards": [
            _idx_to_card(int(hole_cards[seat, 0])),
            _idx_to_card(int(hole_cards[seat, 1])),
        ],
        "community_cards": [
            _idx_to_card(int(deck[board_start + offset]))
            for offset in range(board_count)
        ],
        "players": players,
        "pot": int(state[S_POT]),
        "seat_to_act": seat,
        "street": street_name,
        "can_check": amount_owed == 0,
        "amount_owed": amount_owed,
        "min_raise_to": min_raise_to,
        "current_bet": current_bet,
        "your_stack": int(state[S_STACKS + seat]),
        "your_bet_this_street": your_bet,
        "action_log": action_log,
    }
    if include_dealer:
        state_dict["dealer"] = int(state[S_DEALER])
    return state_dict


class _ActionLogBuilder:
    def __init__(self, n_players: int, dealer: int, include_deal_markers: bool = False):
        self.log: list[dict] = []
        self.include_deal_markers = include_deal_markers
        if n_players == 2:
            sb_seat = dealer
            bb_seat = (dealer + 1) % 2
        else:
            sb_seat = (dealer + 1) % n_players
            bb_seat = (dealer + 2) % n_players
        self.log.append({"action": "small_blind", "seat": sb_seat, "amount": 50})
        self.log.append({"action": "big_blind", "seat": bb_seat, "amount": BIG_BLIND})

    def record(
        self, state_before: np.ndarray, action_id: int, state_after: np.ndarray
    ) -> None:
        seat = int(state_before[S_TOACT])
        current_bet = int(state_before[S_CURBET])
        your_bet = int(state_before[S_BETS + seat])
        amount_owed = max(0, current_bet - your_bet)
        your_stack = int(state_before[S_STACKS + seat])

        if action_id == FOLD:
            self.log.append({"action": "fold", "seat": seat})
        elif action_id == CHECK_CALL:
            if amount_owed > 0:
                self.log.append(
                    {
                        "action": "call",
                        "seat": seat,
                        "amount": min(amount_owed, your_stack),
                    }
                )
            else:
                self.log.append({"action": "check", "seat": seat})
        else:
            self.log.append(
                {
                    "action": (
                        "all_in"
                        if action_id == ALL_IN or int(state_after[S_STACKS + seat]) == 0
                        else "raise"
                    ),
                    "seat": seat,
                    "amount": int(state_after[S_BETS + seat]),
                }
            )
        if (
            self.include_deal_markers
            and not state_after[S_TERMINAL]
            and int(state_after[S_STREET]) > int(state_before[S_STREET])
        ):
            self.log.append({"action": "deal"})


def _assert_feature_parity(
    state: np.ndarray,
    deck: np.ndarray,
    hole_cards: np.ndarray,
    action_log: list[dict],
) -> None:
    if state[S_TERMINAL]:
        return
    seat = int(state[S_TOACT])
    legal = legal_actions(state, seat)
    train_features = encode_state(state, hole_cards, deck, seat, legal)
    for include_dealer in (True, False):
        runtime_features, runtime_legal = encode_state_dict(
            _state_to_engine_dict(state, deck, hole_cards, action_log, include_dealer)
        )

        np.testing.assert_array_equal(
            runtime_legal.astype(np.float32), legal.astype(np.float32)
        )
        np.testing.assert_allclose(
            runtime_features[_PARITY_INDICES],
            train_features[_PARITY_INDICES],
            atol=1e-6,
            rtol=1e-5,
        )


def test_card_index_roundtrip() -> None:
    for idx in range(52):
        assert _card_to_idx(_idx_to_card(idx)) == idx


def test_feature_dim_is_current() -> None:
    assert FEATURE_DIM == 51


def test_random_state_encoder_parity() -> None:
    rng = np.random.default_rng(1234)
    checked = 0

    for include_deal_markers in (False, True):
        for _ in range(30):
            n_players = int(rng.integers(2, 7))
            dealer = int(rng.integers(0, n_players))
            state, deck, hole_cards = new_hand(n_players, dealer, rng)
            log = _ActionLogBuilder(n_players, dealer, include_deal_markers)

            for _ in range(int(rng.integers(0, 12))):
                if state[S_TERMINAL]:
                    break
                _assert_feature_parity(state, deck, hole_cards, log.log)
                seat = int(state[S_TOACT])
                legal = legal_actions(state, seat)
                legal_ids = [action for action in range(len(legal)) if legal[action]]
                if not legal_ids:
                    break
                action = legal_ids[int(rng.integers(0, len(legal_ids)))]
                state_before = state.copy()
                step(state, deck, hole_cards, action)
                log.record(state_before, action, state)
                checked += 1

            if not state[S_TERMINAL]:
                _assert_feature_parity(state, deck, hole_cards, log.log)
                checked += 1

    assert checked >= 80


def test_runtime_legal_mask_respects_last_full_raise() -> None:
    state = np.zeros(59, dtype=np.int32)
    state[S_NPLAYERS] = 3
    state[S_TOACT] = 1
    state[S_POT] = 200
    state[S_CURBET] = 600
    state[S_NRAISES] = 1
    state[S_LAST_FULL_RAISE] = 500
    state[S_STACKS + 0] = 3000
    state[S_STACKS + 1] = 2000
    state[S_STACKS + 2] = 2500
    state[S_BETS + 0] = 600

    runtime_state = {
        "your_cards": ["Ac", "Kd"],
        "community_cards": [],
        "players": [
            {
                "seat": 0,
                "stack": 3000,
                "bet_this_street": 600,
                "is_folded": False,
                "is_all_in": False,
            },
            {
                "seat": 1,
                "stack": 2000,
                "bet_this_street": 0,
                "is_folded": False,
                "is_all_in": False,
            },
            {
                "seat": 2,
                "stack": 2500,
                "bet_this_street": 0,
                "is_folded": False,
                "is_all_in": False,
            },
        ],
        "pot": 200,
        "seat_to_act": 1,
        "street": "preflop",
        "can_check": False,
        "amount_owed": 600,
        "min_raise_to": 1100,
        "current_bet": 600,
        "your_stack": 2000,
        "your_bet_this_street": 0,
        "action_log": [
            {"action": "small_blind", "seat": 1, "amount": 50},
            {"action": "big_blind", "seat": 2, "amount": 100},
            {"action": "raise", "seat": 0, "amount": 600},
        ],
    }

    _, runtime_legal = encode_state_dict(runtime_state)
    train_legal = legal_actions(state, 1)

    np.testing.assert_array_equal(runtime_legal, train_legal)
    assert not runtime_legal[BET_50]
    assert runtime_legal[BET_POT]


def test_runtime_legal_mask_allows_raises_after_four_full_raises() -> None:
    state = np.zeros(59, dtype=np.int32)
    state[S_NPLAYERS] = 3
    state[S_TOACT] = 1
    state[S_POT] = 600
    state[S_CURBET] = 300
    state[S_NRAISES] = 4
    state[S_LAST_FULL_RAISE] = 100
    state[S_STACKS + 0] = 3000
    state[S_STACKS + 1] = 2000
    state[S_STACKS + 2] = 2500
    state[S_BETS + 0] = 300

    runtime_state = {
        "your_cards": ["Ac", "Kd"],
        "community_cards": [],
        "players": [
            {
                "seat": 0,
                "stack": 3000,
                "bet_this_street": 300,
                "is_folded": False,
                "is_all_in": False,
            },
            {
                "seat": 1,
                "stack": 2000,
                "bet_this_street": 0,
                "is_folded": False,
                "is_all_in": False,
            },
            {
                "seat": 2,
                "stack": 2500,
                "bet_this_street": 0,
                "is_folded": False,
                "is_all_in": False,
            },
        ],
        "pot": 600,
        "seat_to_act": 1,
        "street": "preflop",
        "can_check": False,
        "amount_owed": 300,
        "min_raise_to": 400,
        "current_bet": 300,
        "your_stack": 2000,
        "your_bet_this_street": 0,
        "action_log": [],
    }

    runtime_legal = _compute_legal_mask(runtime_state, n_raises=4, last_full_raise=100)
    train_legal = legal_actions(state, 1)

    np.testing.assert_array_equal(runtime_legal, train_legal)
    assert runtime_legal[BET_50]
    assert runtime_legal[BET_POT]


def test_single_active_actor_runtime_mask_matches_training() -> None:
    state = np.zeros(59, dtype=np.int32)
    state[S_NPLAYERS] = 2
    state[S_TOACT] = 0
    state[S_STREET] = 1
    state[S_POT] = 2000
    state[S_STACKS + 0] = 5000
    state[S_STACKS + 1] = 0
    state[S_ALLIN + 1] = 1

    runtime_state = {
        "your_cards": ["Ac", "Kd"],
        "community_cards": ["2d", "7h", "Ts"],
        "players": [
            {
                "seat": 0,
                "stack": 5000,
                "bet_this_street": 0,
                "is_folded": False,
                "is_all_in": False,
            },
            {
                "seat": 1,
                "stack": 0,
                "bet_this_street": 0,
                "is_folded": False,
                "is_all_in": True,
            },
        ],
        "pot": 2000,
        "seat_to_act": 0,
        "street": "flop",
        "can_check": True,
        "amount_owed": 0,
        "min_raise_to": 100,
        "current_bet": 0,
        "your_stack": 5000,
        "your_bet_this_street": 0,
        "action_log": [],
    }

    _, runtime_legal = encode_state_dict(runtime_state)
    train_legal = legal_actions(state, 0)

    np.testing.assert_array_equal(runtime_legal, train_legal)
    assert runtime_legal[BET_50]
    assert runtime_legal[BET_POT]


def test_runtime_reconstructs_current_street_without_deal_markers() -> None:
    action_log = [
        {"action": "small_blind", "seat": 1, "amount": 50},
        {"action": "big_blind", "seat": 2, "amount": 100},
        {"action": "fold", "seat": 3},
        {"action": "fold", "seat": 4},
        {"action": "fold", "seat": 5},
        {"action": "raise", "seat": 0, "amount": 300},
        {"action": "call", "seat": 1, "amount": 250},
        {"action": "call", "seat": 2, "amount": 200},
        {"action": "check", "seat": 1},
        {"action": "check", "seat": 2},
        {"action": "raise", "seat": 0, "amount": 300},
    ]
    players = [
        {
            "seat": 0,
            "stack": 9400,
            "is_folded": False,
            "is_all_in": False,
            "bet_this_street": 300,
        },
        {
            "seat": 1,
            "stack": 9700,
            "is_folded": False,
            "is_all_in": False,
            "bet_this_street": 0,
        },
        {
            "seat": 2,
            "stack": 9700,
            "is_folded": False,
            "is_all_in": False,
            "bet_this_street": 0,
        },
        {
            "seat": 3,
            "stack": 10000,
            "is_folded": True,
            "is_all_in": False,
            "bet_this_street": 0,
        },
        {
            "seat": 4,
            "stack": 10000,
            "is_folded": True,
            "is_all_in": False,
            "bet_this_street": 0,
        },
        {
            "seat": 5,
            "stack": 10000,
            "is_folded": True,
            "is_all_in": False,
            "bet_this_street": 0,
        },
    ]
    state = {
        "your_cards": ["Ac", "Kc"],
        "community_cards": ["2d", "7h", "Ts"],
        "players": players,
        "pot": 1200,
        "seat_to_act": 1,
        "street": "flop",
        "can_check": False,
        "amount_owed": 300,
        "min_raise_to": 600,
        "current_bet": 300,
        "your_stack": 9700,
        "your_bet_this_street": 0,
        "action_log": action_log,
    }

    features, _ = encode_state_dict(state)
    invested = _hand_invested_from_log(action_log, players, "flop")

    assert invested[0] == 600
    assert invested[1] == 300
    assert invested[2] == 300
    assert features[36] == 0.25
    assert features[37] == 1.0
    assert features[38] == 0.0
    assert features[40] == 0.03
    assert features[44] == 0.06
