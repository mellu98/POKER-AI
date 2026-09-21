"""Unit tests for the current runtime bot."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pytest

from bot import bot as bot_module
from bot.bot import decide, _apply_legal_mask, _hand_class, _position, _push_fold
from bot.deep_cfr_lookup import DeepCFRLookup

VALID_ACTIONS = {"fold", "check", "call", "raise", "all_in"}
BIG_BLIND = 100


def _base_state(**overrides) -> dict:
    s = {
        "your_cards": ["Ah", "Kd"],
        "community_cards": [],
        "pot": 150,
        "your_stack": 9900,
        "your_bet_this_street": 0,
        "amount_owed": 100,
        "min_raise_to": 200,
        "can_check": False,
        "seat_to_act": 3,
        "dealer": 0,
        "street": "preflop",
        "hand_num": 1,
        "players": [
            {
                "seat": i,
                "stack": 10000,
                "is_folded": i >= 4,
                "bet_this_street": [0, 50, 100, 0, 0, 0][i],
            }
            for i in range(6)
        ],
    }
    s.update(overrides)
    return s


def _stub_strategy(
    monkeypatch: pytest.MonkeyPatch,
    strategy: list[float] | np.ndarray,
    *,
    sample: float = 0.0,
) -> None:
    class _Lookup:
        def get_strategy(self, state: dict) -> np.ndarray:
            del state
            return np.array(strategy, dtype=np.float32)

    monkeypatch.setattr(bot_module, "_DEEP_CFR", _Lookup())
    monkeypatch.setattr(bot_module._RNG, "random", lambda: sample)


def test_decide_returns_valid_action(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bot_module._RNG, "random", lambda: 0.0)
    for street, cc in [
        ("preflop", []),
        ("flop", ["Ts", "7h", "2d"]),
        ("turn", ["Ts", "7h", "2d", "Qc"]),
        ("river", ["Ts", "7h", "2d", "Qc", "3s"]),
    ]:
        state = _base_state(street=street, community_cards=cc)
        action = decide(state)
        assert "action" in action, f"Missing 'action' key for {street}"
        assert action["action"] in VALID_ACTIONS, f"Invalid action {action} on {street}"


def test_decide_masks_fold_to_check_when_check_is_free(monkeypatch: pytest.MonkeyPatch):
    _stub_strategy(monkeypatch, [1.0, 0.0, 0.0, 0.0, 0.0])
    state = _base_state(
        can_check=True,
        amount_owed=0,
        street="flop",
        community_cards=["Ts", "7h", "2d"],
        your_stack=100,
        min_raise_to=0,
    )
    action = decide(state)
    assert action == {"action": "check"}


def test_decide_warmup_uses_safe_default_action():
    state = {"type": "warmup", "can_check": True}
    assert decide(state) == {"action": "check"}


def test_decide_falls_back_to_safe_default_on_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
):
    def _boom(state: dict) -> np.ndarray:
        del state
        raise RuntimeError("boom")

    monkeypatch.setattr(bot_module._DEEP_CFR, "get_strategy", _boom)
    state = _base_state(
        street="flop",
        community_cards=["Ts", "7h", "2d"],
        amount_owed=100,
        can_check=False,
    )
    assert decide(state) == {"action": "call"}


def test_push_fold_aa_shoves():
    state = _base_state(
        your_cards=["As", "Ac"],
        your_stack=800,
        amount_owed=100,
        players=[
            {
                "seat": i,
                "stack": [800, 10000, 10000, 800, 0, 0][i],
                "is_folded": i >= 4,
                "bet_this_street": [0, 50, 100, 0, 0, 0][i],
            }
            for i in range(6)
        ],
    )
    result = _push_fold(state)
    assert result is not None, "Push/fold should trigger at 8bb"
    assert result["action"] == "all_in", f"AA at 8bb should shove, got {result}"


def test_push_fold_72o_folds_facing_raise():
    state = _base_state(
        your_cards=["7h", "2d"],
        your_stack=600,
        amount_owed=200,
        players=[
            {
                "seat": i,
                "stack": [600, 10000, 10000, 600, 0, 0][i],
                "is_folded": i >= 4,
                "bet_this_street": [0, 50, 200, 0, 0, 0][i],
            }
            for i in range(6)
        ],
    )
    result = _push_fold(state)
    assert result is not None, "Push/fold should trigger at 6bb"
    assert result["action"] == "fold", (
        f"72o facing raise at 6bb should fold, got {result}"
    )


def test_push_fold_skips_deep_stacks():
    state = _base_state(your_stack=9900)
    assert _push_fold(state) is None, "Push/fold should not trigger at 99bb"


def test_decide_uses_push_fold_before_model_lookup(monkeypatch: pytest.MonkeyPatch):
    def _should_not_run(state: dict) -> np.ndarray:
        raise AssertionError(
            "Deep CFR lookup should not run for preflop push/fold spots"
        )

    monkeypatch.setattr(bot_module._DEEP_CFR, "get_strategy", _should_not_run)
    state = _base_state(
        your_cards=["As", "Ac"],
        your_stack=800,
        amount_owed=100,
        players=[
            {
                "seat": i,
                "stack": [800, 10000, 10000, 800, 0, 0][i],
                "is_folded": i >= 4,
                "bet_this_street": [0, 50, 100, 0, 0, 0][i],
            }
            for i in range(6)
        ],
    )
    assert decide(state) == {"action": "all_in"}


def test_hand_class():
    assert _hand_class(["Ah", "Kd"]) == "AKo"
    assert _hand_class(["Ah", "Kh"]) == "AKs"
    assert _hand_class(["Qd", "Qh"]) == "QQ"
    assert _hand_class(["2h", "7d"]) == "72o"
    assert _hand_class(["9s", "Ts"]) == "T9s"


def test_position_mapping_for_6max_and_heads_up():
    assert _position(_base_state(seat_to_act=0, dealer=0)) == "BTN"
    assert _position(_base_state(seat_to_act=1, dealer=0)) == "SB"
    assert _position(_base_state(seat_to_act=2, dealer=0)) == "BB"

    four_handed = {
        "dealer": 0,
        "players": [{"seat": i, "stack": 10000, "is_folded": False} for i in range(4)],
    }
    assert [_position({**four_handed, "seat_to_act": i}) for i in range(4)] == [
        "BTN",
        "SB",
        "BB",
        "CO",
    ]

    five_handed = {
        "dealer": 0,
        "players": [{"seat": i, "stack": 10000, "is_folded": False} for i in range(5)],
    }
    assert [_position({**five_handed, "seat_to_act": i}) for i in range(5)] == [
        "BTN",
        "SB",
        "BB",
        "HJ",
        "CO",
    ]

    hu_state = {
        "seat_to_act": 0,
        "dealer": 0,
        "players": [
            {"seat": 0, "stack": 10000, "is_folded": False},
            {"seat": 1, "stack": 10000, "is_folded": False},
        ],
    }
    assert _position(hu_state) == "BTN"
    hu_state["seat_to_act"] = 1
    assert _position(hu_state) == "BB"


def test_apply_legal_mask_strips_illegal_raise_lines():
    state = _base_state(
        can_check=True,
        amount_owed=0,
        your_stack=100,
        min_raise_to=0,
    )
    masked = _apply_legal_mask(np.array([0.2] * 5, dtype=np.float32), state)
    assert masked[0] == 0.0
    assert masked[2] == 0.0
    assert masked[3] == 0.0
    assert masked[4] == 0.0
    assert masked[1] == 1.0


def test_apply_legal_mask_raises_when_illegal_masking_zeros_strategy():
    state = _base_state(
        can_check=True,
        amount_owed=0,
        your_stack=1000,
        min_raise_to=200,
    )
    with pytest.raises(RuntimeError, match="removed all probability mass"):
        _apply_legal_mask(np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), state)


def test_decide_raise_action_includes_exact_amount(monkeypatch: pytest.MonkeyPatch):
    _stub_strategy(monkeypatch, [0.0, 0.0, 0.0, 1.0, 0.0])
    state = _base_state(
        street="flop",
        community_cards=["Ts", "7h", "2d"],
        can_check=True,
        amount_owed=0,
    )
    action = decide(state)
    assert action == {"action": "raise", "amount": 200}


def test_deep_cfr_lookup_emits_no_runtime_warnings():
    lookup = DeepCFRLookup(REPO / "data" / "deep_cfr_model.npz")
    state = _base_state(
        street="preflop",
        community_cards=[],
        can_check=False,
        amount_owed=100,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        strategy = lookup.get_strategy(state)

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == [], runtime_warnings
    assert strategy is not None
    assert np.isfinite(strategy).all()
    assert np.isclose(strategy.sum(), 1.0)
