"""Curated scenario regressions for the runtime bot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bot import bot as bot_module
from bot.bot import decide
from bot.features import _compute_legal_mask

SCENARIO_PATH = Path(__file__).resolve().parent / "fixtures" / "bot_scenarios.json"
SCENARIOS = json.loads(SCENARIO_PATH.read_text())


def _patch_lookup(
    monkeypatch: pytest.MonkeyPatch, strategy: list[float], sample: float
) -> None:
    class _Lookup:
        def get_strategy(self, state: dict) -> np.ndarray:
            del state
            return np.array(strategy, dtype=np.float32)

    monkeypatch.setattr(bot_module, "_DEEP_CFR", _Lookup())
    monkeypatch.setattr(bot_module._RNG, "random", lambda: sample)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s["name"])
def test_curated_scenarios_produce_expected_actions(
    monkeypatch: pytest.MonkeyPatch,
    scenario: dict,
) -> None:
    _patch_lookup(monkeypatch, scenario["strategy"], scenario["sample"])
    assert decide(scenario["state"]) == scenario["expected_action"]


@pytest.mark.parametrize(
    "scenario",
    [s for s in SCENARIOS if "expected_legal_mask" in s],
    ids=lambda s: s["name"],
)
def test_curated_scenarios_expose_expected_legal_masks(scenario: dict) -> None:
    state = scenario["state"]
    log = state.get("action_log") or []
    players = state.get("players") or []
    current_bet = state.get("current_bet", 0)
    street = state.get("street", "preflop")
    legal_mask = _compute_legal_mask(
        state, *_current_raise_state(log, players, current_bet, street)
    )
    np.testing.assert_array_equal(
        legal_mask.astype(np.int8),
        np.array(scenario["expected_legal_mask"], dtype=np.int8),
    )


def _current_raise_state(
    log: list[dict], players: list[dict], current_bet: int, street: str
) -> tuple[int, int]:
    n_raises, _, last_full_raise = bot_module._current_street_aggression(
        log, players, current_bet, street
    )
    return n_raises, last_full_raise
