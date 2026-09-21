"""Tests for the packaging pipeline.

Verifies that the submission builder concatenates bot/ modules into a valid
single-file bot.py with no internal imports, and that the result loads and
produces valid actions.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
BUILD_BOT = REPO / "build" / "bot.py"
PACKAGE_SCRIPT = REPO / "scripts" / "build_submission.py"
VALIDATOR = REPO / "engine_vendored" / "sandbox" / "validator.py"
SCENARIO_PATH = REPO / "tests" / "fixtures" / "bot_scenarios.json"

VALID_ACTIONS = {"fold", "check", "call", "raise", "all_in"}


def _build() -> Path:
    zip_path = REPO / "build" / "test.zip"
    r = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_SCRIPT),
            "--skip-validate",
            "--out",
            str(zip_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert r.returncode == 0, f"build_submission.py failed: {r.stderr}"
    assert BUILD_BOT.exists(), "build/bot.py not created"
    return zip_path


def test_concat_parses():
    _build()
    text = BUILD_BOT.read_text()
    ast.parse(text)


def test_no_internal_imports():
    _build()
    for i, line in enumerate(BUILD_BOT.read_text().splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "from bot." not in stripped and "import bot." not in stripped, (
            f"Internal import at line {i}: {stripped}"
        )


def test_concat_size():
    _build()
    size = BUILD_BOT.stat().st_size
    assert size < 5 * 1024 * 1024, f"bot.py is {size / 1e6:.1f} MB, limit 5 MB"


def test_package_includes_runtime_data():
    zip_path = _build()
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert "data/preflop_equity.npz" in names
    expected_models = {
        f"data/{path.name}" for path in (REPO / "data").glob("deep_cfr_model*.npz")
    }
    assert expected_models.issubset(names)


def test_concat_decide_works():
    _build()
    spec = importlib.util.spec_from_file_location("concat_bot", str(BUILD_BOT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "decide"), "Concatenated bot.py missing decide()"

    states = [
        {
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
        },
        {
            "your_cards": ["Ts", "9s"],
            "community_cards": ["8h", "7d", "2c"],
            "pot": 300,
            "your_stack": 9700,
            "your_bet_this_street": 0,
            "amount_owed": 0,
            "min_raise_to": 200,
            "can_check": True,
            "seat_to_act": 3,
            "dealer": 0,
            "street": "flop",
            "hand_num": 2,
            "players": [
                {"seat": i, "stack": 10000, "is_folded": i >= 3, "bet_this_street": 0}
                for i in range(6)
            ],
        },
    ]
    for state in states:
        action = mod.decide(state)
        assert "action" in action
        assert action["action"] in VALID_ACTIONS


def test_zip_decide_works_from_extracted_root():
    zip_path = _build()
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        bot_path = Path(tmp) / "bot.py"
        spec = importlib.util.spec_from_file_location("zip_bot", str(bot_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        action = mod.decide(
            {
                "your_cards": ["Ts", "9s"],
                "community_cards": ["8h", "7d", "2c"],
                "pot": 300,
                "your_stack": 9700,
                "your_bet_this_street": 0,
                "amount_owed": 0,
                "min_raise_to": 200,
                "can_check": True,
                "seat_to_act": 3,
                "street": "flop",
                "hand_num": 2,
                "players": [
                    {
                        "seat": i,
                        "stack": 10000,
                        "is_folded": i >= 3,
                        "bet_this_street": 0,
                    }
                    for i in range(6)
                ],
                "action_log": [
                    {"seat": 1, "action": "small_blind", "amount": 50},
                    {"seat": 2, "action": "big_blind", "amount": 100},
                ],
            }
        )
        assert action["action"] in VALID_ACTIONS


def test_packaged_bot_matches_source_bot_on_curated_states(
    monkeypatch: pytest.MonkeyPatch,
):
    _build()
    scenarios = json.loads(SCENARIO_PATH.read_text())

    source_spec = importlib.util.spec_from_file_location(
        "source_bot_for_pkg_test", str(REPO / "bot" / "bot.py")
    )
    source_mod = importlib.util.module_from_spec(source_spec)
    source_spec.loader.exec_module(source_mod)

    built_spec = importlib.util.spec_from_file_location(
        "built_bot_for_pkg_test", str(BUILD_BOT)
    )
    built_mod = importlib.util.module_from_spec(built_spec)
    built_spec.loader.exec_module(built_mod)

    for scenario in scenarios:
        strategy = scenario["strategy"]
        sample = scenario["sample"]

        class _Lookup:
            def get_strategy(self, state: dict):
                del state
                return np.array(strategy, dtype=np.float32)

        for module in (source_mod, built_mod):
            monkeypatch.setattr(module, "_DEEP_CFR", _Lookup())
            monkeypatch.setattr(module._RNG, "random", lambda sample=sample: sample)

        source_action = source_mod.decide(scenario["state"])
        built_action = built_mod.decide(scenario["state"])
        assert built_action == source_action == scenario["expected_action"]


def test_validator_accepts():
    if not VALIDATOR.exists():
        return  # skip if vendored engine not present
    zip_path = _build()
    r = subprocess.run(
        [sys.executable, str(VALIDATOR), str(zip_path), "--json"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"Validator failed: {r.stdout} {r.stderr}"
