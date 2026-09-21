"""Smoke tests for the shipped runtime bot."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tooling.harness.engine import run_match

BOT_PATH = REPO_ROOT / "bot" / "bot.py"
TEMPLATE = REPO_ROOT / "engine_vendored/bots/template/bot.py"
VALIDATOR = REPO_ROOT / "engine_vendored/sandbox/validator.py"


def test_decide_exists() -> None:
    from bot import bot as candidate

    assert hasattr(candidate, "decide"), "bot.bot.decide is missing"
    assert callable(candidate.decide)


def test_runs_ten_hands_vs_template() -> None:
    r = run_match(
        {"cand": str(BOT_PATH), "tmpl": str(TEMPLATE)},
        n_hands=10,
        seed=7,
    )
    # No load errors and no per-hand exceptions
    assert r["bot_errors"]["cand"] == [], r["bot_errors"]
    # Chip invariant: in a 2-bot match the two deltas must sum to 0
    assert r["chip_delta"]["cand"] + r["chip_delta"]["tmpl"] == 0


def test_passes_upstream_validator() -> None:
    proc = subprocess.run(
        [sys.executable, str(VALIDATOR), str(BOT_PATH), "--json"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"validator failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
