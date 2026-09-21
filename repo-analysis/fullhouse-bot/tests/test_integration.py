"""Integration smoke tests for in-process engine matches."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tooling.harness.engine import run_match

pytestmark = pytest.mark.integration

BOT_PATH = str(REPO / "bot" / "bot.py")
ADVERSARIAL_DIR = REPO / "bots" / "adversarial"


def _adversarial_bots() -> list[Path]:
    """All .py files in bots/adversarial/ that have a decide() function."""
    bots = []
    for p in sorted(ADVERSARIAL_DIR.glob("*.py")):
        if p.name.startswith("_"):
            continue
        text = p.read_text()
        if "def decide(" in text:
            bots.append(p)
    return bots


@pytest.mark.parametrize("bot_path", _adversarial_bots(), ids=lambda p: p.name)
def test_no_crashes_vs_adversarial_bot(bot_path: Path) -> None:
    """Each bundled adversarial bot should complete a short match cleanly."""
    r = run_match(
        {"cand": BOT_PATH, "opp": str(bot_path)},
        n_hands=20,
        seed=42,
    )
    cand_errors = r["bot_errors"].get("cand", [])
    assert cand_errors == [], f"Our bot errored vs {bot_path.name}: {cand_errors}"


def test_runs_vs_random_bot_without_errors() -> None:
    """The simplest shipped opponent should not trigger runtime errors."""
    random_bot = ADVERSARIAL_DIR / "random_bot.py"
    if not random_bot.exists():
        pytest.skip("random_bot.py is not present")
    if not any((REPO / "data").glob("deep_cfr_model*.npz")):
        pytest.skip("Deep CFR model data is not present")
    r = run_match(
        {"cand": BOT_PATH, "rnd": str(random_bot)},
        n_hands=100,
        seed=0,
    )
    assert r["bot_errors"]["cand"] == [], r["bot_errors"]
    assert r["chip_delta"]["cand"] + r["chip_delta"]["rnd"] == 0


def test_self_play_chip_conservation() -> None:
    """In self-play, chip deltas must sum to zero."""
    r = run_match(
        {"a": BOT_PATH, "b": BOT_PATH},
        n_hands=50,
        seed=99,
    )
    total = sum(r["chip_delta"].values())
    assert total == 0, f"Chip conservation violated: deltas sum to {total}"


COMPETITOR_DIR = REPO / "bots" / "competitors"


def _competitor_bots() -> list[Path]:
    """All bot directories in bots/competitors/ that have a working bot.py."""
    if not COMPETITOR_DIR.is_dir():
        return []
    bots = []
    for d in sorted(COMPETITOR_DIR.iterdir()):
        bp = d / "bot.py"
        if d.is_dir() and bp.is_file():
            text = bp.read_text()
            if "def decide(" in text:
                bots.append(d)
    return bots


@pytest.mark.parametrize("bot_path", _competitor_bots(), ids=lambda p: p.name)
def test_no_crashes_vs_competitor_bot(bot_path: Path) -> None:
    """Each competitor bot should complete a short match without our bot crashing."""
    r = run_match(
        {"cand": BOT_PATH, "opp": str(bot_path)},
        n_hands=20,
        seed=42,
    )
    cand_errors = r["bot_errors"].get("cand", [])
    assert cand_errors == [], f"Our bot errored vs {bot_path.name}: {cand_errors}"
