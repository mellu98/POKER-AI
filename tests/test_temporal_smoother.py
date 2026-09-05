"""Tests for vision/temporal_smoother.py.

These are pure unit tests: no OpenCV, no screenshots, no real vision pipeline.
They verify that the temporal consensus wrapper smooths single-frame misreads
and stabilises numeric values.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Allow importing from the project root when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vision"))

from temporal_smoother import TemporalSmoother


def _state(
    hole: list[str] | None = None,
    board: list[str] | None = None,
    pot: int = 0,
    to_call: int = 0,
    stack: int = 1000,
    position: str = "BTN",
    stage: str = "preflop",
) -> dict:
    return {
        "hole": hole or [],
        "board": board or [],
        "pot": pot,
        "to_call": to_call,
        "stack": stack,
        "position": position,
        "stage": stage,
        "confidence": {},
        "uncertainty_reasons": [],
        "is_uncertain": False,
    }


class TestTemporalSmoother(unittest.TestCase):
    """Unit tests for TemporalSmoother.update()."""

    def test_stable_sequence_keeps_state_and_high_confidence(self):
        """A run of identical frames reaches consensus and keeps every field."""
        smoother = TemporalSmoother(window_size=5, agreement_threshold=0.6, min_samples=3)
        raw = _state(
            hole=["As", "Kh"],
            board=["Qd", "Jh", "2c"],
            pot=120,
            to_call=20,
            stack=980,
            position="BTN",
            stage="flop",
        )

        for _ in range(5):
            smoothed = smoother.update(raw)

        self.assertEqual(smoothed["hole"], ["As", "Kh"])
        self.assertEqual(smoothed["board"], ["Qd", "Jh", "2c"])
        self.assertEqual(smoothed["pot"], 120)
        self.assertEqual(smoothed["to_call"], 20)
        self.assertEqual(smoothed["stack"], 980)
        self.assertEqual(smoothed["position"], "BTN")
        self.assertEqual(smoothed["stage"], "flop")
        self.assertGreaterEqual(smoothed["confidence"]["hole"], 0.7)
        self.assertGreaterEqual(smoothed["confidence"]["cards"], 0.7)
        self.assertGreaterEqual(smoothed["confidence"]["pot"], 0.7)
        self.assertGreaterEqual(smoothed["confidence"]["to_call"], 0.7)
        self.assertGreaterEqual(smoothed["confidence"]["position"], 0.7)
        self.assertGreaterEqual(smoothed["confidence"]["stage"], 0.7)
        self.assertFalse(smoothed["is_uncertain"])
        self.assertEqual(smoothed["uncertainty_reasons"], [])

    def test_single_outlier_lowers_card_confidence(self):
        """One misread hole card is corrected and makes the hand uncertain."""
        smoother = TemporalSmoother(window_size=5, agreement_threshold=0.6, min_samples=3)
        stable = _state(
            hole=["As", "Kh"],
            board=["Qd", "Jh", "2c"],
            pot=120,
            to_call=20,
            position="BTN",
            stage="flop",
        )
        outlier = _state(
            hole=["As", "Qc"],  # slot 1 misread
            board=["Qd", "Jh", "2c"],
            pot=120,
            to_call=20,
            position="BTN",
            stage="flop",
        )

        for _ in range(2):
            smoother.update(stable)
        smoothed = smoother.update(outlier)

        # Slot 1 is unstable: we keep slot 0 but drop the contested slot.
        self.assertIn("As", smoothed["hole"])
        self.assertLess(len(smoothed["hole"]), 2)
        self.assertLess(smoothed["confidence"]["hole"], 0.7)
        self.assertLess(smoothed["confidence"]["cards"], 0.7)
        self.assertTrue(smoothed["is_uncertain"])
        self.assertTrue(any("hole unstable" in r for r in smoothed["uncertainty_reasons"]))

    def test_board_growth_transitions_stage(self):
        """Board grows from empty to flop consensus and stage updates."""
        smoother = TemporalSmoother(window_size=5, agreement_threshold=0.6, min_samples=3)
        preflop = _state(hole=["As", "Kh"], board=[], pot=20, position="SB", stage="preflop")
        flop = _state(
            hole=["As", "Kh"],
            board=["Qd", "Jh", "2c"],
            pot=60,
            position="SB",
            stage="flop",
        )

        # First frames are preflop.
        for _ in range(3):
            smoothed = smoother.update(preflop)
        self.assertEqual(smoothed["stage"], "preflop")
        self.assertEqual(smoothed["board"], [])

        # New flop cards need min_samples frames before they are accepted.
        for _ in range(2):
            smoothed = smoother.update(flop)
        # With only 2 flop frames in a window of 5, board is still unstable.
        self.assertNotEqual(smoothed["stage"], "flop")

        for _ in range(2):
            smoothed = smoother.update(flop)
        # Now at least 3 frames agree on the flop board.
        self.assertEqual(smoothed["board"], ["Qd", "Jh", "2c"])
        self.assertEqual(smoothed["stage"], "flop")

    def test_numeric_median_ignores_outlier(self):
        """The median stabilises numeric fields against a single outlier."""
        smoother = TemporalSmoother(window_size=5, agreement_threshold=0.6, min_samples=3)
        for pot in [100, 100, 100, 100, 999]:
            smoothed = smoother.update(_state(pot=pot, hole=["As", "Kh"]))

        # Median of [100, 100, 100, 100, 999] is 100.
        self.assertEqual(smoothed["pot"], 100)
        # The outlier makes the pot unstable, so confidence is low.
        self.assertLess(smoothed["confidence"]["pot"], 0.7)
        self.assertTrue(any("pot unstable" in r for r in smoothed["uncertainty_reasons"]))

    def test_numeric_stable_sequence_has_high_confidence(self):
        """Stable numeric values reach high confidence."""
        smoother = TemporalSmoother(window_size=5, agreement_threshold=0.6, min_samples=3)
        for _ in range(5):
            smoothed = smoother.update(_state(pot=150, to_call=30, stack=970))

        self.assertEqual(smoothed["pot"], 150)
        self.assertEqual(smoothed["to_call"], 30)
        self.assertEqual(smoothed["stack"], 970)
        self.assertGreaterEqual(smoothed["confidence"]["pot"], 0.7)
        self.assertGreaterEqual(smoothed["confidence"]["to_call"], 0.7)
        self.assertGreaterEqual(smoothed["confidence"]["stack"], 0.7)


class TestIsStableHelper(unittest.TestCase):
    """Unit tests for TemporalSmoother.is_stable()."""

    def test_discrete_majority_is_stable(self):
        """A clear majority among discrete values is stable."""
        smoother = TemporalSmoother(window_size=5, agreement_threshold=0.6, min_samples=3)
        stable, value = smoother.is_stable(["As", "As", "As", "Kh", "Kh"])
        self.assertTrue(stable)
        self.assertEqual(value, "As")

    def test_discrete_no_majority_is_unstable(self):
        """A tied or scattered vote is unstable."""
        smoother = TemporalSmoother(window_size=5, agreement_threshold=0.6, min_samples=3)
        stable, value = smoother.is_stable(["As", "Kh", "Qd", "Jh", "2c"])
        self.assertFalse(stable)
        self.assertIsNone(value)

    def test_numeric_returns_median(self):
        """Numeric values are summarised by the median."""
        smoother = TemporalSmoother(window_size=5, agreement_threshold=0.6, min_samples=3)
        stable, value = smoother.is_stable([100, 102, 98, 101, 99])
        self.assertTrue(stable)
        self.assertEqual(value, 100)


if __name__ == "__main__":
    unittest.main()
