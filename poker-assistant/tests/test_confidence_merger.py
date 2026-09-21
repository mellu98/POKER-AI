"""Tests for vision/confidence_merger.py.

These are pure unit tests: no OpenCV, no YOLO model, no screenshots.
They verify the confidence-fusion logic between local OCR/template reads and
optional YOLO detections.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

# Allow importing from the project root when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vision"))

from confidence_merger import (
    confidence_from_local_scores,
    merge_card_confidence,
    parse_yolo_label,
)


def _bbox(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class TestConfidenceFromLocalScores(unittest.TestCase):
    """Unit tests for confidence_from_local_scores."""

    def test_min_score_when_all_above_baseline(self):
        """Confidence is the minimum local score when every score is accepted."""
        self.assertAlmostEqual(confidence_from_local_scores([0.9, 0.85, 0.88]), 0.85)

    def test_zero_when_any_score_below_baseline(self):
        """If any accepted card score is below the baseline, confidence drops to 0.0."""
        self.assertEqual(confidence_from_local_scores([0.9, 0.3, 0.88]), 0.0)

    def test_empty_score_list_is_fully_confident(self):
        """No cards means there is nothing to be uncertain about."""
        self.assertEqual(confidence_from_local_scores([]), 1.0)

    def test_ignores_empty_slots_when_cards_provided(self):
        """Empty slots (None cards) do not drag down the confidence of real reads."""
        self.assertAlmostEqual(
            confidence_from_local_scores([0.9, 0.0, 0.85], cards=["As", None, "Kh"]),
            0.85,
        )


class TestParseYoloLabel(unittest.TestCase):
    """Unit tests for YOLO label parsing."""

    def test_parses_plain_card_label(self):
        self.assertEqual(parse_yolo_label("As"), ("As", None))
        self.assertEqual(parse_yolo_label("Ts"), ("Ts", None))

    def test_parses_hero_slot_label(self):
        self.assertEqual(parse_yolo_label("hero_card_1_As"), ("As", "hole"))

    def test_parses_board_slot_label(self):
        self.assertEqual(parse_yolo_label("board_card_3_Ts"), ("Ts", "board"))

    def test_generic_card_label_has_no_card(self):
        self.assertEqual(parse_yolo_label("card"), (None, None))


class TestMergeCardConfidence(unittest.TestCase):
    """Unit tests for merge_card_confidence."""

    def test_local_only_returns_min_local_confidence(self):
        """Without YOLO detections, confidence is driven by local scores."""
        merged, conf, reasons = merge_card_confidence(
            local_cards=["As", "Kh"],
            local_scores=[0.9, 0.85],
            yolo_cards=[],
            yolo_threshold=0.5,
            group_name="hole",
        )
        self.assertEqual(merged, ["As", "Kh"])
        self.assertAlmostEqual(conf, 0.85)
        self.assertEqual(reasons, [])

    def test_local_only_low_score_adds_reason(self):
        """A low local score triggers an uncertainty reason."""
        merged, conf, reasons = merge_card_confidence(
            local_cards=["As", "Kh"],
            local_scores=[0.9, 0.3],
            yolo_cards=[],
            yolo_threshold=0.5,
            group_name="hole",
        )
        self.assertEqual(merged, ["As", "Kh"])
        self.assertEqual(conf, 0.0)
        self.assertIn("low local confidence", reasons)

    def test_yolo_agreement_boosts_confidence(self):
        """When local and YOLO agree, the slot confidence is boosted above local only."""
        yolo_cards = [
            ("As", _bbox(0.0, 0.0, 0.1, 0.1), 0.95),
            ("Kh", _bbox(0.2, 0.0, 0.3, 0.1), 0.90),
        ]
        merged, conf, reasons = merge_card_confidence(
            local_cards=["As", "Kh"],
            local_scores=[0.80, 0.80],
            yolo_cards=yolo_cards,
            yolo_threshold=0.5,
            group_name="hole",
        )
        self.assertEqual(merged, ["As", "Kh"])
        self.assertGreater(conf, 0.80)
        self.assertEqual(reasons, [])

    def test_yolo_disagreement_lowers_confidence_below_threshold(self):
        """A local/YOLO mismatch forces the slot confidence to 0.0."""
        yolo_cards = [
            ("Kh", _bbox(0.0, 0.0, 0.1, 0.1), 0.90),
        ]
        merged, conf, reasons = merge_card_confidence(
            local_cards=["As"],
            local_scores=[0.80],
            yolo_cards=yolo_cards,
            yolo_threshold=0.5,
            group_name="hole",
        )
        self.assertEqual(merged, ["As"])
        self.assertLess(conf, 0.7)
        self.assertEqual(conf, 0.0)
        self.assertIn("hole mismatch local vs YOLO", reasons)

    def test_missing_yolo_card_marks_uncertain(self):
        """A local read with no matching YOLO detection is marked uncertain."""
        yolo_cards = [
            ("As", _bbox(0.0, 0.0, 0.1, 0.1), 0.90),
        ]
        merged, conf, reasons = merge_card_confidence(
            local_cards=["As", "Kh"],
            local_scores=[0.90, 0.85],
            yolo_cards=yolo_cards,
            yolo_threshold=0.5,
            group_name="hole",
        )
        self.assertEqual(merged, ["As", "Kh"])
        self.assertEqual(conf, 0.0)
        self.assertIn("hole card missing in local vs YOLO", reasons)

    def test_missing_local_card_marks_uncertain(self):
        """A YOLO read with no matching local detection is marked uncertain."""
        yolo_cards = [
            ("As", _bbox(0.0, 0.0, 0.1, 0.1), 0.90),
        ]
        merged, conf, reasons = merge_card_confidence(
            local_cards=[None],
            local_scores=[1.0],
            yolo_cards=yolo_cards,
            yolo_threshold=0.5,
            group_name="board",
        )
        self.assertEqual(merged, ["As"])
        self.assertEqual(conf, 0.0)
        self.assertIn("board card missing in local vs YOLO", reasons)

    def test_unparseable_yolo_label_falls_back_to_local(self):
        """A generic 'card' label cannot be compared, so local read is kept."""
        yolo_cards = [
            ("card", _bbox(0.0, 0.0, 0.1, 0.1), 0.90),
        ]
        merged, conf, reasons = merge_card_confidence(
            local_cards=["As"],
            local_scores=[0.80],
            yolo_cards=yolo_cards,
            yolo_threshold=0.5,
            group_name="hole",
        )
        self.assertEqual(merged, ["As"])
        self.assertAlmostEqual(conf, 0.80)
        self.assertIn("YOLO card label unparseable", reasons)


if __name__ == "__main__":
    unittest.main()
