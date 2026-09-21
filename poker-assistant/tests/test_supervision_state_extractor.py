"""Tests for SupervisionStateExtractor confidence fusion (Fase 2).

These tests verify that the supervision wrapper no longer hardcodes confidence
values and that real local-template scores flow through to the returned state.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import yaml

# Allow importing from the project root when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vision"))

from state_extractor import SupervisionStateExtractor, ScreenshotStateExtractor


class TestSupervisionConfidenceKeys(unittest.TestCase):
    """Unit tests for confidence fields returned by SupervisionStateExtractor."""

    def _make_extractor(self) -> SupervisionStateExtractor:
        cfg = {
            "big_blind": 2,
            "vision": {
                "mode": "screenshot",
                "yolo": {"enabled": False},
                "rois": {
                    "hole": [
                        {"x": 10, "y": 10, "w": 60, "h": 80, "rel": False},
                        {"x": 80, "y": 10, "w": 60, "h": 80, "rel": False},
                    ],
                    "board": [
                        {"x": 160, "y": 10, "w": 60, "h": 80, "rel": False},
                        {"x": 230, "y": 10, "w": 60, "h": 80, "rel": False},
                        {"x": 300, "y": 10, "w": 60, "h": 80, "rel": False},
                    ],
                    "pot": {"x": 0, "y": 0, "w": 10, "h": 10, "rel": False},
                    "to_call": {"x": 0, "y": 0, "w": 10, "h": 10, "rel": False},
                    "stack": {"x": 0, "y": 0, "w": 10, "h": 10, "rel": False},
                },
            },
        }
        self.tmp_config = Path(tempfile.gettempdir()) / "test_supervision_config.yaml"
        self.tmp_config.write_text(yaml.safe_dump(cfg))
        return SupervisionStateExtractor(str(self.tmp_config))

    def tearDown(self):
        if hasattr(self, "tmp_config"):
            self.tmp_config.unlink(missing_ok=True)

    def test_extract_returns_confidence_keys(self):
        """SupervisionStateExtractor populates state['confidence'] for cards."""
        extractor = self._make_extractor()

        base_state = {
            "hole": [],
            "board": [],
            "pot": 0,
            "to_call": 0,
            "position": "BTN",
            "stage": "preflop",
            "uncertainty_reasons": [],
            "is_uncertain": False,
            "confidence": {},
        }

        # Mock local reads so the test is independent of OpenCV/Tesseract.
        with mock.patch.object(
            extractor._local,
            "extract",
            return_value=base_state,
        ), mock.patch.object(
            extractor._local,
            "_read_cards_with_confidence",
            side_effect=[
                (["As", "Kh"], [0.92, 0.88]),
                (["Qd", "Jh", "2c"], [0.85, 0.90, 0.87]),
            ],
        ):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            state = extractor.extract(frame)

        self.assertIn("confidence", state)
        self.assertIn("hole", state["confidence"])
        self.assertIn("cards", state["confidence"])
        self.assertAlmostEqual(state["confidence"]["hole"], 0.88, places=5)
        self.assertAlmostEqual(state["confidence"]["cards"], 0.85, places=5)

    def test_low_local_confidence_marks_uncertain(self):
        """A low local score makes the hand uncertain."""
        extractor = self._make_extractor()

        base_state = {
            "hole": [],
            "board": [],
            "pot": 0,
            "to_call": 0,
            "position": "BTN",
            "stage": "preflop",
            "uncertainty_reasons": [],
            "is_uncertain": False,
            "confidence": {},
        }

        with mock.patch.object(
            extractor._local,
            "extract",
            return_value=base_state,
        ), mock.patch.object(
            extractor._local,
            "_read_cards_with_confidence",
            side_effect=[
                (["As", "Kh"], [0.92, 0.30]),
                ([], []),
            ],
        ):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            state = extractor.extract(frame)

        self.assertEqual(state["confidence"]["hole"], 0.0)
        self.assertTrue(state["is_uncertain"])
        self.assertTrue(any("low local confidence" in r for r in state["uncertainty_reasons"]))


class TestSyntheticImageConfidence(unittest.TestCase):
    """Integration test proving that local confidence is not hardcoded to 1.0."""

    def test_read_cards_with_confidence_on_synthetic_frame(self):
        """A synthetic frame with synthetic templates yields real scores < 1.0."""
        from ocr_cards import generate_card_templates

        with tempfile.TemporaryDirectory() as tmpdir:
            templates = generate_card_templates(size=(60, 80), save_dir=tmpdir)
            rng = np.random.default_rng(0)

            # Build a black frame and paste the two hole-card templates exactly.
            frame = np.zeros((200, 300, 3), dtype=np.uint8)
            hole_rois = [
                {"x": 20, "y": 40, "w": 60, "h": 80, "rel": False},
                {"x": 100, "y": 40, "w": 60, "h": 80, "rel": False},
            ]
            expected = ["As", "Kh"]
            for roi, card in zip(hole_rois, expected):
                tmpl = templates[card]
                h, w = tmpl.shape[:2]
                frame[roi["y"] : roi["y"] + h, roi["x"] : roi["x"] + w] = tmpl

            # Add a small amount of noise so the match scores are not exactly 1.0.
            noise = rng.normal(0, 5, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            cfg = {
                "big_blind": 2,
                "vision": {
                    "mode": "screenshot",
                    "template_dir": tmpdir,
                    "yolo": {"enabled": False},
                    "rois": {"hole": hole_rois, "board": []},
                },
            }
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(cfg))

            extractor = ScreenshotStateExtractor(str(config_path))
            # Disable hybrid OCR path so the test is deterministic even when
            # Tesseract is installed; we want to exercise the match_card path.
            extractor.suit_templates = {}
            extractor.rank_templates = {}
            cards, scores = extractor._read_cards_with_confidence(frame, hole_rois)

        self.assertEqual(cards, expected)
        self.assertEqual(len(scores), 2)
        for score in scores:
            self.assertGreater(score, 0.0)
            self.assertLess(score, 1.0)


if __name__ == "__main__":
    unittest.main()
