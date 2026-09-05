"""Tests for the optional YOLO detector scaffolding.

These tests do not require a real YOLO model file or the ultralytics/supervision
packages to be installed. They verify the graceful fallback paths and the
configuration contract defined in Fase 1.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

# Allow importing from the project root when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vision"))

from yolo_detector import PokerYOLODetector


class TestPokerYOLODetector(unittest.TestCase):
    """Unit tests for PokerYOLODetector."""

    def test_disabled_detector_is_available_false(self):
        """When enabled is False the detector reports unavailable."""
        config = {"vision": {"yolo": {"enabled": False}}}
        detector = PokerYOLODetector(config)
        self.assertFalse(detector.available)

    def test_disabled_detector_returns_empty_detections(self):
        """A disabled detector returns an empty supervision.Detections object."""
        config = {"vision": {"yolo": {"enabled": False}}}
        detector = PokerYOLODetector(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        self.assertEqual(len(detections), 0)

    def test_enabled_missing_model_marks_unavailable(self):
        """If enabled but the model file is missing, available becomes False."""
        config = {
            "vision": {
                "yolo": {
                    "enabled": True,
                    "model_path": "vision/models/nonexistent_yolo.pt",
                }
            }
        }
        detector = PokerYOLODetector(config)
        self.assertFalse(detector.available)

    def test_enabled_missing_model_returns_empty_detections(self):
        """An enabled detector with a missing model returns empty detections."""
        config = {
            "vision": {
                "yolo": {
                    "enabled": True,
                    "model_path": "vision/models/nonexistent_yolo.pt",
                }
            }
        }
        detector = PokerYOLODetector(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        self.assertEqual(len(detections), 0)

    def test_detect_cards_returns_empty_when_disabled(self):
        """detect_cards returns an empty list when the detector is unavailable."""
        config = {"vision": {"yolo": {"enabled": False}}}
        detector = PokerYOLODetector(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cards = detector.detect_cards(frame)
        self.assertIsInstance(cards, list)
        self.assertEqual(len(cards), 0)

    def test_detect_button_placeholder(self):
        """detect_button returns the Fase 1 placeholder (None, 0.0)."""
        config = {"vision": {"yolo": {"enabled": False}}}
        detector = PokerYOLODetector(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        seat, confidence = detector.detect_button(frame)
        self.assertIsNone(seat)
        self.assertEqual(confidence, 0.0)

    def test_detect_number_regions_returns_empty_when_disabled(self):
        """detect_number_regions returns an empty list when unavailable."""
        config = {"vision": {"yolo": {"enabled": False}}}
        detector = PokerYOLODetector(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        regions = detector.detect_number_regions(frame)
        self.assertIsInstance(regions, list)
        self.assertEqual(len(regions), 0)

    def test_device_auto_selection_prefers_cuda(self):
        """auto device selection returns cuda when torch reports it available."""
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(PokerYOLODetector._resolve_device("auto"), "cuda")

    def test_device_auto_selection_falls_back_to_mps(self):
        """auto device selection falls back to mps when cuda is unavailable."""
        with mock.patch("torch.cuda.is_available", return_value=False):
            with mock.patch("torch.backends.mps.is_available", return_value=True):
                self.assertEqual(PokerYOLODetector._resolve_device("auto"), "mps")

    def test_device_auto_selection_falls_back_to_cpu(self):
        """auto device selection falls back to cpu when no accelerator exists."""
        with mock.patch("torch.cuda.is_available", return_value=False):
            with mock.patch("torch.backends.mps.is_available", return_value=False):
                self.assertEqual(PokerYOLODetector._resolve_device("auto"), "cpu")

    def test_explicit_device_is_respected(self):
        """When a concrete device is configured, _resolve_device returns it."""
        self.assertEqual(PokerYOLODetector._resolve_device("cpu"), "cpu")
        self.assertEqual(PokerYOLODetector._resolve_device("cuda:0"), "cuda:0")


if __name__ == "__main__":
    unittest.main()
