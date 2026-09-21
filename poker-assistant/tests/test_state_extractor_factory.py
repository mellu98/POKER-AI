"""Focused tests for the state extractor factory.

These tests guard the factory routing introduced in Fase 1 without requiring
vision hardware or a real YOLO model.
"""
import sys
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vision"))

from state_extractor import get_extractor, ManualStateExtractor, ScreenshotStateExtractor, SupervisionStateExtractor


class TestGetExtractor(unittest.TestCase):
    """Factory routing tests."""

    def test_manual_mode_returns_manual_extractor(self):
        """get_extractor('manual') returns a ManualStateExtractor."""
        extractor = get_extractor("manual")
        self.assertIsInstance(extractor, ManualStateExtractor)

    def test_screenshot_mode_with_yolo_disabled_returns_screenshot_extractor(self):
        """With yolo.enabled=false the factory returns ScreenshotStateExtractor."""
        extractor = get_extractor("screenshot", config_path="config.yaml")
        self.assertIsInstance(extractor, ScreenshotStateExtractor)
        self.assertNotIsInstance(extractor, SupervisionStateExtractor)

    def test_webcam_mode_with_yolo_disabled_returns_screenshot_extractor(self):
        """Webcam mode falls back to the screenshot extractor when YOLO is off."""
        extractor = get_extractor("webcam", config_path="config.yaml")
        self.assertIsInstance(extractor, ScreenshotStateExtractor)

    def test_screenshot_mode_with_yolo_enabled_but_missing_model(self):
        """Even with yolo.enabled=true, a missing model keeps local extraction."""
        config_path = Path("config.yaml")
        cfg = yaml.safe_load(config_path.read_text()) or {}
        cfg.setdefault("vision", {})["yolo"] = {
            "enabled": True,
            "model_path": "vision/models/does_not_exist.pt",
        }

        temp_config = Path("config.test_yolo_missing.yaml")
        try:
            temp_config.write_text(yaml.safe_dump(cfg))
            extractor = get_extractor("screenshot", config_path=str(temp_config))
            self.assertIsInstance(extractor, ScreenshotStateExtractor)
            self.assertNotIsInstance(extractor, SupervisionStateExtractor)
        finally:
            temp_config.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
