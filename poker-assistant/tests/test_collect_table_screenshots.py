"""Tests for the table screenshot collector."""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "vision"))

from collect_table_screenshots import _frame_hash, collect_table_screenshots


def test_frame_hash_is_stable():
    """The same frame must always produce the same hash."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert _frame_hash(frame) == _frame_hash(frame)


def test_frame_hash_detects_differences():
    """Different frames should almost never collide."""
    frame_a = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame_b = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    assert _frame_hash(frame_a) != _frame_hash(frame_b)


def test_collection_stops_at_max_and_skips_duplicates():
    """Collector should save max_screenshots unique frames and skip duplicates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        call_count = [0]

        def fake_screenshot(*, window_title=None):
            call_count[0] += 1
            # Return the same frame every time to test duplicate skipping.
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

        with patch("collect_table_screenshots.screenshot", fake_screenshot):
            saved = collect_table_screenshots(
                output_dir=tmpdir,
                window_title="Poker - Opera",
                interval_seconds=0.01,
                max_screenshots=1,
                skip_duplicates=True,
            )

        # With skip_duplicates=True, only one unique frame is saved.
        assert saved == 1
        assert len(list(Path(tmpdir).glob("*.png"))) == 1


def test_collection_saves_multiple_unique_frames():
    """When frames differ, all of them are saved up to max_screenshots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        counter = [0]

        def fake_screenshot(*, window_title=None):
            counter[0] += 1
            # Each frame has a different color.
            value = (counter[0] * 10) % 256
            return np.full((1080, 1920, 3), value, dtype=np.uint8)

        with patch("collect_table_screenshots.screenshot", fake_screenshot):
            saved = collect_table_screenshots(
                output_dir=tmpdir,
                window_title="Poker - Opera",
                interval_seconds=0.01,
                max_screenshots=3,
                skip_duplicates=True,
            )

        assert saved == 3
        assert len(list(Path(tmpdir).glob("*.png"))) == 3


if __name__ == "__main__":
    test_frame_hash_is_stable()
    test_frame_hash_detects_differences()
    test_collection_stops_at_max_and_skips_duplicates()
    test_collection_saves_multiple_unique_frames()
    print("All collector tests passed.")
