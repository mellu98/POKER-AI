"""
Basic vision tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from ocr_cards import generate_card_templates, match_card
from state_extractor import ManualStateExtractor, get_extractor


def test_generate_templates():
    templates = generate_card_templates()
    assert len(templates) == 52
    assert "As" in templates
    print("[PASS] Generated 52 synthetic templates")


def test_match_synthetic():
    """A synthetic template should match itself perfectly."""
    templates = generate_card_templates()
    for name, tmpl in templates.items():
        result = match_card(tmpl, templates)
        assert result == name, f"Expected {name}, got {result}"
    print("[PASS] All synthetic templates self-match")


def test_manual_extractor():
    ext = ManualStateExtractor()
    state = {
        "hole": ["As", "Kh"],
        "board": ["Qd", "Jh", "2c"],
        "pot": 120,
        "to_call": 20,
        "position": "BTN",
        "stage": "flop",
    }
    out = ext.extract(state)
    assert out == state
    print("[PASS] Manual extractor works")


def test_screenshot_capture():
    from capture import screenshot
    frame = screenshot()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    print(f"[PASS] Screenshot captured: {frame.shape}")


if __name__ == "__main__":
    test_generate_templates()
    test_match_synthetic()
    test_manual_extractor()
    test_screenshot_capture()
    print("\n=== ALL VISION TESTS PASSED ===")
