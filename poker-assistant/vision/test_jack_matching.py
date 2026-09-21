"""Test same-rank suit matching for face cards on #26 and #28."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2
import yaml
from llm_vision_extractor import (
    _resolve_relative_roi,
    _crop_suit_from_full_card,
    _detect_card_color,
    _match_template_resized,
)
from capture import crop_roi
from ocr_cards import load_templates_from_dir, extract_split_templates_from_full

CACHE_ROOT = Path("C:/Users/franc/.claude")

CASES = [
    ("26.png", "hole", 1, "Jd"),
    ("28.png", "hole", 1, "Jd"),
    ("28.png", "board", 3, "Jc"),
]


def main():
    with open(ROOT / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f) or {}
    rois = cfg.get("vision", {}).get("rois", {})
    hole_rois = rois.get("hole", [])
    board_rois = rois.get("board", [])
    full = load_templates_from_dir(str(ROOT / "vision" / "templates"))
    _, suit_tmpl = extract_split_templates_from_full(full)

    for img_name, slot, idx, expected in CASES:
        path = CACHE_ROOT / f"image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/{img_name}"
        frame = cv2.imread(str(path))
        roi_list = hole_rois if slot == "hole" else board_rois
        roi = _resolve_relative_roi(roi_list[idx], frame)
        card_crop = crop_roi(frame, roi["x"], roi["y"], roi["w"], roi["h"])
        suit_crop = _crop_suit_from_full_card(card_crop)
        color = _detect_card_color(card_crop)
        rank = expected[0].upper()
        print(f"\n{img_name} {slot}{idx} expected={expected} color={color}")
        for suit in "shdc":
            if (ROOT / "vision" / "templates" / f"{rank}{suit}.png").exists() and suit in suit_tmpl:
                best = max(_match_template_resized(suit_crop, t, (40,40)) for t in suit_tmpl[suit])
                print(f"  vs {rank}{suit}: {best:.2f}")


if __name__ == "__main__":
    main()
