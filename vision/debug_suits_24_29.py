"""Debug suit correction scores on #24-#29."""
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
    _detect_suit_color,
    _best_suit,
    _extract_suit_templates,
)
from capture import crop_roi

CONFIG_PATH = ROOT / "config.yaml"
TEMPLATES_DIR = ROOT / "vision" / "templates"
EXTRA_SUIT_DIR = ROOT / "vision" / "suit_templates"
CACHE_ROOT = Path("C:/Users/franc/.claude")

CASES = [
    ("24.png", {"hole": ["Kh", "Kc"], "board": []}),
    ("25.png", {"hole": ["Ts", "7d"], "board": []}),
    ("26.png", {"hole": ["Qh", "Jd"], "board": ["2s", "Qd", "7h", "Jc", "6h"]}),
    ("27.png", {"hole": ["Tc", "8s"], "board": ["6s", "7c", "Td"]}),
    ("28.png", {"hole": ["9h", "Jd"], "board": ["2s", "Qd", "7h", "Jc", "6h"]}),
    ("29.png", {"hole": ["Ts", "9c"], "board": ["4d", "Th", "Ad"]}),
]


def main():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f) or {}
    rois = cfg.get("vision", {}).get("rois", {})
    hole_rois = rois.get("hole", [])
    board_rois = rois.get("board", [])
    suit_templates = _extract_suit_templates(TEMPLATES_DIR, EXTRA_SUIT_DIR)

    for img_name, expected in CASES:
        img_path = CACHE_ROOT / "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4" / img_name
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        print(f"\n=== {img_name} ===")
        for slot, cards, roi_list in [
            ("hole", expected["hole"], hole_rois),
            ("board", expected["board"], board_rois),
        ]:
            if not cards:
                continue
            print(f"-- {slot} --")
            for i, card in enumerate(cards):
                if i >= len(roi_list):
                    continue
                roi = _resolve_relative_roi(roi_list[i], frame)
                card_crop = crop_roi(frame, roi["x"], roi["y"], roi["w"], roi["h"])
                if card_crop is None or card_crop.size == 0:
                    print(f"  {card}: empty crop")
                    continue
                suit_crop = _crop_suit_from_full_card(card_crop)
                color = _detect_suit_color(suit_crop)
                allowed = None
                if color == "red":
                    allowed = ["h", "d"]
                elif color == "black":
                    allowed = ["s", "c"]
                best, score, margin = _best_suit(suit_crop, suit_templates, threshold=0.0, allowed_suits=allowed)
                print(f"  {card} color={str(color):6} best={best} score={score:.2f} margin={margin:.2f}")


if __name__ == "__main__":
    main()
