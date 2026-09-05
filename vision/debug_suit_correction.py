"""
Debug local suit correction on saved screenshots.

Usage (from repo root):
    .venv\Scripts\python.exe vision\debug_suit_correction.py
"""
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

# (image_file, expected_hole, expected_board)
CASES = [
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/18.png",
        ["4s", "Ad"],
        ["2c", "Kd", "Ts", "As"],
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/19.png",
        ["8c", "Qc"],
        [],
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/21.png",
        ["2c", "Jd"],
        ["6h", "3h", "Ah"],
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/22.png",
        ["4d", "Ah"],
        ["6c", "9h", "8d"],
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/23.png",
        ["2d", "Jd"],
        [],
    ),
]


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


def main():
    cfg = load_config()
    rois = cfg.get("vision", {}).get("rois", {})
    hole_rois = rois.get("hole", [])
    board_rois = rois.get("board", [])

    suit_templates = _extract_suit_templates(TEMPLATES_DIR, EXTRA_SUIT_DIR)
    print(f"Loaded {sum(len(v) for v in suit_templates.values())} suit templates")

    cache_root = Path("C:/Users/franc/.claude")

    for img_rel, exp_hole, exp_board in CASES:
        img_path = cache_root / img_rel
        if not img_path.exists():
            print(f"\n[SKIP] {img_path} not found")
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"\n[SKIP] Cannot read {img_path}")
            continue

        print(f"\n{'='*60}\n{img_path.name}  {frame.shape[1]}x{frame.shape[0]}")

        for slot_name, exp_cards, rois_list in [
            ("hole", exp_hole, hole_rois),
            ("board", exp_board, board_rois),
        ]:
            if not exp_cards:
                continue
            print(f"\n-- {slot_name} --")
            for i, expected in enumerate(exp_cards):
                if i >= len(rois_list):
                    print(f"  {expected}: no ROI")
                    continue
                roi = _resolve_relative_roi(rois_list[i], frame)
                card_crop = crop_roi(frame, roi["x"], roi["y"], roi["w"], roi["h"])
                if card_crop is None or card_crop.size == 0:
                    print(f"  {expected}: empty crop")
                    continue
                suit_crop = _crop_suit_from_full_card(card_crop)
                color = _detect_suit_color(suit_crop)

                # best within color filter
                allowed = None
                if color == "red":
                    allowed = ["h", "d"]
                elif color == "black":
                    allowed = ["s", "c"]

                best_suit, score, margin = _best_suit(
                    suit_crop, suit_templates, threshold=0.35, allowed_suits=allowed
                )
                # also compute all-suit scores for reference
                all_best, all_score, all_margin = _best_suit(
                    suit_crop, suit_templates, threshold=0.35, allowed_suits=None
                )
                rank = expected[0].upper()
                template_exists = (TEMPLATES_DIR / f"{rank}{best_suit}.png").exists() if best_suit else False
                print(
                    f"  {expected}  color={color:6} best={best_suit} score={score:.2f} "
                    f"margin={margin:.2f} all_best={all_best} all_score={all_score:.2f} "
                    f"template_exists={template_exists}"
                )


if __name__ == "__main__":
    main()
