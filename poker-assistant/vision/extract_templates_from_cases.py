"""Extract accurate full-card templates from the user's case screenshots.

Saves only cards that are not already present in vision/templates/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2
import yaml
from llm_vision_extractor import _resolve_relative_roi
from capture import crop_roi

CONFIG_PATH = ROOT / "config.yaml"
TEMPLATES_DIR = ROOT / "vision" / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)
CACHE_ROOT = Path("C:/Users/franc/.claude")

CASES = [
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/18.png",
        {"hole": ["4s", "Ad"], "board": ["2c", "Kd", "Ts", "As"]},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/19.png",
        {"hole": ["8c", "Qc"], "board": []},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/21.png",
        {"hole": ["2c", "Jd"], "board": ["6h", "3h", "Ah"]},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/22.png",
        {"hole": ["4d", "Ah"], "board": ["6c", "9h", "8d"]},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/23.png",
        {"hole": ["2d", "Jd"], "board": []},
    ),
    # nuovi casi #24-#29
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/24.png",
        {"hole": ["Kh", "Kc"], "board": []},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/25.png",
        {"hole": ["Ts", "7d"], "board": ["4d", "Th", "Ad"]},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/26.png",
        {"hole": ["Qh", "Jd"], "board": ["2s", "Qd", "7h", "Jc", "6h"]},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/27.png",
        {"hole": ["Tc", "8s"], "board": ["6s", "7c", "Td"]},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/28.png",
        {"hole": ["9h", "Jd"], "board": ["2s", "Qd", "7h", "Jc", "6h"]},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/29.png",
        {"hole": ["Ts", "9c"], "board": ["4d", "Th", "Ad"]},
    ),
]

STD_SIZE = (60, 80)


def main():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f) or {}
    rois = cfg.get("vision", {}).get("rois", {})
    hole_rois = rois.get("hole", [])
    board_rois = rois.get("board", [])

    saved = []
    skipped = []

    for img_rel, expected in CASES:
        img_path = CACHE_ROOT / img_rel
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[skip] cannot read {img_path}")
            continue

        for slot, cards, roi_list in [
            ("hole", expected["hole"], hole_rois),
            ("board", expected["board"], board_rois),
        ]:
            for i, card in enumerate(cards):
                if i >= len(roi_list):
                    continue
                card = card[0].upper() + card[1].lower()
                out_path = TEMPLATES_DIR / f"{card}.png"
                if out_path.exists():
                    skipped.append(card)
                    continue
                roi = _resolve_relative_roi(roi_list[i], frame)
                crop = crop_roi(frame, roi["x"], roi["y"], roi["w"], roi["h"])
                if crop is None or crop.size == 0:
                    print(f"[skip] empty crop for {card} in {img_path.name}")
                    continue
                resized = cv2.resize(crop, STD_SIZE, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out_path), resized)
                saved.append(card)

    print(f"Saved {len(saved)} new templates: {sorted(set(saved))}")
    print(f"Skipped existing: {len(skipped)} unique: {sorted(set(skipped))}")


if __name__ == "__main__":
    main()
