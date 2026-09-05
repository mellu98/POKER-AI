"""Debug local card recognition on screenshots #24-#29."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2
import yaml
from capture import crop_roi
from ocr_cards import load_templates_from_dir, load_rank_templates, load_suit_templates, recognize_cards_rois
from llm_vision_extractor import _resolve_relative_roi

CONFIG_PATH = ROOT / "config.yaml"
TEMPLATES_DIR = ROOT / "vision" / "templates"
CACHE_ROOT = Path("C:/Users/franc/.claude")

CASES = [
    ("24.png", {"hole": [], "board": []}),  # preflop, no board
    ("25.png", {"hole": ["Ts", "7d"], "board": ["4d", "Th", "Ad"]}),
    ("26.png", {"hole": ["Qc", "Js"], "board": ["2s", "Qd", "7h", "Jc", "6h"]}),
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

    full = load_templates_from_dir(str(TEMPLATES_DIR))
    rank_tmpl = load_rank_templates(str(TEMPLATES_DIR))
    suit_tmpl = load_suit_templates(str(TEMPLATES_DIR))

    for img_name, expected in CASES:
        img_path = CACHE_ROOT / "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4" / img_name
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"\n{img_name}: cannot read")
            continue
        print(f"\n=== {img_name} ===")
        for slot, exp_cards, roi_list in [
            ("hole", expected["hole"], hole_rois),
            ("board", expected["board"], board_rois),
        ]:
            if not exp_cards:
                continue
            resolved = [_resolve_relative_roi(r, frame) for r in roi_list]
            preds = recognize_cards_rois(
                frame, resolved, full,
                rank_templates=rank_tmpl, suit_templates=suit_tmpl,
            )
            print(f"-- {slot} --")
            for i, (exp, pred) in enumerate(zip(exp_cards, preds)):
                ok = pred == exp
                print(f"  ROI{i}: expected={exp} local={pred} {'OK' if ok else 'WRONG'}")


if __name__ == "__main__":
    main()
