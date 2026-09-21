"""Test trained classifier on case screenshots."""
import importlib
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

_card_classifier = importlib.import_module("card_classifier")
CardClassifier = _card_classifier.CardClassifier
_llm_vision_extractor = importlib.import_module("llm_vision_extractor")
_resolve_relative_roi = _llm_vision_extractor._resolve_relative_roi
_capture = importlib.import_module("capture")
crop_roi = _capture.crop_roi

CONFIG_PATH = ROOT / "config.yaml"
CACHE_ROOT = Path("C:/Users/franc/.claude")

CASES = [
    ("image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/18.png", {"hole": ["4s", "Ad"], "board": ["2c", "Kd", "Ts", "As"]}),
    ("image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/19.png", {"hole": ["8c", "Qc"], "board": []}),
    ("image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/21.png", {"hole": ["2c", "Jd"], "board": ["6h", "3h", "Ah"]}),
    ("image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/22.png", {"hole": ["4d", "Ah"], "board": ["6c", "9h", "8d"]}),
    ("image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/23.png", {"hole": ["2d", "Jd"], "board": []}),
]


def main():
    clf = CardClassifier(
        str(ROOT / "vision" / "models" / "card_rank_classifier.joblib"),
        str(ROOT / "vision" / "models" / "card_suit_classifier.joblib"),
        confidence_threshold=0.35,
    )
    if not clf.is_ready:
        print("Classifier not ready")
        return
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    except OSError as exc:
        raise SystemExit(f"config.yaml non leggibile: {exc}") from exc
    rois = cfg.get("vision", {}).get("rois", {})
    hole_rois = rois.get("hole", [])
    board_rois = rois.get("board", [])

    correct = 0
    total = 0
    for img_rel, expected in CASES:
        frame = cv2.imread(str(CACHE_ROOT / img_rel))
        if frame is None:
            print(f"\n=== {Path(img_rel).name}: non leggibile, salto ===")
            continue
        print(f"\n=== {Path(img_rel).name} ===")
        for slot, cards, roi_list in [("hole", expected["hole"], hole_rois), ("board", expected["board"], board_rois)]:
            if not cards:
                continue
            for i, expected_card in enumerate(cards):
                if i >= len(roi_list):
                    continue
                roi = _resolve_relative_roi(roi_list[i], frame)
                if roi is None:
                    continue
                crop = crop_roi(frame, roi["x"], roi["y"], roi["w"], roi["h"])
                pred, conf = clf.predict_card(crop)
                ok = pred == expected_card[0].upper() + expected_card[1].lower()
                status = "OK" if ok else "WRONG"
                print(f"  {slot}{i} expected={expected_card} pred={pred} conf={conf:.2f} [{status}]")
                total += 1
                if ok:
                    correct += 1
    print(f"\nTotal {correct}/{total} = {100*correct/total:.1f}%")


if __name__ == "__main__":
    main()
