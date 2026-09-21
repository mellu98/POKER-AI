"""Compare edge density in board ROI for empty vs populated boards."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2
import yaml
from llm_vision_extractor import _resolve_relative_roi

CACHE_ROOT = Path("C:/Users/franc/.claude")


def board_roi_region(frame, board_rois, pad=0.02):
    h, w = frame.shape[:2]
    xs, ys = [], []
    for r in board_rois:
        roi = _resolve_relative_roi(r, frame)
        if roi is None:
            continue
        xs += [roi["x"], roi["x"] + roi["w"]]
        ys += [roi["y"], roi["y"] + roi["h"]]
    if not xs:
        return frame
    try:
        x0, x1 = max(0, min(xs) - int(pad * w)), min(w, max(xs) + int(pad * w))
        y0, y1 = max(0, min(ys) - int(pad * h)), min(h, max(ys) + int(pad * h))
    except ValueError:
        return frame
    return frame[y0:y1, x0:x1]


def main():
    try:
        with open(ROOT / "config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
    except OSError as exc:
        raise SystemExit(f"config.yaml non leggibile: {exc}") from exc
    board_rois = cfg.get("vision", {}).get("rois", {}).get("board", [])
    for num in (18, 22, 25, 26, 27, 28, 29):
        path = CACHE_ROOT / f"image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/{num}.png"
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"{num}.png: non leggibile, salto")
            continue
        crop = board_roi_region(frame, board_rois)
        if crop is None or crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = edges.sum() / (edges.size * 255)
        # also white pixel ratio
        white = (gray > 200).sum() / gray.size
        print(f"{num}.png: edge_ratio={edge_ratio:.3f} white_ratio={white:.3f}")


if __name__ == "__main__":
    main()
