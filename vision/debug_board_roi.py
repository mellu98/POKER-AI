"""Disegna le board ROI di config.yaml su uno screenshot salvato."""
import sys
from pathlib import Path

import cv2
import yaml


def main(image_path: str, out_path: str = "vision/debug_board_roi.png"):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load {image_path}")
        return
    h, w = frame.shape[:2]

    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}
    # Use effective ROI: site-specific if present, else legacy vision.rois
    site = cfg.get("vision", {}).get("site", "golbet")
    site_cfg = cfg.get("sites", {}).get(site, {})
    board_rois = site_cfg.get("board") or cfg.get("vision", {}).get("rois", {}).get("board", [])

    for i, r in enumerate(board_rois):
        x = int(r["x"] * w)
        y = int(r["y"] * h)
        rw = int(r["w"] * w)
        rh = int(r["h"] * h)
        cv2.rectangle(frame, (x, y), (x + rw, y + rh), (0, 0, 255), 2)
        cv2.putText(frame, f"B{i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(out_path, frame)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vision/debug_board_roi.py <screenshot.png>")
        sys.exit(1)
    main(sys.argv[1])
