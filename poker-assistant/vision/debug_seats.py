"""
Debug overlay: disegna i seat definiti in config.yaml su uno screenshot.
Salva l'immagine in vision/debug_seats_overlay.png per verifica visiva.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import yaml


def main(image_path: str, out_path: str = "vision/debug_seats_overlay.png"):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load {image_path}")
        return
    h, w = frame.shape[:2]

    cfg_path = Path("config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    site_cfg = cfg.get("sites", {}).get("golbet", {})
    seats = site_cfg.get("seats", [])

    for seat in seats:
        idx = seat.get("index")
        x = int(seat.get("x", 0) * w)
        y = int(seat.get("y", 0) * h)
        color = (0, 255, 0) if idx == 0 else (0, 165, 255)
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.putText(frame, str(idx), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # stack_roi
        sroi = seat.get("stack_roi")
        if sroi:
            sx = int(sroi["x"] * w)
            sy = int(sroi["y"] * h)
            sw = int(sroi["w"] * w)
            sh = int(sroi["h"] * h)
            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 1)

    cv2.imwrite(out_path, frame)
    print(f"Saved overlay to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vision/debug_seats.py <screenshot.png>")
        sys.exit(1)
    main(sys.argv[1])
