"""Detect board card positions in screenshots for ROI recalibration."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2
import numpy as np

CACHE_ROOT = Path("C:/Users/franc/.claude")


def detect_board_cards(frame):
    h, w = frame.shape[:2]
    # focus on upper-center region
    x0, x1 = int(w * 0.25), int(w * 0.75)
    y0, y1 = int(h * 0.30), int(h * 0.60)
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # white card background
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        aspect = cw / max(ch, 1)
        if area > 800 and 0.45 < aspect < 1.0 and ch > 25:
            boxes.append((x + x0, y + y0, cw, ch))
    return boxes


def main():
    for num in (18, 22, 25, 26, 27, 28, 29):
        path = CACHE_ROOT / f"image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/{num}.png"
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        boxes = detect_board_cards(frame)
        print(f"\n{num}.png ({w}x{h}) detected {len(boxes)} board cards")
        boxes = sorted(boxes)
        for i, (x, y, cw, ch) in enumerate(boxes[:7]):
            print(f"  card{i}: x={x/w:.3f} y={y/h:.3f} w={cw/w:.3f} h={ch/h:.3f}")


if __name__ == "__main__":
    main()
