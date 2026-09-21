"""Inspect white contours in board region for #25."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2
import numpy as np

CACHE_ROOT = Path("C:/Users/franc/.claude")


def main():
    path = CACHE_ROOT / "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/25.png"
    frame = cv2.imread(str(path))
    h, w = frame.shape[:2]
    x0, x1 = int(w * 0.20), int(w * 0.80)
    y0, y1 = int(h * 0.25), int(h * 0.65)
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    for thresh in (160, 180, 200):
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"\nthresh={thresh}: {len(contours)} contours")
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            print(f"  x={(x+x0)/w:.3f} y={(y+y0)/h:.3f} w={cw/w:.3f} h={ch/h:.3f} area={area} aspect={cw/max(ch,1):.2f}")


if __name__ == "__main__":
    main()
