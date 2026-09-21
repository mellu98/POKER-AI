"""
Calibra automaticamente le board ROI da uno screenshot in cui il flop/turn/river è visibile.
Cerca rettangoli bianchi (le carte) nella zona centrale-alta e restituisce le ROI in coordinate normalizzate.
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def calibrate_board(image_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load {image_path}")
        return
    h, w = frame.shape[:2]

    # Search region: top-center horizontal band
    x0, x1 = int(w * 0.25), int(w * 0.75)
    y0, y1 = int(h * 0.25), int(h * 0.55)
    search = frame[y0:y1, x0:x1]

    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    # White/light card backgrounds (permissive to handle JPEG/compression)
    _, binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

    # Clean up
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        aspect = cw / max(ch, 1)
        if not (0.3 < aspect < 1.4):
            continue
        # Convert back to full-frame coords
        abs_x = x0 + x
        abs_y = y0 + y
        cards.append((abs_x + cw / 2, abs_y + ch / 2, cw, ch, area))

    if not cards:
        print("No board card regions found.")
        return

    # Sort by x center
    cards.sort(key=lambda c: c[0])

    print(f"Found {len(cards)} card-like regions:")
    for cx, cy, cw, ch, area in cards:
        roi = {
            "x": round((cx - cw / 2) / w, 3),
            "y": round((cy - ch / 2) / h, 3),
            "w": round(cw / w, 3),
            "h": round(ch / h, 3),
            "rel": True,
        }
        print(f"  center=({cx:.0f},{cy:.0f}) area={area:.0f} ROI={roi}")

    # Save debug image
    debug = frame.copy()
    for cx, cy, cw, ch, area in cards:
        x = int(cx - cw / 2)
        y = int(cy - ch / 2)
        cv2.rectangle(debug, (x, y), (x + int(cw), y + int(ch)), (0, 0, 255), 2)
    out = Path(image_path).with_suffix(".board_debug.png")
    cv2.imwrite(str(out), debug)
    print(f"Saved debug image to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibrate_board.py <screenshot_with_board.png>")
        sys.exit(1)
    calibrate_board(sys.argv[1])
