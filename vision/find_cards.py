"""
Find card contours in a screenshot using color segmentation.
"""
import cv2
import numpy as np
from pathlib import Path


def find_white_cards(frame: np.ndarray):
    """
    Find rectangular white card regions in the frame.
    Returns list of (x, y, w, h) sorted by y descending (bottom first).
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White-ish range (cards background)
    # These values may need tuning depending on client brightness
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Morphological ops to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    h_frame, w_frame = frame.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000 or area > 30000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0
        if aspect < 0.5 or aspect > 1.0:
            continue
        # Must be in the bottom half of the screen (player cards)
        # or center (board cards). For now accept anywhere.
        cards.append((x, y, w, h))

    # Sort by y descending (bottom of image first) then x
    cards.sort(key=lambda b: (b[1], b[0]), reverse=False)
    return cards, mask


def main():
    img_path = Path(r"C:\Users\franc\.claude\image-cache\4a0bae80-b8b2-4141-9caa-41dee5631ae0\8.png")
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise RuntimeError(f"Could not load {img_path}")

    cards, mask = find_white_cards(frame)
    print(f"Found {len(cards)} card candidates")

    # Draw debug image
    debug = frame.copy()
    for i, (x, y, w, h) in enumerate(cards):
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"  Card {i}: x={x}, y={y}, w={w}, h={h}, aspect={w/h:.2f}")

    out = Path(__file__).parent / "debug" / "found_cards.png"
    cv2.imwrite(str(out), debug)
    print(f"Saved debug to {out}")

    # Save rank and suit crops for the first few cards
    for i, (x, y, w, h) in enumerate(cards[:6]):
        rank_roi = frame[y : y + int(h * 0.35), x : x + int(w * 0.35)]
        suit_roi = frame[y + int(h * 0.65) : y + h, x : x + int(w * 0.35)]
        cv2.imwrite(str(Path(__file__).parent / "debug" / f"found_rank_{i}.png"), rank_roi)
        cv2.imwrite(str(Path(__file__).parent / "debug" / f"found_suit_{i}.png"), suit_roi)


if __name__ == "__main__":
    main()
