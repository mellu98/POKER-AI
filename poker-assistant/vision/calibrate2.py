"""
Second calibration pass — try wider ranges to locate all elements.
"""
import cv2
from pathlib import Path


def crop_and_save(frame, name, x, y, w, h):
    h_f, w_f = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    w = min(w, w_f - x)
    h = min(h, w_f - y)
    roi = frame[y : y + h, x : x + w]
    out = Path(__file__).parent / "debug" / f"{name}.png"
    cv2.imwrite(str(out), roi)
    print(f"Saved {out}  ({x},{y},{w},{h})")
    return roi


def main():
    img_path = Path(r"C:\Users\franc\.claude\image-cache\4a0bae80-b8b2-4141-9caa-41dee5631ae0\8.png")
    frame = cv2.imread(str(img_path))
    h, w = frame.shape[:2]
    print(f"Size {w}x{h}")

    # Wider hole-card search
    crop_and_save(frame, "hole_wide", 800, 550, 450, 200)
    crop_and_save(frame, "hole_right_card", 950, 560, 150, 160)
    crop_and_save(frame, "hole_right_rank", 955, 565, 40, 40)
    crop_and_save(frame, "hole_right_suit", 955, 650, 40, 40)

    # Wider rank/suit left
    crop_and_save(frame, "rank_left_big", 870, 570, 50, 50)
    crop_and_save(frame, "suit_left_big", 870, 640, 50, 50)

    # Pot — seen in board crop at left side, y~250
    crop_and_save(frame, "pot_1", 700, 220, 250, 120)
    crop_and_save(frame, "pot_2", 750, 220, 150, 80)
    crop_and_save(frame, "pot_3", 770, 240, 100, 60)

    # Stack — above "You" label, red label with $998
    crop_and_save(frame, "stack_1", 900, 720, 200, 60)
    crop_and_save(frame, "stack_2", 920, 740, 160, 40)
    crop_and_save(frame, "stack_3", 930, 700, 140, 50)

    # Dealer button — red "D" near Ruth
    crop_and_save(frame, "dealer_1", 1250, 290, 80, 80)
    crop_and_save(frame, "dealer_2", 1280, 300, 60, 60)
    crop_and_save(frame, "dealer_3", 1260, 280, 100, 100)

    # Board (should be empty but let's see)
    crop_and_save(frame, "board_center", 800, 250, 400, 120)

    # Actions buttons already good, but also crop individual buttons
    crop_and_save(frame, "btn_fold", 350, 970, 200, 60)
    crop_and_save(frame, "btn_check", 620, 970, 200, 60)
    crop_and_save(frame, "btn_raise", 890, 970, 200, 60)

    # Look for call button (not present here, but for future)
    crop_and_save(frame, "btn_call", 620, 970, 200, 60)


if __name__ == "__main__":
    main()
