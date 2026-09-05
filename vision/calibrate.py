"""
Vision calibration helper — extracts candidate ROIs from a screenshot
and runs EasyOCR to find cards, pot, stack, etc.
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import easyocr
    READER = easyocr.Reader(["en"], gpu=False)
except Exception as e:
    print(f"EasyOCR init failed: {e}")
    READER = None


def crop_and_save(frame: np.ndarray, name: str, x: int, y: int, w: int, h: int):
    """Crop a region and save it to vision/debug/{name}.png"""
    h_frame, w_frame = frame.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_frame - x)
    h = min(h, h_frame - y)
    roi = frame[y : y + h, x : x + w]
    out = Path(__file__).parent / "debug" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), roi)
    print(f"Saved {out}  ({x},{y},{w},{h})")
    return roi


def ocr_text(roi: np.ndarray) -> list:
    """Run EasyOCR on a ROI and return text results."""
    if READER is None:
        return []
    results = READER.readtext(roi)
    return [(r[1], r[2]) for r in results]  # text, confidence


def main():
    img_path = Path(r"C:\Users\franc\.claude\image-cache\4a0bae80-b8b2-4141-9caa-41dee5631ae0\8.png")
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise RuntimeError(f"Could not load {img_path}")
    h, w = frame.shape[:2]
    print(f"Screenshot size: {w}x{h}")

    # --- Candidate ROIs based on visual estimation for 1999x1086 ---
    # Hole cards (bottom center)
    crop_and_save(frame, "hole_cards", x=860, y=560, w=280, h=160)
    # Individual card left & right (approx)
    crop_and_save(frame, "card_left", x=870, y=570, w=130, h=140)
    crop_and_save(frame, "card_right", x=1000, y=570, w=130, h=140)
    # Rank zones (top-left of each card)
    crop_and_save(frame, "rank_left", x=875, y=575, w=30, h=30)
    crop_and_save(frame, "rank_right", x=1005, y=575, w=30, h=30)
    # Suit zones (bottom-left-ish of each card)
    crop_and_save(frame, "suit_left", x=875, y=650, w=30, h=30)
    crop_and_save(frame, "suit_right", x=1005, y=650, w=30, h=30)

    # Pot (center top-ish)
    crop_and_save(frame, "pot", x=900, y=180, w=200, h=80)

    # Stack "You" (bottom center, under name)
    crop_and_save(frame, "stack_you", x=920, y=780, w=160, h=40)

    # Dealer button on Ruth (right side)
    crop_and_save(frame, "dealer_ruth", x=1280, y=310, w=50, h=50)

    # Board area (center, above pot)
    crop_and_save(frame, "board", x=750, y=250, w=500, h=150)

    # Action buttons (bottom)
    crop_and_save(frame, "actions", x=350, y=950, w=1300, h=80)

    # --- OCR sanity check ---
    if READER:
        print("\n--- OCR Results ---")
        for name in ["pot", "stack_you", "actions", "rank_left", "rank_right", "suit_left", "suit_right"]:
            roi_path = Path(__file__).parent / "debug" / f"{name}.png"
            roi = cv2.imread(str(roi_path))
            if roi is None:
                continue
            texts = ocr_text(roi)
            print(f"{name:15s}: {texts}")


if __name__ == "__main__":
    main()
