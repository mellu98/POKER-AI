"""
Find player cards using white mask + Tesseract OCR on top-left corner.
"""
import cv2
import numpy as np
from pathlib import Path

# Tesseract config
try:
    import pytesseract
    from tesseract_utils import find_tesseract_binary
    pytesseract.pytesseract.tesseract_cmd = find_tesseract_binary()
except Exception:
    pytesseract = None


def find_white_regions(frame: np.ndarray):
    """Find bright regions (card faces) in bottom half."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # threshold for white-ish pixels (lowered to catch shaded cards)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    # focus on bottom half where player cards are
    h, w = mask.shape
    mask[: h // 2, :] = 0
    # morphological close to connect card regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000 or area > 30000:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / ch if ch > 0 else 0
        if aspect < 0.4 or aspect > 1.2:
            continue
        cards.append((x, y, cw, ch))
    # sort by x to get left-to-right
    cards.sort(key=lambda b: b[0])
    return cards, mask


def ocr_rank(roi: np.ndarray) -> str:
    if pytesseract is None:
        return ""
    # preprocess: threshold to black text on white
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    # enlarge for better OCR
    binary = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(
        binary,
        config="--psm 10 -c tessedit_char_whitelist=AKQJT98765432",
    )
    return text.strip()


def main():
    img_path = Path(r"C:\Users\franc\.claude\image-cache\4a0bae80-b8b2-4141-9caa-41dee5631ae0\8.png")
    frame = cv2.imread(str(img_path))
    cards, mask = find_white_regions(frame)
    print(f"Found {len(cards)} card region(s)")

    debug = frame.copy()
    results = []
    for i, (x, y, w, h) in enumerate(cards):
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # extract top-left corner for rank
        rank_roi = frame[y : y + int(h * 0.4), x : x + int(w * 0.4)]
        rank = ocr_rank(rank_roi)
        results.append((x, y, w, h, rank))
        cv2.putText(debug, f"{i}:{rank}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"  Card {i}: x={x}, y={y}, w={w}, h={h}, rank='{rank}'")
        # save corner crop
        cv2.imwrite(str(Path(__file__).parent / "debug" / f"card_corner_{i}.png"), rank_roi)

    out = Path(__file__).parent / "debug" / "found_cards_final.png"
    cv2.imwrite(str(out), debug)
    print(f"Saved debug to {out}")


if __name__ == "__main__":
    main()
