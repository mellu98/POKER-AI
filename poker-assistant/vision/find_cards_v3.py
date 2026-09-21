"""
Find cards using Canny edge detection + contour filtering.
"""
import cv2
import numpy as np
from pathlib import Path


def find_card_contours(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_frame, w_frame = frame.shape[:2]
    cards = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4000 or area > 25000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0
        if aspect < 0.55 or aspect > 0.85:
            continue
        # Additional check: interior should be mostly bright (white card face)
        roi = gray[y:y+h, x:x+w]
        mean_brightness = roi.mean()
        if mean_brightness < 120:
            continue
        cards.append((x, y, w, h))

    # NMS
    cards = sorted(cards, key=lambda b: b[2]*b[3], reverse=True)
    filtered = []
    for c in cards:
        x, y, w, h = c
        keep = True
        for fx, fy, fw, fh in filtered:
            if abs(x - fx) < 40 and abs(y - fy) < 40:
                keep = False
                break
        if keep:
            filtered.append(c)

    return filtered, edges


def main():
    img_path = Path(r"C:\Users\franc\.claude\image-cache\4a0bae80-b8b2-4141-9caa-41dee5631ae0\8.png")
    frame = cv2.imread(str(img_path))

    cards, edges = find_card_contours(frame)
    print(f"Found {len(cards)} card candidates")

    debug = frame.copy()
    for i, (x, y, w, h) in enumerate(cards):
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(f"  Card {i}: x={x}, y={y}, w={w}, h={h}, aspect={w/h:.2f}")

    out = Path(__file__).parent / "debug" / "canny_cards.png"
    cv2.imwrite(str(out), debug)
    cv2.imwrite(str(Path(__file__).parent / "debug" / "canny_edges.png"), edges)
    print(f"Saved debug to {out}")


if __name__ == "__main__":
    main()
