"""
Find cards using Canny edge detection — debug version to tune parameters.
"""
import cv2
import numpy as np
from pathlib import Path


def find_card_contours(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    h_frame, w_frame = frame.shape[:2]
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0
        candidates.append((x, y, w, h, area, aspect))

    # Sort by area desc
    candidates.sort(key=lambda c: c[4], reverse=True)
    return candidates[:30], edges


def main():
    img_path = Path(r"C:\Users\franc\.claude\image-cache\4a0bae80-b8b2-4141-9caa-41dee5631ae0\8.png")
    frame = cv2.imread(str(img_path))

    candidates, edges = find_card_contours(frame)
    print(f"Top 30 contour candidates:")

    debug = frame.copy()
    for i, (x, y, w, h, area, aspect) in enumerate(candidates):
        color = (0, 255, 0) if 0.5 < aspect < 0.9 else (0, 0, 255)
        cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
        cv2.putText(debug, f"{i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        print(f"  {i:2d}: x={x:4d} y={y:4d} w={w:3d} h={h:3d} area={area:6.0f} aspect={aspect:.2f}")

    out = Path(__file__).parent / "debug" / "canny_debug.png"
    cv2.imwrite(str(out), debug)
    print(f"Saved debug to {out}")


if __name__ == "__main__":
    main()
