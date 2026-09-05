"""
Find suit symbols in the bottom-center region (player cards) using template matching.
"""
import cv2
import numpy as np
from pathlib import Path


def extract_suit_template(crop_path: str):
    img = cv2.imread(crop_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No suit contour found")
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    template = gray[y : y + h, x : x + w]
    return template


def main():
    img_path = Path(r"C:\Users\franc\.claude\image-cache\4a0bae80-b8b2-4141-9caa-41dee5631ae0\8.png")
    frame = cv2.imread(str(img_path))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_frame, w_frame = gray.shape

    suit_crop = str(Path(__file__).parent / "debug" / "suit_left.png")
    template = extract_suit_template(suit_crop)
    print(f"Template shape: {template.shape}")

    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.75)
    matches = []
    for pt in zip(*loc[::-1]):
        score = res[pt[1], pt[0]]
        matches.append((int(pt[0]), int(pt[1]), float(score)))

    # NMS + region filter
    matches = sorted(matches, key=lambda m: m[2], reverse=True)
    filtered = []
    for m in matches:
        x, y, s = m
        # Keep only matches in bottom-center region (player cards)
        if not (500 <= y <= 800 and 700 <= x <= 1200):
            continue
        keep = True
        for fx, fy, fs in filtered:
            if abs(x - fx) < 50 and abs(y - fy) < 50:
                keep = False
                break
        if keep:
            filtered.append(m)

    print(f"Found {len(filtered)} suit match(es) in player region")
    debug = frame.copy()
    for i, (x, y, s) in enumerate(filtered):
        th, tw = template.shape
        cv2.rectangle(debug, (x, y), (x + tw, y + th), (0, 0, 255), 2)
        cv2.putText(debug, f"{s:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        print(f"  Match {i}: x={x}, y={y}, score={s:.3f}")

    out = Path(__file__).parent / "debug" / "suits_v2.png"
    cv2.imwrite(str(out), debug)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
