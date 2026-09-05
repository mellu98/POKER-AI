"""
Find suit symbols in the screenshot using template matching.
We extract the club symbol from the calibration crop and match it.
"""
import cv2
import numpy as np
from pathlib import Path


def extract_suit_template(crop_path: str):
    """Extract the black suit symbol from a crop, removing green background."""
    img = cv2.imread(crop_path)
    if img is None:
        raise RuntimeError(f"Cannot load {crop_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # The suit is black (~0), background is green or white.
    # Use threshold to isolate dark pixels.
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    # Find largest contour = suit
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No suit contour found")
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    template = gray[y : y + h, x : x + w]
    return template, mask


def find_matches(frame_gray: np.ndarray, template: np.ndarray, threshold: float = 0.6):
    """Run matchTemplate and return list of (x, y, score)."""
    res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    matches = []
    for pt in zip(*loc[::-1]):
        score = res[pt[1], pt[0]]
        matches.append((int(pt[0]), int(pt[1]), float(score)))
    return matches


def non_max_suppression(matches, min_dist=30):
    """Simple NMS to remove overlapping matches."""
    matches = sorted(matches, key=lambda m: m[2], reverse=True)
    filtered = []
    for m in matches:
        x, y, s = m
        keep = True
        for fx, fy, fs in filtered:
            if abs(x - fx) < min_dist and abs(y - fy) < min_dist:
                keep = False
                break
        if keep:
            filtered.append(m)
    return filtered


def main():
    img_path = Path(r"C:\Users\franc\.claude\image-cache\4a0bae80-b8b2-4141-9caa-41dee5631ae0\8.png")
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise RuntimeError(f"Could not load {img_path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    suit_crop = str(Path(__file__).parent / "debug" / "suit_left.png")
    template, mask = extract_suit_template(suit_crop)
    print(f"Template shape: {template.shape}")

    # Save template for inspection
    cv2.imwrite(str(Path(__file__).parent / "debug" / "suit_template.png"), template)

    matches = find_matches(gray, template, threshold=0.55)
    print(f"Raw matches: {len(matches)}")
    matches = non_max_suppression(matches, min_dist=40)
    print(f"After NMS: {len(matches)}")

    debug = frame.copy()
    for i, (x, y, s) in enumerate(matches):
        h, w = template.shape
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(debug, f"{s:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        print(f"  Match {i}: x={x}, y={y}, score={s:.3f}")

    out = Path(__file__).parent / "debug" / "suit_matches.png"
    cv2.imwrite(str(out), debug)
    print(f"Saved debug to {out}")


if __name__ == "__main__":
    main()
