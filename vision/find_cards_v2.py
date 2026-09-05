"""
Find cards using the top-left corner template (includes rank + suit + white border).
"""
import cv2
import numpy as np
from pathlib import Path


def find_corner_matches(frame_gray: np.ndarray, template: np.ndarray, threshold: float = 0.65):
    res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    matches = []
    for pt in zip(*loc[::-1]):
        score = res[pt[1], pt[0]]
        matches.append((int(pt[0]), int(pt[1]), float(score)))
    return matches


def non_max_suppression(matches, min_dist=60):
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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use card_left corner as template
    corner = cv2.imread(str(Path(__file__).parent / "debug" / "card_left.png"))
    template = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    print(f"Corner template shape: {template.shape}")

    matches = find_corner_matches(gray, template, threshold=0.55)
    print(f"Raw matches: {len(matches)}")
    matches = non_max_suppression(matches, min_dist=80)
    print(f"After NMS: {len(matches)}")

    debug = frame.copy()
    for i, (x, y, s) in enumerate(matches[:20]):
        h, w = template.shape
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug, f"{i}:{s:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(f"  Match {i}: x={x}, y={y}, score={s:.3f}")

    out = Path(__file__).parent / "debug" / "corner_matches.png"
    cv2.imwrite(str(out), debug)
    print(f"Saved debug to {out}")


if __name__ == "__main__":
    main()
