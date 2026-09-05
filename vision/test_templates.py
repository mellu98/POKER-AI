"""Test template matching of generated templates on real card crops."""
import cv2
import numpy as np
from pathlib import Path


def load_templates(directory: str, prefix: str):
    templates = {}
    d = Path(directory)
    for f in d.glob(f"{prefix}_*.png"):
        key = f.stem.replace(f"{prefix}_", "")
        templates[key] = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        print(f"Loaded template {key} shape {templates[key].shape}")
    return templates


def match_best(roi_gray: np.ndarray, templates: dict) -> tuple:
    best_score = -1.0
    best_key = None
    for key, tmpl in templates.items():
        if roi_gray.shape[0] < tmpl.shape[0] or roi_gray.shape[1] < tmpl.shape[1]:
            print(f"Skipping {key}: roi {roi_gray.shape} < tmpl {tmpl.shape}")
            continue
        res = cv2.matchTemplate(roi_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        print(f"  {key}: {max_val:.3f}")
        if max_val > best_score:
            best_score = max_val
            best_key = key
    return best_key, best_score


def main():
    script_dir = Path(__file__).parent
    rank_templates = load_templates(script_dir / "templates", "rank")
    suit_templates = load_templates(script_dir / "templates", "suit")

    img_path = Path(r"C:\Users\franc\.claude\image-cache\4a0bae80-b8b2-4141-9caa-41dee5631ae0\8.png")
    frame = cv2.imread(str(img_path))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract rank ROI from right card (approx)
    rank_roi = gray[530:575, 805:850]
    print(f"Rank ROI shape: {rank_roi.shape}")
    cv2.imwrite(str(script_dir / "debug" / "test_rank_roi.png"), rank_roi)
    rank_key, rank_score = match_best(rank_roi, rank_templates)
    print(f"Rank match: {rank_key} (score {rank_score:.3f})")

    # Extract suit ROI from right card (approx)
    suit_roi = gray[555:590, 810:845]
    print(f"Suit ROI shape: {suit_roi.shape}")
    cv2.imwrite(str(script_dir / "debug" / "test_suit_roi.png"), suit_roi)
    suit_key, suit_score = match_best(suit_roi, suit_templates)
    print(f"Suit match: {suit_key} (score {suit_score:.3f})")

    # Also try on the big suit in center of left card
    big_suit = gray[600:650, 750:800]
    print(f"Big suit ROI shape: {big_suit.shape}")
    cv2.imwrite(str(script_dir / "debug" / "test_big_suit_roi.png"), big_suit)
    suit_key2, suit_score2 = match_best(big_suit, suit_templates)
    print(f"Big suit match: {suit_key2} (score {suit_score2:.3f})")


if __name__ == "__main__":
    main()
