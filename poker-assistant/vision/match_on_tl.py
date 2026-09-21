"""
Match synthetic rank/suit templates on tl_search.png to find exact positions.
"""
import cv2
from pathlib import Path


def load_templates(directory: str, prefix: str):
    templates = {}
    d = Path(directory)
    for f in d.glob(f"{prefix}_*.png"):
        key = f.stem.replace(f"{prefix}_", "")
        templates[key] = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    return templates


def find_best_matches(gray: cv2.Mat, templates: dict, threshold: float = 0.3):
    best = {}
    for key, tmpl in templates.items():
        if gray.shape[0] < tmpl.shape[0] or gray.shape[1] < tmpl.shape[1]:
            continue
        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            best[key] = (max_loc[0], max_loc[1], max_val)
    return best


def main():
    script_dir = Path(__file__).parent
    rank_templates = load_templates(script_dir / "templates", "rank")
    suit_templates = load_templates(script_dir / "templates", "suit")

    tl = cv2.imread(str(script_dir / "debug" / "tl_search.png"))
    gray = cv2.cvtColor(tl, cv2.COLOR_BGR2GRAY)

    print("Rank matches on tl_search:")
    rank_matches = find_best_matches(gray, rank_templates, threshold=0.15)
    for k, (x, y, s) in sorted(rank_matches.items(), key=lambda kv: kv[1][2], reverse=True):
        print(f"  {k}: x={x}, y={y}, score={s:.3f}")

    print("\nSuit matches on tl_search:")
    suit_matches = find_best_matches(gray, suit_templates, threshold=0.15)
    for k, (x, y, s) in sorted(suit_matches.items(), key=lambda kv: kv[1][2], reverse=True):
        print(f"  {k}: x={x}, y={y}, score={s:.3f}")

    # Draw debug
    debug = tl.copy()
    for k, (x, y, s) in rank_matches.items():
        h, w = rank_templates[k].shape
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(debug, k, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    for k, (x, y, s) in suit_matches.items():
        h, w = suit_templates[k].shape
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(debug, k, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(str(script_dir / "debug" / "tl_matches.png"), debug)
    print("\nSaved debug to tl_matches.png")


if __name__ == "__main__":
    main()
