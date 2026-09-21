"""
Interactive reviewer for placeholder card labels produced by data_collector.py.

Usage:
    python vision/review_labels.py --dir vision/dataset/yolo_goldbet

For every unresolved crop in ``<dir>/review/`` (named ``XX_<ts>_<slot>.png``):
1. Opens the crop in the default image viewer (headless-safe).
2. Prompts for the correct card code (e.g. '7d', 'Ts', '10h').
3. Rewrites the corresponding line in the frame's label file, replacing the
   placeholder class id (52) with the correct class, and renames the crop to
   '<card>_<ts>_<slot>.png' following the existing dataset convention.

The mapping between review crops and label lines is stored in
``<dir>/review/index.json`` (written by data_collector.py).

Commands while reviewing: Enter a card code, 's' to skip, 'q' to quit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the class mapping and validation from data_collector.
sys.path.insert(0, str(Path(__file__).parent))

from data_collector import CLASS_NAMES, PLACEHOLDER_CLASS_ID, card_to_class_id, parse_card_code


def _show_image(crop_path: Path) -> None:
    """Open a crop in the default system viewer (best-effort, headless-safe)."""
    try:
        import cv2
        from PIL import Image

        img = cv2.imread(str(crop_path))
        if img is not None:
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show(
                title=crop_path.name
            )
    except Exception as exc:
        print(f"  (could not display image: {exc})")


def _rewrite_label_line(label_path: Path, line_index: int, class_id: int) -> bool:
    """Replace the class id on `line_index` of a YOLO label file."""
    if not label_path.exists():
        print(f"  !! label file missing: {label_path}")
        return False
    lines = label_path.read_text().splitlines()
    if not (0 <= line_index < len(lines)):
        print(f"  !! line {line_index} out of range in {label_path.name} "
              f"({len(lines)} lines) — was the file edited?")
        return False
    parts = lines[line_index].split()
    if not parts or int(parts[0]) != PLACEHOLDER_CLASS_ID:
        print(f"  !! line {line_index} is not a placeholder; left unchanged.")
        return False
    parts[0] = str(class_id)
    lines[line_index] = " ".join(parts)
    label_path.write_text("\n".join(lines) + "\n")
    return True


def review(output_dir: str) -> dict:
    """Iterate unresolved review crops and prompt for the correct card code."""
    out_dir = Path(output_dir)
    review_dir = out_dir / "review"
    index_path = review_dir / "index.json"

    if not index_path.exists():
        print(f"[review] No review index at {index_path}. "
              f"Run data_collector.py first.")
        return {"resolved": 0, "skipped": 0, "remaining": 0}

    index = json.loads(index_path.read_text())
    pending = {k: v for k, v in index.items() if not v.get("resolved")}
    if not pending:
        print("[review] Nothing to review — all crops resolved.")
        return {"resolved": 0, "skipped": 0, "remaining": 0}

    print(f"[review] {len(pending)} crops need review. "
          f"Enter card code (e.g. 7d, Ts, 10h), 's' to skip, 'q' to quit.\n")

    resolved = skipped = 0
    for crop_name, entry in list(pending.items()):
        proposed = entry.get("proposed")
        print(f"- {crop_name}  slot={entry['slot']}  "
              f"proposed={proposed or '?'} (conf {entry.get('confidence', 0):.2f})")
        print(f"  frame: {entry.get('frame', '?')}")
        _show_image(review_dir / crop_name)

        try:
            raw = input("  card> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[review] Interrupted.")
            break

        if raw.lower() == "q":
            break
        if raw.lower() == "s" or not raw:
            skipped += 1
            continue

        card = parse_card_code(raw)
        if card is None:
            print(f"  !! invalid code '{raw}' (expected like 7d, Ts, 10h) — skipped.")
            skipped += 1
            continue

        class_id = card_to_class_id(card)
        label_path = out_dir / entry["label_file"]
        if not _rewrite_label_line(label_path, entry["line"], class_id):
            skipped += 1
            continue

        # Rename the crop from XX_<ts>_<slot>.png to <card>_<ts>_<slot>.png.
        new_name = f"{card}_{crop_name[3:]}"
        (review_dir / crop_name).rename(review_dir / new_name)
        entry["resolved"] = True
        entry["card"] = card
        index[new_name] = index.pop(crop_name)
        resolved += 1
        print(f"  ok -> class {class_id} ({CLASS_NAMES[class_id]})")

    index_path.write_text(json.dumps(index, indent=2))
    remaining = sum(1 for v in index.values() if not v.get("resolved"))
    print(f"\n[review] Resolved: {resolved}, skipped: {skipped}, "
          f"still pending: {remaining}")
    if remaining == 0:
        print("[review] All placeholders resolved — dataset is ready for training.")
    return {"resolved": resolved, "skipped": skipped, "remaining": remaining}


def main():
    parser = argparse.ArgumentParser(
        description="Resolve placeholder card labels from data_collector.py."
    )
    parser.add_argument("--dir", type=str, default="vision/dataset/yolo_goldbet",
                        help="Dataset directory containing review/index.json")
    args = parser.parse_args()
    review(args.dir)


if __name__ == "__main__":
    main()
