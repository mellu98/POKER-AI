"""
Capture card templates in 10 seconds while playing.

Usage:
    python vision/capture_templates_quick.py

What it does:
    1. Takes an instant screenshot
    2. Shows each card ROI one by one
    3. You type the card name (e.g. As, 7h, Kd) or press ENTER to skip
    4. Saves templates to vision/templates/

Card name format: rank + suit
    Ranks: A K Q J T 9 8 7 6 5 4 3 2   (T = 10)
    Suits: s h d c   (spades hearts diamonds clubs)
    Examples: As = Ace of spades, Th = 10 of hearts
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import yaml
from capture import screenshot, crop_roi


def main():
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("ERRORE: config.yaml non trovato. Esegui prima il calibratore.")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    rois = cfg.get("vision", {}).get("rois", {})
    hole_rois = rois.get("hole", [])
    board_rois = rois.get("board", [])

    if not hole_rois and not board_rois:
        print("ERRORE: nessun ROI configurato in config.yaml")
        sys.exit(1)

    print("Catturo screenshot in 3 secondi... metti il tavolo poker in primo piano!")
    import time
    time.sleep(3)

    frame = screenshot()
    print(f"Screenshot catturato: {frame.shape[1]}x{frame.shape[0]}")

    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)

    def _resolve(roi):
        if roi.get("rel"):
            h, w = frame.shape[:2]
            return {
                "x": int(roi["x"] * w),
                "y": int(roi["y"] * h),
                "w": int(roi["w"] * w),
                "h": int(roi["h"] * h),
            }
        return roi

    all_rois = [("hole", i, _resolve(r)) for i, r in enumerate(hole_rois)]
    all_rois += [("board", i, _resolve(r)) for i, r in enumerate(board_rois)]

    import threading

    saved = 0
    for roi_type, idx, roi in all_rois:
        label = f"{roi_type}{idx + 1}"
        crop = crop_roi(frame, roi["x"], roi["y"], roi["w"], roi["h"])
        if crop is None or crop.size == 0:
            continue

        resized = cv2.resize(crop, (60, 80))
        cv2.imshow("template_preview", resized)
        print(f"\n{label}: scrivi il nome della carta (es. As, 7h) o premi ENTER per saltare:")

        result = []
        def _ask():
            try:
                result.append(input("> ").strip().upper())
            except EOFError:
                result.append("")
        t = threading.Thread(target=_ask)
        t.start()
        while t.is_alive():
            cv2.waitKey(50)

        cv2.destroyWindow("template_preview")
        name = result[0] if result else ""

        if name:
            path = templates_dir / f"{name}.png"
            cv2.imwrite(str(path), resized)
            print(f"  -> salvata {path}")
            saved += 1

    print(f"\nFatto! {saved} template salvati in {templates_dir}")
    print("Lancia di nuovo quando vedi nuove carte.")


if __name__ == "__main__":
    main()
