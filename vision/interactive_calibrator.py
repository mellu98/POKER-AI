"""
Interactive Vision Calibrator for Poker Assistant.

Usage:
    python vision/interactive_calibrator.py --image screenshot.png
    python vision/interactive_calibrator.py               (live screenshot)

Hotkeys (while the image window is focused):
    1-2          : set next hole-card ROI
    3-7          : set next board-card ROI
    p            : set pot ROI
    c            : set to_call ROI
    s            : set stack ROI
    d            : set dealer ROI
    t            : capture card templates from hole/board ROIs
    r            : run recognition test on all ROIs
    S (shift+s)  : save config.yaml
    q / ESC      : quit

Mouse:
    Click & drag to draw a rectangle for the current mode.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ocr_cards import match_card, load_templates_from_dir, generate_card_templates


class InteractiveCalibrator:
    ROI_COLORS = {
        "hole": (0, 255, 0),
        "board": (255, 0, 0),
        "pot": (0, 0, 255),
        "to_call": (0, 255, 255),
        "stack": (255, 255, 0),
        "dealer": (255, 0, 255),
    }

    def __init__(self, image_path: str | None = None):
        self.image_path = image_path
        self.original_frame: np.ndarray | None = None
        self.display_frame: np.ndarray | None = None
        self.scale = 1.0
        self.rois: list[dict] = []          # drawn regions
        self.drawing = False
        self.ix = self.iy = 0
        self.current_rect: tuple | None = None
        self.mode = "hole"
        self.mode_index = 1
        self.templates_dir = Path(__file__).parent / "templates"
        self.templates_dir.mkdir(exist_ok=True)
        self._load_templates()
        self._load_existing_rois()

    def _load_existing_rois(self, config_path: str = "config.yaml"):
        """Pre-populate ROIs from an existing config.yaml so the user can skip drawing."""
        import yaml
        path = Path(config_path)
        if not path.exists():
            return
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        vision = cfg.get("vision", {})
        rois = vision.get("rois", {})
        for roi_type, entries in rois.items():
            if isinstance(entries, list):
                for i, entry in enumerate(entries):
                    self.rois.append({
                        "type": roi_type,
                        "label": f"{roi_type}{i + 1}",
                        "x": entry["x"], "y": entry["y"],
                        "w": entry["w"], "h": entry["h"],
                    })
                    if roi_type == "board" and i + 2 > self.mode_index:
                        self.mode_index = i + 2
            elif isinstance(entries, dict):
                self.rois.append({
                    "type": roi_type,
                    "label": roi_type,
                    "x": entries["x"], "y": entries["y"],
                    "w": entries["w"], "h": entries["h"],
                })

    def _load_templates(self):
        if self.templates_dir.exists():
            from ocr_cards import load_rank_templates, load_suit_templates
            self.templates = load_templates_from_dir(str(self.templates_dir))
            self.rank_templates = load_rank_templates(str(self.templates_dir))
            self.suit_templates = load_suit_templates(str(self.templates_dir))
        else:
            self.templates = generate_card_templates()
            self.rank_templates = {}
            self.suit_templates = {}

    def _load_image(self):
        if self.image_path:
            self.original_frame = cv2.imread(str(self.image_path))
            if self.original_frame is None:
                raise RuntimeError(f"Could not load image: {self.image_path}")
        else:
            from capture import screenshot
            self.original_frame = screenshot()

        h, w = self.original_frame.shape[:2]
        max_h = 650  # fit on screen with title bar + taskbar
        if h > max_h:
            self.scale = max_h / h
            new_w = int(w * self.scale)
            new_h = int(h * self.scale)
            self.display_frame = cv2.resize(self.original_frame, (new_w, new_h))
        else:
            self.scale = 1.0
            self.display_frame = self.original_frame.copy()

    def _draw_rois(self, canvas: np.ndarray):
        for roi in self.rois:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            x, y, w, h = int(x * self.scale), int(y * self.scale), int(w * self.scale), int(h * self.scale)
            color = self.ROI_COLORS.get(roi["type"], (200, 200, 200))
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
            label = roi.get("label", roi["type"])
            cv2.putText(canvas, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.current_rect:
            x, y, w, h = self.current_rect
            x, y, w, h = int(x * self.scale), int(y * self.scale), int(w * self.scale), int(h * self.scale)
            color = self.ROI_COLORS.get(self.mode, (200, 200, 200))
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)

    def _draw_hud(self, canvas: np.ndarray):
        lines = [
            f"Mode: {self.mode}{self.mode_index if self.mode in ('hole','board') else ''}",
            "Keys: 1-2=hole 3-7=board p=pot c=call s=stack d=dealer",
            "t=templates  r=test  S=save  q=quit",
            f"ROIs: {len(self.rois)}  |  Templates: {len(self.templates)}",
        ]
        y0 = 20
        for i, line in enumerate(lines):
            y = y0 + i * 20
            cv2.putText(canvas, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    def _mouse_callback(self, event, x, y, flags, param):
        # convert display coordinates back to original image coordinates
        x = int(x / self.scale)
        y = int(y / self.scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.current_rect = (x, y, 0, 0)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                x0, y0 = min(self.ix, x), min(self.iy, y)
                w, h = abs(x - self.ix), abs(y - self.iy)
                self.current_rect = (x0, y0, w, h)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.current_rect and self.current_rect[2] > 5 and self.current_rect[3] > 5:
                x, y, w, h = self.current_rect
                label = f"{self.mode}{self.mode_index}"
                self.rois.append({
                    "type": self.mode,
                    "label": label,
                    "x": x, "y": y, "w": w, "h": h,
                })
                if self.mode in ("hole", "board"):
                    self.mode_index += 1
            self.current_rect = None

    def _capture_templates(self):
        """Ask the user for a card name per hole/board ROI and save the crop."""
        import threading
        for roi in self.rois:
            if roi["type"] not in ("hole", "board"):
                continue
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            crop = self.original_frame[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            resized = cv2.resize(crop, (60, 80))
            cv2.imshow("template_preview", resized)
            print(f"\nROI {roi['label']}: type card name (e.g. As) or press ENTER to skip:")

            # Read input in a background thread so the main thread can keep
            # calling cv2.waitKey and the window stays responsive on Windows.
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
                path = self.templates_dir / f"{name}.png"
                cv2.imwrite(str(path), resized)
                print(f"  -> saved template {path}")
        self._load_templates()

    def _run_test(self):
        """Run recognition on every ROI and overlay results."""
        print("\n--- Recognition Test ---")
        canvas = self.original_frame.copy()
        has_split = bool(self.rank_templates and self.suit_templates)
        for roi in self.rois:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            crop = self.original_frame[y:y + h, x:x + w]
            if roi["type"] in ("hole", "board"):
                card = None
                if has_split:
                    from ocr_cards import recognize_card_split
                    card = recognize_card_split(
                        crop, self.rank_templates, self.suit_templates
                    )
                if card is None:
                    card = match_card(crop, self.templates)
                text = card or "?"
            else:
                from state_extractor import ScreenshotStateExtractor
                val = ScreenshotStateExtractor._read_number(self.original_frame, roi)
                text = str(val) if val is not None else "?"
            print(f"  {roi['label']:12s}: {text}")
            cv2.putText(canvas, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if self.scale != 1.0:
            canvas = cv2.resize(canvas, (self.display_frame.shape[1], self.display_frame.shape[0]))
        cv2.imshow("calibrator", canvas)
        cv2.waitKey(0)

    def _save_config(self, config_path: str = "config.yaml"):
        import yaml
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}

        cfg.setdefault("vision", {})
        cfg["vision"]["template_dir"] = str(
            self.templates_dir.relative_to(Path.cwd())
        )
        cfg["vision"]["mode"] = "screenshot"

        rois = {
            "hole": [],
            "board": [],
            "pot": None,
            "to_call": None,
            "stack": None,
            "dealer": None,
        }
        for roi in self.rois:
            entry = {"x": roi["x"], "y": roi["y"],
                     "w": roi["w"], "h": roi["h"]}
            t = roi["type"]
            if t == "hole":
                rois["hole"].append(entry)
            elif t == "board":
                rois["board"].append(entry)
            elif t in rois:
                rois[t] = entry

        cfg["vision"]["rois"] = rois

        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Config saved to {config_path}")

    def run(self):
        self._load_image()
        cv2.namedWindow("calibrator")
        cv2.setMouseCallback("calibrator", self._mouse_callback)

        try:
            while True:
                # Detect if the user closed the window with the X button
                if cv2.getWindowProperty("calibrator", cv2.WND_PROP_VISIBLE) < 1:
                    break

                canvas = self.display_frame.copy()
                self._draw_rois(canvas)
                self._draw_hud(canvas)
                cv2.imshow("calibrator", canvas)
                key = cv2.waitKey(20) & 0xFF

                if key == ord("q") or key == 27:
                    break
                elif key == ord("1"):
                    self.mode, self.mode_index = "hole", 1
                elif key == ord("2"):
                    self.mode, self.mode_index = "hole", 2
                elif key in (ord("3"), ord("4"), ord("5"), ord("6"), ord("7")):
                    self.mode, self.mode_index = "board", key - ord("2")
                elif key == ord("p"):
                    self.mode = "pot"
                elif key == ord("c"):
                    self.mode = "to_call"
                elif key == ord("s"):
                    self.mode = "stack"
                elif key == ord("d"):
                    self.mode = "dealer"
                elif key == ord("t"):
                    self._capture_templates()
                elif key == ord("r"):
                    self._run_test()
                elif key == ord("S"):
                    self._save_config()
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Poker Vision Calibrator"
    )
    parser.add_argument("--image", type=str,
                        help="Path to screenshot image to calibrate against")
    args = parser.parse_args()

    cal = InteractiveCalibrator(image_path=args.image)
    cal.run()


if __name__ == "__main__":
    main()
