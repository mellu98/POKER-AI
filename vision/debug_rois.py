"""
debug_rois.py — Visualizza le ROI calibrate sopra uno screenshot.

Uso:
    python vision/debug_rois.py

Cattura uno screenshot dalla finestra configurata, disegna tutte le ROI
(hole, board, pot, to_call, stack, seats, dealer_button) e salva
l'immagine annotata in vision/debug/roi_overlay.png.

Utile per verificare che la calibrazione sia precisa.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import yaml
from capture import screenshot


def _load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_roi(roi: dict, frame: np.ndarray):
    if roi is None:
        return None
    h, w = frame.shape[:2]
    if roi.get("rel"):
        return {
            "x": int(roi["x"] * w),
            "y": int(roi["y"] * h),
            "w": int(roi["w"] * w),
            "h": int(roi["h"] * h),
        }
    return roi


def _draw_roi(frame, roi, color, label, thickness=2):
    r = _resolve_roi(roi, frame)
    if r is None:
        return
    x, y, w, h = r["x"], r["y"], r["w"], r["h"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    cv2.putText(
        frame,
        label,
        (x, max(y - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def main():
    cfg = _load_config()
    vision_cfg = cfg.get("vision", {})
    site = vision_cfg.get("site", "golbet")
    site_cfg = cfg.get("sites", {}).get(site, {})

    window_title = vision_cfg.get("window_title", "Poker - Opera")
    print(f"[debug_rois] Capturing window: {window_title}")
    frame = screenshot(window_title=window_title)
    if frame is None:
        print(f"[debug_rois] Could not capture window '{window_title}'")
        sys.exit(1)

    print(f"[debug_rois] Screenshot: {frame.shape[1]}x{frame.shape[0]}")

    # Copia pulita per i test OCR (prima di disegnare sopra)
    raw_frame = frame.copy()
    out_dir = Path(__file__).parent / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_screenshot.png"
    cv2.imwrite(str(raw_path), raw_frame)
    print(f"[debug_rois] Saved raw screenshot to: {raw_path}")

    # Colori BGR
    colors = {
        "hole": (0, 255, 0),      # verde
        "board": (255, 0, 0),     # blu
        "pot": (0, 165, 255),     # arancione
        "to_call": (0, 0, 255),   # rosso
        "stack": (255, 255, 0),   # ciano
        "dealer_button": (255, 0, 255),  # magenta
        "seat": (128, 128, 128),  # grigio
        "stack_roi": (0, 255, 255), # giallo
    }

    # Disegna ROI principali
    for key in ("hole", "board"):
        rois = site_cfg.get(key, [])
        for i, roi in enumerate(rois):
            _draw_roi(frame, roi, colors[key], f"{key}-{i}")

    for key in ("pot", "to_call", "stack", "dealer_button"):
        roi = site_cfg.get(key)
        if roi:
            _draw_roi(frame, roi, colors[key], key.upper(), thickness=3)

    # Disegna seat
    for seat in site_cfg.get("seats", []):
        idx = seat.get("index", 0)
        x, y = seat.get("x", 0), seat.get("y", 0)
        h, w = frame.shape[:2]
        abs_x, abs_y = int(x * w), int(y * h)
        cv2.circle(frame, (abs_x, abs_y), 8, colors["seat"], -1)
        cv2.putText(
            frame,
            f"S{idx}",
            (abs_x + 10, abs_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colors["seat"],
            2,
            cv2.LINE_AA,
        )
        stack_roi = seat.get("stack_roi")
        if stack_roi:
            _draw_roi(frame, stack_roi, colors["stack_roi"], f"stack-S{idx}")

    # Salva overlay
    out_path = out_dir / "roi_overlay.png"
    cv2.imwrite(str(out_path), frame)
    print(f"[debug_rois] Saved annotated screenshot to: {out_path}")

    # Mostra (cv2.imshow non funziona con opencv-python-headless)
    try:
        cv2.imshow("Calibrated ROIs (press any key to close)", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        from PIL import Image
        print("[debug_rois] OpenCV headless: apro l'immagine con il viewer di sistema.")
        Image.open(str(out_path)).show()


if __name__ == "__main__":
    main()
