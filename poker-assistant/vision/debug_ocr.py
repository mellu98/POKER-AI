"""
debug_ocr.py — Isola e testa l'OCR su ogni ROI per diagnosticare letture mancanti.

Uso:
    python vision/debug_ocr.py [percorso_screenshot]

Se non viene passato uno screenshot, usa vision/debug/raw_screenshot.png.
Salva i crop in vision/debug/crops/ e stampa i risultati OCR.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import yaml
from local_table_state import _crop_roi, _ocr_number


def _load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "debug" / "raw_screenshot.png")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[debug_ocr] Could not load {image_path}")
        sys.exit(1)

    cfg = _load_config()
    site = cfg.get("vision", {}).get("site", "golbet")
    site_cfg = cfg.get("sites", {}).get(site, {})
    legacy_rois = cfg.get("vision", {}).get("rois", {})

    out_dir = Path(__file__).parent / "debug" / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _roi_or_legacy(key: str):
        return site_cfg.get(key) or legacy_rois.get(key)

    print(f"[debug_ocr] Image: {image_path} ({frame.shape[1]}x{frame.shape[0]})")
    print("-" * 60)

    # ROI singoli
    for key in ("pot", "to_call", "stack", "dealer_button"):
        roi = _roi_or_legacy(key)
        crop = _crop_roi(frame, roi)
        if crop is None or crop.size == 0:
            print(f"{key:12}: ROI vuota")
            continue
        crop_path = out_dir / f"{key}.png"
        cv2.imwrite(str(crop_path), crop)
        value = _ocr_number(crop)
        print(f"{key:12}: value={value}  -> {crop_path}")

    # Hole / board
    for key in ("hole", "board"):
        rois = _roi_or_legacy(key) or []
        for i, roi in enumerate(rois):
            crop = _crop_roi(frame, roi)
            if crop is None:
                continue
            crop_path = out_dir / f"{key}_{i}.png"
            cv2.imwrite(str(crop_path), crop)
            print(f"{key}_{i:11}: saved {crop_path}")

    # Seat stacks
    print("-" * 60)
    for seat in site_cfg.get("seats", []):
        idx = seat.get("index")
        stack_roi = seat.get("stack_roi")
        if stack_roi is None:
            continue
        crop = _crop_roi(frame, stack_roi)
        if crop is None or crop.size == 0:
            print(f"stack-S{idx}: ROI vuota")
            continue
        crop_path = out_dir / f"stack_S{idx}.png"
        cv2.imwrite(str(crop_path), crop)
        value = _ocr_number(crop)
        print(f"stack-S{idx:2}: value={value}  -> {crop_path}")


if __name__ == "__main__":
    main()
