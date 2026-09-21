"""
Collect full-table screenshots for training/validating a YOLO poker detector.

Uso:
    python vision/collect_table_screenshots.py

Comportamento:
- Legge `vision.table_collection` da `config.yaml`.
- Cattura la finestra del tavolo (es. "Poker - Opera") o il monitor principale.
- Salva i frame in `captures/raw/` con timestamp.
- Ignora frame duplicati consecutivi (stesso hash) per non sprecare spazio.
- Si ferma con Ctrl+C o quando raggiunge `max_screenshots`.

Le immagini raccolte servono per:
1. Addestrare un modello YOLO custom su Roboflow/Ultralytics.
2. Validare il detector locale senza rischiare soldi reali.
"""
from __future__ import annotations

import hashlib
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

# Permette import relativi quando lo script viene lanciato da solo
sys.path.insert(0, str(Path(__file__).parent))

from capture import screenshot


def _load_config(config_path: str = "config.yaml") -> dict:
    """Load application config if available."""
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[collect] Could not load {config_path}: {exc}")
        return {}


def _frame_hash(frame) -> str:
    """Return a quick perceptual hash for duplicate detection."""
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()


def collect_table_screenshots(
    output_dir: str = "captures/raw",
    window_title: Optional[str] = None,
    interval_seconds: float = 2.0,
    max_screenshots: Optional[int] = None,
    skip_duplicates: bool = True,
) -> int:
    """Collect screenshots until interrupted or max reached.

    Returns the number of unique screenshots saved.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[collect] Saving screenshots to: {out_path.absolute()}")
    print(f"[collect] Window title: {window_title or 'full screen'}")
    print(f"[collect] Interval: {interval_seconds}s")
    if max_screenshots:
        print(f"[collect] Max screenshots: {max_screenshots}")
    print("[collect] Press Ctrl+C to stop.\n")

    saved = 0
    last_hash: Optional[str] = None

    try:
        while max_screenshots is None or saved < max_screenshots:
            frame = screenshot(window_title=window_title)
            if frame is None:
                print("[collect] Screenshot failed, retrying...")
                time.sleep(interval_seconds)
                continue

            frame_hash = _frame_hash(frame)
            if skip_duplicates and frame_hash == last_hash:
                time.sleep(interval_seconds)
                continue
            last_hash = frame_hash

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = out_path / f"table_{timestamp}.png"
            cv2.imwrite(str(filename), frame)
            saved += 1

            print(
                f"[collect] Saved {saved}: {filename.name} "
                f"({frame.shape[1]}x{frame.shape[0]})"
            )

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n[collect] Stopped by user.")
    except Exception as exc:
        print(f"\n[collect] Error: {exc}")
        traceback.print_exc()

    print(f"[collect] Total screenshots saved: {saved}")
    return saved


def main():
    cfg = _load_config()
    collection_cfg = cfg.get("vision", {}).get("table_collection", {})

    output_dir = collection_cfg.get("output_dir", "captures/raw")
    window_title = collection_cfg.get("window_title")
    if window_title is None:
        # Fallback sulla finestra principale configurata per la vision
        window_title = cfg.get("vision", {}).get("window_title")
    interval = float(collection_cfg.get("interval_seconds", 2.0))
    max_shots = collection_cfg.get("max_screenshots")
    skip_dups = bool(collection_cfg.get("skip_duplicates", True))

    collect_table_screenshots(
        output_dir=output_dir,
        window_title=window_title,
        interval_seconds=interval,
        max_screenshots=max_shots,
        skip_duplicates=skip_dups,
    )


if __name__ == "__main__":
    main()
