"""
calibrate_table.py — Calibrazione automatica del tavolo tramite LLM vision.

Uso:
    python vision/calibrate_table.py --site golbet --window "Poker - Opera"
    python vision/calibrate_table.py --site golbet --images img1.png img2.png

Il processo:
1. Cattura 3-5 screenshot del tavolo (o usa immagini fornite).
2. Invia ogni screenshot a un modello vision economico (OpenRouter).
3. Il modello restituisce le coordinate relative (0..1) di:
   - hole cards, board cards, pot, to_call, hero stack
   - i 9 seat (o 6 per 6max)
   - dealer button
4. Aggrega i risultati e scrive la configurazione in config.yaml
   sotto la chiave `sites.{site}`.

Vantaggio: una sola calibrazione per sito, poi tutto locale e velocissimo.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

# Carica .env se disponibile
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Prova a importare capture; se fallisce, usiamo solo immagini esplicite.
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from capture import screenshot
except Exception:
    screenshot = None  # type: ignore


DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #

def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_config(config_path: Path, cfg: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def _get_api_key(cfg: dict) -> Optional[str]:
    key = (
        cfg.get("vision", {})
        .get("llm", {})
        .get("api_key")
    )
    if key:
        return key
    return os.getenv("OPENROUTER_API_KEY")


def _encode_frame(frame: np.ndarray, max_dim: int = 1024, quality: int = 85) -> str:
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    return base64.b64encode(buf).decode("ascii")


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


# --------------------------------------------------------------------------- #
# Chiamata LLM
# --------------------------------------------------------------------------- #

def _call_openrouter(
    api_key: str,
    api_url: str,
    model: str,
    b64_image: str,
    prompt: str,
    timeout: float = 60.0,
) -> dict:
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }

    t0 = time.time()
    resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    print(f"[calibrate] API latency: {(time.time() - t0) * 1000:.0f}ms, status={resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return _extract_json(text)


def _extract_json(text: str) -> dict:
    text = text.strip()
    # Prova parse diretto
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Estrai da blocco markdown
    import re
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Primo { ... }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise RuntimeError(f"Could not parse JSON from model response:\n{text[:800]}")


CALIBRATION_PROMPT = """You are a poker table layout analyzer for the site Golbet.it.

Look at the screenshot and identify the table layout. Return ONLY a valid JSON object. No explanation, no markdown, no text before or after the JSON.

Required JSON structure (compact, all coordinates 0.0 to 1.0 relative to image size):

{"table_type":"9max","hero_seat":0,"hole":[{"x":0.45,"y":0.67,"w":0.05,"h":0.11},{"x":0.50,"y":0.67,"w":0.05,"h":0.11}],"board":[{"x":0.35,"y":0.42,"w":0.06,"h":0.13},{"x":0.41,"y":0.42,"w":0.06,"h":0.13},{"x":0.47,"y":0.42,"w":0.06,"h":0.13},{"x":0.54,"y":0.42,"w":0.06,"h":0.13},{"x":0.60,"y":0.42,"w":0.06,"h":0.13}],"pot":{"x":0.40,"y":0.36,"w":0.20,"h":0.08},"to_call":{"x":0.30,"y":0.81,"w":0.16,"h":0.05},"stack":{"x":0.30,"y":0.74,"w":0.16,"h":0.05},"dealer_button":{"x":0.62,"y":0.28,"w":0.04,"h":0.04},"seats":[{"index":0,"x":0.50,"y":0.85,"stack_roi":{"x":0.46,"y":0.78,"w":0.08,"h":0.04}},{"index":1,"x":0.12,"y":0.65,"stack_roi":{"x":0.08,"y":0.58,"w":0.08,"h":0.04}},{"index":2,"x":0.12,"y":0.35,"stack_roi":{"x":0.08,"y":0.28,"w":0.08,"h":0.04}},{"index":3,"x":0.50,"y":0.15,"stack_roi":{"x":0.46,"y":0.08,"w":0.08,"h":0.04}},{"index":4,"x":0.88,"y":0.35,"stack_roi":{"x":0.84,"y":0.28,"w":0.08,"h":0.04}},{"index":5,"x":0.88,"y":0.65,"stack_roi":{"x":0.84,"y":0.58,"w":0.08,"h":0.04}},{"index":6,"x":0.30,"y":0.22,"stack_roi":{"x":0.26,"y":0.15,"w":0.08,"h":0.04}},{"index":7,"x":0.70,"y":0.22,"stack_roi":{"x":0.66,"y":0.15,"w":0.08,"h":0.04}},{"index":8,"x":0.50,"y":0.50,"stack_roi":{"x":0.46,"y":0.43,"w":0.08,"h":0.04}}]}

Rules:
- All coordinates are RELATIVE to image size, in range 0.0 to 1.0.
- table_type must be "6max" or "9max".
- hero_seat is the seat index where the player's hole cards are visible (usually bottom-center).
- hole: two ROIs for the player's hole cards.
- board: five ROIs for community cards (even if empty now, provide expected locations).
- pot: ROI around the main pot number (e.g. "Piatto: €X.XX").
- to_call: ROI around the call/bet amount shown on the action button or near it.
- stack: ROI around the player's own stack amount.
- dealer_button: ROI where the dealer button (circle with "D") typically appears.
- seats: one entry per seat. Provide stack_roi for each seat where an opponent stack would be shown.
- If a seat is empty, still provide its expected position and stack_roi.

Return ONLY the JSON object. Do not wrap it in markdown. Do not add comments.
"""


# --------------------------------------------------------------------------- #
# Aggregazione
# --------------------------------------------------------------------------- #

def _aggregate_roi(rois: list[dict]) -> dict:
    """Calcola la mediana di x, y, w, h su più rilevazioni."""
    return {
        "x": _clamp01(_median([r["x"] for r in rois])),
        "y": _clamp01(_median([r["y"] for r in rois])),
        "w": _clamp01(_median([r["w"] for r in rois])),
        "h": _clamp01(_median([r["h"] for r in rois])),
        "rel": True,
    }


def _aggregate_detections(detections: list[dict]) -> dict:
    """Aggrega N rilevazioni LLM in una singola configurazione stabile."""
    table_types = [d.get("table_type", "9max") for d in detections]
    table_type = max(set(table_types), key=table_types.count)

    hero_seats = [d.get("hero_seat", 0) for d in detections if d.get("hero_seat") is not None]
    hero_seat = int(_median(hero_seats)) if hero_seats else 0

    result: dict[str, Any] = {
        "table_type": table_type,
        "hero_seat": hero_seat,
        "hole": [],
        "board": [],
        "pot": None,
        "to_call": None,
        "stack": None,
        "dealer_button": None,
        "seats": [],
    }

    # Hole cards
    n_hole = max(len(d.get("hole", [])) for d in detections)
    for i in range(n_hole):
        rois = [d["hole"][i] for d in detections if i < len(d.get("hole", []))]
        if rois:
            result["hole"].append(_aggregate_roi(rois))

    # Board cards
    n_board = max(len(d.get("board", [])) for d in detections)
    for i in range(n_board):
        rois = [d["board"][i] for d in detections if i < len(d.get("board", []))]
        if rois:
            result["board"].append(_aggregate_roi(rois))

    # Singoli ROIs
    for key in ("pot", "to_call", "stack", "dealer_button"):
        rois = [d[key] for d in detections if d.get(key)]
        if rois:
            result[key] = _aggregate_roi(rois)

    # Seats
    n_seats = 9 if table_type == "9max" else 6
    for i in range(n_seats):
        seat_rois = []
        stack_rois = []
        for d in detections:
            for s in d.get("seats", []):
                if s.get("index") == i:
                    seat_rois.append({"x": s["x"], "y": s["y"], "w": 0.0, "h": 0.0})
                    stack = s.get("stack_roi")
                    if stack:
                        stack_rois.append(stack)
        if seat_rois:
            center = _aggregate_roi(seat_rois)
            seat: dict[str, Any] = {
                "index": i,
                "x": center["x"],
                "y": center["y"],
            }
            if stack_rois:
                seat["stack_roi"] = _aggregate_roi(stack_rois)
            result["seats"].append(seat)

    return result


# --------------------------------------------------------------------------- #
# Acquisizione screenshot
# --------------------------------------------------------------------------- #

def _capture_window(window_title: str, count: int, delay: float = 2.0) -> list[np.ndarray]:
    if screenshot is None:
        raise RuntimeError("capture.screenshot not available. Provide --images instead.")
    frames = []
    for i in range(count):
        print(f"[calibrate] Capturing screenshot {i + 1}/{count}...")
        frame = screenshot(window_title=window_title)
        if frame is None:
            raise RuntimeError(f"Could not capture window '{window_title}'")
        frames.append(frame)
        if i < count - 1:
            time.sleep(delay)
    return frames


def _load_images(paths: list[str]) -> list[np.ndarray]:
    frames = []
    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            raise RuntimeError(f"Could not load image: {p}")
        frames.append(frame)
    return frames


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Auto-calibrate poker table layout via LLM vision")
    parser.add_argument("--site", default="golbet", help="Site identifier (default: golbet)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--window", default=None, help="Window title to capture from")
    parser.add_argument("--images", nargs="+", default=None, help="Existing screenshot paths")
    parser.add_argument("--shots", type=int, default=3, help="Number of screenshots to capture")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="OpenRouter API URL")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path

    cfg = _load_config(config_path)
    api_key = _get_api_key(cfg)
    if not api_key:
        print("[calibrate] ERROR: No OpenRouter API key found.")
        print("Set OPENROUTER_API_KEY env var or add vision.llm.api_key to config.yaml")
        sys.exit(1)

    # Carica screenshot
    if args.images:
        frames = _load_images(args.images)
    elif args.window:
        frames = _capture_window(args.window, args.shots)
    else:
        window_title = cfg.get("vision", {}).get("window_title")
        if not window_title:
            print("[calibrate] ERROR: Provide --window, --images, or set vision.window_title in config.yaml")
            sys.exit(1)
        frames = _capture_window(window_title, args.shots)

    print(f"[calibrate] Calibrating {args.site} from {len(frames)} screenshots...")

    detections = []
    for i, frame in enumerate(frames):
        b64 = _encode_frame(frame)
        try:
            det = _call_openrouter(api_key, args.api_url, args.model, b64, CALIBRATION_PROMPT)
            detections.append(det)
            print(f"[calibrate] Detection {i + 1} OK: table_type={det.get('table_type')}, hero_seat={det.get('hero_seat')}")
        except Exception as e:
            print(f"[calibrate] Detection {i + 1} failed: {e}")

    if len(detections) < 2:
        print("[calibrate] ERROR: Need at least 2 successful detections to aggregate.")
        sys.exit(1)

    calibration = _aggregate_detections(detections)

    # Backup vecchia config
    backup_path = config_path.with_suffix(".yaml.bak")
    _save_config(backup_path, cfg)
    print(f"[calibrate] Backup saved to {backup_path}")

    # Scrive sotto sites.{site}
    if "sites" not in cfg:
        cfg["sites"] = {}
    cfg["sites"][args.site] = calibration
    cfg["vision"]["site"] = args.site

    # Se manca window_title, lo lasciamo com'è; altrimenti aggiorniamo rois legacy
    # con quelli calibrati per compatibilità immediata.
    legacy_rois = cfg.setdefault("vision", {}).setdefault("rois", {})
    legacy_rois["hole"] = calibration.get("hole", legacy_rois.get("hole", []))
    legacy_rois["board"] = calibration.get("board", legacy_rois.get("board", []))
    for key in ("pot", "to_call", "stack", "dealer_button"):
        if calibration.get(key):
            legacy_rois[key] = calibration[key]

    _save_config(config_path, cfg)
    print(f"[calibrate] Config saved to {config_path}")
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
