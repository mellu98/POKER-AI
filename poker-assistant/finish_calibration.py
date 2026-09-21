"""Finalizza la calibrazione: board dal CV (già validato in 2 run), hero hole
cards dal modello vision (bounding boxes), pot derivato dal board.

Uso:  python3 finish_calibration.py
"""
import base64
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "vision"))

_capture = __import__("capture")
screenshot = _capture.screenshot

# Board rilevata dal CV in entrambe le run di calibrate_live (stabile):
BOARD_REL = [
    {"x": 0.391, "y": 0.398, "w": 0.040, "h": 0.097, "rel": True},
    {"x": 0.437, "y": 0.398, "w": 0.040, "h": 0.097, "rel": True},
    {"x": 0.483, "y": 0.398, "w": 0.041, "h": 0.097, "rel": True},
    {"x": 0.529, "y": 0.398, "w": 0.041, "h": 0.097, "rel": True},
    {"x": 0.575, "y": 0.398, "w": 0.040, "h": 0.097, "rel": True},
]


def ask_llm_hero_boxes(frame: "np.ndarray") -> list[dict] | None:
    """Una chiamata vision: bounding box delle 2 hole cards hero (0-1000)."""
    import os

    import requests
    from dotenv import load_dotenv

    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        print("!! OPENROUTER_API_KEY mancante")
        return None

    b64 = _capture_encode(frame)
    system = (
        "You locate objects in poker table screenshots. Find the TWO HERO HOLE CARDS: "
        "the player's own two cards at the BOTTOM CENTER of the table (just above the "
        "hero avatar/stack, below the community board). They may slightly overlap. "
        "Return ONLY JSON: {\"hero_cards\": [{\"x\": X, \"y\": Y, \"w\": W, \"h\": H}, ...]} "
        "with x,y = top-left corner and w,h = size, all integers normalized 0-1000 "
        "relative to the full image. Two entries, left card first."
    )
    payload = {
        "model": "google/gemini-3.5-flash-lite",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": "Locate the two hero hole cards."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload, timeout=30.0,
        )
    except requests.RequestException as exc:
        print(f"!! richiesta fallita: {exc}")
        return None
    if not resp.ok:
        print(f"!! HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    text = (resp.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        print(f"!! nessun JSON: {text[:200]}")
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        print(f"!! JSON invalido: {text[:200]}")
        return None
    boxes = data.get("hero_cards")
    if not isinstance(boxes, list) or len(boxes) < 2:
        print(f"!! box mancanti: {text[:200]}")
        return None
    out = []
    for b in boxes[:2]:
        try:
            out.append({
                "x": max(0.0, min(1.0, float(b["x"]) / 1000.0)),
                "y": max(0.0, min(1.0, float(b["y"]) / 1000.0)),
                "w": max(0.005, min(1.0, float(b["w"]) / 1000.0)),
                "h": max(0.005, min(1.0, float(b["h"]) / 1000.0)),
            })
        except (KeyError, TypeError, ValueError):
            return None
    return out


def _capture_encode(frame) -> str:
    """Encode del frame come il production path (max 1280, JPEG q92)."""
    fh, fw = frame.shape[:2]
    scale = min(1.0, 1280 / max(fh, fw))
    try:
        if scale < 1.0:
            frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    except (cv2.error, ValueError, ZeroDivisionError):
        raise RuntimeError("encode fallito") from None
    if not ok:
        raise RuntimeError("encode fallito")
    return base64.b64encode(buf).decode()


def roi_lines(anchor: str, rois: list[dict]) -> list[str]:
    lines = [f"    {anchor}"]
    for r in rois:
        lines.append(f"      - x: {r['x']}")
        lines.append(f"        y: {r['y']}")
        lines.append(f"        w: {r['w']}")
        lines.append(f"        h: {r['h']}")
        lines.append("        rel: true")
    return lines


def replace_block(lines: list[str], start_marker: str, new_block: list[str],
                  end_markers: list[str]) -> list[str]:
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith(start_marker))
    except StopIteration:
        print(f"  !! marker {start_marker!r} non trovato")
        return lines
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if any(lines[i].startswith(m) for m in end_markers):
            end = i
            break
    return lines[:start] + new_block + lines[end:]


def main() -> None:
    frame = screenshot(window_title="Poker")
    if frame is None:
        raise SystemExit("finestra 'Poker' non trovata")
    fh, fw = frame.shape[:2]
    print(f"frame {fw}x{fh}")

    print(">> chiedo le box delle hero cards al modello vision...")
    boxes = ask_llm_hero_boxes(frame)
    if boxes is None:
        raise SystemExit("box LLM non ottenute: rilancia")
    print(f"  box LLM (relative): {boxes}")

    # LLM vede il frame DOWNSCALED: le sue box sono relative -> valide direttamente
    hole_rel = [
        {"x": round(b["x"], 3), "y": round(b["y"], 3),
         "w": round(b["w"], 3), "h": round(b["h"], 3), "rel": True}
        for b in boxes
    ]

    # sanity: le hero cards devono stare nel terzo inferiore, entro il centro largo
    for r in hole_rel:
        if not (0.45 <= r["y"] <= 0.85 and 0.25 <= r["x"] <= 0.75):
            print(f"  !! box implausibile {r}: fuori dalla fascia hero")
            raise SystemExit("box LLM non plausibili")

    # pot: sopra il board (board CV stabile)
    bcx = sum(r["x"] + r["w"] / 2 for r in BOARD_REL) / len(BOARD_REL)
    by = min(r["y"] for r in BOARD_REL)
    pot_rel = {"x": round(bcx - 0.08, 3), "y": round(max(0.0, by - 0.10), 3),
               "w": 0.16, "h": 0.08, "rel": True}

    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    lines = text.split("\n")

    lines = replace_block(lines, "    hole: &id001",
                          roi_lines("hole: &id001", hole_rel),
                          ["    board:", "    pot:", "    to_call:"])
    lines = replace_block(lines, "    board: &id002",
                          roi_lines("board: &id002", BOARD_REL),
                          ["    pot:"])
    pot_block = [
        "    pot: &id003",
        f"      x: {pot_rel['x']}", f"      y: {pot_rel['y']}",
        f"      w: {pot_rel['w']}", f"      h: {pot_rel['h']}", "      rel: true",
    ]
    lines = replace_block(lines, "    pot: &id003", pot_block, ["    to_call:"])

    new_text = "\n".join(lines)
    new_cfg = yaml.safe_load(new_text)
    if new_cfg["sites"]["golbet"]["hole"] != new_cfg["vision"]["rois"]["hole"]:
        raise SystemExit("alias rotti: abort")
    if new_cfg["sites"]["golbet"]["board"] != new_cfg["vision"]["rois"]["board"]:
        raise SystemExit("alias rotti: abort")

    (ROOT / "config.yaml.bak2").write_text(text, encoding="utf-8")
    (ROOT / "config.yaml").write_text(new_text, encoding="utf-8")
    print("✓ config.yaml aggiornato (backup config.yaml.bak2)")

    debug = frame.copy()

    def _draw(rois: list[dict], color: tuple[int, int, int]) -> None:
        for r in rois:
            try:
                x, y = int(r["x"] * fw), int(r["y"] * fh)
                w, h = int(r["w"] * fw), int(r["h"] * fh)
            except (KeyError, TypeError, ValueError):
                continue
            cv2.rectangle(debug, (x, y), (x + w, y + h), color, 4)

    _draw(hole_rel, (0, 255, 0))
    _draw(BOARD_REL, (255, 0, 0))
    Path(ROOT / "captures").mkdir(exist_ok=True)
    cv2.imwrite(str(ROOT / "captures/finish_calib_debug.png"), debug)
    print("✓ overlay captures/finish_calib_debug.png")
    print(f"\nhole : {hole_rel}")
    print(f"pot  : {pot_rel}")


if __name__ == "__main__":
    main()
