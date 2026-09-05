"""Test focused board-only LLM prompt on #25 and #26."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2
import yaml
import numpy as np
import llm_vision_extractor
from llm_vision_extractor import LLMVisionExtractor, _encode_frame_to_base64, _resolve_relative_roi
from capture import crop_roi

CACHE_ROOT = Path("C:/Users/franc/.claude")


def crop_board_region(frame, board_rois, pad=0.05):
    h, w = frame.shape[:2]
    # tight upper-center board region
    x0, x1 = int(w * 0.32), int(w * 0.68)
    y0, y1 = int(h * 0.37), int(h * 0.57)
    return frame[y0:y1, x0:x1]


def ask_board(extractor, board_crop):
    b64 = _encode_frame_to_base64(board_crop, max_dim=1024, jpeg_quality=90)
    headers = {"Authorization": f"Bearer {extractor.api_key}", "Content-Type": "application/json"}
    system = (
        "You are looking at a cropped region of a poker table that contains ONLY the community cards (board).\n"
        "List the visible community cards from left to right.\n"
        "Return ONLY a JSON array like [\"4d\", \"Th\", \"Ad\"].\n"
        "If no community cards are visible, return [].\n"
        "Do not guess or invent cards. Use rank+suit notation (e.g. Kd, Th, 7s)."
    )
    payload = {
        "model": extractor.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": "What are the community cards in this crop?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    }
    import requests, time
    t0 = time.time()
    resp = requests.post(extractor.api_url, headers=headers, json=payload, timeout=30.0)
    print(f"  time={time.time()-t0:.1f}s size={len(b64)//1024}KB status={resp.status_code}")
    if not resp.ok:
        print(resp.text[:200])
        return None
    return resp.json()["choices"][0]["message"]["content"]


def main():
    with open(ROOT / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f) or {}
    api_key = cfg.get("vision", {}).get("llm", {}).get("api_key")
    extractor = LLMVisionExtractor(api_key=api_key, model="google/gemini-3.1-flash-lite")
    board_rois = cfg.get("vision", {}).get("rois", {}).get("board", [])
    for num in (25, 26):
        path = CACHE_ROOT / f"image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/{num}.png"
        frame = cv2.imread(str(path))
        board_crop = crop_board_region(frame, board_rois)
        print(f"\n=== {num}.png board crop ===")
        text = ask_board(extractor, board_crop)
        print(f"  board: {text}")


if __name__ == "__main__":
    main()
