"""Test raw LLM extraction at higher resolution on #25 and #26."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2
import yaml
import llm_vision_extractor
from llm_vision_extractor import LLMVisionExtractor, _encode_frame_to_base64

CACHE_ROOT = Path("C:/Users/franc/.claude")


def call_raw(extractor, frame, max_dim):
    b64 = _encode_frame_to_base64(frame, max_dim=max_dim, jpeg_quality=85)
    headers = {"Authorization": f"Bearer {extractor.api_key}", "Content-Type": "application/json"}
    payload = {
        "model": extractor.model,
        "messages": [
            {"role": "system", "content": llm_vision_extractor.SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract the full poker table state from this screenshot."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
    }
    import requests, time
    t0 = time.time()
    resp = requests.post(extractor.api_url, headers=headers, json=payload, timeout=30.0)
    print(f"  max_dim={max_dim} time={time.time()-t0:.1f}s size={len(b64)//1024}KB status={resp.status_code}")
    if not resp.ok:
        print(resp.text[:200])
        return None
    return resp.json()["choices"][0]["message"]["content"]


def main():
    with open(ROOT / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f) or {}
    api_key = cfg.get("vision", {}).get("llm", {}).get("api_key")
    extractor = LLMVisionExtractor(api_key=api_key, model="google/gemini-3.1-flash-lite")
    for num in (25, 26):
        path = CACHE_ROOT / f"image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/{num}.png"
        frame = cv2.imread(str(path))
        print(f"\n=== {num}.png ===")
        for max_dim in (1024, 1536):
            text = call_raw(extractor, frame, max_dim)
            print(f"  raw: {text[:300]}")


if __name__ == "__main__":
    main()
