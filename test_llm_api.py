"""Test script: make a real API call to OpenRouter with a dummy image."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vision"))

from llm_vision_extractor import LLMVisionExtractor, _encode_frame_to_base64
import numpy as np
import cv2

API_KEY = os.getenv("OPENROUTER_API_KEY")


def main():
    if not API_KEY:
        print("SKIPPED: OPENROUTER_API_KEY is not configured")
        return

    # Create a synthetic black image (not a poker table, but tests the API plumbing)
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(frame, "TEST IMAGE", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    print("[test] Creating LLMVisionExtractor...")
    ext = LLMVisionExtractor(api_key=API_KEY, cooldown_seconds=0.0)

    print("[test] Calling OpenRouter API...")
    try:
        state = ext.extract(frame)
        print(f"[test] SUCCESS: {state}")
    except Exception as e:
        print(f"[test] ERROR: {e}")


if __name__ == "__main__":
    main()
