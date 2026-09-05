"""Test script: make a real API call with a synthetic poker table image."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vision"))

from llm_vision_extractor import LLMVisionExtractor
import numpy as np
import cv2

API_KEY = os.getenv("OPENROUTER_API_KEY")


def draw_poker_table():
    """Draw a very crude synthetic poker table with card text."""
    img = np.full((500, 800, 3), (20, 80, 20), dtype=np.uint8)  # green felt

    # Hole cards (bottom center)
    cv2.putText(img, "A", (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.putText(img, "s", (340, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.putText(img, "K", (420, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.putText(img, "h", (460, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

    # Board cards (top center)
    cv2.putText(img, "Qd", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.putText(img, "Jh", (320, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.putText(img, "2c", (440, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

    # Pot
    cv2.putText(img, "Pot: 120", (300, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

    # To call
    cv2.putText(img, "Call: 20", (300, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

    return img


def main():
    if not API_KEY:
        print("SKIPPED: OPENROUTER_API_KEY is not configured")
        return

    frame = draw_poker_table()
    cv2.imwrite("synthetic_table.png", frame)
    print("[test] Saved synthetic_table.png")

    ext = LLMVisionExtractor(api_key=API_KEY, cooldown_seconds=0.0)

    print("[test] Calling OpenRouter API with synthetic table...")
    try:
        state = ext.extract(frame)
        print(f"[test] SUCCESS!")
        for k, v in state.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"[test] ERROR: {e}")


if __name__ == "__main__":
    main()
