"""Test LLM vision on a saved screenshot."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vision"))

import cv2
from llm_vision_extractor import LLMVisionExtractor


def main(image_path: str):
    extractor = LLMVisionExtractor(config_path="config.yaml")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load {image_path}")
        return
    state = extractor.extract(frame)
    print("--- LLM state ---")
    for k, v in state.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_llm_on_image.py <screenshot.png>")
        sys.exit(1)
    main(sys.argv[1])
