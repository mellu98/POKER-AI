"""Test pot OCR with a tighter ROI."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2
from llm_vision_extractor import LLMVisionExtractor, _resolve_relative_roi
from capture import crop_roi

CACHE_ROOT = Path("C:/Users/franc/.claude")

TIGHT_POT_ROI = {"x": 0.40, "y": 0.36, "w": 0.20, "h": 0.08, "rel": True}


def main():
    extractor = LLMVisionExtractor(config_path=str(ROOT / "config.yaml"))
    for num in range(24, 30):
        path = CACHE_ROOT / f"image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/{num}.png"
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        val = extractor._ocr_number_in_roi(frame, TIGHT_POT_ROI)
        print(f"{num}.png: tight pot OCR={val}")


if __name__ == "__main__":
    main()
