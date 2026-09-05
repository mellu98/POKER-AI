"""Debug pot OCR on #24-#29."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2
import yaml
from llm_vision_extractor import LLMVisionExtractor, _resolve_relative_roi
from capture import crop_roi

CACHE_ROOT = Path("C:/Users/franc/.claude")


def main():
    with open(ROOT / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f) or {}
    extractor = LLMVisionExtractor(config_path=str(ROOT / "config.yaml"))
    pot_roi = cfg.get("vision", {}).get("rois", {}).get("pot")
    for num in range(24, 30):
        path = CACHE_ROOT / f"image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/{num}.png"
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        val = extractor._ocr_number_in_roi(frame, pot_roi)
        print(f"{num}.png: pot OCR={val}")


if __name__ == "__main__":
    main()
