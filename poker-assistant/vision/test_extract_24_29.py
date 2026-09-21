"""Run full LLMVisionExtractor.extract() on #24-#29."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2
from llm_vision_extractor import LLMVisionExtractor

CACHE_ROOT = Path("C:/Users/franc/.claude")

CASES = [
    ("24.png", {"hole": ["Kh", "Kc"], "board": []}),
    ("25.png", {"hole": ["Ts", "7d"], "board": []}),
    ("26.png", {"hole": ["Qh", "Jd"], "board": ["2s", "Qd", "7h", "Jc", "6h"]}),
    ("27.png", {"hole": ["Tc", "8s"], "board": ["6s", "7c", "Td"]}),
    ("28.png", {"hole": ["9h", "Jd"], "board": ["2s", "Qd", "7h", "Jc", "6h"]}),
    ("29.png", {"hole": ["Ts", "9c"], "board": ["4d", "Th", "Ad"]}),
]


def main():
    extractor = LLMVisionExtractor(config_path=str(ROOT / "config.yaml"))
    for img_name, expected in CASES:
        path = CACHE_ROOT / f"image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/{img_name}"
        frame = cv2.imread(str(path))
        print(f"\n=== {img_name} ===")
        try:
            state = extractor.extract(frame)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        print(f"  EXTR: hole={state['hole']} board={state['board']} pot={state['pot']} stage={state['stage']}")
        print(f"  REAL: hole={expected['hole']} board={expected['board']}")


if __name__ == "__main__":
    main()
