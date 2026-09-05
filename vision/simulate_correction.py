"""Simulate the new _correct_suits on the user's latest screenshots."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2
import yaml
from llm_vision_extractor import LLMVisionExtractor

CACHE_ROOT = Path("C:/Users/franc/.claude")

CASES = [
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/18.png",
        {"hole": ["4s", "As"], "board": ["2c", "Kh", "Ts"]},
        {"hole": ["4s", "Ad"], "board": ["2c", "Kd", "Ts", "As"]},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/19.png",
        {"hole": ["8c", "Qh"], "board": []},
        {"hole": ["8c", "Qc"], "board": []},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/21.png",
        {"hole": ["2c", "Jh"], "board": ["6h", "3h", "Ah"]},
        {"hole": ["2c", "Jd"], "board": ["6h", "3h", "Ah"]},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/22.png",
        {"hole": ["4h", "Ah"], "board": ["6c", "9d", "8d"]},
        {"hole": ["4d", "Ah"], "board": ["6c", "9h", "8d"]},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/23.png",
        {"hole": ["2d", "Jh"], "board": []},
        {"hole": ["2d", "Jd"], "board": []},
    ),
    # nuovi casi #24-#29
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/24.png",
        {"hole": ["Kh", "Ks"], "board": []},
        {"hole": ["Kh", "Kc"], "board": []},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/25.png",
        {"hole": ["Ts", "7d"], "board": ["Kd", "4h", "2d", "Qd"]},
        {"hole": ["Ts", "7d"], "board": ["4d", "Th", "Ad"]},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/26.png",
        {"hole": ["Qc", "Js"], "board": ["2s", "Jh", "Ac"]},
        {"hole": ["Qh", "Jd"], "board": ["2s", "Qd", "7h", "Jc", "6h"]},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/27.png",
        {"hole": ["Ts", "8s"], "board": ["6c", "7c", "Td"]},
        {"hole": ["Tc", "8s"], "board": ["6s", "7c", "Td"]},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/28.png",
        {"hole": ["9s", "Jh"], "board": ["2s", "Qh", "7s", "Js", "6h"]},
        {"hole": ["9h", "Jd"], "board": ["2s", "Qd", "7h", "Jc", "6h"]},
    ),
    (
        "image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/29.png",
        {"hole": ["Tc", "9c"], "board": ["4d", "Td", "Ad"]},
        {"hole": ["Ts", "9c"], "board": ["4d", "Th", "Ad"]},
    ),
]


def main():
    with open(ROOT / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f) or {}
    api_key = cfg.get("vision", {}).get("llm", {}).get("api_key")
    extractor = LLMVisionExtractor(
        api_key=api_key,
        model="google/gemini-3.1-flash-lite",
        config_path=str(ROOT / "config.yaml"),
    )

    for img_rel, llm_state, expected in CASES:
        img_path = CACHE_ROOT / img_rel
        frame = cv2.imread(str(img_path))
        simulated = {
            "hole": list(llm_state.get("hole", [])),
            "board": list(llm_state.get("board", [])),
            "pot": 0,
            "to_call": 0,
            "stack": 1000,
            "stage": "preflop",
            "position": "BTN",
        }
        corrected = extractor._correct_suits(frame, simulated)
        print(f"\n{img_path.name}")
        print(f"  LLM:   hole={simulated['hole']} board={simulated['board']}")
        print(f"  FIX:   hole={corrected['hole']} board={corrected['board']}")
        print(f"  REAL:  hole={expected['hole']} board={expected['board']}")


if __name__ == "__main__":
    main()
