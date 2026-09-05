"""Test EasyOCR on rank and pot crops."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import cv2
from pathlib import Path

try:
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
except Exception as e:
    print(f"EasyOCR init failed: {e}")
    exit(1)

for name in ["rank_left_big", "pot_3", "stack_2", "btn_fold", "btn_check", "btn_raise"]:
    p = Path(__file__).parent / "debug" / f"{name}.png"
    img = cv2.imread(str(p))
    if img is None:
        print(f"{name}: missing image")
        continue
    results = reader.readtext(img)
    texts = [r[1] for r in results]
    print(f"{name}: {texts}")
