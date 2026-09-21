import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import cv2
from pathlib import Path
import easyocr
reader = easyocr.Reader(["en"], gpu=False)

for name in ["tl_search", "whole_left_card", "pot_3"]:
    p = Path(__file__).parent / "debug" / f"{name}.png"
    img = cv2.imread(str(p))
    if img is None:
        print(f"{name}: missing")
        continue
    results = reader.readtext(img)
    texts = [r[1] for r in results]
    print(f"{name}: {texts}")
