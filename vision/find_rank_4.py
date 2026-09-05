"""
Scan tl_search.png with a sliding window to find the '4' rank.
"""
import cv2
import pytesseract
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\franc\Desktop\POKER-AI\poker-assistant\tesseract\tesseract.exe"

tl = cv2.imread(str(Path(__file__).parent / "debug" / "tl_search.png"))
gray = cv2.cvtColor(tl, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

found = []
for y in range(0, h - 30, 5):
    for x in range(0, w - 30, 5):
        roi = gray[y : y + 30, x : x + 30]
        # preprocess
        _, binary = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(
            binary,
            config="--psm 10 -c tessedit_char_whitelist=AKQJT98765432",
        )
        text = text.strip()
        if text == "4":
            found.append((x, y))
            print(f"Found '4' at tl_search offset ({x}, {y})")
            cv2.imwrite(str(Path(__file__).parent / "debug" / f"found4_{x}_{y}.png"), roi)

print(f"Total matches: {len(found)}")
