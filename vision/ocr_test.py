import cv2
import pytesseract
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\franc\Desktop\POKER-AI\poker-assistant\tesseract\tesseract.exe"

img_path = Path(r"C:\Users\franc\.claude\image-cache\4a0bae80-b8b2-4141-9caa-41dee5631ae0\8.png")
frame = cv2.imread(str(img_path))
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Crop the right card top-left corner where '4' should be
# Based on tl_search (720,520,150,100), the '4' is around x=820, y=535
x, y, w, h = 820, 530, 40, 45
roi = gray[y:y+h, x:x+w]
cv2.imwrite(str(Path(__file__).parent / "debug" / "ocr_4_raw.png"), roi)

# Preprocess: threshold to isolate black text
_, binary = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY_INV)
binary = cv2.resize(binary, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(str(Path(__file__).parent / "debug" / "ocr_4_processed.png"), binary)

text = pytesseract.image_to_string(
    binary,
    config="--psm 10 -c tessedit_char_whitelist=AKQJT98765432",
)
print(f"OCR result for '4': '{text.strip()}'")

# Try on the pot '8'
pot = gray[250:310, 760:820]
_, pot_bin = cv2.threshold(pot, 120, 255, cv2.THRESH_BINARY_INV)
pot_bin = cv2.resize(pot_bin, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(str(Path(__file__).parent / "debug" / "ocr_8_processed.png"), pot_bin)
text2 = pytesseract.image_to_string(
    pot_bin,
    config="--psm 10 -c tessedit_char_whitelist=0123456789$",
)
print(f"OCR result for pot: '{text2.strip()}'")
