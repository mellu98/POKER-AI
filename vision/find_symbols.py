"""
Find black symbols (rank + suit) inside tl_search.png using contours.
"""
import cv2
from pathlib import Path

tl = cv2.imread(str(Path(__file__).parent / "debug" / "tl_search.png"))
gray = cv2.cvtColor(tl, cv2.COLOR_BGR2GRAY)
# isolate dark pixels (text/symbols)
_, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

debug = tl.copy()
found = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 50 or area > 2000:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    found.append((x, y, w, h, area))
    cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.putText(debug, f"{area}", (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

found.sort(key=lambda b: b[0])
print(f"Found {len(found)} symbol(s)")
for i, (x, y, w, h, area) in enumerate(found):
    print(f"  {i}: x={x}, y={y}, w={w}, h={h}, area={area}")
    cv2.imwrite(str(Path(__file__).parent / "debug" / f"symbol_{i}.png"), tl[y:y+h, x:x+w])

cv2.imwrite(str(Path(__file__).parent / "debug" / "tl_symbols.png"), debug)
print("Saved debug to tl_symbols.png")
