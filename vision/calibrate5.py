import cv2
from pathlib import Path

img_path = Path(r"C:\Users\franc\.claude\image-cache\4a0bae80-b8b2-4141-9caa-41dee5631ae0\8.png")
frame = cv2.imread(str(img_path))

def crop(name, x, y, w, h):
    roi = frame[y:y+h, x:x+w]
    out = Path(__file__).parent / "debug" / f"{name}.png"
    cv2.imwrite(str(out), roi)
    print(f"Saved {name} ({x},{y},{w},{h})")

# Based on tl_search observations:
# Right card rank "4" is around x~820, y~535
crop("rank_right_v2", 815, 530, 45, 45)
crop("suit_right_v2", 818, 558, 35, 35)
# Left card small suit is around x~735, y~558
crop("suit_left_v2", 730, 555, 35, 35)
# Left card rank (covered now, but for completeness)
crop("rank_left_v2", 730, 530, 45, 45)

# Stack red label — try area just above "You" label
# You is at y~780, red label just above, maybe y=745-770
crop("stack_v1", 880, 745, 160, 35)
crop("stack_v2", 900, 750, 120, 30)
crop("stack_v3", 920, 755, 100, 25)

# Dealer button — red circle with D. Try near Ruth's avatar
# Ruth is at x~1200, y~300. D is slightly right/up.
crop("dealer_v1", 1300, 290, 50, 50)
crop("dealer_v2", 1320, 300, 40, 40)
crop("dealer_v3", 1280, 280, 60, 60)

# Pot refined — just the number
crop("pot_v1", 750, 240, 80, 80)
crop("pot_v2", 760, 250, 60, 60)
