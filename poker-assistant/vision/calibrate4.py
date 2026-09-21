import cv2
from pathlib import Path

img_path = Path(r"C:\Users\franc\.claude\image-cache\4a0bae80-b8b2-4141-9caa-41dee5631ae0\8.png")
frame = cv2.imread(str(img_path))

def crop(name, x, y, w, h):
    roi = frame[y:y+h, x:x+w]
    out = Path(__file__).parent / "debug" / f"{name}.png"
    cv2.imwrite(str(out), roi)
    print(f"Saved {name} ({x},{y},{w},{h})")

# Right card rank (approx from tl_search)
crop("right_rank_exact", 805, 528, 45, 45)
# Right card suit (small one under rank)
crop("right_suit_exact", 808, 558, 35, 35)
# Left card rank (maybe covered, but try)
crop("left_rank_guess", 728, 530, 45, 45)
# Left card suit (small one)
crop("left_suit_guess", 732, 558, 35, 35)

# Pot exact (from pot_3 we know it includes $8)
crop("pot_exact", 750, 220, 150, 100)

# Stack exact (red label above You)
crop("stack_exact", 880, 720, 200, 50)
crop("stack_exact2", 900, 730, 160, 40)

# Dealer button (red D) near Ruth
crop("dealer_exact", 1270, 290, 60, 60)
