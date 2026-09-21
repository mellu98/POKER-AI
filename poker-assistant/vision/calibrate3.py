import cv2
from pathlib import Path

img_path = Path(r"C:\Users\franc\.claude\image-cache\4a0bae80-b8b2-4141-9caa-41dee5631ae0\8.png")
frame = cv2.imread(str(img_path))

def crop(name, x, y, w, h):
    roi = frame[y:y+h, x:x+w]
    out = Path(__file__).parent / "debug" / f"{name}.png"
    cv2.imwrite(str(out), roi)
    print(f"Saved {name} ({x},{y},{w},{h})")

# Whole left card area
crop("whole_left_card", 700, 500, 250, 250)
# Top-left corner search
crop("tl_search", 720, 520, 150, 100)
# Rank guess 1
crop("rank_guess1", 750, 540, 60, 50)
# Rank guess 2 (more left)
crop("rank_guess2", 720, 530, 60, 50)
# Rank guess 3 (higher)
crop("rank_guess3", 750, 520, 60, 50)

# Right card search (maybe it's more to the right?)
crop("right_card_search", 900, 500, 300, 250)
crop("right_card_guess", 950, 540, 150, 150)

# Stack search (red label above You)
crop("stack_search1", 880, 700, 240, 80)
crop("stack_search2", 900, 710, 200, 60)
crop("stack_search3", 920, 720, 160, 40)

# Dealer button search
crop("dealer_search", 1240, 280, 80, 80)
