"""Save board crop for visual inspection."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

import cv2

CACHE_ROOT = Path("C:/Users/franc/.claude")
OUT_DIR = ROOT / "vision" / "debug_board_crops"
OUT_DIR.mkdir(exist_ok=True)

for num in (25, 26):
    path = CACHE_ROOT / f"image-cache/1d3f0527-6075-4c16-bd0d-401136adb5a4/{num}.png"
    frame = cv2.imread(str(path))
    h, w = frame.shape[:2]
    x0, x1 = int(w * 0.32), int(w * 0.68)
    y0, y1 = int(h * 0.25), int(h * 0.45)
    crop = frame[y0:y1, x0:x1]
    out = OUT_DIR / f"{num}_board_crop_mid.png"
    cv2.imwrite(str(out), crop)
    print(f"saved {out}")
