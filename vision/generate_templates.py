"""
Generate synthetic rank and suit templates using Pillow.
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).parent / "templates"
OUT_DIR.mkdir(exist_ok=True)

RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
SUITS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}


def _find_font(size: int):
    # Try common Windows fonts that support suit symbols
    candidates = [
        "C:/Windows/Fonts/seguisym.ttf",   # Segoe UI Symbol
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/verdana.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def make_template(text: str, size: tuple, font_size: int, color: tuple = (0, 0, 0)):
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _find_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (size[0] - tw) // 2
    y = (size[1] - th) // 2 - bbox[1]  # adjust for ascent
    draw.text((x, y), text, font=font, fill=color)
    return img


def main():
    # Rank templates
    for rank in RANKS:
        img = make_template(rank, (40, 40), 32)
        img.save(OUT_DIR / f"rank_{rank}.png")
        print(f"Saved rank_{rank}.png")

    # Suit templates (black for spades/clubs, red for hearts/diamonds)
    for code, sym in SUITS.items():
        color = (200, 0, 0) if code in ("h", "d") else (0, 0, 0)
        img = make_template(sym, (30, 30), 24, color)
        img.save(OUT_DIR / f"suit_{code}.png")
        print(f"Saved suit_{code}.png")

    print(f"\nAll templates saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
