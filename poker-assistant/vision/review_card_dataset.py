"""
Offline dataset review — correct wrong guesses from fast/auto capture mode.

Usage:
    python vision/review_card_dataset.py

For each saved card image:
    - opens the image in your default image viewer
    - type the correct label (e.g. As, 7h, Td) and press ENTER to rename
    - press ENTER empty to keep current label
    - type 'd' and press ENTER to delete
    - type 'q' to quit

The script only touches files in vision/dataset/cards/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image


RANKS = set("A23456789TJQK")
SUITS = set("shdc")

CARDS_DIR = Path(__file__).parent / "dataset" / "cards"


def parse_label(text: str) -> str | None:
    text = text.strip().upper()
    if len(text) != 2:
        return None
    r, s = text[0], text[1].lower()
    if r in RANKS and s in SUITS:
        return f"{r}{s}"
    return None


def current_label_from_filename(path: Path) -> str:
    stem = path.stem
    if len(stem) >= 2:
        r, s = stem[0].upper(), stem[1].lower()
        if r in RANKS and s in SUITS:
            return f"{r}{s}"
    return "XX"


def review():
    if not CARDS_DIR.exists():
        print("[review] Nessun dataset trovato.")
        return

    files = sorted(CARDS_DIR.glob("*.png"))
    if not files:
        print("[review] Nessuna immagine da revisionare.")
        return

    print(f"\n[review] {len(files)} immagini da revisionare.")
    print("Comandi: scrivi la carta (As, 7h...) per rinominare, INVIO=ok, d=cancella, q=esci\n")

    kept = 0
    renamed = 0
    deleted = 0

    for i, path in enumerate(files, 1):
        current = current_label_from_filename(path)

        # Open in default system viewer (works even with headless OpenCV)
        try:
            Image.open(path).show(title=f"{path.name} — {current}")
        except Exception as e:
            print(f"[review] Non riesco ad aprire l'immagine: {e}")

        prompt = f"[{i}/{len(files)}] {path.name} (etichetta: {current}): "
        answer = input(prompt).strip().lower()

        if answer == "q":
            print("[review] Uscita anticipata.")
            break
        if answer == "d":
            path.unlink()
            print("  cancellata")
            deleted += 1
            continue

        new_label = parse_label(answer)
        if new_label is None:
            print(f"  mantenuta {current}")
            kept += 1
            continue

        if new_label == current:
            print("  già corretta")
            kept += 1
            continue

        # Keep the timestamp/slot suffix after the old label
        rest = path.stem[2:] if len(path.stem) >= 2 else f"_{path.stem}"
        new_name = f"{new_label}{rest}.png"
        new_path = path.with_name(new_name)

        # Avoid overwriting an existing file
        counter = 1
        while new_path.exists():
            new_name = f"{new_label}{rest}_{counter}.png"
            new_path = path.with_name(new_name)
            counter += 1

        path.rename(new_path)
        print(f"  rinominata -> {new_path.name}")
        renamed += 1

    print(f"\n[review] Fatto. Mantenute: {kept}, Rinonimate: {renamed}, Cancellate: {deleted}")
    print("[review] Ora puoi rilanciare il trainer con: python vision/train_full_card_classifier.py")


if __name__ == "__main__":
    review()
