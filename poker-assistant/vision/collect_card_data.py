"""
Interactive card dataset collector — now with FAST auto-capture mode.

Cattura rapida di carte dal tavolo Goldbet e le salva in un dataset per
addestrare il classificatore locale.

Uso:
    python vision/collect_card_data.py

Comandi principali:
    SPAZIO  -> cattura (modalità dipende da M/F)
    M       -> passa a modalità manuale (etichetti ogni slot)
    F       -> passa a modalità fast (auto-cattura con guess)
    R       -> apre lo script di review offline
    T       -> addestra i modelli con i dati raccolti
    Q       -> esci

Modalità FAST:
    - Premi SPAZIO durante la mano.
    - Lo script riconosce automaticamente le carte con i template/classificatore
      già presenti e le salva SENZA chiederti nulla.
    - Ci mette meno di 1 secondo per cattura.
    - Le carte non riconosciute vengono salvate come 'XX' per essere corrette
      in review.

Modalità MANUALE:
    - Premi SPAZIO, poi scrivi la carta per ogni slot (es. As, 7h, Td).
    - Meglio per le prime carte o per correggere qualcosa al volo.

I file vengono salvati in:
    vision/dataset/cards/<GUESS>_<timestamp>_<slot>.png
"""
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import yaml
from capture import crop_roi, screenshot
from ocr_cards import load_templates_from_dir, match_card
from PIL import Image

RANKS = set("A23456789TJQK")
SUITS = set("shdc")

VISION_DIR = Path(__file__).parent
DATASET_DIR = VISION_DIR / "dataset"
CARDS_DIR = DATASET_DIR / "cards"
TEMPLATES_DIR = VISION_DIR / "templates"
RANK_MODEL = VISION_DIR / "models" / "card_rank_classifier.joblib"
SUIT_MODEL = VISION_DIR / "models" / "card_suit_classifier.joblib"

FULL_SIZE = (60, 80)


def ensure_dirs():
    CARDS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_roi(roi: dict, frame: np.ndarray) -> dict:
    if roi.get("rel"):
        h, w = frame.shape[:2]
        try:
            return {
                "x": int(roi["x"] * w),
                "y": int(roi["y"] * h),
                "w": int(roi["w"] * w),
                "h": int(roi["h"] * h),
            }
        except (KeyError, TypeError, ValueError):
            print(f"[collect] ROI malformato ({roi}): uso i valori grezzi")
            return roi
    return roi


def parse_label(text: str) -> str | None:
    text = text.strip().upper()
    if len(text) != 2:
        return None
    r, s = text[0], text[1].lower()
    if r in RANKS and s in SUITS:
        return f"{r}{s}"
    return None


def show_image(full_crop: np.ndarray, title: str = "card"):
    """Open a card crop in the default system viewer (headless-safe)."""
    rgb = cv2.cvtColor(full_crop, cv2.COLOR_BGR2RGB) if full_crop.ndim == 3 else full_crop
    Image.fromarray(rgb).show(title=title)


class Guesser:
    """Tries to guess a card crop using templates first, then classifier."""

    def __init__(self):
        self.templates = {}
        if TEMPLATES_DIR.exists():
            self.templates = load_templates_from_dir(str(TEMPLATES_DIR))
            print(f"[collect] Loaded {len(self.templates)} templates for guessing.")

        self.classifier = None
        if RANK_MODEL.exists() and SUIT_MODEL.exists():
            try:
                from card_classifier import CardClassifier
                self.classifier = CardClassifier(
                    str(RANK_MODEL), str(SUIT_MODEL), confidence_threshold=0.35
                )
                if self.classifier.is_ready:
                    print("[collect] Loaded full-card classifier for guessing.")
                else:
                    self.classifier = None
            except (ImportError, OSError, ValueError, AttributeError) as e:
                print(f"[collect] Could not load classifier for guessing: {e}")

    def guess(self, crop: np.ndarray) -> tuple[str | None, str]:
        """Return (card_label, source). source is 'template', 'classifier', or 'unknown'."""
        if crop is None or crop.size == 0:
            return None, "unknown"

        # Try template matching first (full card)
        if self.templates:
            tmpl_guess = match_card(crop, self.templates)
            if tmpl_guess:
                return tmpl_guess, "template"

        # Fallback to classifier
        if self.classifier is not None:
            pred, conf = self.classifier.predict_card(crop)
            if pred is not None:
                return pred, f"classifier({conf:.2f})"

        return None, "unknown"


def save_card(slot: str, full_crop: np.ndarray, label: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    stem = f"{label}_{ts}_{slot}"
    full_path = CARDS_DIR / f"{stem}.png"
    cv2.imwrite(str(full_path), full_crop)
    return full_path


def is_similar(a: np.ndarray | None, b: np.ndarray | None, threshold: float = 8.0) -> bool:
    """Return True if two card crops are visually similar (skip duplicates)."""
    if a is None or b is None or a.size == 0 or b.size == 0:
        return False
    thumb = (30, 40)
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
    a_small = cv2.resize(a_gray, thumb, interpolation=cv2.INTER_AREA)
    b_small = cv2.resize(b_gray, thumb, interpolation=cv2.INTER_AREA)
    try:
        diff = float(np.mean(np.abs(a_small.astype(float) - b_small.astype(float))))
    except (ValueError, TypeError):
        return False
    return diff < threshold


def manual_capture(frame: np.ndarray, slots: list[tuple[str, int, dict]], saved_total_ref: list[int]):
    for roi_type, idx, roi_cfg in slots:
        slot = f"{roi_type}{idx}"
        roi = resolve_roi(roi_cfg, frame)
        crop = crop_roi(frame, roi["x"], roi["y"], roi["w"], roi["h"])
        if crop is None or crop.size == 0:
            print(f"[collect] {slot}: ROI vuoto, salto.")
            continue

        full_crop = cv2.resize(crop, FULL_SIZE, interpolation=cv2.INTER_AREA)
        show_image(full_crop, title=f"{slot} — inserisci carta")

        label_text = input(
            f"\n{slot}: carta? (es. As, 7h, Td). INVIO=skip, d=delete last: "
        ).strip()

        if label_text.lower() == "d":
            continue

        label = parse_label(label_text)
        if label is None:
            if label_text:
                print(f"[collect] Etichetta non valida: '{label_text}'")
            continue

        path = save_card(slot, full_crop, label)
        saved_total_ref[0] += 1
        print(f"[collect] Salvata {label} -> {path.name}")


def fast_capture(frame: np.ndarray, slots: list[tuple[str, int, dict]],
                 guesser: Guesser, saved_total_ref: list[int]):
    print("[collect] FAST capture in corso...")
    summary = []
    for roi_type, idx, roi_cfg in slots:
        slot = f"{roi_type}{idx}"
        roi = resolve_roi(roi_cfg, frame)
        crop = crop_roi(frame, roi["x"], roi["y"], roi["w"], roi["h"])
        if crop is None or crop.size == 0:
            summary.append((slot, None, "empty"))
            continue

        full_crop = cv2.resize(crop, FULL_SIZE, interpolation=cv2.INTER_AREA)
        guess, source = guesser.guess(full_crop)
        label = guess if guess is not None else "XX"

        save_card(slot, full_crop, label)
        saved_total_ref[0] += 1
        summary.append((slot, label, source))
        print(f"  {slot}: {label} ({source})")

    print(f"[collect] Fast capture salvate: {len([s for s in summary if s[1]])}")


def auto_capture_once(
    frame: np.ndarray,
    slots: list[tuple[str, int, dict]],
    guesser: Guesser,
    last_crops: dict[str, np.ndarray],
    saved_total_ref: list[int],
) -> dict[str, np.ndarray]:
    """Capture one auto frame: save only changed/non-duplicate card slots."""
    saved_this_round = 0
    for roi_type, idx, roi_cfg in slots:
        slot = f"{roi_type}{idx}"
        roi = resolve_roi(roi_cfg, frame)
        crop = crop_roi(frame, roi["x"], roi["y"], roi["w"], roi["h"])
        if crop is None or crop.size == 0:
            continue

        full_crop = cv2.resize(crop, FULL_SIZE, interpolation=cv2.INTER_AREA)

        # Skip if this slot hasn't changed enough
        if is_similar(full_crop, last_crops.get(slot)):
            continue

        guess, _ = guesser.guess(full_crop)
        label = guess if guess is not None else "XX"
        save_card(slot, full_crop, label)
        saved_total_ref[0] += 1
        saved_this_round += 1
        last_crops[slot] = full_crop

    if saved_this_round:
        print(f"[collect] Auto saved {saved_this_round} new cards (total {saved_total_ref[0]})")
    return last_crops


def auto_capture_loop(
    cfg: dict,
    slots: list[tuple[str, int, dict]],
    guesser: Guesser,
    stop_event: threading.Event,
    interval: float = 2.0,
):
    """Background thread: capture while the user plays."""
    window_title = cfg.get("vision", {}).get("window_title")
    saved_total = [0]
    last_crops: dict[str, np.ndarray] = {}

    print(f"[collect] AUTO avviato: cattura ogni {interval}s. Gioca pure.")
    while not stop_event.is_set():
        try:
            frame = screenshot(window_title=window_title)
            if frame is not None:
                last_crops = auto_capture_once(frame, slots, guesser, last_crops, saved_total)
        except (OSError, ValueError, cv2.error) as e:
            print(f"[collect] Auto capture error: {e}")

        # Sleep in small chunks so stop is responsive
        try:
            chunks = int(interval * 10)
        except (TypeError, ValueError):
            chunks = 20
        for _ in range(chunks):
            if stop_event.is_set():
                break
            time.sleep(0.1)

    print(f"[collect] AUTO fermato. Totale salvate: {saved_total[0]}")


def run_training():
    print("\n[collect] Avvio training...")
    script = VISION_DIR / "train_full_card_classifier.py"
    try:
        subprocess.run([sys.executable, str(script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[collect] Training fallito: {e}")


def run_review():
    print("\n[collect] Apro review offline...")
    script = VISION_DIR / "review_card_dataset.py"
    try:
        subprocess.run([sys.executable, str(script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[collect] Review fallito: {e}")
    except FileNotFoundError:
        print("[collect] Review script non trovato.")


def capture_loop(cfg: dict, mode: str = "fast"):
    window_title = cfg.get("vision", {}).get("window_title")
    rois = cfg.get("vision", {}).get("rois", {})
    hole_rois = rois.get("hole", [])
    board_rois = rois.get("board", [])

    slots = []
    for i, r in enumerate(hole_rois):
        slots.append(("hole", i + 1, r))
    for i, r in enumerate(board_rois):
        slots.append(("board", i + 1, r))

    if not slots:
        print("[collect] ERRORE: nessun ROI configurato in config.yaml")
        sys.exit(1)

    guesser = Guesser()
    saved_total = [0]

    print("\n[collect] Comandi:")
    print("  SPAZIO = cattura singola")
    print("  A      = modalità AUTO (cattura continua mentre giochi)")
    print("  F      = modalità fast (cattura singola automatica)")
    print("  M      = modalità manuale")
    print("  R      = review offline")
    print("  T      = train")
    print("  Q      = esci")
    print(f"\n[collect] Modalità attuale: {mode.upper()}\n")

    auto_thread: threading.Thread | None = None
    stop_event = threading.Event()

    while True:
        if auto_thread is not None and auto_thread.is_alive():
            # While auto is running, just wait for Enter to stop
            input("[AUTO] Cattura in corso... premi INVIO per fermare: ")
            stop_event.set()
            auto_thread.join(timeout=2)
            auto_thread = None
            continue

        key = input(
            f"\n[{mode.upper()}] SPAZIO=cattura A=auto F=fast M=manuale R=review T=train Q=esci: "
        ).strip().lower()

        if key == "q":
            if auto_thread is not None and auto_thread.is_alive():
                stop_event.set()
                auto_thread.join(timeout=2)
            break
        if key == "m":
            mode = "manual"
            print("[collect] Modalità MANUALE")
            continue
        if key == "f":
            mode = "fast"
            print("[collect] Modalità FAST")
            continue
        if key == "a":
            stop_event.clear()
            auto_thread = threading.Thread(
                target=auto_capture_loop,
                args=(cfg, slots, guesser, stop_event, 2.0),
                daemon=True,
            )
            auto_thread.start()
            continue
        if key == "r":
            run_review()
            continue
        if key == "t":
            run_training()
            continue
        if key != " ":
            print("[collect] Comando non riconosciuto.")
            continue

        # Capture
        print("[collect] Catturo screenshot...")
        frame = screenshot(window_title=window_title)
        if frame is None:
            print("[collect] Screenshot fallito.")
            continue
        print(f"[collect] Screenshot: {frame.shape[1]}x{frame.shape[0]}")

        if mode == "fast":
            fast_capture(frame, slots, guesser, saved_total)
        else:
            manual_capture(frame, slots, saved_total)

        print(f"[collect] Totale immagini salvate: {saved_total[0]}")


def main():
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("[collect] ERRORE: config.yaml non trovato.")
        sys.exit(1)

    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except OSError as exc:
        sys.exit(f"[collect] config.yaml non leggibile: {exc}")

    ensure_dirs()

    # Default to fast mode so the user can keep up with 15s hands
    capture_loop(cfg, mode="fast")
    print(f"\n[collect] Dataset in: {CARDS_DIR}")
    print("[collect] Esci.")


if __name__ == "__main__":
    main()
