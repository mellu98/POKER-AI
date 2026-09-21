"""
Calibrazione automatica — trova carte e piatto sull'immagine del tavolo
e salva gli ROI come coordinate percentuali (robuste a qualunque dimensione).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "vision"))

import cv2
import numpy as np
import yaml
from capture import screenshot


def find_card_rois(frame: np.ndarray):
    """Trova le carte bianche sul tavolo verde. Ritorna lista di dict {x,y,w,h}."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Carte bianche: bassa saturazione, alto valore
    lower = np.array([0, 0, 150])
    upper = np.array([180, 100, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Chiudi buchi piccoli
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards = []
    fh, fw = frame.shape[:2]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / h if h > 0 else 0
        # Filtra per dimensione tipica carta da poker sullo schermo
        if 4000 < area < 50000 and 0.55 < aspect < 0.95 and 60 < w < 250 and 80 < h < 350:
            # Escludi aree troppo ai bordi estremi (fumetti, UI)
            if x > 20 and y > 20 and x + w < fw - 20 and y + h < fh - 20:
                cards.append({"x": x, "y": y, "w": w, "h": h})
    return cards


def to_relative(roi: dict, frame):
    h, w = frame.shape[:2]
    return {
        "x": round(roi["x"] / w, 4),
        "y": round(roi["y"] / h, 4),
        "w": round(roi["w"] / w, 4),
        "h": round(roi["h"] / h, 4),
        "rel": True,
    }


def main():
    print("Catturo screenshot del tavolo...")
    frame = screenshot(window_title="Free Poker")
    if frame is None:
        print("Errore: finestra 'Free Poker' non trovata.")
        sys.exit(1)

    print(f"Dimensione screenshot: {frame.shape[1]}x{frame.shape[0]}")

    cards = find_card_rois(frame)
    print(f"Trovate {len(cards)} carte candidate")

    if not cards:
        print("Nessuna carta trovata. Assicurati che il tavolo sia visibile.")
        sys.exit(1)

    fh, fw = frame.shape[:2]

    # Classifica: hole (in basso) vs board (in alto)
    hole = [c for c in cards if c["y"] > fh * 0.55]
    board = [c for c in cards if c["y"] <= fh * 0.55]

    # Prendi massimo 5 board
    board = sorted(board, key=lambda c: c["x"])[:5]

    # Se hole non rilevate automaticamente, stima dal centro del board
    board_center_x = None
    if board:
        board_center_x = sum(c["x"] + c["w"] / 2 for c in board) / len(board)

    if len(hole) < 2 and board_center_x:
        # Stima hole cards sotto il centro del board
        hole_w = int(fw * 0.055)
        hole_h = int(fh * 0.13)
        hole_y = int(fh * 0.58)
        gap = int(fw * 0.01)
        hole = [
            {"x": int(board_center_x - hole_w - gap), "y": hole_y, "w": hole_w, "h": hole_h},
            {"x": int(board_center_x + gap), "y": hole_y, "w": hole_w, "h": hole_h},
        ]
    else:
        hole = sorted(hole, key=lambda c: c["x"])[:2]

    print(f"  Hole cards: {len(hole)}")
    print(f"  Board cards: {len(board)}")

    # Piatto: sopra il board, centrato sul board
    pot = None
    if board:
        by = min(c["y"] for c in board)
        pot_x = int(board_center_x - fw * 0.08) if board_center_x else int(fw * 0.35)
        pot = {
            "x": pot_x,
            "y": max(0, by - int(fh * 0.10)),
            "w": int(fw * 0.16),
            "h": int(fh * 0.08),
        }
    else:
        pot = {
            "x": int(fw * 0.35),
            "y": int(fh * 0.30),
            "w": int(fw * 0.16),
            "h": int(fh * 0.08),
        }

    # Stack: sotto le hole, centrato
    stack = None
    if hole:
        hy = max(c["y"] + c["h"] for c in hole)
        stack_x = int(board_center_x - fw * 0.08) if board_center_x else int(fw * 0.35)
        stack = {
            "x": stack_x,
            "y": min(fh - 50, hy + int(fh * 0.03)),
            "w": int(fw * 0.16),
            "h": int(fh * 0.05),
        }

    # to_call: sotto lo stack, centrato
    to_call = None
    if stack:
        to_call = {
            "x": stack["x"],
            "y": min(fh - 30, stack["y"] + stack["h"] + int(fh * 0.02)),
            "w": int(fw * 0.16),
            "h": int(fh * 0.05),
        }

    # Converti a relative
    hole_rel = [to_relative(c, frame) for c in hole]
    board_rel = [to_relative(c, frame) for c in board]
    pot_rel = to_relative(pot, frame) if pot else None
    stack_rel = to_relative(stack, frame) if stack else None
    to_call_rel = to_relative(to_call, frame) if to_call else None

    # Salva
    config_path = Path("config.yaml")
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        config = {}

    config.setdefault("vision", {})
    config["vision"]["rois"] = {
        "hole": hole_rel,
        "board": board_rel,
        "pot": pot_rel,
        "to_call": to_call_rel,
        "stack": stack_rel,
        "dealer": None,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\nSalvato in {config_path}")
    print("ROI (relativi):")
    print(f"  hole: {hole_rel}")
    print(f"  board: {board_rel}")
    print(f"  pot: {pot_rel}")

    # Salva immagine di debug
    debug = frame.copy()
    for c in hole:
        cv2.rectangle(debug, (c["x"], c["y"]), (c["x"]+c["w"], c["y"]+c["h"]), (0, 255, 0), 2)
    for c in board:
        cv2.rectangle(debug, (c["x"], c["y"]), (c["x"]+c["w"], c["y"]+c["h"]), (255, 0, 0), 2)
    if pot:
        cv2.rectangle(debug, (pot["x"], pot["y"]), (pot["x"]+pot["w"], pot["y"]+pot["h"]), (0, 0, 255), 2)
    if stack:
        cv2.rectangle(debug, (stack["x"], stack["y"]), (stack["x"]+stack["w"], stack["y"]+stack["h"]), (0, 255, 255), 2)
    if to_call:
        cv2.rectangle(debug, (to_call["x"], to_call["y"]), (to_call["x"]+to_call["w"], to_call["y"]+to_call["h"]), (255, 255, 0), 2)

    cv2.imwrite("auto_calib_debug.png", debug)
    print("\nImmagine di debug salvata: auto_calib_debug.png")
    print("Apri l'immagine e verifica che i rettangoli siano corretti.")


if __name__ == "__main__":
    main()
