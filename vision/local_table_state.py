"""
local_table_state.py — Estrazione locale dello stato multi-seat.

Legge dallo screenshot:
  - stack di ogni seat
  - quale seat è attivo (ha un giocatore seduto)
  - posizione del dealer button
  - pot, to_call, hero stack (usando le ROI calibrate)

Tutto locale, nessuna chiamata LLM. Deve essere veloce (< 100ms).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from seat_layout import SeatLayout


@dataclass
class SeatState:
    index: int
    position: str
    stack: Optional[int]
    is_active: bool
    is_hero: bool
    has_button: bool
    name: Optional[str] = None


@dataclass
class TableState:
    hero_seat: int
    button_seat: Optional[int]
    table_type: str = "9max"
    seats: List[SeatState] = field(default_factory=list)
    pot: Optional[int] = None
    to_call: Optional[int] = None
    hero_stack: Optional[int] = None

    def table_type_from_config(self) -> str:
        return self.table_type

    def active_count(self) -> int:
        return sum(1 for s in self.seats if s.is_active)

    def effective_stack(self) -> Optional[int]:
        """Stack effettivo = minimo stack attivo tra gli avversari."""
        stacks = [s.stack for s in self.seats if s.is_active and s.stack is not None]
        if not stacks:
            return None
        return min(stacks)


# --------------------------------------------------------------------------- #
# OCR helper (self-contained per evitare dipendenze circolari)
# --------------------------------------------------------------------------- #

def _resolve_relative_roi(roi: dict, frame: np.ndarray) -> Optional[dict]:
    if roi is None:
        return None
    if roi.get("rel"):
        h, w = frame.shape[:2]
        return {
            "x": int(roi["x"] * w),
            "y": int(roi["y"] * h),
            "w": int(roi["w"] * w),
            "h": int(roi["h"] * h),
        }
    return roi


def _crop_roi(frame: np.ndarray, roi: dict) -> Optional[np.ndarray]:
    roi = _resolve_relative_roi(roi, frame)
    if roi is None:
        return None
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    if w <= 0 or h <= 0:
        return None
    return frame[y : y + h, x : x + w]


def _ocr_number(crop: np.ndarray, whitelist: str = "0123456789$€.,") -> Optional[int]:
    """OCR di un numero monetario. Ritorna valore in centesimi."""
    if crop is None or crop.size == 0:
        return None
    try:
        import pytesseract
    except Exception:
        return None

    # Resolve Tesseract binary (Windows bundle, macOS Homebrew, or PATH)
    from tesseract_utils import find_tesseract_binary
    pytesseract.pytesseract.tesseract_cmd = find_tesseract_binary()

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    # Prova più strategie di preprocessamento e prendi il risultato migliore
    best_value: Optional[int] = None
    best_len = 0

    strategies = [
        ("bin_inv_180", lambda: cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)[1]),
        ("bin_inv_150", lambda: cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]),
        ("bin_200", lambda: cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]),
        ("adaptive", lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY_INV, 11, 2)),
    ]

    for name, make_binary in strategies:
        binary = make_binary()
        binary = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        # Piccola pulizia
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        text = pytesseract.image_to_string(
            binary,
            config=f"--psm 7 -c tessedit_char_whitelist={whitelist}",
        ).strip()

        if not text:
            continue

        import re
        matches = re.findall(r"[\d.,]+", text)
        if not matches:
            continue
        token = matches[-1].replace(",", ".")
        # Normalizza multipli punti (es. ".." -> ".")
        token = re.sub(r"\.{2,}", ".", token)
        # Se inizia con punto, aggiungi uno zero
        if token.startswith("."):
            token = "0" + token
        if token.endswith("."):
            token = token[:-1]
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            continue

        if "." in token:
            cents = int(round(value * 100))
        else:
            cents = int(value)

        # Preferisci il risultato con più cifre (più informativo)
        digits = len(re.sub(r"\D", "", token))
        if digits > best_len:
            best_len = digits
            best_value = cents

    return best_value


def _ocr_text(crop: np.ndarray) -> str:
    """OCR generico di testo (per leggere 'Check', 'Call', ecc.)."""
    if crop is None or crop.size == 0:
        return ""
    try:
        import pytesseract
    except Exception:
        return ""

    tesseract_bin = Path(__file__).parent.parent / "tesseract" / "tesseract.exe"
    if tesseract_bin.exists():
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_bin)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(binary, config="--psm 7").strip().lower()
    return text


# --------------------------------------------------------------------------- #
# Button detection
# --------------------------------------------------------------------------- #

def _find_button_in_roi(frame: np.ndarray, roi: dict, debug: bool = False) -> Tuple[bool, float]:
    """
    Cerca il dealer button (cerchio rosso/arancione/giallo con D) dentro la ROI.
    Ritorna (trovato, confidence).
    """
    crop = _crop_roi(frame, roi)
    if crop is None or crop.size == 0:
        return False, 0.0

    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return False, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Rosso in HSV ha due range
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Arancione
    lower_orange = np.array([10, 120, 100])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Giallo/oro (Goldbet usa un D in un cerchio dorato)
    lower_yellow = np.array([20, 120, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    color_mask = cv2.bitwise_or(red_mask, orange_mask)
    color_mask = cv2.bitwise_or(color_mask, yellow_mask)

    # Piccola pulizia
    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circularity = 4 * math.pi * area / (perim * perim)
        # Cerchio perfetto = 1.0; accettiamo > 0.55
        if circularity > 0.55:
            score = min(1.0, circularity * (min(area, 2000) / 2000))
            if score > best_score:
                best_score = score

    if debug:
        print(f"[button] best_score={best_score:.2f}")

    return best_score > 0.45, best_score


# --------------------------------------------------------------------------- #
# Table state extractor
# --------------------------------------------------------------------------- #

class LocalTableStateExtractor:
    """
    Estrae lo stato completo del tavolo da uno screenshot usando solo
    tecniche locali (OCR + template matching).
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.cfg = self._load_config()
        self.site = self.cfg.get("vision", {}).get("site", "golbet")
        self.table_cfg = self.cfg.get("table", {})
        self.table_type = self.table_cfg.get("type", "9max")
        self.hero_seat = int(self.table_cfg.get("hero_seat", 0))
        self.max_seats = 6 if self.table_type == "6max" else 9

        site_cfg = self.cfg.get("sites", {}).get(self.site, {})
        self.site_rois = site_cfg

        # Fallback su rois legacy se il sito non è ancora calibrato
        self.legacy_rois = self.cfg.get("vision", {}).get("rois", {})

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _site_roi(self, key: str):
        return self.site_rois.get(key) or self.legacy_rois.get(key)

    def _read_number_at(self, frame: np.ndarray, roi: dict) -> Optional[int]:
        crop = _crop_roi(frame, roi)
        return _ocr_number(crop)

    def _read_to_call(self, frame: np.ndarray) -> Optional[int]:
        """Legge l'importo da call. Se il bottone dice 'Check', ritorna 0."""
        roi = self._site_roi("to_call")
        crop = _crop_roi(frame, roi)
        if crop is None or crop.size == 0:
            return None
        text = _ocr_text(crop)
        if "check" in text:
            return 0
        return _ocr_number(crop)

    def detect_button_seat(self, frame: np.ndarray) -> Optional[int]:
        """
        Trova il seat con il dealer button cercando nella ROI di ogni seat.
        Ritorna l'indice del seat con confidence più alta, o None.
        """
        seat, _ = self.detect_button_seat_with_score(frame)
        return seat

    def detect_button_seat_with_score(self, frame: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Trova il seat con il dealer button e restituisce anche il confidence score.
        """
        seats = self.site_rois.get("seats", [])
        if not seats:
            return None, 0.0

        best_seat: Optional[int] = None
        best_score = 0.0
        for seat in seats:
            idx = seat.get("index")
            if idx is None:
                continue
            # Cerca il bottone in una piccola area attorno al seat center
            x, y = seat.get("x", 0.0), seat.get("y", 0.0)
            # ROI 12% dello schermo centrata sul seat (il bottone può essere
            # spostato rispetto all'avatar)
            roi = {
                "x": max(0.0, x - 0.06),
                "y": max(0.0, y - 0.06),
                "w": 0.12,
                "h": 0.12,
                "rel": True,
            }
            found, score = _find_button_in_roi(frame, roi)
            if found and score > best_score:
                best_score = score
                best_seat = idx

        # Se non trovato nei seat, prova la ROI generica dealer_button
        if best_seat is None:
            dealer_roi = self._site_roi("dealer_button")
            if dealer_roi:
                found, score = _find_button_in_roi(frame, dealer_roi)
                if found:
                    # Non sappiamo a quale seat appartiene; restituiamo None
                    # ma logghiamo.
                    print(f"[local_table] Dealer button detected in generic ROI (score={score:.2f})")

        return best_seat, best_score

    def read_seat_stacks(self, frame: np.ndarray) -> Dict[int, Optional[int]]:
        """Legge lo stack di ogni seat dalla ROI calibrata."""
        stacks: Dict[int, Optional[int]] = {}
        seats = self.site_rois.get("seats", [])
        for seat in seats:
            idx = seat.get("index")
            stack_roi = seat.get("stack_roi")
            if idx is None or not stack_roi:
                continue
            stacks[idx] = self._read_number_at(frame, stack_roi)
        return stacks

    def detect_active_seats(self, frame: np.ndarray, stacks: Dict[int, Optional[int]]) -> List[int]:
        """
        Determina quali seat sono attivi. Per ora: uno seat è attivo se ha
        uno stack leggibile. In futuro si può aggiungere rilevamento carte
        coperte o avatar.
        """
        active = []
        for idx, stack in stacks.items():
            if stack is not None and stack >= 0:
                active.append(idx)
        return active

    def extract(self, frame: np.ndarray) -> TableState:
        """Estrae lo stato completo del tavolo."""
        # Numero base
        pot = self._read_number_at(frame, self._site_roi("pot"))
        to_call = self._read_to_call(frame)
        hero_stack = self._read_number_at(frame, self._site_roi("stack"))

        # Seat stacks e attività
        stacks = self.read_seat_stacks(frame)
        active_seats = self.detect_active_seats(frame, stacks)

        # Button
        button_seat = self.detect_button_seat(frame)

        # Costruisci layout
        layout = SeatLayout(
            table_type=self.table_type,
            button_seat=button_seat if button_seat is not None else self.hero_seat,
            hero_seat=self.hero_seat,
        )

        seat_states: List[SeatState] = []
        for idx in range(self.max_seats):
            seat_states.append(
                SeatState(
                    index=idx,
                    position=layout.position_for(idx),
                    stack=stacks.get(idx),
                    is_active=(idx in active_seats),
                    is_hero=(idx == self.hero_seat),
                    has_button=(button_seat is not None and idx == button_seat),
                )
            )

        return TableState(
            hero_seat=self.hero_seat,
            button_seat=button_seat,
            table_type=self.table_type,
            seats=seat_states,
            pot=pot,
            to_call=to_call,
            hero_stack=hero_stack,
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python local_table_state.py <screenshot.png>")
        sys.exit(1)

    frame = cv2.imread(sys.argv[1])
    if frame is None:
        print(f"Could not load {sys.argv[1]}")
        sys.exit(1)

    extractor = LocalTableStateExtractor()
    state = extractor.extract(frame)
    print(f"Hero seat: {state.hero_seat}, Button seat: {state.button_seat}")
    print(f"Pot: {state.pot}, To call: {state.to_call}, Hero stack: {state.hero_stack}")
    print(f"Effective stack: {state.effective_stack()}")
    for s in state.seats:
        marker = ""
        if s.is_hero:
            marker += " [HERO]"
        if s.has_button:
            marker += " [BTN]"
        print(f"  seat {s.index} {s.position}: stack={s.stack} active={s.is_active}{marker}")
