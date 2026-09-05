"""
Card recognition via template matching.

For a real poker client, you should replace the synthetic templates with
actual card images captured from your target software.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union


def generate_card_templates(size=(60, 80), save_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Generate synthetic card templates for testing.
    Returns a dict {card_str: template_image}.
    """
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = {"s": "Spades", "h": "Hearts", "d": "Diamonds", "c": "Clubs"}
    templates = {}

    for rank in ranks:
        for suit_char in suits:
            card_str = f"{rank}{suit_char}"
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            # White text on black background
            text = f"{rank}\n{suit_char}"
            y_offset = 25
            for line in text.split("\n"):
                cv2.putText(
                    img, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                )
                y_offset += 30
            templates[card_str] = img

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, f"{card_str}.png"), img)

    return templates


def extract_template_from_roi(
    frame: np.ndarray, roi_cfg: dict, label: str, output_dir: str
) -> str:
    """Crop a ROI, resize to standard template size, and save as {label}.png."""
    from capture import crop_roi
    crop = crop_roi(frame, roi_cfg["x"], roi_cfg["y"], roi_cfg["w"], roi_cfg["h"])
    if crop is None or crop.size == 0:
        raise ValueError("Empty ROI, cannot extract template")
    resized = cv2.resize(crop, (60, 80))
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{label}.png")
    cv2.imwrite(path, resized)
    return path


def load_templates_from_dir(directory: str) -> Dict[str, np.ndarray]:
    """Load .png templates from a directory. Filename = card name (e.g. As.png)."""
    templates = {}
    if not os.path.isdir(directory):
        return templates
    for fname in os.listdir(directory):
        if fname.endswith(".png"):
            card = fname.replace(".png", "")
            # Normalizza: rank maiuscolo, suit minuscolo (es. 'TD' -> 'Td', 'as' -> 'As')
            if len(card) == 2:
                card = card[0].upper() + card[1].lower()
            templates[card] = cv2.imread(os.path.join(directory, fname))
    return templates


def extract_split_templates_from_full(
    templates: Dict[str, np.ndarray],
) -> tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    """
    Estrae rank e suit dai template completi di carte già catturati.
    Mantiene TUTTE le varianti (es. rank rosso e nero) per un matching più robusto.
    """
    from collections import defaultdict

    rank_templates: Dict[str, List[np.ndarray]] = defaultdict(list)
    suit_templates: Dict[str, List[np.ndarray]] = defaultdict(list)
    for card_name, img in templates.items():
        if img is None or len(card_name) != 2:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        h, w = gray.shape
        # Rank: angolo alto-sinistra (~22% altezza, ~22% larghezza)
        rank_roi = gray[0 : int(h * 0.22), 0 : int(w * 0.22)]
        # Suit: sotto il rank in alto a sinistra (~25-45% altezza, ~10-28% larghezza)
        suit_roi = gray[
            int(h * 0.25) : int(h * 0.45), int(w * 0.10) : int(w * 0.28)
        ]
        rank = card_name[0].upper()
        suit = card_name[1].lower()
        rank_templates[rank].append(rank_roi)
        suit_templates[suit].append(suit_roi)
    return dict(rank_templates), dict(suit_templates)


def load_rank_templates(directory: str) -> Dict[str, List[np.ndarray]]:
    """Load rank_*.png templates (e.g. rank_A.png -> ['A': [img]])."""
    templates: Dict[str, List[np.ndarray]] = {}
    d = Path(directory)
    for f in d.glob("rank_*.png"):
        key = f.stem.replace("rank_", "")
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates.setdefault(key, []).append(img)
    return templates


def load_suit_templates(directory: str) -> Dict[str, List[np.ndarray]]:
    """Load suit templates from suits/ subdirectory or suit_*.png files."""
    templates: Dict[str, List[np.ndarray]] = {}
    d = Path(directory)
    suits_dir = d / "suits"
    if suits_dir.exists():
        for f in suits_dir.glob("*.png"):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.setdefault(f.stem, []).append(img)
    for f in d.glob("suit_*.png"):
        key = f.stem.replace("suit_", "")
        if key not in templates:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.setdefault(key, []).append(img)
    return templates


def _match_template_resized(
    image: np.ndarray, template: np.ndarray, std_size: tuple = (40, 40)
) -> float:
    """Resize both image and template to std_size, then matchTemplate."""
    if image is None or template is None or image.size == 0 or template.size == 0:
        return -1.0
    img_resized = cv2.resize(image, std_size)
    tmpl_resized = cv2.resize(template, std_size)
    res = cv2.matchTemplate(img_resized, tmpl_resized, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val


def recognize_card_split(
    roi: np.ndarray,
    rank_templates: Dict[str, List[np.ndarray]],
    suit_templates: Dict[str, List[np.ndarray]],
    threshold: float = 0.35,
) -> Optional[str]:
    """
    Recognize a card by matching rank (top-left) and small suit (below rank, top-left) independently.

    Args:
        roi: card ROI (BGR or grayscale).
        rank_templates: dict of rank symbol image lists (e.g. {'A': [img1, img2]}).
        suit_templates: dict of suit symbol image lists (e.g. {'s': [img1, img2]}).
        threshold: minimum matchTemplate score to accept.

    Returns:
        Card string like 'As' or '7h', or None if no good match.
    """
    card, _ = recognize_card_split_with_confidence(
        roi, rank_templates, suit_templates, threshold
    )
    return card


def recognize_card_split_with_confidence(
    roi: np.ndarray,
    rank_templates: Dict[str, List[np.ndarray]],
    suit_templates: Dict[str, List[np.ndarray]],
    threshold: float = 0.35,
) -> tuple[Optional[str], float]:
    """
    Same as :func:`recognize_card_split` but returns ``(card, confidence)``.

    Confidence is ``min(best_rank_score, best_suit_score)`` when both parts are
    accepted; otherwise ``0.0``.
    """
    if roi is None or roi.size == 0:
        return None, 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    h, w = gray.shape

    # Crop rank from top-left corner (~22% height, ~22% width)
    rank_roi = gray[0 : int(h * 0.22), 0 : int(w * 0.22)]
    # Crop small suit from just below rank in top-left (~20-45% height, ~22% width)
    suit_roi = gray[
        int(h * 0.20) : int(h * 0.45), 0 : int(w * 0.22)
    ]

    best_rank = None
    best_rank_score = -1.0
    for rank, tmpl_list in rank_templates.items():
        if not isinstance(tmpl_list, list):
            tmpl_list = [tmpl_list]
        for tmpl in tmpl_list:
            score = _match_template_resized(rank_roi, tmpl, std_size=(40, 40))
            if score > best_rank_score:
                best_rank_score = score
                best_rank = rank

    best_suit = None
    best_suit_score = -1.0
    for suit, tmpl_list in suit_templates.items():
        if not isinstance(tmpl_list, list):
            tmpl_list = [tmpl_list]
        for tmpl in tmpl_list:
            score = _match_template_resized(suit_roi, tmpl, std_size=(40, 40))
            if score > best_suit_score:
                best_suit_score = score
                best_suit = suit

    if (
        best_rank is not None
        and best_suit is not None
        and best_rank_score > threshold
        and best_suit_score > threshold
    ):
        return f"{best_rank}{best_suit}", min(best_rank_score, best_suit_score)

    return None, 0.0


def recognize_card_hybrid(
    roi: np.ndarray,
    suit_templates: Dict[str, List[np.ndarray]],
    threshold: float = 0.30,
) -> Optional[str]:
    """
    Riconoscimento ibrido: OCR per il rank, template matching per il suit.
    Molto più robusto del matching completo perché l'OCR legge il testo
    indipendentemente dal font/dimensione.
    """
    card, _ = recognize_card_hybrid_with_confidence(roi, suit_templates, threshold)
    return card


def recognize_card_hybrid_with_confidence(
    roi: np.ndarray,
    suit_templates: Dict[str, List[np.ndarray]],
    threshold: float = 0.30,
) -> tuple[Optional[str], float]:
    """
    Same as :func:`recognize_card_hybrid` but returns ``(card, confidence)``.

    Confidence is the best suit-template score when a rank is also found via OCR;
    otherwise ``0.0``.
    """
    if roi is None or roi.size == 0:
        return None, 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    h, w = gray.shape

    # ---- 1. Rank con OCR ----
    rank_roi = gray[0 : int(h * 0.25), 0 : int(w * 0.25)]
    if rank_roi.size == 0:
        return None, 0.0

    # Preprocess per OCR: binarizza e ingrandisci
    _, binary = cv2.threshold(rank_roi, 180, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    try:
        import pytesseract
        text = pytesseract.image_to_string(
            binary,
            config="--psm 10 -c tessedit_char_whitelist=AKQJT98765432",
        ).strip()
    except Exception:
        text = ""

    rank = text[0].upper() if text else None
    if rank and rank == "1":
        rank = "T"  # Tesseract a volte legge '1' come 'T' (10)

    # ---- 2. Suit con template matching ----
    suit_roi = gray[int(h * 0.18) : int(h * 0.48), 0 : int(w * 0.25)]
    if suit_roi.size == 0:
        return None, 0.0

    best_suit = None
    best_suit_score = -1.0
    for suit, tmpl_list in suit_templates.items():
        if not isinstance(tmpl_list, list):
            tmpl_list = [tmpl_list]
        for tmpl in tmpl_list:
            score = _match_template_resized(suit_roi, tmpl, std_size=(30, 30))
            if score > best_suit_score:
                best_suit_score = score
                best_suit = suit

    if rank and best_suit and best_suit_score > threshold:
        return f"{rank}{best_suit}", best_suit_score
    return None, 0.0


def match_card(roi: np.ndarray, templates: Dict[str, np.ndarray]) -> Optional[str]:
    """
    Try to match a card ROI against a set of templates.
    Returns the best matching card string or None.
    """
    card, _ = match_card_with_confidence(roi, templates)
    return card


def match_card_with_confidence(
    roi: np.ndarray, templates: Dict[str, np.ndarray], threshold: float = 0.6
) -> tuple[Optional[str], float]:
    """
    Same as :func:`match_card` but returns ``(card, confidence)``.

    Confidence is the best template-matching score when it is above the
    threshold; otherwise ``0.0``.
    """
    if roi is None or roi.size == 0:
        return None, 0.0

    best_match = None
    best_score = -1.0

    # Resize ROI to a standard size for fair comparison
    std_size = (60, 80)
    roi_resized = cv2.resize(roi, std_size)

    for card_str, template in templates.items():
        if template is None:
            continue
        tmpl_resized = cv2.resize(template, std_size)
        # Use normalized cross-correlation
        result = cv2.matchTemplate(roi_resized, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
            best_match = card_str

    # Threshold — tune this based on your real templates
    if best_score > threshold:
        return best_match, best_score
    return None, 0.0


def recognize_cards_rois(
    frame: np.ndarray,
    rois: List[dict],
    templates: Dict[str, np.ndarray],
    rank_templates: Dict[str, List[np.ndarray]] = None,
    suit_templates: Dict[str, List[np.ndarray]] = None,
) -> List[Optional[str]]:
    """
    Recognize cards in a list of ROIs.

    Tenta prima il riconoscimento ibrido (OCR rank + template suit),
    poi il fallback con matching completo sui template interi.
    """
    results = recognize_cards_rois_with_confidence(
        frame,
        rois,
        templates,
        rank_templates=rank_templates,
        suit_templates=suit_templates,
    )
    return [card for card, _ in results]


def recognize_cards_rois_with_confidence(
    frame: np.ndarray,
    rois: List[dict],
    templates: Dict[str, np.ndarray],
    rank_templates: Dict[str, List[np.ndarray]] = None,
    suit_templates: Dict[str, List[np.ndarray]] = None,
    card_classifier=None,
) -> List[tuple[Optional[str], float]]:
    """
    Recognize cards in a list of ROIs and return ``(card, confidence)`` pairs.

    Tries the same logic as :func:`recognize_cards_rois`:

    - hybrid OCR-rank + template-suit when ``suit_templates`` are available;
    - full-card template matching as a fallback;
    - optional ``CardClassifier`` fallback for any slot that template matching
      could not resolve.

    Confidence semantics:

    - ``recognize_card_hybrid``: best suit-template score when OCR finds a rank;
    - ``match_card``: best full-template score when above threshold;
    - ``CardClassifier``: the classifier's own joint rank+suit confidence.
    """
    from capture import crop_roi

    has_hybrid = suit_templates is not None and len(suit_templates) > 0
    results = []
    for roi_cfg in rois:
        roi = crop_roi(frame, roi_cfg["x"], roi_cfg["y"], roi_cfg["w"], roi_cfg["h"])
        card, conf = None, 0.0
        if has_hybrid:
            card, conf = recognize_card_hybrid_with_confidence(roi, suit_templates)
        if card is None:
            card, conf = match_card_with_confidence(roi, templates)
        if card is None and card_classifier is not None:
            pred, clf_conf = card_classifier.predict_card(roi)
            if pred is not None:
                card = pred
                conf = clf_conf
        results.append((card, conf))
    return results
