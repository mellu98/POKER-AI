"""Confidence fusion between local OCR/template reads and optional YOLO detections.

This module is intentionally dependency-light: it only needs NumPy for bounding
box handling. All logic is expressed as pure functions so it is easy to unit test.

Expected YOLO label conventions (Fase 2):
- Plain card values: ``"As"``, ``"Ts"``, ``"Kh"`` ...
- Slot-tied labels: ``"hero_card_1_As"``, ``"board_card_3_Ts"`` ...
- Generic labels: ``"card"`` — parsed as present but unidentifiable.

The helper :func:`parse_yolo_label` extracts a normalized card string and an
optional slot hint (``"hole"`` or ``"board"``) from these conventions.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np


# Card value regex: rank (A/K/Q/J/T or digit) followed by suit (s/h/d/c).
_CARD_RE = re.compile(r"[AKQJT98765432][shdc]", re.IGNORECASE)
_SLOT_RE = re.compile(r"(?:hole|hero)_card_(\d+)|board_card_(\d+)", re.IGNORECASE)


def _normalize_card(card: Optional[str]) -> Optional[str]:
    """Normalize a card string to Rank+lowercase-suit."""
    if not card:
        return None
    card = card.strip()
    if len(card) != 2:
        return None
    return card[0].upper() + card[1].lower()


def parse_yolo_label(label: str) -> tuple[Optional[str], Optional[str]]:
    """Parse a YOLO class label into ``(card, slot_hint)``.

    Returns
    -------
    card:
        Normalized card string like ``"As"`` when the label contains one;
        ``None`` for generic labels such as ``"card"``.
    slot_hint:
        ``"hole"`` or ``"board"`` when the label encodes a slot index;
        ``None`` otherwise.
    """
    card_match = _CARD_RE.search(label)
    card = _normalize_card(card_match.group(0)) if card_match else None

    slot_match = _SLOT_RE.search(label)
    slot_hint: Optional[str] = None
    if slot_match:
        slot_hint = "hole" if slot_match.group(1) is not None else "board"

    return card, slot_hint


def confidence_from_local_scores(
    scores: list[float],
    cards: Optional[list[Optional[str]]] = None,
    baseline: float = 0.35,
) -> float:
    """Derive a single confidence value from per-slot local scores.

    Empty slots (``None`` cards or zero scores with no card) are ignored so that
    an empty board slot does not artificially force confidence to 0.0.

    Parameters
    ----------
    scores:
        Per-slot scores returned by the local card reader.
    cards:
        Optional parallel list of recognized cards. Used to ignore empty slots.
    baseline:
        Minimum acceptable score. If any real read is below this, confidence is
        forced to 0.0.

    Returns
    -------
    ``min(accepted scores)`` when all accepted scores are above the baseline;
    ``0.0`` if any accepted score is below; ``1.0`` when there are no accepted
    cards at all.
    """
    accepted: list[float] = []
    for i, score in enumerate(scores):
        if cards is not None:
            card = cards[i] if i < len(cards) else None
            if card is None:
                continue
        accepted.append(float(score))

    if not accepted:
        return 1.0

    if any(s < baseline for s in accepted):
        return 0.0

    return min(accepted)


def _bbox_center(bbox: np.ndarray) -> tuple[float, float]:
    """Return the center point of a ``[x1, y1, x2, y2]`` bounding box."""
    x1, y1, x2, y2 = bbox
    return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0


def merge_card_confidence(
    local_cards: list[Optional[str]],
    local_scores: list[float],
    yolo_cards: list[tuple[str, np.ndarray, float]],
    yolo_threshold: float,
    group_name: str = "cards",
) -> tuple[list[Optional[str]], float, list[str]]:
    """Fuse local card reads with optional YOLO detections.

    Parameters
    ----------
    local_cards:
        Per-slot card strings from local OCR/template matching. ``None`` means
        the slot was empty or unreadable.
    local_scores:
        Per-slot confidence scores parallel to ``local_cards``.
    yolo_cards:
        List of ``(label, bbox, confidence)`` tuples from YOLO. The bounding box
        is used to left-to-right align YOLO cards with local slots.
    yolo_threshold:
        Minimum YOLO confidence to consider a detection. Detections below this
        are ignored.
    group_name:
        Used to make uncertainty reasons human-readable (e.g. ``"hole"`` or
        ``"board"``).

    Returns
    -------
    ``(merged_cards, min_confidence, uncertainty_reasons)``.
    ``merged_cards`` has the same length as ``local_cards``; empty slots are
    represented by ``None``.
    """
    reasons: list[str] = []

    # Normalize local reads.
    norm_local = [_normalize_card(c) for c in local_cards]
    padded_scores = list(local_scores) + [0.0] * (len(norm_local) - len(local_scores))

    # Filter and sort YOLO detections by left-to-right position.
    filtered_yolo = [
        (label, bbox, float(conf))
        for label, bbox, conf in yolo_cards
        if conf >= yolo_threshold
    ]
    sorted_yolo = sorted(filtered_yolo, key=lambda item: _bbox_center(item[1])[0])

    # No usable YOLO data: rely entirely on local scores.
    if not sorted_yolo:
        local_conf = confidence_from_local_scores(local_scores, cards=norm_local)
        if local_conf < 0.7:
            reasons.append("low local confidence")
        return norm_local, local_conf, reasons

    merged: list[Optional[str]] = []
    slot_confidences: list[float] = []

    for i, local_card in enumerate(norm_local):
        local_score = padded_scores[i] if i < len(padded_scores) else 0.0
        yolo_entry = sorted_yolo[i] if i < len(sorted_yolo) else None

        if yolo_entry is None:
            # YOLO saw cards elsewhere but missed this slot.
            merged.append(local_card)
            slot_confidences.append(0.0)
            reasons.append(f"{group_name} card missing in local vs YOLO")
            continue

        yolo_label, _bbox, yolo_conf = yolo_entry
        yolo_card, _slot_hint = parse_yolo_label(yolo_label)

        if yolo_card is None:
            # Generic label: cannot verify, fall back to local read.
            merged.append(local_card)
            slot_confidences.append(local_score)
            reasons.append("YOLO card label unparseable")
            continue

        if local_card is None:
            # Local missed a card that YOLO saw.
            merged.append(yolo_card)
            slot_confidences.append(0.0)
            reasons.append(f"{group_name} card missing in local vs YOLO")
            continue

        if local_card == yolo_card:
            # Agreement: boost confidence (use the higher of the two).
            merged.append(local_card)
            slot_confidences.append(max(local_score, yolo_conf))
        else:
            # Disagreement: keep local read but mark fully uncertain.
            merged.append(local_card)
            slot_confidences.append(0.0)
            reasons.append(f"{group_name} mismatch local vs YOLO")

    # Empty slots (no local card, no YOLO card) should not drag confidence down.
    non_empty_slot_confidences = [
        conf for card, conf in zip(merged, slot_confidences) if card is not None
    ]
    if not non_empty_slot_confidences:
        min_confidence = 1.0
    else:
        min_confidence = min(non_empty_slot_confidences)

    return merged, min_confidence, reasons
