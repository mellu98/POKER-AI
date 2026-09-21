"""
Data collector + assisted labeling for fine-tuning a YOLO card detector.

Usage:
    python vision/data_collector.py --n 100 --interval 3 --out vision/dataset/yolo_goldbet
    python vision/data_collector.py --yaml-only --out vision/dataset/yolo_goldbet

Flow:
1. Captures N frames from the Goldbet table window (config ``vision.window_title``).
2. Saves each full frame to ``<out>/images/frame_<ts>.png``.
3. For each card ROI (2 hole + 5 board, calibrated percent coordinates):
   - Recognizes the card with the existing ``ocr_cards`` pipeline (hybrid
     OCR-rank + template-suit, with full-card template fallback).
   - If recognized with confidence >= threshold: writes the YOLO line with the
     correct class id and saves the crop as ``<out>/crops/<card>_<ts>_<slot>.png``
     (same naming convention as ``vision/dataset/cards/``).
   - If uncertain/failed: writes the bbox with the PLACEHOLDER class id (52)
     and saves the crop to ``<out>/review/XX_<ts>_<slot>.png`` so it can be
     resolved later with ``vision/review_labels.py``.
   - If the slot looks empty (e.g. preflop board): skipped entirely.
4. Writes ``<out>/labels/frame_<ts>.txt`` in YOLO format (class cx cy w h,
   normalized to the frame size).
5. Emits ``<out>/data.yaml`` with the 52-class TeogopK ordering.

IMPORTANT: label lines with class id 52 are placeholders. Run
``python vision/review_labels.py --dir <out>`` before training.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Allow running as a standalone script (same pattern as other vision modules).
sys.path.insert(0, str(Path(__file__).parent))

from capture import crop_roi, screenshot
from ocr_cards import (
    SUITS_BY_COLOR,
    classify_suit_by_shape,
    corner_glyph_contour,
    detect_suit_color,
    extract_split_templates_from_full,
    generate_card_templates,
    load_rank_templates,
    load_suit_templates,
    load_templates_from_dir,
    match_card_with_confidence,
    ocr_rank,
)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

# ---------------------------------------------------------------------------
# Card code mapping
#
# Internal format: rank in "23456789TJQKA" + suit in "shdc" (e.g. "7d", "Ts").
# The YOLO model (TeogopK) uses "10" instead of "T" and orders classes as
# lexicographically-sorted ranks x suits [c, d, h, s]:
#   10c,10d,10h,10s, 2c,2d,2h,2s, ..., 9c..9s, Ac..As, Jc..Js, Kc..Ks, Qc..Qs
# ---------------------------------------------------------------------------
OUR_RANKS = "23456789TJQKA"
OUR_SUITS = "shdc"
YOLO_RANKS = ["10", "2", "3", "4", "5", "6", "7", "8", "9", "A", "J", "K", "Q"]
YOLO_SUITS = ["c", "d", "h", "s"]
CLASS_NAMES = [r + s for r in YOLO_RANKS for s in YOLO_SUITS]  # 52 entries

# Placeholder class written to the label file when recognition is uncertain.
# It is OUT of the 52-class range on purpose: review_labels.py rewrites it.
PLACEHOLDER_CLASS_ID = len(CLASS_NAMES)  # 52

# Minimum confidence to accept an auto-proposed label.
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# A real card crop is mostly white; an empty slot is green felt. Fraction of
# near-white pixels above which we consider a card present in the ROI.
CARD_WHITE_FRACTION = 0.15


def to_yolo_card(card: str) -> str:
    """Convert internal card code to the YOLO model naming (rank T -> 10)."""
    rank, suit = card[0].upper(), card[1].lower()
    return ("10" if rank == "T" else rank) + suit


def card_to_class_id(card: str) -> int:
    """Map an internal card code (e.g. 'Ts') to its YOLO class id."""
    return CLASS_NAMES.index(to_yolo_card(card))


def parse_card_code(text: str) -> str | None:
    """Validate user-entered card code ('7d', 'Ts', '10H' ...) -> '7d'/'Ts'."""
    text = text.strip()
    if len(text) not in (2, 3):
        return None
    rank = text[:-1].upper()
    suit = text[-1].lower()
    if rank == "10":
        rank = "T"  # canonical internal representation
    if rank in OUR_RANKS and suit in OUR_SUITS:
        return f"{rank}{suit}"
    return None


# ---------------------------------------------------------------------------
# Config / ROI helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        print(f"[data] Config not found: {config_path}, using defaults.")
        return {}
    try:
        import yaml

        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[data] Could not load {config_path}: {exc}")
        return {}


def _card_rois(cfg: dict, key: str) -> list:
    """Site-calibrated card ROIs, falling back to legacy vision.rois."""
    site = cfg.get("vision", {}).get("site")
    site_cfg = cfg.get("sites", {}).get(site, {}) if site else {}
    rois = site_cfg.get(key) or cfg.get("vision", {}).get("rois", {}).get(key, [])
    return rois or []


def resolve_roi(roi: dict, frame: np.ndarray) -> dict:
    """Convert a percent ('rel: true') ROI to integer pixel coordinates."""
    try:
        if roi.get("rel"):
            h, w = frame.shape[:2]
            return {
                "x": int(roi["x"] * w),
                "y": int(roi["y"] * h),
                "w": int(roi["w"] * w),
                "h": int(roi["h"] * h),
            }
        return {"x": int(roi["x"]), "y": int(roi["y"]),
                "w": int(roi["w"]), "h": int(roi["h"])}
    except (KeyError, TypeError, ValueError):
        return {"x": 0, "y": 0, "w": 0, "h": 0}


def _normalize_bbox(x: int, y: int, w: int, h: int,
                    frame_w: int, frame_h: int) -> tuple[float, float, float, float]:
    """Pixel bbox -> YOLO normalized (cx, cy, w, h), clamped to [0, 1]."""
    cx = (x + w / 2) / frame_w
    cy = (y + h / 2) / frame_h
    nw = w / frame_w
    nh = h / frame_h
    clamp = lambda v: min(max(v, 0.0), 1.0)
    return clamp(cx), clamp(cy), clamp(nw), clamp(nh)


def _looks_like_card(crop: np.ndarray) -> bool:
    """Heuristic: a card crop is mostly near-white, felt is not."""
    if crop is None or crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    try:
        return float(np.mean(gray > 180)) > CARD_WHITE_FRACTION
    except (ValueError, TypeError):
        return False


def snap_to_card(crop: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the card's white body inside a loose ROI crop.

    Slot ROIs are calibrated once but card positions drift between frames
    (deal animation, action bar appearing, avatar decorations poking in).
    The card body is the big near-white blob; snapping the bbox to it makes
    recognition see the card top-left (where rank/suit live) and keeps YOLO
    boxes tight. Returns (x, y, w, h) relative to the crop, or None.
    """
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    mask = (gray > 180).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    # Largest white component (skip background label 0).
    areas = stats[1:, cv2.CC_STAT_AREA]
    try:
        i = 1 + int(np.argmax(areas))
        x, y, w, h, area = (int(v) for v in stats[i])
    except (ValueError, TypeError, IndexError):
        return None
    if area < 0.3 * crop.shape[0] * crop.shape[1]:
        return None
    # Re-expand a few px: the white-body snap can clip the colored card edge
    # (glow border) and leave the rank glyph touching the crop boundary,
    # which breaks OCR.
    pad = 3
    ch, cw = crop.shape[:2]
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(cw - x, w + 2 * pad)
    h = min(ch - y, h + 2 * pad)
    return x, y, w, h


# ---------------------------------------------------------------------------
# Tesseract setup (graceful on macOS — never crashes the module import)
# ---------------------------------------------------------------------------

def _setup_tesseract() -> bool:
    """Point pytesseract at a valid binary. Returns False if unavailable."""
    try:
        import pytesseract
        from tesseract_utils import find_tesseract_binary

        pytesseract.pytesseract.tesseract_cmd = find_tesseract_binary()
        return True
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        print(f"[data] Tesseract unavailable, OCR-rank disabled ({exc}). "
              f"Falling back to template matching only.")
        return False


# ---------------------------------------------------------------------------
# Card guesser — reuses ocr_cards recognition to PROPOSE labels
# ---------------------------------------------------------------------------

class CardGuesser:
    """Proposes card labels for crops using the existing ocr_cards pipeline.

    Same recognition order as ``recognize_cards_rois_with_confidence``:
    hybrid (OCR rank + suit template) first, full-card template fallback.
    Proposals are NOT ground truth — low-confidence crops go to review.
    """

    def __init__(self, cfg: dict):
        self.ocr_enabled = _setup_tesseract()
        self.templates, self.rank_templates, self.suit_templates = self._load_templates(cfg)
        # Reference suit-glyph contours from real full-card templates, for
        # classify_suit_by_shape (color-constrained matchShapes).
        self.suit_refs: dict[str, list] = {}
        for card, img in self.templates.items():
            if img is None or len(card) != 2:
                continue
            c = corner_glyph_contour(img)
            if c is not None:
                self.suit_refs.setdefault(card[1], []).append(c)

    @staticmethod
    def _load_templates(cfg: dict):
        """Load real card templates; fall back to synthetic ones (Windows-style
        'vision\\templates' paths from config are normalized for macOS)."""
        tmpl_dir = cfg.get("vision", {}).get("template_dir", "")
        candidates = [Path(str(tmpl_dir).replace("\\", "/")),
                      PROJECT_ROOT / "vision" / "templates",
                      PROJECT_ROOT / "vision" / "templates_goldbet"]
        for candidate in candidates:
            if candidate.exists():
                templates = load_templates_from_dir(str(candidate))
                if templates:
                    rank_templates = load_rank_templates(str(candidate))
                    suit_templates = load_suit_templates(str(candidate))
                    # Augment with rank/suit crops auto-split from full templates
                    # (same trick as state_extractor for robust hybrid matching).
                    auto_rank, auto_suit = extract_split_templates_from_full(templates)
                    for rank, imgs in auto_rank.items():
                        rank_templates.setdefault(rank, []).extend(imgs)
                    for suit, imgs in auto_suit.items():
                        suit_templates.setdefault(suit, []).extend(imgs)
                    # Drop blank templates (uniform image matches EVERYTHING at
                    # 1.0 with TM_CCOEFF and poisons the argmax — this was the
                    # root cause of diamonds labeled as hearts).
                    suit_templates = {
                        s: [t for t in tl if t is not None and t.std() >= 10]
                        for s, tl in suit_templates.items()
                    }
                    suit_templates = {s: tl for s, tl in suit_templates.items() if tl}
                    print(f"[data] Loaded {len(templates)} full templates from {candidate} "
                          f"(suit templates: { {s: len(tl) for s, tl in sorted(suit_templates.items())} })")
                    return templates, rank_templates, suit_templates
        print("[data] No real templates found; using synthetic templates "
              "(proposals will be unreliable until real ones are captured).")
        return generate_card_templates(), {}, {}

    def guess(self, crop: np.ndarray) -> tuple[str | None, float, str]:
        """Return (card, confidence, source). source: 'hybrid', 'template' or ''.

        Hard guard: a real card ALWAYS has ink (red or black rank/suit glyphs).
        No ink → not a card — reject immediately. Without this, tesseract
        hallucinates ranks on pure-white animation flashes and blank suit
        templates "match" at 1.0.
        """
        if detect_suit_color(crop) is None:
            return None, 0.0, ""

        # Two INDEPENDENT signals must agree before auto-labeling.
        # Tesseract alone misreads glyphs touching the crop edge (5→2, K→7);
        # template matching alone can't tell suits apart (5c ≈ 5s at ~1.0).
        # Agreement on rank AND color group is required; the suit comes from
        # the shape classifier, which never crosses color groups.
        color = detect_suit_color(crop)

        # Primary: rank via OCR + suit via color-constrained matchShapes on
        # the corner glyph.
        primary = None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        rank = ocr_rank(gray) if self.ocr_enabled else None
        if rank:
            suit, suit_conf = classify_suit_by_shape(crop, self.suit_refs)
            if suit and suit_conf >= 0.5:
                primary = (f"{rank}{suit}", suit_conf)

        # Template: full-card match gated by ink color (rank + pip layout).
        template = None
        card, conf = match_card_with_confidence(crop, self.templates)
        if color and card and card[1] in SUITS_BY_COLOR[color]:
            template = (card, conf)

        if primary and template:
            p_card, p_conf = primary
            t_card, t_conf = template
            if p_card[0] == t_card[0]:
                return p_card, min(p_conf, t_conf), "hybrid"
        return None, 0.0, ""


# ---------------------------------------------------------------------------
# Capture / labeling
# ---------------------------------------------------------------------------

def capture_frame(window_title: str | None = None) -> tuple[np.ndarray | None, str]:
    """Grab the poker table window and return (BGR frame, timestamp string).

    Returns (None, ts) when the window is not found: screenshot() silently
    falls back to a full-screen grab, which would label whatever happens to
    be on screen (web pages included) as cards. Never collect that.
    """
    ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    if window_title and platform.system() == "Darwin":
        from capture import _find_window_rect_mac

        if _find_window_rect_mac(window_title) is None:
            print(f"[data] Window {window_title!r} not on screen — frame skipped "
                  f"(no full-screen fallback).")
            return None, ts
    frame = screenshot(window_title=window_title)
    return frame, ts


def label_frame(
    frame: np.ndarray,
    ts: str,
    guesser: CardGuesser,
    cfg: dict,
    out_dir: Path,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Label one frame: write the YOLO txt, save crops and review entries.

    Returns a stats dict and appends entries to ``review/index.json`` entries
    for every crop that needs manual review.
    """
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    crops_dir = out_dir / "crops"
    review_dir = out_dir / "review"
    for d in (images_dir, labels_dir, crops_dir, review_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Save the full frame.
    frame_path = images_dir / f"frame_{ts}.png"
    cv2.imwrite(str(frame_path), frame)

    # Build the 7 card slots: hole1..2, board1..5.
    slots = [(f"hole{i + 1}", roi) for i, roi in enumerate(_card_rois(cfg, "hole"))]
    slots += [(f"board{i + 1}", roi) for i, roi in enumerate(_card_rois(cfg, "board"))]

    frame_h, frame_w = frame.shape[:2]
    lines: list[str] = []
    review_entries: dict = {}
    stats = {"auto": 0, "review": 0, "empty": 0}

    for slot, roi_cfg in slots:
        px = resolve_roi(roi_cfg, frame)
        crop = crop_roi(frame, px["x"], px["y"], px["w"], px["h"])
        if not _looks_like_card(crop):
            stats["empty"] += 1  # e.g. preflop board slots
            continue

        # Snap to the card's white body so recognition and the YOLO box track
        # the actual card, not the nominal (possibly drifted) ROI.
        snap = snap_to_card(crop)
        if snap is not None:
            sx, sy, sw, sh = snap
            crop = crop[sy:sy + sh, sx:sx + sw]
            px = {"x": px["x"] + sx, "y": px["y"] + sy, "w": sw, "h": sh}

        cx, cy, nw, nh = _normalize_bbox(px["x"], px["y"], px["w"], px["h"],
                                         frame_w, frame_h)
        card, conf, _ = guesser.guess(crop)

        if card and conf >= threshold:
            class_id = card_to_class_id(card)
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats["auto"] += 1
            # Confident crop -> same naming convention as vision/dataset/cards.
            crop_path = crops_dir / f"{card}_{ts}_{slot}.png"
            cv2.imwrite(str(crop_path), crop)
        else:
            # Uncertain -> placeholder class + crop into review/.
            class_id = PLACEHOLDER_CLASS_ID
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats["review"] += 1
            crop_name = f"XX_{ts}_{slot}.png"
            cv2.imwrite(str(review_dir / crop_name), crop)
            review_entries[crop_name] = {
                "label_file": f"labels/frame_{ts}.txt",
                "line": len(lines) - 1,  # 0-based index of the placeholder line
                "slot": slot,
                "frame": f"images/frame_{ts}.png",
                "proposed": card,
                "confidence": round(conf, 4),
                "resolved": False,
            }

    label_path = labels_dir / f"frame_{ts}.txt"
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return {"stats": stats, "review_entries": review_entries,
            "frame": str(frame_path), "label": str(label_path)}


def collect(
    output_dir: str,
    n: int,
    interval: float,
    config_path: str = str(DEFAULT_CONFIG),
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    window_title: str | None = None,
) -> dict:
    """Capture N frames at `interval` seconds and write assisted YOLO labels."""
    out_dir = Path(output_dir)
    cfg = _load_config(Path(config_path))
    if window_title is None:
        window_title = cfg.get("vision", {}).get("window_title")
    guesser = CardGuesser(cfg)

    print(f"[data] Output dir : {out_dir.absolute()}")
    print(f"[data] Window title: {window_title or 'full screen'}")
    print(f"[data] Frames     : {n} @ {interval}s (confidence threshold {threshold})")
    print("[data] Press Ctrl+C to stop early.\n")

    totals = {"frames": 0, "auto": 0, "review": 0, "empty": 0}
    review_index: dict = {}
    index_path = out_dir / "review" / "index.json"

    try:
        for i in range(1, n + 1):
            frame, ts = capture_frame(window_title)
            if frame is None:
                print(f"[data] [{i}/{n}] capture failed, skipping.")
                time.sleep(interval)
                continue

            result = label_frame(frame, ts, guesser, cfg, out_dir, threshold)
            s = result["stats"]
            totals["frames"] += 1
            totals["auto"] += s["auto"]
            totals["review"] += s["review"]
            totals["empty"] += s["empty"]
            review_index.update(result["review_entries"])

            print(f"[data] [{i}/{n}] frame_{ts}: "
                  f"auto={s['auto']} review={s['review']} empty={s['empty']}")

            if i < n:
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[data] Stopped by user.")

    # Persist the review index (mapping review crops -> label lines).
    if review_index:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(review_index, indent=2))

    write_dataset_yaml(str(out_dir))

    print("\n[data] === Summary ===")
    print(f"[data] Frames captured     : {totals['frames']}")
    print(f"[data] Cards auto-labeled  : {totals['auto']}")
    print(f"[data] Cards needing review: {totals['review']}")
    print(f"[data] Empty slots skipped : {totals['empty']}")
    if totals["review"]:
        print(f"[data] -> Run: python vision/review_labels.py --dir {out_dir}")
    return totals


def write_dataset_yaml(output_dir: str) -> Path:
    """Emit data.yaml for YOLOv8 training with the 52 TeogopK classes."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / "data.yaml"
    names = ", ".join(CLASS_NAMES)
    content = (
        f"# YOLOv8 dataset config — generated by vision/data_collector.py\n"
        f"# Class order matches the TeogopK model (rank 'T' is '10').\n"
        f"# NOTE: split images/ into images/train and images/val and update\n"
        f"#       train/val below before training. Review all placeholder\n"
        f"#       class-{PLACEHOLDER_CLASS_ID} lines first (review_labels.py).\n"
        f"path: {out_dir.absolute()}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: [{names}]\n"
    )
    yaml_path.write_text(content)
    print(f"[data] Wrote {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Collect Goldbet table frames + assisted YOLO card labels."
    )
    parser.add_argument("--n", type=int, default=100,
                        help="Number of frames to capture (default: 100)")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Seconds between captures (default: 3)")
    parser.add_argument("--out", type=str, default="vision/dataset/yolo_goldbet",
                        help="Output dataset directory")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Path to config.yaml")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Min confidence to auto-accept a proposed label")
    parser.add_argument("--window-title", type=str, default=None,
                        help="Override the poker window title from config")
    parser.add_argument("--yaml-only", action="store_true",
                        help="Only (re)write data.yaml, do not capture")
    args = parser.parse_args()

    if args.yaml_only:
        write_dataset_yaml(args.out)
        return

    collect(
        output_dir=args.out,
        n=args.n,
        interval=args.interval,
        config_path=args.config,
        threshold=args.threshold,
        window_title=args.window_title,
    )


if __name__ == "__main__":
    main()
