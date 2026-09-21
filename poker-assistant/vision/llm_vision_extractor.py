"""
LLM Vision Extractor — uses a multimodal LLM via OpenRouter to read poker table state from screenshots.

Usage:
    extractor = LLMVisionExtractor(api_key="sk-or-v1-...")
    state = extractor.extract(frame)  # frame is a BGR numpy array

Environment:
    OPENROUTER_API_KEY — used if api_key is not passed explicitly.
"""

import base64
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from local_table_state import LocalTableStateExtractor

# Carica .env automaticamente se python-dotenv è installato (opzionale)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError as exc:
    print(
        f"[llm_vision] python-dotenv non disponibile ({exc}); uso le variabili d'ambiente di sistema"
    )

# requests may not be installed; give a helpful error.
try:
    import requests
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "The 'requests' package is required for LLM vision. "
        "Install it: pip install requests"
    ) from exc

DEFAULT_MODEL = "google/gemini-3.5-flash-lite"
DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are an expert poker table OCR assistant. Analyze the screenshot carefully and extract the exact current table state.

Return ONLY a JSON object with this exact structure:
{
    "hole": ["As", "Kh"],
    "board": ["Qd", "Jh", "2c"],
    "pot": 120,
    "to_call": 20,
    "stack": 980,
    "stage": "flop",
    "button_seat": 3,
    "opponent_stacks": {
        "SB": 150,
        "BB": 200,
        "UTG": 80,
        "MP": 120,
        "CO": 95
    }
}

VISUAL GUIDE for 9-max Goldbet tables:
- HERO (you) sits at the BOTTOM CENTER, seat index 0.
- Your hole cards are at the BOTTOM CENTER, slightly overlapping. There are ALWAYS exactly two.
- The board (community cards) is at the TOP CENTER in a horizontal row. These are shared cards, NOT your hole cards.
- Each card shows a LARGE rank (A,K,Q,J,T,9-2) with a SMALL suit symbol directly BELOW it.
- Seat indices are counted CLOCKWISE starting from 0 = HERO (bottom-center), then
  1 = bottom-right, 2 = right-side, 3 = top-right, 4 = top-center, 5 = top-left,
  6 = left-side, 7 = left-mid, 8 = bottom-left.
- A seat showing "Vuoto" or greyed out is empty and has no stack.

BUTTON RECOGNITION — critical, report this as button_seat:
- Look for a small circular yellow/gold chip with the letter "D" (dealer button) near a player's seat.
- Report the SEAT INDEX (0-8) where the "D" chip is located as "button_seat".
- The D chip belongs to the player whose seat it is CLOSEST to. Do not confuse it with chips belonging to adjacent players.
- If you cannot see the "D" chip clearly, report "button_seat": null.

DEALER BUTTON POSITION GUIDE for 9-max Goldbet (seat indices):
- seat 0 = HERO (bottom-center, your cards). If the D chip is right next to YOUR cards, you have the button: button_seat = 0.
- seat 1 = bottom-right player. D chip near bottom-right = seat 1.
- seat 2 = right-side player. D chip on the right = seat 2.
- seat 3 = top-right player. D chip near top-right = seat 3.
- seat 4 = top-center player. D chip at the top = seat 4.
- seat 5 = top-left player. D chip near top-left = seat 5.
- seat 6 = left-side player. D chip on the left = seat 6.
- seat 7 = left-mid player. D chip on the left-middle = seat 7.
- seat 8 = bottom-left player. D chip near bottom-left = seat 8.
- When in doubt, identify which player the D chip is nearest to and map that to the seat index above.

SUIT RECOGNITION — pay extreme attention:
- ♠ Spades (s) = BLACK, pointed symbol with a stem.
- ♣ Clubs (c)  = BLACK, three-leaf clover symbol.
- ♥ Hearts (h) = RED, heart shape.
- ♦ Diamonds (d) = RED, diamond/rhombus shape.
- Color is the EASIEST and most reliable cue: BLACK cards are spades or clubs. RED cards are hearts or diamonds. Never call a red card black or a black card red.
- If you are unsure between spades and clubs, look at the symbol shape. Spades have a pointed top; clubs are rounder with three lobes.
- If you are unsure between hearts and diamonds, look at the symbol shape. Hearts are rounded with a dip at the top; diamonds are sharp angular diamonds.
- Verify EVERY card individually: rank + color + shape. Do not rush.

GOLDBET.IT SPECIFIC:
- The suit symbol is drawn SMALL, directly below the large rank number in the TOP-LEFT corner of each card.
- HEART vs DIAMOND on Goldbet: hearts are round/curved with a clear notch at the top; diamonds are straight-edged, pointed at all four corners, like a square rotated 45°. If the red symbol looks ANGULAR, it is a DIAMOND (d). If it looks like two curved lobes meeting at the top, it is a HEART (h).
- SPADE vs CLUB on Goldbet: spades are black pointed symbols with a stem; clubs are black three-leaf clovers. Zoom in on the corner symbol.
- The green felt and table shadows can make black suits look darker; do not rely only on brightness.
- Mentally zoom in on each card before deciding the suit. If the symbol is unclear, mark it with "?" rather than guessing.
- Board cards are laid horizontally at the top center; hero hole cards are at the bottom center, overlapping slightly.
- Count board cards exactly: 0 = preflop, 3 = flop, 4 = turn, 5 = river. Do not omit a board card and do not invent one.
- The main pot is labeled "Piatto:" at the top center. Use ONLY that amount for "pot". Ignore the smaller chip/bet amounts near seats or below the board.
- The amount to call is shown on the central action button (Call/Raise) or as the current bet chip; use that for "to_call", not the pot.

CRITICAL RULES:
- Use standard 2-char notation: rank (2-9,T,J,Q,K,A) + suit (s,h,d,c). Example: As, Td, 3h, Kc, Qs.
- NEVER guess a suit. If the symbol is unclear, write the rank + "?" (e.g. "Q?").
- pot / to_call / stack: integers only. Strip $, €, commas.
- IMPORTANT: If the value shows decimals (e.g. €0.05, $0.10), convert to the smallest currency unit (cents) by multiplying by 100. Example: €0.05 -> 5, €0.20 -> 20, €5 -> 500. Never return decimal values.
- stage: preflop (0 board cards), flop (3), turn (4), river (5). The stage is determined by the number of board cards.
- button_seat: integer seat index (0-8) where the dealer "D" chip is located. This is REQUIRED.
- opponent_stacks: object mapping each active opponent POSITION (not name) to their stack in cents. Include only opponents who are still in the hand (not folded, not empty seats). Omit a position if its stack is unreadable.
- num_active: number of active players including the hero. You may omit this; it will be computed from opponent_stacks.
- Always return the best data you can extract from the visible poker table, even if some details are unclear.
- Do not include any markdown, explanation, or text outside the JSON.

TOCALL RECOGNITION for Goldbet:
- Look at the action buttons at the BOTTOM RIGHT of the screen.
- If the middle button says "CHECK" or "GRATIS", to_call = 0.
- If the middle button says "CALL" or "CHIAMA", read the number on that button. That is to_call.
- If there is no number on the call button, estimate from the current bet chips in front of other players.
- Preflop when you are SB and no one raised: to_call = BB - SB (example BB=2, SB=1 -> to_call=1).
- Preflop when you are BB and no one raised: to_call = 0 (you can check).
"""


def _resize_frame(frame: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """Resize so the longest side is at most max_dim, preserving aspect ratio."""
    try:
        h, w = frame.shape[:2]
        if max(h, w) <= max_dim:
            return frame
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except (ValueError, OverflowError):
        # degradazione graduale: frame non ridimensionato (payload piu' grande ma valido)
        return frame


def _encode_frame_to_base64(
    frame: np.ndarray, max_dim: int = 1280, jpeg_quality: int = 92
) -> str:
    """
    Resize and encode an OpenCV BGR image to base64 JPEG.

    PNG is too large for fast vision calls. JPEG @ q=92 keeps suit symbols
    sharp while keeping payload reasonable. Resize first to keep total payload
    under ~250KB and API latency low.
    """
    frame = _resize_frame(frame, max_dim)
    try:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
    except (ValueError, cv2.error):
        raise RuntimeError("Failed to encode image to JPEG") from None
    if not success:
        raise RuntimeError("Failed to encode image to JPEG")
    return base64.b64encode(buffer).decode("utf-8")


def _resolve_relative_roi(roi: dict, frame: np.ndarray) -> dict | None:
    """Convert a relative ROI config to absolute pixel coordinates."""
    if roi is None:
        return None
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
            # ROI malformato nel config: il caller salta i ROI None
            return None
    return roi


def _extract_suit_templates(
    templates_dir: Path, extra_suit_dir: Path | None = None
) -> dict[str, list[np.ndarray]]:
    """Load full card templates and extract only the suit ROI crops for c/s/h/d.

    If extra_suit_dir is provided, also load any additional suit-symbol crops
    stored there (one sub-folder per suit). More samples = more robust matching
    across screenshot variations.
    """
    from ocr_cards import extract_split_templates_from_full, load_templates_from_dir

    templates: dict[str, np.ndarray] = load_templates_from_dir(str(templates_dir))
    _, suit_templates = extract_split_templates_from_full(templates)

    if extra_suit_dir and extra_suit_dir.exists():
        for suit_sub in extra_suit_dir.iterdir():
            if not suit_sub.is_dir():
                continue
            suit = suit_sub.name.lower()
            if suit not in ("s", "h", "d", "c"):
                continue
            for f in suit_sub.glob("*.png"):
                img = cv2.imread(str(f))
                if img is None or img.size == 0:
                    continue
                suit_templates.setdefault(suit, []).append(img)

    return suit_templates


def _match_template_resized(
    image: np.ndarray, template: np.ndarray, std_size: tuple = (60, 60)
) -> float:
    """Resize both images to std_size and return normalized cross-correlation score."""
    if image is None or template is None or image.size == 0 or template.size == 0:
        return -1.0
    try:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if template.ndim == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(image, std_size, interpolation=cv2.INTER_AREA)
        tmpl_resized = cv2.resize(template, std_size, interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(img_resized, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return float(max_val)
    except (cv2.error, ValueError):
        # sentinel di errore gia' usato dai caller (-1.0 = match non valido)
        return -1.0


def _best_suit(
    suit_crop: np.ndarray,
    suit_templates: dict[str, list[np.ndarray]],
    threshold: float = 0.35,
    allowed_suits: list[str] | None = None,
) -> tuple[str | None, float, float]:
    """Return (best_suit, best_score, margin) for a suit ROI.

    If allowed_suits is provided, only those suit templates are considered.
    This lets us use color (red vs black) to increase separation and avoid
    cross-color false matches.
    """
    if suit_crop is None or suit_crop.size == 0:
        return None, 0.0, 0.0
    scores: list[tuple[float, str]] = []
    for suit, tmpl_list in suit_templates.items():
        if allowed_suits is not None and suit not in allowed_suits:
            continue
        for tmpl in tmpl_list:
            score = _match_template_resized(suit_crop, tmpl, std_size=(80, 80))
            scores.append((score, suit))
    if not scores:
        return None, 0.0, 0.0
    scores.sort(reverse=True)
    best_score, best_suit = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else 0.0
    margin = best_score - second_score
    if best_suit is not None and best_score >= threshold:
        return best_suit, best_score, margin
    return None, best_score, margin


def _crop_suit_from_full_card(card_crop: np.ndarray) -> np.ndarray:
    """Extract only the suit symbol crop from a full card (below the rank), keeping color.

    Goldbet places the small suit symbol directly below the large rank in the
    top-left corner. We use a slightly generous crop to account for alignment
    differences across card sizes.
    """
    h, w = card_crop.shape[:2]
    try:
        y0, y1 = int(h * 0.28), int(h * 0.55)
        x0, x1 = int(w * 0.08), int(w * 0.30)
        return card_crop[y0:y1, x0:x1]
    except (ValueError, IndexError):
        # crop vuoto: i caller gia' gestiscono size == 0
        return np.zeros((0, 0, 3), dtype=np.uint8)


def _detect_suit_color(suit_crop: np.ndarray) -> str | None:
    """
    Detect whether a suit symbol is red (hearts/diamonds) or black (spades/clubs).

    Uses HSV first with permissive thresholds, then a simple RGB fallback.
    """
    if suit_crop is None or suit_crop.size == 0:
        return None
    if suit_crop.ndim == 2:
        return None

    hsv = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    bg = (v > 210) & (s < 60)
    fg = ~bg
    if fg.sum() > 0:
        red = ((h < 18) | (h > 165)) & (s > 30) & (v > 40)
        black = (v < 130) & (s < 130)
        try:
            red_count = int(red[fg].sum())
            black_count = int(black[fg].sum())
        except (ValueError, OverflowError):
            red_count = 0
            black_count = 0
        if red_count > black_count * 1.3 and red_count > 10:
            return "red"
        if black_count > red_count * 1.3 and black_count > 10:
            return "black"

    # RGB fallback
    b, g, r = cv2.split(suit_crop)
    fg = (b > 40) | (g > 40) | (r > 40)
    if fg.sum() < 10:
        return None
    try:
        mr = float(r[fg].mean())
        mg = float(g[fg].mean())
        mb = float(b[fg].mean())
    except (ValueError, ZeroDivisionError):
        return None
    if mr > max(mg, mb) + 15:
        return "red"
    if max(mr, mg, mb) < 140:
        return "black"
    return None


def _detect_card_color(card_crop: np.ndarray) -> str | None:
    """
    Detect card color using only the small suit-symbol region.

    The top-left corner contains a large black/red rank letter and a small suit
    symbol below it. Using the whole corner dilutes the signal with the rank
    letter, so we look only at the suit crop.
    """
    if card_crop is None or card_crop.size == 0:
        return None
    if card_crop.ndim == 2:
        return None
    suit_crop = _crop_suit_from_full_card(card_crop)
    return _detect_suit_color(suit_crop)


def _build_composite_image(
    frame: np.ndarray,
    items: list[tuple[str, dict]],
    cell_size: tuple[int, int] = (80, 120),
    cols: int = 4,
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """
    Build a single image tiling card crops with numbered labels.

    Args:
        frame: full screenshot.
        items: list of (label, roi_dict) for each card to verify.

    Returns:
        composite image, list of (number, label) for reference.
    """
    n = len(items)
    if n == 0:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8), []
    rows = (n + cols - 1) // cols
    canvas = (
        np.ones((rows * cell_size[1], cols * cell_size[0], 3), dtype=np.uint8) * 255
    )
    labels: list[tuple[int, str]] = []
    from capture import crop_roi

    for i, (label, roi_cfg) in enumerate(items):
        roi = _resolve_relative_roi(roi_cfg, frame)
        if roi is None:
            continue
        crop = crop_roi(frame, roi["x"], roi["y"], roi["w"], roi["h"])
        if crop is None or crop.size == 0:
            continue
        crop = cv2.resize(crop, cell_size, interpolation=cv2.INTER_AREA)
        r = i // cols
        c = i % cols
        y0, x0 = r * cell_size[1], c * cell_size[0]
        canvas[y0 : y0 + cell_size[1], x0 : x0 + cell_size[0]] = crop
        cv2.putText(
            canvas,
            f"{i + 1}",
            (x0 + 2, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        labels.append((i + 1, label))
    return canvas, labels


def _parse_suits_array(text: str | None) -> list[str] | None:
    """Parse a JSON array of suit letters like ["s","h","d","c"]."""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return [str(x).lower() for x in arr]
    except json.JSONDecodeError:
        print(
            f"[llm_vision] suits array non e' JSON valido, provo con i fallback: {text[:80]}"
        )
    # Fallback: extract quoted letters
    matches = re.findall(r"['\"]([shdc])['\"]", text)
    if matches:
        return matches
    # Fallback: first char of each comma-separated token
    if "," in text:
        return [tok.strip().lower()[0] for tok in text.split(",") if tok.strip()]
    return None


def _extract_json_from_text(text: str | None) -> dict:
    """
    Extract a JSON object from text. Handles markdown code blocks, None,
    empty strings, raw lists, and other model quirks via fallback strategies.
    """
    if text is None:
        raise RuntimeError(
            "Model returned empty content (None). "
            "Likely causes: content filter, model refusal, or token limit hit."
        )

    text = text.strip()
    if not text:
        raise RuntimeError("Model returned an empty string.")

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract from ```json ... ``` (or plain ``` ... ```) block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: find the first {...} block (greedy)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 4: raw list like ["As", "Kh"] — wrap as hole (focused mode)
    if text.startswith("["):
        match = re.match(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                hole_list = json.loads(match.group(0))
                if isinstance(hole_list, list):
                    return {"hole": hole_list}
            except json.JSONDecodeError:
                pass

    raise RuntimeError(
        f"Failed to parse JSON from model response. Raw text:\n{text[:800]}"
    )


def _normalize_card(card: str) -> str:
    """Normalize a card string to Rank+lowercase-suit, e.g. 'TD' -> 'Td', '10d' -> 'Td'."""
    if not card:
        return card
    card = str(card).strip()
    # Some models (notably gpt-4o-mini) return "10" instead of "T"
    if card.upper().startswith("10") and len(card) == 3:
        card = "T" + card[2:]
    if len(card) != 2:
        return card
    return card[0].upper() + card[1].lower()


class LLMVisionExtractor:
    """
    Extracts poker table state by sending screenshots to an OpenRouter vision model.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        api_url: str = DEFAULT_API_URL,
        cooldown_seconds: float = 1.5,
        timeout: float = 20.0,
        position_override: str | None = None,
        position_model: str | None = None,
        position_refresh_seconds: float = 60.0,
        window_title: str | None = None,
        config_path: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.api_url = api_url
        self.cooldown = cooldown_seconds
        self.window_title = window_title or "Free Poker"
        self.timeout = timeout
        self.config_path = config_path
        self.position_override = position_override
        self.position_model = position_model
        self.position_refresh_seconds = position_refresh_seconds
        self._last_call_time = 0.0
        self._last_frame_hash: str | None = None
        self._cached_state: dict | None = None
        self._cached_position: str | None = None
        self._cached_button_seat: int | None = None
        self._last_button_time: float = 0.0
        self._button_ttl: float = 2.0  # seconds: re-detect button at most this often
        self._last_hole_key: str = ""
        self._last_position_time: float = 0.0
        self._position_thread_running: bool = False

        # Suit memory per mano (voto temporale): i rank non cambiano durante
        # una mano, i semi si'. Si congela la coppia di semi piu' votata.
        self._frozen_ranks: list[str] = []
        self._hand_votes: dict[tuple[str, str], int] = {}
        self._board_suit_votes: dict[tuple[int, str], dict[str, int]] = {}

        # Local helpers
        self._local_table: Any | None = None
        self._hero_seat = 0
        self._table_type = "9max"
        self._load_config(config_path)

        # Local suit-correction support (clubs vs spades)
        self._hole_rois: list[dict] = []
        self._board_rois: list[dict] = []
        self._suit_templates: dict[str, list[np.ndarray]] = {}
        self._load_suit_correction_resources(config_path)

        # Jev auditor (opzionale, text-only): audit di coerenza in background
        self._jev_auditor: Any | None = None
        self._load_jev_auditor(config_path)

    def _load_jev_auditor(self, config_path: str | None) -> None:
        """Crea il JevAuditor se vision.jev.enabled è true nel config."""
        if config_path is None:
            return
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            return
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            jev_cfg = cfg.get("vision", {}).get("jev", {}) or {}
            if not jev_cfg.get("enabled", False):
                return
            from jev_decisions import JevAuditor, JevClient

            client = JevClient(
                api_key=self.api_key,
                model=str(jev_cfg.get("model", "typesafe/jev-1.13")),
            )
            interval = jev_cfg.get("check_interval_seconds", 10.0)
            self._jev_auditor = JevAuditor(
                client, check_interval_seconds=float(interval)
            )
            print(f"[llm_vision] Jev auditor attivo ({client.model}, ogni {interval}s)")
        except (
            ImportError,
            ValueError,
            TypeError,
            KeyError,
            OSError,
            yaml.YAMLError,
        ) as e:
            print(f"[llm_vision] Jev auditor non caricato: {e}")

    def _load_config(self, config_path: str | None) -> None:
        """Load table config once for position computation."""
        if config_path is None:
            return
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            return
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            table_cfg = cfg.get("table", {})
            self._table_type = table_cfg.get("type", "9max")
            self._hero_seat = int(table_cfg.get("hero_seat", 0))
            self._local_table = LocalTableStateExtractor(config_path)
        except Exception as e:
            print(f"[llm_vision] Could not load local table extractor: {e}")

    def _load_suit_correction_resources(self, config_path: str | None) -> None:
        """Load ROIs and suit templates for local clubs/spades correction."""
        if config_path is None:
            return
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            return
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            rois = cfg.get("vision", {}).get("rois", {})
            self._hole_rois = rois.get("hole", [])
            self._board_rois = rois.get("board", [])
            self._pot_roi = rois.get("pot")
            self._to_call_roi = rois.get("to_call")
            self._stack_roi = rois.get("stack")

            templates_dir = Path(
                cfg.get("vision", {}).get("template_dir", "vision/templates")
            )
            if not templates_dir.is_absolute():
                templates_dir = cfg_path.parent / templates_dir
            extra_suit_dir = templates_dir.parent / "suit_templates"
            if templates_dir.exists():
                self._suit_templates = _extract_suit_templates(
                    templates_dir, extra_suit_dir=extra_suit_dir
                )
                print(
                    f"[llm_vision] Suit correction loaded: "
                    f"{sum(len(v) for v in self._suit_templates.values())} suit templates"
                )
        except Exception as e:
            print(f"[llm_vision] Could not load suit correction resources: {e}")

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _frame_hash(frame: np.ndarray) -> str:
        """Simple perceptual hash: resize to tiny and hash bytes."""
        tiny = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY) if tiny.ndim == 3 else tiny
        return gray.tobytes().hex()[:64]

    def _should_skip(self, frame: np.ndarray) -> bool:
        """Return True if we should reuse the cached result."""
        now = time.time()
        if now - self._last_call_time < self.cooldown:
            h = self._frame_hash(frame)
            if h == self._last_frame_hash:
                return True
        return False

    @staticmethod
    def _post_with_retry(
        url: str,
        headers: dict,
        json_payload: dict,
        timeout: float,
        max_retries: int = 2,
    ) -> requests.Response:
        """Post with exponential backoff on read-timeout/connection errors."""
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return requests.post(
                    url, headers=headers, json=json_payload, timeout=timeout
                )
            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
            ) as exc:
                last_exc = exc
                if attempt == max_retries:
                    break
                sleep = 1.0 * (attempt + 1)
                print(
                    f"[llm_vision] API timeout/connection error, "
                    f"retrying in {sleep}s... ({exc})"
                )
                time.sleep(sleep)
        assert last_exc is not None  # sempre impostato se arriviamo qui
        raise last_exc

    def _call_api(self, frame: np.ndarray, focused: bool = False) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "OpenRouter API key not configured. "
                "Set OPENROUTER_API_KEY env var, create a .env file, pass api_key=..., "
                "or set vision.llm.api_key in config.yaml."
            )

        t0 = time.time()
        b64_img = _encode_frame_to_base64(frame)
        t_encode = time.time() - t0
        payload_kb = len(b64_img) // 1024

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if focused:
            system = SYSTEM_PROMPT + (
                "\n\nFOCUSED TASK: Look ONLY at the TWO hole cards at the BOTTOM CENTER of the screen. "
                "Ignore all other cards. Pay extreme attention to the suit COLOR (black=spades/clubs, red=hearts/diamonds) "
                "and the suit SYMBOL shape. Return ONLY the two hole cards."
            )
            user_text = 'What are the two hole cards at the bottom center? Return ONLY ["Xs", "Yh"] format.'
        else:
            system = SYSTEM_PROMPT
            user_text = "Extract the full poker table state from this screenshot."

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                        },
                    ],
                },
            ],
            "temperature": 0.05 if focused else 0.1,
            "max_tokens": 256,
        }

        t1 = time.time()
        resp = self._post_with_retry(
            self.api_url, headers=headers, json_payload=payload, timeout=self.timeout
        )
        t_api = time.time() - t1
        print(
            f"[llm_vision] encode={t_encode * 1000:.0f}ms api={t_api * 1000:.0f}ms "
            f"payload={payload_kb}KB total={(t_encode + t_api) * 1000:.0f}ms"
        )
        if not resp.ok:
            print(f"[llm_vision] HTTP {resp.status_code} BODY: {resp.text[:500]}")
        resp.raise_for_status()
        data = resp.json()

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected API response structure: {data}") from exc

        # Nota: bool(text) evita AttributeError se il modello ha risposto None
        focused_array_retry = focused and bool(text) and text.strip().startswith("[")
        try:
            state = _extract_json_from_text(text)
        except RuntimeError as exc:
            # _extract_json_from_text solleva RuntimeError (non JSONDecodeError):
            # il ramo focused qui sotto era irraggiungibile prima della correzione.
            # Focused mode may return a raw array like ["As", "Kh"]
            if focused_array_retry:
                try:
                    hole_list = json.loads(text.strip())
                    state = {"hole": hole_list}
                except json.JSONDecodeError:
                    raise RuntimeError(
                        f"Failed to parse JSON from model response. Raw text:\n{text[:800]}"
                    ) from exc
            else:
                raise RuntimeError(
                    f"Failed to parse JSON from model response. Raw text:\n{text[:800]}"
                ) from exc

        return state

    def _verify_red_suit(self, card_crop: np.ndarray, rank: str) -> str | None:
        """
        Focused LLM call to disambiguate diamond (d) vs heart (h).

        Sends a single card crop to the vision model and asks only about the
        suit symbol. Cheap because max_tokens=8 and the image is tiny.
        """
        if not self.api_key:
            return None
        try:
            b64_img = _encode_frame_to_base64(card_crop, max_dim=512, jpeg_quality=90)
            system = (
                "You are looking at a single playing card crop. "
                f"The card rank is {rank}. Focus ONLY on the suit symbol.\n\n"
                "Is the suit a heart (♥) or a diamond (♦)?\n"
                "A heart is rounded with two curves and a dip at the top.\n"
                "A diamond is a sharp angular rhombus.\n\n"
                "Reply with EXACTLY one letter: 'h' for heart or 'd' for diamond. "
                "No explanation, no punctuation."
            )
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Heart or diamond?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_img}"
                                },
                            },
                        ],
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 8,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            resp = self._post_with_retry(
                self.api_url, headers=headers, json_payload=payload, timeout=15.0
            )
            if not resp.ok:
                print(
                    f"[llm_vision] red-suit verify HTTP {resp.status_code}: {resp.text[:200]}"
                )
                return None
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            if text is None:
                return None
            text = text.strip().lower()
            if text.startswith("h") or "♥" in text:
                return "h"
            if text.startswith("d") or "♦" in text:
                return "d"
            return None
        except Exception as e:
            print(f"[llm_vision] red-suit verify failed: {e}")
            return None

    def _verify_suit(self, card_crop: np.ndarray, rank: str, color: str) -> str | None:
        """
        Focused LLM call to resolve a single ambiguous suit.

        Asks the model to choose between the two suits of the given color.
        """
        if not self.api_key:
            return None
        if color == "red":
            suits = ("h", "d")
            names = ("heart (♥)", "diamond (♦)")
            desc = (
                "A heart is rounded with two curves and a dip at the top.\n"
                "A diamond is a sharp angular rhombus."
            )
            ask = "Heart or diamond?"
        elif color == "black":
            suits = ("s", "c")
            names = ("spade (♠)", "club (♣)")
            desc = (
                "A spade is a black pointed symbol with a stem.\n"
                "A club is a black three-leaf clover."
            )
            ask = "Spade or club?"
        else:
            return None

        try:
            b64_img = _encode_frame_to_base64(card_crop, max_dim=1024, jpeg_quality=90)
            system = (
                f"You are looking at a single playing card crop. "
                f"The card rank is {rank}. Ignore the large portrait/face in the center. "
                f"Look ONLY at the small suit symbol located directly below the rank letter "
                f"in the TOP-LEFT corner of the card.\n\n"
                f"Is that small suit symbol a {names[0]} or a {names[1]}?\n{desc}\n\n"
                f"Reply with EXACTLY one letter: '{suits[0]}' for {names[0]} "
                f"or '{suits[1]}' for {names[1]}. No explanation, no punctuation."
            )
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": ask},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_img}"
                                },
                            },
                        ],
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 8,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            resp = self._post_with_retry(
                self.api_url, headers=headers, json_payload=payload, timeout=15.0
            )
            if not resp.ok:
                print(
                    f"[llm_vision] single-suit verify HTTP {resp.status_code}: {resp.text[:200]}"
                )
                return None
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            if text is None:
                return None
            text = text.strip().lower()
            for suit, name in zip(suits, names, strict=True):
                if text.startswith(suit) or name.split()[0] in text:
                    return suit
            return None
        except Exception as e:
            print(f"[llm_vision] single-suit verify failed: {e}")
            return None

    @staticmethod
    def _ocr_number_in_roi(frame: np.ndarray, roi: dict) -> int | None:
        """Local OCR fallback for a numeric ROI (pot, to_call, stack).

        Returns the value in cents (e.g. €0.26 -> 26, €5 -> 500) or None.
        """
        try:
            import pytesseract
        except Exception:
            return None

        resolved = _resolve_relative_roi(roi, frame)
        if resolved is None:
            return None

        # Resolve Tesseract binary (Windows bundle, macOS Homebrew, or PATH)
        from tesseract_utils import find_tesseract_binary

        pytesseract.pytesseract.tesseract_cmd = find_tesseract_binary()

        try:
            from capture import crop_roi
        except Exception:
            return None

        crop = crop_roi(
            frame, resolved["x"], resolved["y"], resolved["w"], resolved["h"]
        )
        if crop is None or crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        text = pytesseract.image_to_string(
            binary,
            config="--psm 7 -c tessedit_char_whitelist=0123456789$€.,",
        ).strip()

        # Extract the last numeric token, supporting . and , as decimal separators
        matches = re.findall(r"[\d.,]+", text)
        if not matches:
            return None
        token = matches[-1].replace(",", ".")
        try:
            value = float(token)
        except ValueError:
            return None

        # Convert to cents
        try:
            if "." in token:
                return round(value * 100)
            return int(value * 100)
        except (ValueError, OverflowError):
            return None

    def _ocr_text_in_roi(self, frame: np.ndarray, roi: dict) -> str:
        """Local OCR fallback that returns raw text from a ROI."""
        try:
            import pytesseract
        except Exception:
            return ""

        resolved = _resolve_relative_roi(roi, frame)
        if resolved is None:
            return ""

        tesseract_bin = Path(__file__).parent.parent / "tesseract" / "tesseract.exe"
        if tesseract_bin.exists():
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_bin)

        try:
            from capture import crop_roi
        except Exception:
            return ""

        crop = crop_roi(
            frame, resolved["x"], resolved["y"], resolved["w"], resolved["h"]
        )
        if crop is None or crop.size == 0:
            return ""

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return pytesseract.image_to_string(binary, config="--psm 7").strip().lower()

    def _fallback_numbers_from_ocr(self, frame: np.ndarray, state: dict) -> dict:
        """Use local OCR as fallback, but don't override a plausible LLM read."""
        corrected = dict(state)

        def _close_enough(
            ocr_val: int | None, llm_val: int | None, tol: float = 0.30
        ) -> bool:
            if ocr_val is None or llm_val is None:
                return False
            if llm_val == 0:
                return ocr_val == 0
            return abs(ocr_val - llm_val) / llm_val <= tol

        if self._pot_roi is not None:
            try:
                ocr_pot = self._ocr_number_in_roi(frame, self._pot_roi)
                llm_pot = corrected.get("pot")
                if ocr_pot is not None and ocr_pot >= 0:
                    if llm_pot is None:
                        print(
                            f"[llm_vision] Pot OCR fallback (LLM missing): None -> {ocr_pot}"
                        )
                        corrected["pot"] = ocr_pot
                    elif llm_pot == 0 and ocr_pot > 0:
                        print(
                            f"[llm_vision] Pot OCR override (LLM zero): 0 -> {ocr_pot}"
                        )
                        corrected["pot"] = ocr_pot
                    elif _close_enough(ocr_pot, llm_pot):
                        print(f"[llm_vision] Pot OCR confirm: {llm_pot} -> {ocr_pot}")
                        corrected["pot"] = ocr_pot
                    else:
                        print(
                            f"[llm_vision] Pot OCR ignored (mismatch): LLM={llm_pot}, OCR={ocr_pot}"
                        )
            except Exception as e:
                print(f"[llm_vision] Pot OCR fallback failed: {e}")

        if self._to_call_roi is not None:
            try:
                ocr_to_call = self._ocr_number_in_roi(frame, self._to_call_roi)
                llm_to_call = corrected.get("to_call", 0)
                if ocr_to_call is not None and ocr_to_call >= 0:
                    # Strong signal: button text explicitly says Check/Gratis.
                    text = self._ocr_text_in_roi(frame, self._to_call_roi) or ""
                    is_check = "check" in text.lower() or "gratis" in text.lower()

                    if ocr_to_call == 0 and llm_to_call > 0 and is_check:
                        print(
                            f"[llm_vision] ToCall OCR override (Check): {llm_to_call} -> 0"
                        )
                        corrected["to_call"] = 0
                    elif ocr_to_call > 0 and llm_to_call == 0:
                        # OCR sees a bet but LLM says 0. Only trust OCR if it is a
                        # plausible call size (not a random stack/chip misread).
                        if (
                            _close_enough(ocr_to_call, corrected.get("pot"), tol=0.5)
                            or ocr_to_call <= 4
                        ):
                            print(
                                f"[llm_vision] ToCall OCR override: {llm_to_call} -> {ocr_to_call}"
                            )
                            corrected["to_call"] = ocr_to_call
                        else:
                            print(
                                f"[llm_vision] ToCall OCR ignored (implausible): LLM={llm_to_call}, OCR={ocr_to_call}"
                            )
                    elif (
                        ocr_to_call > 0
                        and llm_to_call > 0
                        and _close_enough(ocr_to_call, llm_to_call)
                    ):
                        print(
                            f"[llm_vision] ToCall OCR confirm: {llm_to_call} -> {ocr_to_call}"
                        )
                        corrected["to_call"] = ocr_to_call
                    else:
                        print(
                            f"[llm_vision] ToCall OCR ignored (mismatch): LLM={llm_to_call}, OCR={ocr_to_call}"
                        )
            except Exception as e:
                print(f"[llm_vision] ToCall OCR fallback failed: {e}")

        return corrected

    def _is_board_empty(self, frame: np.ndarray) -> bool:
        """
        Heuristic: true if the board ROIs contain almost no visible board cards.

        Goldbet board cards have a LIGHT GRAY (not bright white) background.
        Empty slots show either green felt (preflop) or blue/brown card backs.
        We count a slot as card-like when a large portion of the ROI is distinctly
        lighter than the green felt.
        """
        if not self._board_rois:
            return False

        card_like = 0
        details = []
        for r in self._board_rois:
            roi = _resolve_relative_roi(r, frame)
            if roi is None:
                continue
            crop = frame[roi["y"] : roi["y"] + roi["h"], roi["x"] : roi["x"] + roi["w"]]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Gray card backgrounds are ~120-180; green felt is ~60-100.
            if gray.size == 0:
                continue
            try:
                light_ratio = float((gray > 120).sum()) / gray.size
            except (ValueError, ZeroDivisionError):
                continue
            details.append(f"{light_ratio:.2f}")
            if light_ratio > 0.25:
                card_like += 1

        empty = card_like < 2  # board is 0/3/4/5; 1 bright ROI is suspicious
        if empty:
            print(
                f"[llm_vision] Board region looks empty (light ratios={', '.join(details)}, "
                f"card_like={card_like})"
            )
        return empty

    def _verify_uncertain_suits(
        self,
        frame: np.ndarray,
        state: dict,
        uncertain_items: list[tuple[str, str, dict, int]],
    ) -> dict:
        """
        One-shot LLM verification for cards where local suit matching is unsure.

        Sends a composite image of all uncertain cards and asks the model for
        only the suit letter of each, given the rank. This fixes cross-color
        misreads and low-margin same-color swaps in a single API call.
        """
        if not uncertain_items:
            return state

        composite, labels = _build_composite_image(
            frame,
            [(card, roi) for card, _, roi, _ in uncertain_items],
            cell_size=(120, 180),
            cols=3,
        )
        if composite.size == 0 or not labels:
            return state

        ranks_list = [card[0].upper() for card, _, _, _ in uncertain_items]
        try:
            b64_img = _encode_frame_to_base64(composite, max_dim=1280, jpeg_quality=90)
            system = (
                "You are looking at a grid of numbered playing card crops. "
                "Each crop shows ONE card. The RANK of each card is known. "
                "Your job is to identify ONLY the SUIT symbol of each card.\n\n"
                "Suits:\n"
                "- s = spades (♠), black pointed symbol\n"
                "- h = hearts (♥), red heart shape\n"
                "- d = diamonds (♦), red angular rhombus\n"
                "- c = clubs (♣), black three-leaf clover\n\n"
                f"The ranks in order (1 to {len(ranks_list)}) are: {ranks_list}\n\n"
                "Return ONLY a JSON array with the suit letter for each card, in the same order.\n"
                f"Example for {len(ranks_list)} cards: {json.dumps(['s'] * len(ranks_list))}\n"
                "No explanation, no markdown."
            )
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is the suit of each numbered card?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_img}"
                                },
                            },
                        ],
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 64,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            resp = self._post_with_retry(
                self.api_url, headers=headers, json_payload=payload, timeout=20.0
            )
            if not resp.ok:
                print(
                    f"[llm_vision] uncertain-suits verify HTTP {resp.status_code}: {resp.text[:200]}"
                )
                return state
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            suits = _parse_suits_array(text)
            if suits is None or len(suits) != len(uncertain_items):
                print(
                    f"[llm_vision] Could not parse uncertain suits array: {text[:200]}"
                )
                return state

            corrected = dict(state)
            for (card, slot, _, idx), suit in zip(uncertain_items, suits, strict=True):
                if suit in ("s", "h", "d", "c") and suit != card[1].lower():
                    new_card = card[0].upper() + suit
                    target = corrected.setdefault(slot, [])
                    if idx < len(target):
                        print(
                            f"[llm_vision] Uncertain-suit verify: {card} -> {new_card}"
                        )
                        target[idx] = new_card
            return corrected
        except Exception as e:
            print(f"[llm_vision] uncertain-suits verify failed: {e}")
            return state

    def _correct_suits(
        self,
        frame: np.ndarray,
        state: dict,
        threshold: float = 0.35,
        min_margin: float = 0.08,
    ) -> dict:
        """
        Local color + same-rank template-matching fix for LLM suit confusion.

        - Detects suit color (red/black) and builds same-rank candidate templates.
        - Makes a local correction ONLY when the match is confident.
        - Sends all unsure cards (cross-color, low margin, missing template) to a
          single focused composite LLM call for verification.
        """
        if not self._suit_templates or (not self._hole_rois and not self._board_rois):
            return state

        from pathlib import Path

        from capture import crop_roi
        from ocr_cards import extract_split_templates_from_full, load_templates_from_dir

        corrected = dict(state)
        templates_dir = Path("vision/templates")

        full_templates = load_templates_from_dir(str(templates_dir))
        _, full_suit_templates = extract_split_templates_from_full(full_templates)

        def candidate_suit_templates(
            rank: str, allowed: list[str]
        ) -> dict[str, list[np.ndarray]]:
            candidates: dict[str, list[np.ndarray]] = {}
            for suit in allowed:
                if not (templates_dir / f"{rank}{suit}.png").exists():
                    continue
                imgs: list[np.ndarray] = []
                if suit in full_suit_templates:
                    imgs.extend(full_suit_templates[suit])
                if suit in self._suit_templates:
                    imgs.extend(self._suit_templates[suit])
                if imgs:
                    candidates[suit] = imgs
            return candidates

        uncertain_items: list[tuple[str, str, dict, int]] = []

        def fix_card_list(cards: list[str], rois: list[dict], slot: str) -> list[str]:
            out: list[str] = []
            for i, card in enumerate(cards):
                if len(card) != 2 or card[1].lower() not in ("s", "h", "d", "c"):
                    out.append(card)
                    continue
                if i >= len(rois):
                    out.append(card)
                    continue
                roi = _resolve_relative_roi(rois[i], frame)
                if roi is None:
                    out.append(card)
                    continue
                card_crop = crop_roi(frame, roi["x"], roi["y"], roi["w"], roi["h"])
                if card_crop is None or card_crop.size == 0:
                    out.append(card)
                    continue

                rank = card[0].upper()
                current_suit = card[1].lower()
                color = _detect_card_color(card_crop)

                if color is None:
                    uncertain_items.append((card, slot, roi, i))
                    out.append(card)
                    continue

                # suit_crop is used later only for face-card verification
                suit_crop = _crop_suit_from_full_card(card_crop)

                allowed = {"red": ["h", "d"], "black": ["s", "c"]}[color]
                candidates = candidate_suit_templates(rank, allowed)
                if not candidates:
                    uncertain_items.append((card, slot, roi, i))
                    out.append(card)
                    continue

                best_suit, score, margin = _best_suit(
                    suit_crop, candidates, threshold=0.0, allowed_suits=None
                )
                new_card: str | None = None

                if current_suit not in allowed:
                    # Cross-color misread: trust local color and pick the best
                    # same-color suit if the template match is decent.
                    if best_suit is not None and score >= 0.35 and margin >= 0.10:
                        new_card = f"{rank}{best_suit}"
                        print(
                            f"[llm_vision] Cross-color correction: {card} -> {new_card} "
                            f"(score={score:.2f}, margin={margin:.2f}, color={color})"
                        )
                    else:
                        # Not confident enough; leave for possible verification.
                        uncertain_items.append((card, slot, roi, i))
                elif len(candidates) >= 2:
                    required_margin = 0.12 if color == "black" else 0.10
                    if (
                        best_suit is not None
                        and best_suit != current_suit
                        and score >= 0.45
                        and margin >= required_margin
                    ):
                        new_card = f"{rank}{best_suit}"
                    else:
                        # Same color but not confident enough.
                        uncertain_items.append((card, slot, roi, i))
                else:
                    # Only one candidate of this color exists.
                    only_suit = next(iter(candidates))
                    if current_suit != only_suit:
                        # Missing the other same-color template; verify rather than force.
                        uncertain_items.append((card, slot, roi, i))

                if new_card is not None:
                    print(
                        f"[llm_vision] Suit correction: {card} -> {new_card} "
                        f"(score={score:.2f}, margin={margin:.2f}, color={color})"
                    )
                    out.append(new_card)
                else:
                    out.append(card)
            return out

        corrected["hole"] = fix_card_list(
            state.get("hole", []), self._hole_rois, "hole"
        )
        corrected["board"] = fix_card_list(
            state.get("board", []), self._board_rois, "board"
        )

        if uncertain_items:
            # Verify uncertain cards when suits matter for the decision:
            # - hole cards are always verified (they affect every decision)
            # - board cards are verified on flop/turn/river (any suit can become
            #   relevant for flush draws); on empty boards we skip them.
            hole_uncertain = [it for it in uncertain_items if it[1] == "hole"]

            board_color_counts: dict[str, int] = {}
            for card in corrected.get("board", []):
                if len(card) == 2:
                    color = {"s": "black", "c": "black", "h": "red", "d": "red"}.get(
                        card[1].lower()
                    )
                    if color:
                        board_color_counts[color] = board_color_counts.get(color, 0) + 1

            board_verify = []
            board_len = len(corrected.get("board", []))
            for it in uncertain_items:
                card, slot, _, _ = it
                if slot != "board" or len(card) != 2:
                    continue
                color = {"s": "black", "c": "black", "h": "red", "d": "red"}.get(
                    card[1].lower()
                )
                # Always verify board suits once the flop is out; otherwise only
                # when two cards of the same color are already present.
                if board_len >= 3 or (color and board_color_counts.get(color, 0) >= 2):
                    board_verify.append(it)

            to_verify = hole_uncertain + board_verify
            if to_verify:
                corrected = self._verify_uncertain_suits(frame, corrected, to_verify)
            else:
                print(
                    f"[llm_vision] {len(uncertain_items)} uncertain suit(s) left uncorrected "
                    f"(no flush relevance): {[c for c, _, _, _ in uncertain_items]}"
                )

        return corrected

    def _verify_missing_suits(self, frame: np.ndarray, state: dict) -> dict:
        """
        Focused LLM verification for cards with no exact template.

        Builds a composite image of all missing-template cards and asks the
        model for only the suit letter of each. This fixes cases where the
        main LLM call confuses suits for ranks we don't have templates for.
        One API call covers all missing cards.
        """
        from pathlib import Path

        templates_dir = Path("vision/templates")
        missing_items: list[
            tuple[str, str, dict, int]
        ] = []  # label, slot, roi, list_idx

        for slot, rois, cards in (
            ("hole", self._hole_rois, state.get("hole", [])),
            ("board", self._board_rois, state.get("board", [])),
        ):
            for i, card in enumerate(cards):
                if len(card) != 2 or i >= len(rois):
                    continue
                if not (templates_dir / f"{card}.png").exists():
                    missing_items.append((card, slot, rois[i], i))

        if not missing_items:
            return state

        composite, labels = _build_composite_image(
            frame, [(c, r) for c, _, r, _ in missing_items]
        )
        if composite.size == 0 or not labels:
            return state

        # Ask LLM for suits only
        try:
            b64_img = _encode_frame_to_base64(composite, max_dim=1024, jpeg_quality=85)
            ranks_list = [c[0].upper() for c, _, _, _ in missing_items]
            system = (
                "You are looking at a grid of numbered playing card crops. "
                "Each crop shows ONE card. The RANK of each card is known. "
                "Your job is to identify ONLY the SUIT symbol of each card.\n\n"
                "Suits:\n"
                "- s = spades (♠), black pointed symbol\n"
                "- h = hearts (♥), red heart shape\n"
                "- d = diamonds (♦), red angular rhombus\n"
                "- c = clubs (♣), black three-leaf clover\n\n"
                f"The ranks in order (1 to {len(ranks_list)}) are: {ranks_list}\n\n"
                "Return ONLY a JSON array with the suit letter for each card, in the same order.\n"
                f"Example for {len(ranks_list)} cards: {json.dumps(['s'] * len(ranks_list))}\n"
                "No explanation, no markdown."
            )
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is the suit of each numbered card?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_img}"
                                },
                            },
                        ],
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 64,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            resp = self._post_with_retry(
                self.api_url, headers=headers, json_payload=payload, timeout=20.0
            )
            if not resp.ok:
                print(
                    f"[llm_vision] missing-suits verify HTTP {resp.status_code}: {resp.text[:200]}"
                )
                return state
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            suits = _parse_suits_array(text)
            if suits is None or len(suits) != len(missing_items):
                print(f"[llm_vision] Could not parse suits array: {text[:200]}")
                return state

            corrected = dict(state)
            for (card, slot, _, idx), suit in zip(missing_items, suits, strict=True):
                if suit in ("s", "h", "d", "c") and suit != card[1].lower():
                    new_card = card[0].upper() + suit
                    target = corrected.setdefault(slot, [])
                    if idx < len(target):
                        print(f"[llm_vision] Missing-suit verify: {card} -> {new_card}")
                        target[idx] = new_card
            return corrected
        except Exception as e:
            print(f"[llm_vision] missing-suits verify failed: {e}")
            return state

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def extract(self, frame: np.ndarray | None = None) -> dict:
        """
        Extract table state from a frame.
        If *frame* is None, captures a fresh screenshot.
        """
        if frame is None:
            # capture may not be importable if run standalone; be defensive.
            try:
                from capture import screenshot
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "Could not import capture.screenshot. "
                    "Make sure the vision/ folder is on PYTHONPATH."
                ) from exc
            frame = screenshot(window_title=self.window_title)
            if frame is None:
                raise RuntimeError("Screenshot failed: window not found")

        if self._should_skip(frame):
            assert (
                self._cached_state is not None
            )  # impostato dall'ultima extract() riuscita
            return self._cached_state

        raw_state = self._call_api(frame, focused=False)

        # Normalize / sanitize first pass
        state = self._normalize_state(raw_state)

        # ---- Board hallucination guard ----
        # Local per-slot white detection is more reliable than trusting the LLM
        # when the board area is actually empty (e.g. between hands or preflop).
        try:
            if self._is_board_empty(frame) and state.get("board"):
                print(
                    f"[llm_vision] Board looks empty; clearing hallucinated board {state['board']}"
                )
                state["board"] = []
                state["stage"] = "preflop"
        except Exception as e:
            print(f"[llm_vision] Board empty check failed: {e}")

        # ---- Local OCR fallback DISABLED ----
        # OCR on pot/to_call proved unreliable and was overriding correct LLM
        # reads (e.g. correct pot=12 overwritten to 3). Trust LLM numbers.
        # try:
        #     state = self._fallback_numbers_from_ocr(frame, state)
        # except Exception as e:
        #     print(f"[llm_vision] Number OCR fallback failed: {e}")

        # ---- Local suit correction ----
        # Fast local color + template matching to fix cross-color suit swaps
        # (e.g. red heart read as black club) without extra API calls.
        # Same-color uncertainty (h/d or s/c) is verified only when needed.
        pre_correction = {
            "hole": list(state.get("hole", [])),
            "board": list(state.get("board", [])),
        }
        try:
            state = self._correct_suits(frame, state)
        except Exception as e:
            print(f"[llm_vision] Suit correction failed: {e}")

        # Sanity post-correzione: se la correzione locale ha introdotto duplicati
        # (dentro uno slot o tra hole e board), la correzione e' sbagliata:
        # ripristina le carte lette dall'LLM.
        cards_now = list(state.get("hole", [])) + list(state.get("board", []))
        cards_pre = pre_correction["hole"] + pre_correction["board"]
        if len(cards_now) != len(set(cards_now)) and len(cards_pre) == len(
            set(cards_pre)
        ):
            print(
                f"[llm_vision] Suit correction ha introdotto duplicati "
                f"(hole={state['hole']}, board={state['board']}) -> "
                f"ripristino lettura LLM (hole={pre_correction['hole']}, "
                f"board={pre_correction['board']})"
            )
            state["hole"] = list(pre_correction["hole"])
            state["board"] = list(pre_correction["board"])

        # ---- Suit memory per mano ----
        # I semi non cambiano durante una mano: voto temporale sulla coppia
        # di semi piu' vista per gli stessi rank + suit piu' votato per
        # posizione del board. Reset al cambio di mano.
        try:
            state = self._apply_hand_suit_memory(state)
        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as e:
            print(f"[llm_vision] Hand suit memory failed: {e}")

        # ---- Focused retry on hole cards DISABLED ----
        # The focused retry often pulled board cards into the hole list when
        # the LLM reported duplicates. We now trust the main LLM call and log
        # any remaining duplicates as a warning instead of burning another
        # 2-3s API call.
        all_cards = state["hole"] + state["board"]
        duplicates = len(all_cards) != len(set(all_cards))
        unknown_suit = any("?" in str(c) for c in state["hole"])
        if duplicates or unknown_suit:
            print(
                f"[llm_vision] WARNING: inconsistency detected "
                f"(dupes={duplicates}, unknown={unknown_suit}, hole={state['hole']}, board={state['board']})"
            )

        # Cache result and timestamp for cooldown / frame-hash skip
        self._last_call_time = time.time()
        self._last_frame_hash = self._frame_hash(frame)
        self._cached_state = state

        print(
            f"[llm_vision] hole={state['hole']} board={state['board']} "
            f"pot={state['pot']} to_call={state['to_call']} stage={state['stage']}"
        )

        # ---- Button seat: LLM primary, local CV fallback only when LLM fails ----
        # Local CV on Goldbet produces false positives (crowns, chips); trusting
        # it to override a valid LLM read caused wrong positions. Use local only
        # if the LLM did not return a button.
        local_button, local_score = self._detect_button_local(frame)
        llm_button = state.get("button_seat")
        if llm_button is not None:
            state["button_seat"] = llm_button
        elif local_button is not None and local_score >= 0.55:
            print(
                f"[llm_vision] Button fallback to local CV: seat={local_button} "
                f"score={local_score:.2f}"
            )
            state["button_seat"] = local_button
        else:
            state["button_seat"] = self._cached_button_seat

        # ---- Position: compute locally from button_seat ----
        # The LLM reports button_seat; SeatLayout derives hero position.
        # This is faster and more reliable than a separate position-only LLM call.
        state["position"] = self._position_from_button(
            state.get("button_seat"), state.get("hole", [])
        )

        # ---- Jev audit (opzionale): background, allega jev_audit allo stato ----
        if self._jev_auditor is not None:
            self._jev_auditor.maybe_check_async(state)
            state = self._jev_auditor.attach(state)

        return state

    def _detect_button_local(self, frame: np.ndarray) -> tuple[int | None, float]:
        """Use local CV to find the dealer button seat and confidence."""
        if self._local_table is None:
            return None, 0.0
        try:
            return self._local_table.detect_button_seat_with_score(frame)
        except (
            cv2.error,
            ValueError,
            KeyError,
            AttributeError,
            IndexError,
            TypeError,
        ) as e:
            print(f"[llm_vision] Local button detection failed: {e}")
            return None, 0.0

    def _position_from_button(self, button_seat: int | None, hole: list[str]) -> str:
        """Compute hero position from button_seat using SeatLayout.

        Position is kept stable within the same hand (same hole cards) unless
        the button seat changes by a real table rotation.
        """
        if button_seat is None:
            return self._cached_position or "BTN"

        # New hand detection: if hole cards changed significantly, allow button update.
        hole_key = ",".join(sorted(hole))
        new_hand = hole_key != self._last_hole_key
        self._last_hole_key = hole_key

        # If same hand and we already have a cached button, keep stable position
        # to avoid LLM jitter. Only update if button changed between hands.
        if (
            not new_hand
            and self._cached_button_seat is not None
            and self._cached_position is not None
            and button_seat != self._cached_button_seat
        ):
            # Button did not really move during the hand; trust cache.
            return self._cached_position

        try:
            from seat_layout import SeatLayout

            layout = SeatLayout(
                table_type=self._table_type,
                button_seat=button_seat,
                hero_seat=self._hero_seat,
            )
            pos = layout.hero_position()
            self._cached_button_seat = button_seat
            if pos != self._cached_position:
                if self._cached_position:
                    print(
                        f"[llm_vision] position changed: {self._cached_position} -> {pos} (button seat {button_seat})"
                    )
                else:
                    print(
                        f"[llm_vision] position detected: {pos} (button seat {button_seat})"
                    )
                self._cached_position = pos
            else:
                print(
                    f"[llm_vision] position confirmed: {pos} (button seat {button_seat})"
                )
            return pos
        except (
            ImportError,
            ValueError,
            KeyError,
            AttributeError,
            IndexError,
            TypeError,
        ):
            print(f"[llm_vision] position compute failed:\n{traceback.format_exc()}")
            if self._cached_position:
                return self._cached_position
            return "BTN"

    def _maybe_refresh_position_async(self, frame: np.ndarray) -> None:
        """Start a background thread to refresh position if cache is stale."""
        import threading

        now = time.time()
        if self._position_thread_running:
            return
        if now - self._last_position_time < self.position_refresh_seconds:
            return
        if frame is None:
            return
        self._position_thread_running = True
        frame_copy = frame.copy()
        t = threading.Thread(
            target=self._refresh_position_worker,
            args=(frame_copy,),
            daemon=True,
        )
        t.start()

    def _refresh_position_worker(self, frame: np.ndarray) -> None:
        """Background worker: call position_model to get the D-button position."""
        try:
            pos = self._call_api_position_only(
                frame, model=self.position_model or self.model
            )
            if pos:
                if pos != self._cached_position:
                    print(
                        f"[llm_vision] position changed: {self._cached_position} -> {pos}"
                    )
                else:
                    print(f"[llm_vision] position confirmed: {pos}")
                self._cached_position = pos
                self._last_position_time = time.time()
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            print(f"[llm_vision] position refresh failed: {e}")
        finally:
            self._position_thread_running = False

    def _apply_hand_suit_memory(self, state: dict) -> dict:
        """Congela i semi per mano tramite voto temporale.

        I rank sono stabili tra le letture della stessa mano mentre i semi
        oscillano (downscale + suit-correction rumorosa): si vota la coppia
        di semi piu' vista per gli stessi rank, e il suit piu' votato per
        ogni posizione del board. Reset completo al cambio di mano.
        Override solo con >= 2 voti (maggioranza), mai introdurre duplicati.
        """
        hole = [c for c in state.get("hole", []) if isinstance(c, str) and len(c) == 2]
        board = [
            c for c in state.get("board", []) if isinstance(c, str) and len(c) == 2
        ]
        hole_ok = len(hole) == 2 and "?" not in hole[0] + hole[1]
        if not hole_ok:
            return state

        ranks_now = sorted(c[0] for c in hole)

        # Nuova mano: rank diversi da quelli votati finora -> reset voti
        if self._frozen_ranks and ranks_now != self._frozen_ranks:
            print(
                f"[llm_vision] Nuova mano: {self._frozen_ranks} -> {ranks_now}, "
                f"reset voti semi"
            )
            self._hand_votes = {}
            self._board_suit_votes = {}
            self._frozen_ranks = []
        if not self._frozen_ranks:
            self._frozen_ranks = ranks_now

        merged_hole = list(hole)
        merged_board = list(board)

        # --- voto hole: coppia di semi piu' vista per questi rank ---
        a, b = sorted(hole)
        key = (a, b)
        self._hand_votes[key] = self._hand_votes.get(key, 0) + 1
        best_pair = max(self._hand_votes.items(), key=lambda kv: kv[1])[0]
        if best_pair != key and self._hand_votes[best_pair] >= 2:
            # preserva l'ordine di lettura: mappa i semi vincenti sui rank
            # attuali (gestisce anche le coppie, es. KK)
            pool: dict[str, list[str]] = {}
            for c in best_pair:
                pool.setdefault(c[0], []).append(c)
            merged_hole = []
            for c in hole:
                candidates = pool.get(c[0])
                if candidates:
                    merged_hole.append(candidates.pop(0))
                else:
                    merged_hole.append(c)
            print(
                f"[llm_vision] Suit memory hole (voto {self._hand_votes[best_pair]}x): "
                f"{hole} -> {merged_hole}"
            )

        # --- voto board: suit piu' votato per (posizione, rank) ---
        changed = False
        for i, c in enumerate(board):
            votes = self._board_suit_votes.setdefault((i, c[0]), {})
            votes[c[1]] = votes.get(c[1], 0) + 1
            best_suit, best_n = max(votes.items(), key=lambda kv: kv[1])
            if best_n >= 2 and best_suit != c[1]:
                merged_board[i] = c[0] + best_suit
                changed = True
        if changed:
            print(f"[llm_vision] Suit memory board (voto): {board} -> {merged_board}")

        # sicurezza: mai introdurre duplicati (intra o cross-slot)
        all_merged = merged_hole + merged_board
        if len(all_merged) == len(set(all_merged)):
            state["hole"] = merged_hole
            state["board"] = merged_board
        return state

    def _call_api_position_only(self, frame: np.ndarray, model: str) -> str | None:
        """
        Focused, model-agnostic call to extract ONLY the position (D button).
        Uses max_tokens=8 for speed and a position-specific prompt.
        """
        if not self.api_key:
            return None

        t0 = time.time()
        b64_img = _encode_frame_to_base64(frame)
        t_encode = time.time() - t0
        payload_kb = len(b64_img) // 1024

        system = (
            "You are a poker position detector for 247 Free Poker (5-handed table).\n\n"
            "The PLAYER is at BOTTOM-CENTER. The OTHER 4 seats are:\n"
            "  - BOTTOM-LEFT\n"
            "  - BOTTOM-RIGHT\n"
            "  - TOP-LEFT\n"
            "  - TOP-RIGHT\n\n"
            "Find the small circular chip with the letter 'D' (the dealer button). "
            "It is on ONE of the 5 seats, OR on the player's seat (bottom-center).\n\n"
            "Look CAREFULLY at the screenshot and identify WHERE the D chip is located. "
            "Then return ONLY the position code of the PLAYER based on this lookup:\n\n"
            "  - D on PLAYER (bottom-center)  -> BTN\n"
            "  - D on BOTTOM-LEFT             -> SB\n"
            "  - D on BOTTOM-RIGHT            -> BB\n"
            "  - D on TOP-RIGHT               -> CO\n"
            "  - D on TOP-LEFT                -> MP\n\n"
            "Output ONLY the 2-3 letter code (e.g. 'SB'), no explanation."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Find the D button. Return ONLY the position code.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 8,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        t1 = time.time()
        resp = self._post_with_retry(
            self.api_url, headers=headers, json_payload=payload, timeout=30.0
        )
        t_api = time.time() - t1
        print(
            f"[llm_position] model={model} encode={t_encode * 1000:.0f}ms "
            f"api={t_api * 1000:.0f}ms payload={payload_kb}KB"
        )
        if not resp.ok:
            print(f"[llm_position] HTTP {resp.status_code} BODY: {resp.text[:300]}")
            return None
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        if text is None:
            return None
        text = text.strip().upper()
        for pos in ("SB", "BB", "BTN", "CO", "MP", "UTG"):
            if pos in text:
                return pos
        return None

    def get_position(self, frame: np.ndarray) -> str | None:
        """
        Quickly extract ONLY the player's position (D button) from a frame.
        Returns one of: SB, BB, BTN, CO, MP, UTG, or None on failure.
        Uses minimal max_tokens (8) and a focused prompt to be ~3x faster than full extract.
        """
        if not self.api_key:
            return None

        t0 = time.time()
        b64_img = _encode_frame_to_base64(frame)
        t_encode = time.time() - t0
        payload_kb = len(b64_img) // 1024

        system = (
            "You are a poker table assistant. Find the dealer button (a small circular chip with the letter 'D') "
            "near one of the seats.\n\n"
            "Return ONLY the player's position relative to the dealer button. One of: SB, BB, BTN, CO, MP, UTG.\n\n"
            "Rules:\n"
            "- YOU have the D -> BTN\n"
            "- D is at seat immediately to your LEFT -> SB\n"
            "- D is two seats to your LEFT -> BB\n"
            "- D is one seat to your RIGHT -> CO\n"
            "- D is two seats to your RIGHT -> MP\n"
            "- D is three seats to your RIGHT -> UTG\n\n"
            "Return ONLY the 2-3 letter code, no explanation."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Find the D button. Return ONLY the position code (e.g. BTN).",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 8,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        t1 = time.time()
        try:
            resp = self._post_with_retry(
                self.api_url,
                headers=headers,
                json_payload=payload,
                timeout=self.timeout,
            )
            t_api = time.time() - t1
            print(
                f"[llm_position] encode={t_encode * 1000:.0f}ms api={t_api * 1000:.0f}ms "
                f"payload={payload_kb}KB total={(t_encode + t_api) * 1000:.0f}ms"
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            if text is None:
                return None
            text = text.strip().upper()
            for pos in ("SB", "BB", "BTN", "CO", "MP", "UTG"):
                if pos in text:
                    return pos
            return None
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            print(f"[llm_position] Failed: {e}")
            return None

    def _normalize_state(self, raw_state: dict) -> dict:
        """Sanitize raw LLM response into a clean state dict."""
        # Only hard-fail if the model explicitly says it's not a poker table AND gave no useful data
        if "error" in raw_state and not raw_state.get("hole"):
            raise RuntimeError(f"LLM vision error: {raw_state['error']}")

        state = {
            "hole": [],
            "board": [],
            "pot": 0,
            "to_call": 0,
            "stack": 1000,
            "stage": "preflop",
            "position": "BTN",
            "button_seat": None,
            "opponent_stacks": {},
            "num_active": 1,
            "effective_stack": 1000,
        }

        if isinstance(raw_state.get("hole"), list):
            state["hole"] = [
                _normalize_card(str(c))
                for c in raw_state["hole"]
                if c and isinstance(c, str)
            ]
        if isinstance(raw_state.get("board"), list):
            state["board"] = [
                _normalize_card(str(c))
                for c in raw_state["board"]
                if c and isinstance(c, str)
            ]

        # Sanitize: board can never contain hole cards (LLM sometimes duplicates)
        if state["hole"] and state["board"]:
            hole_set = set(state["hole"])
            cleaned_board = [c for c in state["board"] if c not in hole_set]
            if len(cleaned_board) != len(state["board"]):
                print(
                    f"[llm_vision] Sanitized board: removed hole duplicates -> {cleaned_board}"
                )
                state["board"] = cleaned_board

        # Sanitize: board community cards are always unique
        if state["board"]:
            seen = set()
            unique_board = []
            for c in state["board"]:
                if c not in seen:
                    seen.add(c)
                    unique_board.append(c)
            if len(unique_board) != len(state["board"]):
                print(
                    f"[llm_vision] Sanitized board: removed board duplicates -> {unique_board}"
                )
                state["board"] = unique_board

        for key in ("pot", "to_call", "stack"):
            val = raw_state.get(key)
            try:
                if isinstance(val, (int, float)):
                    state[key] = int(val)
                elif isinstance(val, str):
                    cleaned = "".join(ch for ch in val if ch.isdigit())
                    if cleaned:
                        state[key] = int(cleaned)
            except (ValueError, OverflowError):
                continue

        stage = raw_state.get("stage", "preflop")
        if stage in ("preflop", "flop", "turn", "river"):
            state["stage"] = stage

        # Stage sanity check: board length overrides the model label.
        # 1 or 2 board cards is never a valid poker state; treat as preflop.
        n_board = len(state["board"])
        if n_board == 0:
            state["stage"] = "preflop"
        elif n_board == 3:
            state["stage"] = "flop"
        elif n_board == 4:
            state["stage"] = "turn"
        elif n_board == 5:
            state["stage"] = "river"
        else:
            print(f"[llm_vision] Invalid board length {n_board}; clearing board")
            state["board"] = []
            state["stage"] = "preflop"

        pos = raw_state.get("position", "BTN")
        if pos in ("SB", "BB", "BTN", "CO", "MP", "UTG", "UTG+1", "UTG+2", "LJ", "HJ"):
            state["position"] = pos

        # Parse dealer button seat if provided
        btn = raw_state.get("button_seat")
        if isinstance(btn, int):
            state["button_seat"] = btn
        elif isinstance(btn, str) and btn.isdigit():
            try:
                state["button_seat"] = int(btn)
            except (ValueError, OverflowError):
                state["button_seat"] = None

        # Parse opponent stacks and compute effective stack / active count
        opp = raw_state.get("opponent_stacks")
        opponent_stacks: dict = {}
        if isinstance(opp, dict):
            for k, v in opp.items():
                try:
                    if isinstance(v, (int, float)):
                        opponent_stacks[str(k)] = int(v)
                    elif isinstance(v, str):
                        cleaned = "".join(ch for ch in v if ch.isdigit())
                        if cleaned:
                            opponent_stacks[str(k)] = int(cleaned)
                except (ValueError, OverflowError):
                    continue
        state["opponent_stacks"] = opponent_stacks
        state["num_active"] = 1 + len(opponent_stacks)
        if opponent_stacks:
            state["effective_stack"] = min(opponent_stacks.values())
        else:
            state["effective_stack"] = state.get("stack", 1000)

        # Config override takes precedence (useful when LLM can't read position reliably)
        if self.position_override and self.position_override in (
            "SB",
            "BB",
            "BTN",
            "CO",
            "MP",
            "UTG",
            "UTG+1",
            "UTG+2",
            "LJ",
            "HJ",
        ):
            state["position"] = self.position_override

        return state
