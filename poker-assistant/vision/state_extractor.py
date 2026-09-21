"""
State Extractor — turns a visual frame (or manual input) into a structured table state.

Example output:
    {
        "hole": ["As", "Kh"],
        "board": ["Qd", "Jh", "2c"],
        "pot": 120,
        "to_call": 20,
        "position": "BTN",   # or "SB"
        "stage": "flop",
    }
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from card_classifier import CardClassifier
from confidence_merger import merge_card_confidence, parse_yolo_label
from consistency_checker import ConsistencyChecker, format_report
from llm_vision_extractor import LLMVisionExtractor
from local_table_state import LocalTableStateExtractor
from seat_layout import SeatLayout
from yolo_detector import PokerYOLODetector


class ManualStateExtractor:
    """Reads state from a user-provided dictionary (bypasses vision)."""

    def extract(self, state_dict: dict) -> dict:
        required = {"hole", "board", "pot", "to_call", "position", "stage"}
        missing = required - set(state_dict.keys())
        if missing:
            raise ValueError(f"Missing state keys: {missing}")
        return state_dict


class SupervisionStateExtractor:
    """Wraps the local screenshot extractor and optionally validates with YOLO.

    In Fase 1 this class simply delegates to :class:`ScreenshotStateExtractor`
    and logs whether the optional YOLO detector is available. Real confidence
    fusion between OCR and YOLO will be added in Fase 2.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.cfg = self._load_config()
        self._local = ScreenshotStateExtractor(config_path)
        self._yolo = PokerYOLODetector(self.cfg)
        if self._yolo.available:
            print(
                "[supervision] YOLO detector available and will be used for validation."
            )
        else:
            print(
                "[supervision] YOLO detector unavailable; running local extractor only."
            )

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except OSError:
            return {}

    def extract(self, frame=None) -> dict:
        from capture import screenshot

        if frame is None:
            window_title = self.cfg.get("vision", {}).get("window_title", "Free Poker")
            frame = screenshot(window_title=window_title)
            if frame is None:
                raise RuntimeError("Screenshot failed: window not found")

        # Base state from the local extractor (numbers, position, stage, etc.)
        state = self._local.extract(frame)

        # Re-read cards with real per-slot confidence
        hole_rois = self._local._effective_card_rois("hole")
        board_rois = self._local._effective_card_rois("board")
        hole_cards, hole_scores = self._local._read_cards_with_confidence(
            frame, hole_rois
        )
        board_cards, board_scores = self._local._read_cards_with_confidence(
            frame, board_rois
        )

        # Optional YOLO validation
        yolo_threshold = (
            self._yolo.confidence_threshold if self._yolo.available else 0.5
        )
        yolo_cards = self._yolo.detect_cards(frame) if self._yolo.available else []
        hole_yolo, board_yolo = self._split_yolo_cards(
            yolo_cards, hole_rois, board_rois, frame
        )

        # Fuse local + YOLO confidence
        merged_hole, hole_conf, hole_reasons = merge_card_confidence(
            hole_cards, hole_scores, hole_yolo, yolo_threshold, group_name="hole"
        )
        merged_board, board_conf, board_reasons = merge_card_confidence(
            board_cards, board_scores, board_yolo, yolo_threshold, group_name="board"
        )

        # Update state with merged cards and confidence
        state["hole"] = [c for c in merged_hole if c is not None]
        state["board"] = [c for c in merged_board if c is not None]

        if "confidence" not in state or not isinstance(state["confidence"], dict):
            state["confidence"] = {}
        state["confidence"]["hole"] = hole_conf
        state["confidence"]["cards"] = min(hole_conf, board_conf)

        reasons = list(state.get("uncertainty_reasons", []))
        reasons.extend(hole_reasons)
        reasons.extend(board_reasons)
        state["uncertainty_reasons"] = reasons
        state["is_uncertain"] = bool(reasons)

        return state

    def _split_yolo_cards(
        self,
        yolo_cards: list[tuple[str, np.ndarray, float]],
        hole_rois: list,
        board_rois: list,
        frame: np.ndarray,
    ) -> tuple[
        list[tuple[str, np.ndarray, float]], list[tuple[str, np.ndarray, float]]
    ]:
        """Partition YOLO card detections into hole and board groups."""
        hole_yolo: list[tuple[str, np.ndarray, float]] = []
        board_yolo: list[tuple[str, np.ndarray, float]] = []
        for label, bbox, conf in yolo_cards:
            try:
                cx = (float(bbox[0]) + float(bbox[2])) / 2.0
                cy = (float(bbox[1]) + float(bbox[3])) / 2.0
            except (TypeError, ValueError, IndexError):
                continue
            if self._local._point_in_any_roi(cx, cy, hole_rois, frame):
                hole_yolo.append((label, bbox, conf))
            elif self._local._point_in_any_roi(cx, cy, board_rois, frame):
                board_yolo.append((label, bbox, conf))
            else:
                _, slot_hint = parse_yolo_label(label)
                if slot_hint == "hole":
                    hole_yolo.append((label, bbox, conf))
                elif slot_hint == "board":
                    board_yolo.append((label, bbox, conf))
        return hole_yolo, board_yolo


class ScreenshotStateExtractor:
    """
    Extracts state from a screenshot using configured ROIs and template matching.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.cfg = self._load_config()
        self.templates = None  # populated on first use
        self.rank_templates = None
        self.suit_templates = None
        # Optional: local HOG+SVM full-card classifier (opt-in via config)
        self._card_classifier: CardClassifier | None = None
        nn_cfg = self.cfg.get("vision", {}).get("nn", {})
        if nn_cfg.get("enabled"):
            rank_path = Path(
                nn_cfg.get(
                    "rank_model_path", "vision/models/card_rank_classifier.joblib"
                )
            )
            suit_path = Path(
                nn_cfg.get(
                    "suit_model_path", "vision/models/card_suit_classifier.joblib"
                )
            )
            try:
                self._card_classifier = CardClassifier(
                    str(rank_path),
                    str(suit_path),
                    confidence_threshold=nn_cfg.get("confidence_threshold", 0.5),
                )
                if self._card_classifier.is_ready:
                    print(
                        f"[extract] Full-card classifier enabled: rank={rank_path}, suit={suit_path}"
                    )
                else:
                    print("[extract] Full-card classifier could not load; disabled.")
                    self._card_classifier = None
            except (ImportError, OSError, ValueError, AttributeError) as e:
                print(f"[extract] Full-card classifier requested but unavailable: {e}")

        # Site-specific calibration (optional). Se manca, fallback su rois legacy.
        self.site = self.cfg.get("vision", {}).get("site")
        self.site_cfg = (
            self.cfg.get("sites", {}).get(self.site, {}) if self.site else {}
        )
        self._table_extractor = LocalTableStateExtractor(config_path)

    def _effective_roi(self, key: str):
        """Restituisce la ROI calibrata per il sito, o quella legacy."""
        return self.site_cfg.get(key) or self.cfg.get("vision", {}).get("rois", {}).get(
            key
        )

    def _effective_card_rois(self, key: str) -> list:
        """Restituisce la lista di ROI per carte (hole/board) calibrata o legacy."""
        site_cards = self.site_cfg.get(key)
        if site_cards:
            return site_cards
        return self.cfg.get("vision", {}).get("rois", {}).get(key, [])

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except OSError:
            return {}

    def _ensure_templates_loaded(self) -> None:
        """Lazy-load card templates (real ones if available, else synthetic)."""
        if self.templates is not None:
            return
        vision = self.cfg.get("vision", {})
        tmpl_dir = vision.get("template_dir")
        from ocr_cards import (
            extract_split_templates_from_full,
            generate_card_templates,
            load_rank_templates,
            load_suit_templates,
            load_templates_from_dir,
        )

        if tmpl_dir and Path(tmpl_dir).exists():
            self.templates = load_templates_from_dir(tmpl_dir)
            self.rank_templates = load_rank_templates(tmpl_dir)
            self.suit_templates = load_suit_templates(tmpl_dir)
            auto_rank, auto_suit = extract_split_templates_from_full(self.templates)
            for rank, img in auto_rank.items():
                self.rank_templates[rank] = img
            for suit, img in auto_suit.items():
                self.suit_templates[suit] = img
            missing_ranks = sorted(
                set("A23456789TJQK") - set(self.rank_templates.keys())
            )
            missing_suits = sorted(set("shdc") - set(self.suit_templates.keys()))
            print(
                f"[extract] Auto-extracted {len(auto_rank)} ranks, "
                f"{len(auto_suit)} suits from full templates"
            )
            if missing_ranks:
                print(f"[extract] Missing rank templates: {missing_ranks}")
            if missing_suits:
                print(f"[extract] Missing suit templates: {missing_suits}")
        else:
            self.templates = generate_card_templates()

    def extract(self, frame=None) -> dict:
        """
        If frame is None, captures a fresh screenshot.
        Returns structured table state.
        """
        vision = self.cfg.get("vision", {})
        rois = vision.get("rois", {})

        if not rois:
            raise RuntimeError(
                "No vision ROIs configured. Please set 'vision.rois' in config.yaml"
            )

        from capture import screenshot

        if frame is None:
            frame = screenshot(window_title="Free Poker")

        if frame is None:
            raise RuntimeError("Screenshot failed: window not found")

        print(f"[extract] Screenshot: {frame.shape[1]}x{frame.shape[0]}")

        # Lazy-load templates (prefer real templates if available)
        self._ensure_templates_loaded()

        # Usa ROI calibrati per il sito se disponibili
        hole = [
            self._normalize_card(c)
            for c in self._read_cards(frame, self._effective_card_rois("hole"))
        ]
        board = [
            self._normalize_card(c)
            for c in self._read_cards(frame, self._effective_card_rois("board"))
        ]
        pot = self._read_number(frame, self._effective_roi("pot"))
        to_call = self._read_number(frame, self._effective_roi("to_call"))
        stack = self._read_number(frame, self._effective_roi("stack"))

        position = vision.get("position", "BTN")
        stage = self._infer_stage(board)
        estimated_fields: set[str] = set()
        uncertainty_reasons: list[str] = []
        confidence = {
            "hole": 1.0,
            "cards": 1.0,
            "to_call": 1.0,
            "position": 1.0,
            "stage": 1.0,
        }

        # Stato completo del tavolo (multi-seat, button, stacks avversari)
        table_state = self._table_extractor.extract(frame)
        if table_state.button_seat is not None:
            try:
                layout = SeatLayout(
                    table_type=table_state.table_type_from_config(),
                    button_seat=table_state.button_seat,
                    hero_seat=table_state.hero_seat,
                )
                position = layout.hero_position()
            except (ImportError, ValueError, KeyError, TypeError) as e:
                print(f"[extract] Seat layout failed: {e}")

        # --- Fallback per software che NON mostrano to_call esplicito ---
        # (es. 247freepoker.com: i bottoni dicono solo "Call" senza importo)
        if (to_call is None or to_call == 0) and pot and pot > 0:
            raw_text = self._read_text(frame, rois.get("to_call")).lower()
            bb = self.cfg.get("big_blind", 2)

            # Se l'OCR legge la parola "Call" nel ROI, sappiamo che c'è da pagare
            if "call" in raw_text:
                if stage == "preflop" and position == "SB":
                    to_call = bb - bb // 2
                    print(
                        f"[extract] FALLBACK: OCR vede 'Call' e sei SB preflop -> to_call={to_call}"
                    )
                else:
                    to_call = bb
                    print(
                        f"[extract] FALLBACK: OCR vede 'Call' -> to_call stimato={to_call}"
                    )
                estimated_fields.add("to_call")
                uncertainty_reasons.append("to_call estimated")
                confidence["to_call"] = 0.0
            elif stage == "preflop" and position == "SB" and pot >= bb * 2:
                # Euristica: preflop SB con pot abbastanza grande = dobbiamo completare
                to_call = bb - bb // 2
                print(
                    f"[extract] FALLBACK: Preflop SB con pot={pot} -> to_call={to_call}"
                )
                estimated_fields.add("to_call")
                uncertainty_reasons.append("to_call estimated")
                confidence["to_call"] = 0.0

        # Debug: logga le carte rilevate
        print(
            f"[extract] Hole: {hole} | Board: {board} | Pot: {pot} | ToCall: {to_call}"
        )
        if len(set(hole)) != len(hole):
            print(f"[extract] WARNING: duplicate hole cards detected: {hole}")
        if len(set(board)) != len(board):
            print(f"[extract] WARNING: duplicate board cards detected: {board}")

        return {
            "hole": hole,
            "board": board,
            "pot": pot or 0,
            "to_call": to_call or 0,
            "to_call_source": "estimated"
            if "to_call" in estimated_fields
            else "observed",
            "estimated_fields": sorted(estimated_fields),
            "uncertainty_reasons": uncertainty_reasons,
            "is_uncertain": bool(uncertainty_reasons),
            "confidence": confidence,
            "stack": stack or table_state.hero_stack or vision.get("stack", 1000),
            "position": position,
            "stage": stage,
            "table_type": table_state.table_type,
            "button_seat": table_state.button_seat,
            "num_active": table_state.active_count(),
            "effective_stack": table_state.effective_stack(),
            "opponent_stacks": {
                s.position: s.stack
                for s in table_state.seats
                if s.is_active and not s.is_hero and s.stack is not None
            },
            "seat_positions": {s.index: s.position for s in table_state.seats},
        }

    @staticmethod
    def _normalize_card(card: str) -> str:
        """Normalize card to Rank+lowercase-suit (e.g. 'TD' -> 'Td', '2S' -> '2s')."""
        if not card:
            return card
        if len(card) == 2:
            return card[0].upper() + card[1].lower()
        return card

    @staticmethod
    def _resolve_roi(roi: dict | None, frame: np.ndarray) -> dict | None:
        """Converte un ROI percentuale (rel=True) in assoluto."""
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
                return None
        return roi

    @staticmethod
    def _resolve_rois(rois: list[dict], frame: np.ndarray) -> list[dict]:
        """Converte una lista di ROI percentuali in assoluti."""
        resolved = [ScreenshotStateExtractor._resolve_roi(r, frame) for r in rois if r]
        return [roi for roi in resolved if roi is not None]

    @staticmethod
    def _point_in_any_roi(
        x: float, y: float, rois: list[dict], frame: np.ndarray
    ) -> bool:
        """Return True if (x, y) lies inside any resolved ROI."""
        for roi in ScreenshotStateExtractor._resolve_rois(rois, frame):
            if (
                roi["x"] <= x < roi["x"] + roi["w"]
                and roi["y"] <= y < roi["y"] + roi["h"]
            ):
                return True
        return False

    def _read_cards_with_confidence(
        self, frame, rois: list[dict]
    ) -> tuple[list[str | None], list[float]]:
        """Read cards from ROIs and return per-slot cards + confidence scores."""
        if not rois:
            return [], []
        self._ensure_templates_loaded()
        resolved = self._resolve_rois(rois, frame)
        from ocr_cards import recognize_cards_rois_with_confidence

        templates = self.templates or {}
        rank_templates = self.rank_templates or {}
        suit_templates = self.suit_templates or {}

        cards_scores = recognize_cards_rois_with_confidence(
            frame,
            resolved,
            templates,
            rank_templates=rank_templates,
            suit_templates=suit_templates,
            card_classifier=self._card_classifier,
        )
        cards: list[str | None] = []
        scores: list[float] = []
        for card, score in cards_scores:
            norm = self._normalize_card(card) if isinstance(card, str) else None
            cards.append(norm)
            try:
                scores.append(float(score) if norm is not None else 0.0)
            except (TypeError, ValueError):
                scores.append(0.0)
        return cards, scores

    def _read_cards(self, frame, rois: list[dict]) -> list[str]:
        """Read cards from ROIs and filter out empty slots."""
        cards, _ = self._read_cards_with_confidence(frame, rois)
        return [c for c in cards if c is not None]

    @staticmethod
    def _read_text(frame, roi: dict | None) -> str:
        """Extract raw text from a ROI using Tesseract OCR."""
        roi = ScreenshotStateExtractor._resolve_roi(roi, frame)
        if roi is None:
            return ""
        from capture import crop_roi

        crop = crop_roi(frame, roi["x"], roi["y"], roi["w"], roi["h"])
        if crop is None or crop.size == 0:
            return ""

        try:
            import pytesseract
        except ImportError:
            return ""

        # Resolve Tesseract binary (Windows bundle, macOS Homebrew, or PATH)
        from tesseract_utils import find_tesseract_binary

        pytesseract.pytesseract.tesseract_cmd = find_tesseract_binary()

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        text = pytesseract.image_to_string(
            binary,
            config="--psm 7",
        ).strip()
        return text

    @staticmethod
    def _read_number(frame, roi: dict | None) -> int | None:
        """Extract a numeric value from a ROI using Tesseract OCR."""
        roi = ScreenshotStateExtractor._resolve_roi(roi, frame)
        if roi is None:
            return None
        from capture import crop_roi

        crop = crop_roi(frame, roi["x"], roi["y"], roi["w"], roi["h"])
        if crop is None or crop.size == 0:
            return None

        try:
            import pytesseract
        except ImportError:
            return None

        # Resolve Tesseract binary (Windows bundle, macOS Homebrew, or PATH)
        from tesseract_utils import find_tesseract_binary

        pytesseract.pytesseract.tesseract_cmd = find_tesseract_binary()

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        text = pytesseract.image_to_string(
            binary,
            config="--psm 7 -c tessedit_char_whitelist=0123456789$.,",
        ).strip()

        cleaned = "".join(ch for ch in text if ch.isdigit() or ch in ",.")
        if not cleaned:
            return None
        try:
            return int(cleaned.replace(",", "").replace(".", ""))
        except ValueError:
            return None

    @staticmethod
    def _infer_stage(board: list[str]) -> str:
        n = len(board)
        if n == 0:
            return "preflop"
        elif n == 3:
            return "flop"
        elif n == 4:
            return "turn"
        elif n == 5:
            return "river"
        return "unknown"


class HybridStateExtractor:
    """
    Hybrid extractor: local template OCR for cards/numbers (fast, ~100ms),
    LLM only for position with 60s cache (slow but rarely needed).
    """

    DEFAULT_POSITION_TTL = 60.0  # seconds

    def __init__(
        self,
        config_path: str = "config.yaml",
        position_ttl: float = DEFAULT_POSITION_TTL,
    ):
        self.config_path = Path(config_path)
        self.cfg = self._load_config()
        self._screenshot = ScreenshotStateExtractor(config_path)
        self._llm: LLMVisionExtractor | None = None
        self._position_cache: str = "BTN"
        self._last_position_time: float = 0.0
        self._position_ttl = position_ttl
        self._consistency = ConsistencyChecker(
            big_blind=self.cfg.get("big_blind", 2),
        )

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except OSError:
            return {}

    def _ensure_llm(self) -> LLMVisionExtractor | None:
        if self._llm is None:
            vision_cfg = self.cfg.get("vision", {})
            llm_cfg = vision_cfg.get("llm", {})
            api_key = llm_cfg.get("api_key") or None
            if not api_key:
                print("[hybrid] No LLM api_key configured; position will use cache.")
                return None
            self._llm = LLMVisionExtractor(
                api_key=api_key,
                model=llm_cfg.get("model", "openai/gpt-4o-mini"),
                cooldown_seconds=llm_cfg.get("cooldown_seconds", 5.0),
                timeout=15.0,
                position_override=None,
            )
        return self._llm

    def extract(self, frame=None) -> dict:
        # 1. Local OCR + multi-seat detection (veloce, nessuna chiamata LLM)
        state = self._screenshot.extract(frame)

        local_position = state.get("position")
        local_button = state.get("button_seat")

        # 2. Se il bottone locale non è stato trovato, fallback LLM per la posizione
        now = time.time()
        use_llm = local_button is None or local_position not in (
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
        )

        if use_llm and now - self._last_position_time > self._position_ttl:
            llm = self._ensure_llm()
            if llm is not None:
                try:
                    if frame is None:
                        from capture import screenshot

                        frame = screenshot(window_title="Free Poker")
                    if frame is not None:
                        pos = llm.get_position(frame)
                        if pos:
                            self._position_cache = pos
                            self._last_position_time = now
                            print(f"[hybrid] Position refreshed via LLM: {pos}")
                except (
                    requests.RequestException,
                    KeyError,
                    IndexError,
                    ValueError,
                    RuntimeError,
                ) as e:
                    print(f"[hybrid] Position refresh failed: {e}")

        # Usa posizione locale se valida, altrimenti cache LLM
        if local_position in (
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
            state["position"] = local_position
        else:
            state["position"] = self._position_cache
            print(f"[hybrid] Using cached LLM position: {self._position_cache}")

        # 3. Controlli di coerenza rapidi
        report = self._consistency.check(state)
        print(format_report(report))

        # 4. Fallback LLM completo solo se i controlli falliscono e non abbiamo
        #    già fatto una chiamata recente. Questo preserva la velocità.
        if (
            report.needs_llm_fallback
            and now - self._last_position_time > self._position_ttl
        ):
            llm = self._ensure_llm()
            if llm is not None:
                try:
                    if frame is None:
                        from capture import screenshot

                        frame = screenshot(window_title="Free Poker")
                    if frame is not None:
                        print(
                            "[hybrid] Running full LLM fallback due to consistency failures"
                        )
                        llm_state = llm.extract(frame)
                        # Fonde: carta/numberi locali sono solitamente più affidabili,
                        # ma usiamo LLM per posizione e board se locali sono vuoti.
                        if not state.get("hole") and llm_state.get("hole"):
                            state["hole"] = llm_state["hole"]
                        if not state.get("board") and llm_state.get("board"):
                            state["board"] = llm_state["board"]
                        if state.get("button_seat") is None and llm_state.get(
                            "position"
                        ):
                            state["position"] = llm_state["position"]
                            self._position_cache = llm_state["position"]
                            self._last_position_time = now
                        self._last_position_time = now
                except (
                    requests.RequestException,
                    KeyError,
                    IndexError,
                    ValueError,
                    RuntimeError,
                    OSError,
                ) as e:
                    print(f"[hybrid] Full LLM fallback failed: {e}")

        return state

    def set_position(self, position: str):
        """Manually override the position (e.g. when user knows dealer position)."""
        valid = {"SB", "BB", "BTN", "CO", "MP", "UTG", "UTG+1", "UTG+2", "LJ", "HJ"}
        if position in valid:
            self._position_cache = position
            self._last_position_time = time.time()


def get_extractor(mode: str = "manual", config_path: str = "config.yaml"):
    """Factory."""
    if mode == "manual":
        return ManualStateExtractor()
    if mode in ("screenshot", "webcam"):
        cfg = yaml.safe_load(Path(config_path).read_text()) or {}
        yolo_cfg = cfg.get("vision", {}).get("yolo", {})
        yolo_enabled = bool(yolo_cfg.get("enabled", False))
        model_path = yolo_cfg.get("model_path", "vision/models/poker_yolo.pt")
        if yolo_enabled and Path(model_path).exists():
            return SupervisionStateExtractor(config_path)
        return ScreenshotStateExtractor(config_path)
    if mode == "llm":
        cfg = yaml.safe_load(Path(config_path).read_text()) or {}
        vision_cfg = cfg.get("vision", {})
        llm_cfg = vision_cfg.get("llm", {})
        return LLMVisionExtractor(
            api_key=llm_cfg.get("api_key") or None,
            model=llm_cfg.get("model", "openai/gpt-4o-mini"),
            cooldown_seconds=llm_cfg.get("cooldown_seconds", 1.5),
            position_override=vision_cfg.get("position"),
            position_model=llm_cfg.get("position_model"),
            position_refresh_seconds=llm_cfg.get("position_refresh_seconds", 60.0),
            window_title=vision_cfg.get("window_title"),
            config_path=config_path,
        )
    if mode == "hybrid":
        return HybridStateExtractor(config_path)
    raise ValueError(f"Unknown extractor mode: {mode}")
