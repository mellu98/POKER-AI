"""Optional YOLO-based detector scaffolding.

This module provides an opt-in wrapper around an ultralytics YOLO model with
supervision output formatting. When the model is missing, the dependencies are
not installed, or the configured latency budget is exceeded, the detector
disables itself and returns empty detections so the rest of the application
keeps working unchanged.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Optional dependencies: keep the module importable even if they are missing.
SUPERVISION_AVAILABLE = False
ULTRALYTICS_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import supervision as sv

    SUPERVISION_AVAILABLE = True
except Exception:  # pragma: no cover - dependency may not be installed
    sv = None  # type: ignore[assignment]

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except Exception:  # pragma: no cover - dependency may not be installed
    YOLO = None  # type: ignore[misc, assignment]

try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - dependency may not be installed
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _empty_detections():
    """Return an empty supervision.Detections object or a compatible fallback."""
    if SUPERVISION_AVAILABLE and sv is not None:
        return sv.Detections.empty()
    # Minimal duck-typed fallback when supervision is not installed.
    return _EmptyDetections()


class _EmptyDetections:
    """Duck-typed stand-in for sv.Detections.empty()."""

    def __init__(self):
        self.xyxy = np.empty((0, 4), dtype=np.float32)
        self.confidence = np.empty((0,), dtype=np.float32)
        self.class_id = np.empty((0,), dtype=np.int64)
        self.data = {}

    def __len__(self) -> int:
        return 0

    def __iter__(self):
        return iter([])


def _looks_like_card(label: str) -> bool:
    """Return True when a YOLO label encodes a card value like 'As' or 'Ts'."""
    return bool(_CARD_LABEL_RE.search(label))


_CARD_LABEL_RE = re.compile(r"[AKQJT98765432][shdc]", re.IGNORECASE)


class PokerYOLODetector:
    """Opt-in YOLO detector with graceful degradation.

    Parameters
    ----------
    config:
        Application configuration dictionary. The relevant keys live under
        ``vision.yolo``.
    """

    def __init__(self, config: dict):
        self.config = config
        yolo_cfg = config.get("vision", {}).get("yolo", {})

        self.enabled: bool = bool(yolo_cfg.get("enabled", False))
        self.model_path: str = yolo_cfg.get(
            "model_path", "vision/models/poker_yolo.pt"
        )
        self.confidence_threshold: float = float(
            yolo_cfg.get("confidence_threshold", 0.5)
        )
        self.device: str = self._resolve_device(yolo_cfg.get("device", "auto"))
        self.use_for_cards: bool = bool(yolo_cfg.get("use_for_cards", True))
        self.use_for_numbers: bool = bool(yolo_cfg.get("use_for_numbers", True))
        self.use_for_button: bool = bool(yolo_cfg.get("use_for_button", True))
        self.validation_mode: bool = bool(yolo_cfg.get("validation_mode", True))
        self.inference_size: int = int(yolo_cfg.get("inference_size", 640))
        self.max_latency_ms: float = float(yolo_cfg.get("max_latency_ms", 300.0))

        self._model: Any | None = None
        self._available: bool = False

        if not self.enabled:
            logger.debug("PokerYOLODetector disabled by configuration.")
            return

        if not ULTRALYTICS_AVAILABLE:
            logger.warning(
                "YOLO requested but ultralytics is not installed; detector disabled."
            )
            return

        if not _TORCH_AVAILABLE:
            logger.warning(
                "YOLO requested but torch is not installed; detector disabled."
            )
            return

        model_file = Path(self.model_path)
        if not model_file.exists():
            logger.warning(
                "YOLO requested but model file not found: %s; detector disabled.",
                self.model_path,
            )
            return

        self._available = True

    @property
    def available(self) -> bool:
        """True when the detector is enabled and ready to run inference."""
        return self._available and self._model is not None

    def _ensure_model(self) -> bool:
        """Lazy-load the YOLO model on first use.

        Returns True if a model is available (or was already loaded), False if
        loading failed and the detector should be considered unavailable.
        """
        if not self._available:
            return False
        if self._model is not None:
            return True

        try:
            self._model = YOLO(self.model_path)
            self._model.to(self.device)
            logger.info(
                "Loaded YOLO model from %s on device %s", self.model_path, self.device
            )
            return True
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.warning("Failed to load YOLO model %s: %s", self.model_path, exc)
            self._available = False
            return False

    def _predict(self, frame: np.ndarray):
        """Run YOLO inference and return the ultralytics result object.

        Returns ``(results, elapsed_ms)``. If inference is unavailable or fails,
        returns ``(None, 0.0)`` and disables the detector when appropriate.
        """
        if not self._ensure_model():
            return None, 0.0

        start = time.perf_counter()
        try:
            results = self._model.predict(
                frame,
                imgsz=self.inference_size,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - model runtime errors
            logger.warning("YOLO inference failed: %s", exc)
            return None, 0.0

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms > self.max_latency_ms:
            logger.warning(
                "YOLO inference latency %.1f ms exceeded budget %.1f ms; "
                "disabling detector.",
                elapsed_ms,
                self.max_latency_ms,
            )
            self._available = False
            return None, elapsed_ms

        return results, elapsed_ms

    def detect(self, frame: np.ndarray) -> Any:
        """Run YOLO inference and return supervision-formatted detections.

        Parameters
        ----------
        frame:
            BGR image as a numpy array.

        Returns
        -------
        sv.Detections or compatible empty object when unavailable.
        """
        results, _elapsed_ms = self._predict(frame)
        if results is None:
            return _empty_detections()

        if SUPERVISION_AVAILABLE and sv is not None:
            try:
                return sv.Detections.from_ultralytics(results[0])
            except Exception as exc:  # pragma: no cover - supervision errors
                logger.warning("supervision conversion failed: %s", exc)
                return _empty_detections()

        return _empty_detections()

    def detect_cards(self, frame: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
        """Filter YOLO detections for card classes.

        Returns
        -------
        List of ``(label, bbox, confidence)`` tuples. Empty when unavailable.
        """
        if not self.use_for_cards or not self.available:
            return []

        results, _elapsed_ms = self._predict(frame)
        if results is None:
            return []

        cards: list[tuple[str, np.ndarray, float]] = []
        if SUPERVISION_AVAILABLE and sv is not None:
            try:
                detections = sv.Detections.from_ultralytics(results[0])
                names = getattr(results[0], "names", {})
                for i in range(len(detections)):
                    class_id = detections.class_id[i]
                    label = names.get(class_id, str(class_id))
                    if "card" in label.lower() or _looks_like_card(label):
                        cards.append(
                            (
                                label,
                                detections.xyxy[i],
                                float(detections.confidence[i]),
                            )
                        )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("detect_cards filtering failed: %s", exc)

        return cards

    def detect_button(self, frame: np.ndarray) -> tuple[Optional[int], float]:
        """Placeholder for dealer-button detection.

        Real implementation is planned for Fase 3. Always returns ``(None, 0.0)``
        in Fase 1.
        """
        return None, 0.0

    def detect_number_regions(
        self, frame: np.ndarray
    ) -> list[tuple[str, np.ndarray, float]]:
        """Placeholder for numeric-region detection.

        Returns an empty list in Fase 1; real fusion logic comes in Fase 2.
        """
        return []

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve the compute device for YOLO inference.

        ``auto`` selects ``cuda``, ``mps``, or ``cpu`` in that order. Any other
        value is returned unchanged.
        """
        if device != "auto":
            return device

        if _TORCH_AVAILABLE and torch is not None:
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"

        return "cpu"
