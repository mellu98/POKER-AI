"""YOLOv8 card detector — STUB на M1.

Pipeline (M3):
  1. Скачать предобученную модель на playing cards (52 классов).
     Хорошие источники:
       - https://github.com/TeogopK/Playing-Cards-Object-Detection
       - https://github.com/cadyze/card-vision
     Или fine-tune на своём столе (~500 размеченных кадров через Roboflow).
  2. Положить веса в models/cards_yolov8.pt.
  3. Detector.detect(frame) → [DetectedCard(code='Ah', conf=0.97, bbox=...), ...].
  4. Adapter в game_state: фильтровать по conf > 0.5, dedupe, сортировать.

Сейчас: API-stub, чтобы остальной код компилировался и тестировался.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .game_state import Card

DEFAULT_MODEL_PATH = Path("models/cards_yolov8.pt")


@dataclass
class DetectedCard:
    card: Card
    confidence: float
    bbox: tuple[int, int, int, int]  # x1,y1,x2,y2


class CardDetector:
    """YOLOv8 wrapper. STUB: реальная инициализация в M3."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, conf_threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self._model = None

    @property
    def is_ready(self) -> bool:
        return self.model_path.exists()

    def load(self):
        """TODO M3: from ultralytics import YOLO; self._model = YOLO(self.model_path)."""
        if not self.is_ready:
            raise FileNotFoundError(
                f"Веса не найдены: {self.model_path}\n"
                "См. поэтапные шаги в poker_agent/detector.py docstring (M3)."
            )
        # from ultralytics import YOLO
        # self._model = YOLO(str(self.model_path))
        raise NotImplementedError("M3: подключи ultralytics YOLO здесь")

    def detect(self, frame) -> list[DetectedCard]:
        """Прогнать фрейм через модель и вернуть карты выше conf_threshold."""
        if self._model is None:
            self.load()
        # results = self._model(frame, conf=self.conf_threshold)[0]
        # return [DetectedCard(...) for box in results.boxes]
        raise NotImplementedError("M3")


def board_from_detections(dets: list[DetectedCard]) -> list[Card]:
    """Извлечь упорядоченный борд (3/4/5) из набора детекций.

    Эвристика M3: сортируем по x-координате bbox-а (слева→направо),
    дедуплицируем по коду карты, берём top-N с наибольшей confidence.
    """
    if not dets:
        return []
    # dedupe по card.code, оставляем макс conf
    by_code: dict[str, DetectedCard] = {}
    for d in dets:
        prev = by_code.get(d.card.code)
        if prev is None or d.confidence > prev.confidence:
            by_code[d.card.code] = d
    sorted_dets = sorted(by_code.values(), key=lambda d: d.bbox[0])
    return [d.card for d in sorted_dets[:5]]
