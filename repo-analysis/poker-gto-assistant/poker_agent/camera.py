"""Camera ingest — iPhone Continuity Camera через OpenCV.

Setup:
  1. iPhone должен быть рядом, на том же Apple ID, со включённым Bluetooth+WiFi.
  2. macOS Sequoia + iOS 17+ — Continuity Camera работает out of the box.
  3. iPhone появится как обычная webcam в системе. cv2.VideoCapture найдёт его по index.
  4. Чтобы понять index — запусти `list_cameras()`.

API:
  cam = Camera(index=1)
  for frame in cam.frames():
      ...
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # noqa: N816


def list_cameras(max_check: int = 5) -> list[int]:
    """Перебрать первые N indexes и вернуть рабочие."""
    if cv2 is None:
        return []
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            available.append(i)
            cap.release()
    return available


class Camera:
    def __init__(self, index: int = 0, fps: int = 10):
        if cv2 is None:
            raise RuntimeError("opencv-python не установлен")
        self.index = index
        self.fps = fps
        self._cap = None

    def open(self):
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Не открылась камера index={self.index}")

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def frames(self) -> Iterator:
        """Yield BGR frames at self.fps."""
        if self._cap is None:
            self.open()
        delay = 1.0 / self.fps
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame
            time.sleep(delay)

    def grab_one(self):
        """Одиночный кадр для UI snapshot."""
        if self._cap is None:
            self.open()
        ok, frame = self._cap.read()
        return frame if ok else None


@contextmanager
def camera(index: int = 0, fps: int = 10):
    cam = Camera(index=index, fps=fps)
    cam.open()
    try:
        yield cam
    finally:
        cam.close()
