"""Salva uno screenshot pulito del tavolo per calibrazione."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "vision"))

from capture import screenshot
import cv2

frame = screenshot(window_title="Free Poker")
if frame is None:
    print("Errore: finestra non trovata")
    sys.exit(1)

print(f"Dimensione originale: {frame.shape[1]}x{frame.shape[0]}")

# Ridimensiona a 1999x1249 come fa state_extractor
if frame.shape[1] != 1999 or frame.shape[0] != 1249:
    frame = cv2.resize(frame, (1999, 1249), interpolation=cv2.INTER_LINEAR)
    print(f"Ridimensionato a: {frame.shape[1]}x{frame.shape[0]}")

cv2.imwrite("calib_frame.png", frame)
print("Salvato calib_frame.png")
