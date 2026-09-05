"""Test LLM vision on a saved Goldbet screenshot."""
import sys
from pathlib import Path
import yaml
import cv2

sys.path.insert(0, str(Path(__file__).parent / "vision"))

from llm_vision_extractor import LLMVisionExtractor

# Load API key from config
config_path = Path(__file__).parent / "config.yaml"
cfg = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
llm_cfg = cfg.get("vision", {}).get("llm", {})
api_key = llm_cfg.get("api_key") or None
model = llm_cfg.get("model", "google/gemini-3.1-flash-lite")

if not api_key:
    print("ERROR: No API key found in config.yaml")
    sys.exit(1)

# Load image
img_path = Path.home() / "Desktop" / "screenshotgoldbet.jpeg"
if not img_path.exists():
    img_path = Path.home() / "Desktop" / "screenshotgoldbet2.jpeg"

print(f"Loading image: {img_path}")
frame = cv2.imread(str(img_path))
if frame is None:
    print("ERROR: Could not load image")
    sys.exit(1)

print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")

# Extract state via LLM
extractor = LLMVisionExtractor(
    api_key=api_key,
    model=model,
    cooldown_seconds=0,  # no cooldown for single test
)

try:
    state = extractor.extract(frame)
    print("\n=== LLM VISION RESULT ===")
    for k, v in state.items():
        print(f"  {k}: {v}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
