# Poker AI Assistant - Handoff Document

**Date:** 2026-01-27
**Status:** Alpha / Integrated

## Overview

This project is a real-time Poker AI Assistant designed to capture a poker table window, detect cards via template matching and OCR, calculate hand strength and equity, and provide strategic advice via a transparent overlay.

The system is modularized into:

1. **Capture**: Finds window, grabs screen, maps regions.
2. **Detection**: Recognizes cards (Suit/Rank) and parses game state (Pot, Bet, Stack).
3. **Strategy**: Evaluates expected value (EV) using Hand Strength, Equity (Monte Carlo), and Pot Odds.
4. **UI**: Displays recommendations on a transparent PyQt5 overlay.

## Environment & Setup

- **Python Version**: 3.10+ recommended.
- **Virtual Environment**: `venv` (located in project root).
- **Dependencies**: Listed in `requirements.txt`. Key libs: `opencv-python`, `numpy`, `mss`, `pytesseract`, `PyQt5`, `pywin32`.
- **Tesseract**: Requires Tesseract OCR installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` (configurable in `src/detection/text_reader.py`).

## Core Components

### 1. Capture System (`src/capture/`)

- `window_finder.py`: Locates "PokerStars" (configurable) window handle.
- `anchor_manager.py`: Finds a static template (anchor) to align all other regions (cards, text).
- **Critical**: Run `tools/calibrate_anchors.py` if the table size/layout changes.

### 2. Detection System (`src/detection/`)

- `card_detector.py`: Uses template matching (`cv2.matchTemplate`) to identify cards in `models/templates/`.
- `text_reader.py`: Wraps Tesseract to read numbers (Stack, Pot).
- `game_state.py`: Aggregates all info into a `GameState` object.

### 3. Strategy Engine (`src/strategy/`)

- `hand_evaluator.py`: Determines hand rank and granular strength.
- `equity_calculator.py`: Monte Carlo sims for Win %.
- `pot_odds.py`: Calculates pot odds and required equity.
- `board_analyzer.py` (NEW): Analyzes board texture (dry/wet/dynamic).
- `preflop_strategy.py` (NEW): GTO-based preflop ranges.
- `postflop_strategy.py` (NEW): Texture-aware postflop decision logic.
- `decision_engine.py`: Orchestrates Preflop/Postflop logic and heuristics.

### 4. UI (`src/ui/`)

- `display_manager.py`: Manages the PyQt5 application. Handles painting of the overlay.
- `overlay.py`: Draws the transparent HUD with new EV and Pot Odds display.

### 5. Utilities & Infrastructure (`src/utils/`)

- `performance.py` (NEW): Tracks frame timing and identifies bottlenecks.
- `session_logger.py` (NEW): Logs all decisions to JSONL for analysis.
- `logger.py`: Centralized logging.
- `config_loader.py`: JSON config management.

## Current Status & Integration

- **`main.py`**: The fully integrated loop. It initializes the UI in the main thread and runs the Game Logic in a generic `QThread`.
- **Verification**: `test_full_system.py` passes all integration checks (imports, classes, window mocking).
- **Strategy Verification**: `test_strategy.py` passes logic checks (Equity thresholds, Hand ID).

## How to Run

### 1. Activate Environment

```bash
.\venv\Scripts\Activate.ps1
```

### 2. Run the Assistant

```bash
python src/main.py
```

- Ensure a poker client (e.g., PokerStars) or a screenshot of one is visible.

- The overlay should appear (transparent w/ HUD box).

### 3. Run Tests

```bash
# Full Integration Test
python test_full_system.py

# Strategy Logic Test
python test_strategy.py
```

## Known Issues / Optimization Opportunities

1. **Anchor Calibration**: The `anchor_config.json` is user-specific. A new user **must** run calibration or the system won't find the table.
2. **Tesseract Path**: Hardcoded in `text_reader.py`. Should be moved to `settings.json`.
3. **Performance**: Monte Carlo simulations (1000 iter) take ~50-100ms. If frame rate drops, consider optimizing `equity_calculator.py` (e.g., lookup tables or vectorization).
4. **OCR Reliability**: Tesseract can be flaky with small fonts. Consider training a custom small-CNN for digits if reliability is low.
5. **Multi-Table**: Currently supports only one active window.

## File Structure

```
project/
├── config/             # JSON configs
├── models/             # Templates (Card images)
├── src/
│   ├── capture/        # Screen & Window logic
│   ├── detection/      # Computer Vision (Cards/Text)
│   ├── strategy/       # Poker Logic (Math/Equity)
│   ├── ui/             # PyQt5 Overlay
│   ├── utils/          # Logger, ConfigLoader
│   └── main.py         # Entry point
├── tools/              # calibration & template creation scripts
├── tests/              # Additional unit tests
├── test_full_system.py # Integration test
└── test_strategy.py    # Strategy test
```

## Next Steps for AI Agent

1. **Refine Ranges**: Adjust GTO ranges based on specific opponent tendencies (Exploitative Strategy).
2. **Machine Learning**: Use `session_logger` data to train a model for opponent hand prediction.
3. **Advanced Overlay**: visualize opponent ranges on the HUD.
4. **Multi-Table Support**: Expand `window_finder` to handle multiple instances.
