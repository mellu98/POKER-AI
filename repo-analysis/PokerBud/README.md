# Poker Analytical Helper

Desktop Texas Hold'em advisor that reads your table (screenshot or manual input), runs local poker math, and asks Claude for a clear Fold / Call / Raise recommendation.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Vision capture** — Claude reads cards, pot, stacks, and street from a screenshot (OCR fallback available)
- **Local math panel** — made hand / preflop tier, pot odds, required equity, SPR
- **Manual input** — enter a spot by hand when you already know the details
- **Region picker** — drag-select the poker table area to capture
- **Hand history** — auto-save sessions, stats, CSV export
- **Hotkey** — `Ctrl+Shift+A` to capture and analyze

## Screenshots / workflow

1. Save your Anthropic API key in **Settings**
2. Set a capture region around the table (**Pick on Screen**)
3. Open **Analysis** → **Capture & Analyze** (or press `Ctrl+Shift+A`)
4. Read the local math panel + AI recommendation side by side

## Requirements

- Python 3.10+
- Windows (uses `ImageGrab` for screen capture)
- [Anthropic API key](https://console.anthropic.com/)
- Optional: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) if you disable Vision and rely on OCR

## Install

```bash
git clone https://github.com/Daniel-Xiao-Wang/poker-analytical-helper.git
cd poker-analytical-helper
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python poker_helper.py
```

You can also set the API key via environment variable instead of the Settings UI:

```bash
set ANTHROPIC_API_KEY=sk-ant-...
python poker_helper.py
```

API key and capture region are stored in `config.json` (gitignored).

## Project layout

```
poker_helper.py      # GUI + screenshot / Claude pipeline
poker_math.py        # Local hand strength, pot odds, SPR
test_poker_math.py   # Sanity checks for poker math
hand_history/        # Saved hand JSON
requirements.txt
```

## Tests

```bash
python test_poker_math.py
```

## Privacy & fair play

- Screenshots and hand data stay on your machine unless you send them to the Anthropic API for analysis.
- Use this as a **study / review tool**. Using real-time assistance on real-money sites may violate that site's terms of service — know the rules of any room you play.

## License

MIT — see [LICENSE](LICENSE).
