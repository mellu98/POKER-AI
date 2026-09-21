# Poker GTO Assistant

A **Game Theory Optimal** advisor for home Texas Hold'em games. Two front-ends share one philosophy: Monte Carlo equity + preflop charts + (planned) TexasSolver postflop. Available as a Streamlit Python app and a pair of zero-dependency single-file HTML tools.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-7%20py%20%2B%2055%20js-2a6db2.svg)](#tests)

⚠ **For home games only.** Use at any commercial table (casino, poker room, online site) violates rules and will get you banned. The ethics on a home table are yours to decide.

---

## What this is

A single-laptop poker assistant designed for one specific setup: an iPhone on a tripod pointed at the community-cards portion of a home poker table, streaming video over Apple's Continuity Camera into a Mac. The Mac runs a Streamlit UI where the user enters their hole cards, stack sizes, and actions; the assistant returns a GTO-aligned recommendation with EV.

It is built around four small, replaceable engines:

- **Preflop charts** — RFI / 3-bet / squeeze ranges by position and stack depth (hardcoded sets in `poker_agent/preflop.py`; a CSV loader is on the roadmap).
- **Equity** — Monte Carlo equity vs villain range, using `treys` for fast hand evaluation in Python and a pure-JS port in the browser builds.
- **TexasSolver** — postflop GTO solver invoked as a subprocess; returns mixed-strategy frequencies. Wired as a stub (see [Roadmap](#roadmap)).
- **YOLOv8 detector** — board-cards classifier (stub for M3 milestone; manual input works without it).

The adviser routes between them based on street and game state.

### Two front-ends

| File | What it is | Runtime |
|---|---|---|
| `poker_agent/ui.py` | Streamlit web UI for the Python engine | Python 3.11 + Streamlit |
| `poker_assistant.html` | Standalone simple-mode advisor (`adviseSimple`) — phone-friendly, no server | Any modern browser |
| `poker_advanced.html` | Standalone advanced mode (`advise()`) — board texture, SPR, facing-raise logic | Any modern browser |

The two HTML files are self-contained — open them locally, no toolchain. The JS test suites (`tests/test_*.js`) extract the in-page script blocks and run them headless under Node.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   iPhone     │────▶│ YOLOv8 card  │────▶│  Game State  │
│  (Continuity)│     │   detector   │     │   Builder    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                     ┌──────────────┐            │
        manual ────▶ │ Streamlit UI │ ◀──────────┤
        input        └──────┬───────┘            │
                            │                    ▼
                            │            ┌──────────────┐
                            │            │   Adviser    │
                            │            │  ┌─────────┐ │
                            │            │  │ Preflop │ │
                            │◀───────────┤  │ charts  │ │
                            │            │  └─────────┘ │
                            │            │  ┌─────────┐ │
                            │            │  │TexasSolv│ │
                            │            │  │ (post)  │ │
                            │            │  └─────────┘ │
                            │            │  ┌─────────┐ │
                            │            │  │ equity  │ │
                            │            │  └─────────┘ │
                            │            └──────────────┘
                            ▼
                  recommendation + EV
```

## Stack

- **Python 3.11+**
- `streamlit` — UI
- `treys` — hand eval + Monte Carlo equity
- `pydantic` — validated models for `Card` / `Hand` / `Board` / `GameState`
- `opencv-python` + `ultralytics` — YOLOv8 board detection (M3 milestone)
- `texas-solver` — invoked via `subprocess` for postflop GTO

## Repo layout

```
poker_agent/
├── camera.py            # iPhone Continuity Camera capture
├── detector.py          # YOLOv8 board detection (stub for M3)
├── game_state.py        # Card / Hand / Board / GameState models
├── equity.py            # Monte Carlo equity vs range
├── preflop.py           # RFI ranges (hardcoded; CSV loader on roadmap)
├── solver.py            # TexasSolver subprocess wrapper (stub)
├── adviser.py           # routes between preflop charts / solver / equity
└── ui.py                # Streamlit app
poker_assistant.html     # standalone simple-mode browser advisor
poker_advanced.html      # standalone advanced-mode browser advisor
tests/
├── test_smoke.py        # python smoke tests (pytest)
├── test_js_eval.js      # JS hand evaluator + equity sanity
├── test_simple.js       # adviseSimple coverage (poker_assistant.html)
└── test_v2.js           # advise() coverage (poker_advanced.html)
models/                  # YOLO weights (gitignored; created when needed)
```

## Quick start

### Streamlit (Python)

```bash
git clone https://github.com/mkzung/poker-gto-assistant.git
cd poker-gto-assistant
pip install -r requirements.txt
streamlit run poker_agent/ui.py
```

The UI launches at `http://localhost:8501`. Enter:
- Position + stack depth
- Hole cards (two cards from a 52-card deck)
- Board state (preflop / flop / turn / river)
- Action history

The recommendation panel returns: action (fold / call / raise / shove), bet sizing, equity, and EV vs the assumed villain range.

### Browser (zero dependencies)

```bash
# from the repo root, open either file directly:
open poker_assistant.html   # simple mode
open poker_advanced.html    # advanced mode
```

No build step, no server — they ship as single HTML files with inline CSS+JS.

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| **M1** | Streamlit UI + manual input + equity + preflop charts | ✓ MVP |
| **M2** | TexasSolver integration (postflop GTO) | 🟡 stub wired, calibration pending |
| **M3** | YOLOv8 fine-tune on playing cards (~500 labeled frames from a real home table) | 🔴 not started |
| **M4** | iPhone Continuity Camera live capture loop | 🔴 not started |
| **M5** | Hand-history logging + post-session review | 🔴 not started |

## Tests

```bash
# Python side (7 smoke tests)
python -m pytest tests/test_smoke.py

# Browser side — extract <script> blocks and run them headless
node tests/test_js_eval.js   # 5-card eval + equity sanity
node tests/test_simple.js    # adviseSimple coverage (14 tests)
node tests/test_v2.js        # advise() coverage (55 tests)
```

Tests are smoke-level for Python (equity Monte Carlo convergence, chart membership, GameState validation) and behavioural for JS (hand-strength ordering, board texture, SPR buckets, facing-raise advice, end-to-end `advise()`).

## Legal / ethical note

This assistant is built for **friendly home games** where everyone at the table is aware of the setup. It is not designed for, and should not be used at, any commercial poker table (casino, poker room, online site). Doing so violates the rules of every operator and will result in a ban; in many jurisdictions it also crosses into territory that may be legally actionable. The ethics of using it even at home are yours to decide — disclose to your friends.

## License

MIT — see [LICENSE](LICENSE).

---

*Author: [Max Gorbuk](https://github.com/mkzung) — CS founder, Researcher at Stanford GSB Venture Capital Initiative.*
