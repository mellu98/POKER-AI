"""
Poker Analytical Helper — screenshot / manual analysis with Claude + local math.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageGrab, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from poker_math import (
    format_cards,
    normalize_cards,
    parse_card_tokens,
    summarize_situation,
)

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
HISTORY_DIR = APP_DIR / "hand_history"
HISTORY_FILE = HISTORY_DIR / "poker_hands.json"
DEFAULT_MODEL = "claude-sonnet-5"


def _message_text(message) -> str:
    """Join text blocks from an API response (skips thinking blocks)."""
    parts = []
    for block in message.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    defaults = {
        "api_key": "",
        "bbox": None,
        "model": DEFAULT_MODEL,
        "use_vision": True,
        "hotkey_enabled": True,
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            defaults.update(data)
        except (json.JSONDecodeError, OSError):
            pass
    # Environment override (never written back unless user saves)
    env_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if env_key and not defaults.get("api_key"):
        defaults["api_key"] = env_key
    return defaults


def save_config(config: dict) -> None:
    to_save = {
        "api_key": config.get("api_key", ""),
        "bbox": config.get("bbox"),
        "model": config.get("model", DEFAULT_MODEL),
        "use_vision": config.get("use_vision", True),
        "hotkey_enabled": config.get("hotkey_enabled", True),
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class PokerAnalyzer:
    """Capture / parse / recommend poker spots."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.client: Optional[anthropic.Anthropic] = None
        api_key = (self.config.get("api_key") or "").strip()
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)

        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        self.history_file = HISTORY_FILE
        self.hand_history: list = []
        self.load_history()

    def set_api_key(self, api_key: str) -> None:
        self.config["api_key"] = api_key.strip()
        self.client = anthropic.Anthropic(api_key=self.config["api_key"]) if api_key.strip() else None
        save_config(self.config)

    def load_history(self) -> None:
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.hand_history = data if isinstance(data, list) else []
                # Drop accidental empty entries from older clear bug
                self.hand_history = [h for h in self.hand_history if h and "game_state" in h]
            except (json.JSONDecodeError, OSError):
                self.hand_history = []
        else:
            self.hand_history = []

    def persist_history(self) -> None:
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.hand_history, f, indent=2)
        except OSError as e:
            print(f"Error saving history: {e}")

    def save_hand_to_history(self, result: dict) -> None:
        if not result or "game_state" not in result:
            return
        self.hand_history.append(result)
        self.persist_history()

    def clear_history(self) -> None:
        self.hand_history = []
        self.persist_history()

    def get_history(self, limit: Optional[int] = None) -> list:
        if limit:
            return self.hand_history[-limit:]
        return list(self.hand_history)

    def history_stats(self) -> dict:
        hands = [h for h in self.hand_history if h.get("recommendation")]
        actions: dict[str, int] = {}
        for h in hands:
            action = (h["recommendation"].get("action") or "Unknown").strip().title()
            if "error" in h["recommendation"]:
                action = "Error"
            actions[action] = actions.get(action, 0) + 1
        return {"total": len(hands), "actions": actions}

    def export_history_csv(self, filepath: str) -> None:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Position", "Your Cards", "Community Cards",
                "Pot", "To Call", "Stack", "Round", "Action", "Raise Amount",
                "Made Hand", "Reasoning",
            ])
            for hand in self.hand_history:
                gs = hand.get("game_state", {})
                rec = hand.get("recommendation", {})
                local = hand.get("local_summary") or {}
                made = (local.get("made_hand") or {}).get("category", "")
                writer.writerow([
                    hand.get("timestamp", ""),
                    gs.get("position", ""),
                    ", ".join(gs.get("your_cards") or []),
                    ", ".join(gs.get("community_cards") or []),
                    gs.get("pot_size", ""),
                    gs.get("current_bet", ""),
                    gs.get("your_stack", ""),
                    gs.get("betting_round", ""),
                    rec.get("action", ""),
                    rec.get("raise_amount", ""),
                    made,
                    rec.get("reasoning", ""),
                ])

    def capture_screen(self, bbox=None) -> Image.Image:
        return ImageGrab.grab(bbox=tuple(bbox)) if bbox else ImageGrab.grab()

    def preprocess_image(self, image: Image.Image):
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # Adaptive threshold handles uneven felt lighting better
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
        )
        return cv2.fastNlMeansDenoising(thresh, h=10)

    def extract_text_from_image(self, image) -> str:
        if isinstance(image, Image.Image):
            processed = self.preprocess_image(image)
        else:
            processed = image
        try:
            return pytesseract.image_to_string(processed)
        except Exception as e:
            return f"[OCR unavailable: {e}]"

    def parse_game_state(self, ocr_text: str) -> dict:
        game_state = {
            "pot_size": None,
            "your_stack": None,
            "your_cards": [],
            "community_cards": [],
            "current_bet": None,
            "position": None,
            "players_in_hand": None,
            "betting_round": None,
        }

        pot_match = re.search(r"Pot:?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)", ocr_text, re.I)
        if pot_match:
            game_state["pot_size"] = float(pot_match.group(1).replace(",", ""))

        stack_match = re.search(
            r"(?:Stack|Chips|Balance):?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)", ocr_text, re.I
        )
        if stack_match:
            game_state["your_stack"] = float(stack_match.group(1).replace(",", ""))

        bet_match = re.search(
            r"(?:(?:To\s*)?Call|Bet|Raise):?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)", ocr_text, re.I
        )
        if bet_match:
            game_state["current_bet"] = float(bet_match.group(1).replace(",", ""))

        players_match = re.search(r"(\d+)\s*players?", ocr_text, re.I)
        if players_match:
            game_state["players_in_hand"] = int(players_match.group(1))

        cards = parse_card_tokens(ocr_text)
        if len(cards) >= 2:
            game_state["your_cards"] = cards[:2]
            game_state["community_cards"] = cards[2:7]

        position_map = [
            ("button", "BTN"), ("btn", "BTN"), ("dealer", "BTN"),
            ("small blind", "SB"), ("sb", "SB"),
            ("big blind", "BB"), ("bb", "BB"),
            ("utg+1", "UTG+1"), ("utg", "UTG"),
            ("hijack", "HJ"), ("hj", "HJ"),
            ("cutoff", "CO"), ("co", "CO"),
            ("middle", "MP"), ("mp", "MP"),
        ]
        lower = ocr_text.lower()
        for needle, label in position_map:
            if re.search(rf"\b{re.escape(needle)}\b", lower):
                game_state["position"] = label
                break

        for round_name in ("River", "Turn", "Flop", "Pre-Flop", "Preflop"):
            if round_name.lower().replace("-", "") in lower.replace("-", "").replace(" ", ""):
                game_state["betting_round"] = "Preflop" if "pre" in round_name.lower() else round_name
                break

        if not game_state["betting_round"]:
            n = len(game_state["community_cards"])
            game_state["betting_round"] = {0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"}.get(n)

        return game_state

    def _image_to_base64_png(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        # Shrink huge captures for faster / cheaper vision calls
        max_side = 1600
        img = image.convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        img.save(buf, format="PNG", optimize=True)
        return base64.standard_b64encode(buf.getvalue()).decode("ascii")

    def extract_game_state_with_vision(self, image: Image.Image) -> dict:
        """Use Claude vision to read the table — far more reliable than OCR alone."""
        if not self.client:
            return {"error": "API key not set"}

        b64 = self._image_to_base64_png(image)
        prompt = """You are reading a Texas Hold'em poker client screenshot.
Extract the CURRENT player's situation as JSON only (no markdown), with keys:
{
  "position": "BTN|SB|BB|UTG|UTG+1|MP|HJ|CO|Unknown",
  "your_cards": ["Ah","Kd"],
  "community_cards": ["Jh","6d","2c"],
  "pot_size": 100.0,
  "current_bet": 25.0,
  "your_stack": 200.0,
  "players_in_hand": 3,
  "betting_round": "Preflop|Flop|Turn|River",
  "notes": "brief OCR/visual uncertainty notes"
}
Rules:
- Cards use rank then suit letter: s/h/d/c (e.g. Ah, Td, 9c).
- your_cards = hero hole cards only (usually 2).
- community_cards = board only (0–5).
- Numbers are numeric or null if unknown.
- current_bet = amount hero must call (0 if checked to hero).
"""
        message = self.client.messages.create(
            model=self.config.get("model", DEFAULT_MODEL),
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        response_text = _message_text(message)
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            return {"error": "Vision response was not JSON", "raw": response_text}
        data = json.loads(json_match.group())
        data["your_cards"] = normalize_cards(data.get("your_cards") or [])
        data["community_cards"] = normalize_cards(data.get("community_cards") or [])
        return data

    def get_poker_recommendation(self, game_state: dict, local_summary: Optional[dict] = None) -> dict:
        if not self.client:
            return {"error": "API key not set. Add it in Settings."}

        local_summary = local_summary or summarize_situation(game_state)
        made = local_summary.get("made_hand") or {}
        preflop = local_summary.get("preflop") or {}
        odds = local_summary.get("pot_odds") or {}

        prompt = f"""You are an expert Texas Hold'em cash-game advisor. Be decisive and practical.

Game State:
- Position: {game_state.get('position', 'Unknown')}
- Hero cards: {', '.join(game_state.get('your_cards') or ['Unknown'])}
- Board: {', '.join(game_state.get('community_cards') or ['None'])}
- Pot: {game_state.get('pot_size', 'Unknown')}
- To call: {game_state.get('current_bet', 'Unknown')}
- Hero stack: {game_state.get('your_stack', 'Unknown')}
- Street: {game_state.get('betting_round', 'Unknown')}
- Players in hand: {game_state.get('players_in_hand', 'Unknown')}

Local math (use as support, not gospel):
- Made hand: {made.get('category', 'n/a')} ({made.get('display', '')})
- Preflop tier: {preflop.get('label', 'n/a')}
- Pot odds: {odds.get('ratio', 'n/a')} — {odds.get('note', '')}
- SPR: {local_summary.get('spr', 'n/a')}

Respond with JSON only:
{{
  "action": "Fold" | "Check" | "Call" | "Bet" | "Raise",
  "raise_amount": number or null,
  "sizing_note": "short sizing rationale or null",
  "reasoning": "2-4 sentences",
  "pot_odds": "string",
  "estimated_equity": "string",
  "confidence": "Low" | "Medium" | "High"
}}"""

        try:
            message = self.client.messages.create(
                model=self.config.get("model", DEFAULT_MODEL),
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = _message_text(message)
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"action": "Unknown", "reasoning": response_text, "raw_response": response_text}
        except Exception as e:
            return {"error": f"API call failed: {e}"}

    def analyze_screenshot(self, bbox=None, manual_input=None, image: Optional[Image.Image] = None) -> dict:
        ocr_text = None
        source = "manual"

        if manual_input:
            game_state = dict(manual_input)
            game_state["your_cards"] = normalize_cards(game_state.get("your_cards") or [])
            game_state["community_cards"] = normalize_cards(game_state.get("community_cards") or [])
        else:
            screenshot = image or self.capture_screen(bbox)
            use_vision = self.config.get("use_vision", True) and self.client

            if use_vision:
                source = "vision"
                try:
                    game_state = self.extract_game_state_with_vision(screenshot)
                    if "error" in game_state:
                        vision_err = game_state.get("error")
                        source = "ocr_fallback"
                        ocr_text = self.extract_text_from_image(screenshot)
                        game_state = self.parse_game_state(ocr_text)
                        game_state["vision_error"] = vision_err
                except Exception as e:
                    source = "ocr_fallback"
                    ocr_text = self.extract_text_from_image(screenshot)
                    game_state = self.parse_game_state(ocr_text)
                    game_state["vision_error"] = str(e)
            else:
                source = "ocr"
                ocr_text = self.extract_text_from_image(screenshot)
                game_state = self.parse_game_state(ocr_text)

            if ocr_text:
                game_state["raw_ocr"] = ocr_text

        local_summary = summarize_situation(game_state)
        recommendation = self.get_poker_recommendation(game_state, local_summary)

        result = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "game_state": game_state,
            "local_summary": local_summary,
            "recommendation": recommendation,
        }
        self.save_hand_to_history(result)
        return result


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class PokerAnalyzerGUI:
    """Felt-table themed desktop UI."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Poker Analytical Helper")
        self.root.geometry("1080x780")
        self.root.minsize(900, 640)

        # Felt / chip aesthetic — not purple-default
        self.colors = {
            "bg": "#0c1f17",
            "felt": "#145c38",
            "rail": "#1a3328",
            "fg": "#f2efe6",
            "muted": "#a8b5a8",
            "gold": "#d4a017",
            "cream": "#f5ecd7",
            "chip_red": "#c0392b",
            "chip_blue": "#2c6eab",
            "success": "#3ecf8e",
            "warning": "#e8b84a",
            "error": "#e74c3c",
            "text_bg": "#07140f",
            "button": "#1e4d36",
            "button_hover": "#2a6b4a",
            "raise": "#3ecf8e",
            "call": "#e8b84a",
            "fold": "#e74c3c",
            "check": "#5dade2",
        }

        self.root.configure(bg=self.colors["bg"])
        self.config = load_config()
        self.analyzer = PokerAnalyzer(self.config)
        self.bbox = tuple(self.config["bbox"]) if self.config.get("bbox") else None
        self._analyzing = False
        self._preview_photo = None

        self.setup_styles()
        self.setup_gui()
        self._bind_hotkeys()

        if self.analyzer.client:
            self.status_var.set("Ready — press Ctrl+Shift+A to capture & analyze")
        else:
            self.status_var.set("Add your Anthropic API key in Settings to get started")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TNotebook", background=self.colors["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=self.colors["rail"],
            foreground=self.colors["muted"],
            padding=[18, 10],
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.colors["felt"])],
            foreground=[("selected", self.colors["gold"])],
        )
        style.configure("TFrame", background=self.colors["bg"])
        style.configure(
            "TLabel",
            background=self.colors["bg"],
            foreground=self.colors["fg"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "TButton",
            background=self.colors["button"],
            foreground=self.colors["fg"],
            borderwidth=0,
            focuscolor="none",
            padding=[14, 8],
            font=("Segoe UI", 10, "bold"),
        )
        style.map("TButton", background=[("active", self.colors["button_hover"])])
        style.configure(
            "Accent.TButton",
            background=self.colors["gold"],
            foreground="#1a1200",
        )
        style.map("Accent.TButton", background=[("active", "#e0b020")])
        style.configure(
            "TLabelframe",
            background=self.colors["rail"],
            foreground=self.colors["gold"],
            borderwidth=1,
            relief="flat",
        )
        style.configure(
            "TLabelframe.Label",
            background=self.colors["rail"],
            foreground=self.colors["gold"],
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "TCombobox",
            fieldbackground=self.colors["text_bg"],
            background=self.colors["button"],
            foreground=self.colors["fg"],
        )

    def setup_gui(self):
        header = tk.Frame(self.root, bg=self.colors["felt"], height=64)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="POKER ANALYTICAL HELPER",
            bg=self.colors["felt"],
            fg=self.colors["cream"],
            font=("Georgia", 18, "bold"),
        ).pack(side="left", padx=24, pady=16)

        self.api_badge = tk.Label(
            header,
            text="API ●" if self.analyzer.client else "API ○",
            bg=self.colors["felt"],
            fg=self.colors["success"] if self.analyzer.client else self.colors["muted"],
            font=("Segoe UI", 10, "bold"),
        )
        self.api_badge.pack(side="right", padx=24)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=12)

        self.analysis_tab = tk.Frame(self.notebook, bg=self.colors["bg"])
        self.history_tab = tk.Frame(self.notebook, bg=self.colors["bg"])
        self.settings_tab = tk.Frame(self.notebook, bg=self.colors["bg"])
        self.notebook.add(self.analysis_tab, text="  Analysis  ")
        self.notebook.add(self.history_tab, text="  History  ")
        self.notebook.add(self.settings_tab, text="  Settings  ")

        self.setup_analysis_tab()
        self.setup_history_tab()
        self.setup_settings_tab()

    def setup_analysis_tab(self):
        toolbar = ttk.LabelFrame(self.analysis_tab, text="  Capture  ", padding=12)
        toolbar.pack(fill="x", padx=12, pady=(12, 8))

        btn_row = tk.Frame(toolbar, bg=self.colors["rail"])
        btn_row.pack(fill="x")

        ttk.Button(btn_row, text="Capture & Analyze", command=self.capture_and_analyze, style="Accent.TButton").pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(btn_row, text="Manual Input", command=self.show_manual_input).pack(side="left", padx=8)
        ttk.Button(btn_row, text="Pick Region", command=self.pick_region).pack(side="left", padx=8)

        self.vision_var = tk.BooleanVar(value=bool(self.config.get("use_vision", True)))
        tk.Checkbutton(
            btn_row,
            text="Use Vision (recommended)",
            variable=self.vision_var,
            command=self._toggle_vision,
            bg=self.colors["rail"],
            fg=self.colors["fg"],
            selectcolor=self.colors["text_bg"],
            activebackground=self.colors["rail"],
            activeforeground=self.colors["gold"],
            font=("Segoe UI", 9),
        ).pack(side="right")

        # Split: preview | results
        body = tk.Frame(self.analysis_tab, bg=self.colors["bg"])
        body.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        left = ttk.LabelFrame(body, text="  Local Math  ", padding=10)
        left.pack(side="left", fill="both", padx=(0, 8), expand=False)
        left.configure(width=280)

        self.math_text = tk.Text(
            left,
            width=34,
            height=22,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg=self.colors["text_bg"],
            fg=self.colors["cream"],
            relief="flat",
            padx=10,
            pady=10,
            state="disabled",
        )
        self.math_text.pack(fill="both", expand=True)
        self._set_math_placeholder()

        right = ttk.LabelFrame(body, text="  Recommendation  ", padding=10)
        right.pack(side="left", fill="both", expand=True)

        self.action_banner = tk.Label(
            right,
            text="Waiting for a hand…",
            bg=self.colors["rail"],
            fg=self.colors["muted"],
            font=("Georgia", 16, "bold"),
            pady=12,
        )
        self.action_banner.pack(fill="x", pady=(0, 8))

        self.result_text = scrolledtext.ScrolledText(
            right,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg=self.colors["text_bg"],
            fg=self.colors["fg"],
            insertbackground=self.colors["gold"],
            selectbackground=self.colors["felt"],
            relief="flat",
            padx=12,
            pady=12,
        )
        self.result_text.pack(fill="both", expand=True)
        self._configure_result_tags()

        status_frame = tk.Frame(self.analysis_tab, bg=self.colors["rail"], height=32)
        status_frame.pack(side="bottom", fill="x")
        status_frame.pack_propagate(False)
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            status_frame,
            textvariable=self.status_var,
            bg=self.colors["rail"],
            fg=self.colors["muted"],
            font=("Segoe UI", 9),
            anchor="w",
            padx=14,
        ).pack(fill="both", expand=True)

    def setup_history_tab(self):
        top = tk.Frame(self.history_tab, bg=self.colors["bg"])
        top.pack(fill="x", padx=12, pady=12)

        ttk.Button(top, text="Refresh", command=self.refresh_history).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Export CSV", command=self.export_history).pack(side="left", padx=6)
        ttk.Button(top, text="Clear History", command=self.clear_history).pack(side="left", padx=6)

        self.stats_var = tk.StringVar(value="")
        tk.Label(
            top,
            textvariable=self.stats_var,
            bg=self.colors["bg"],
            fg=self.colors["gold"],
            font=("Segoe UI", 10, "bold"),
        ).pack(side="right")

        hist_frame = ttk.LabelFrame(self.history_tab, text="  Recent Hands  ", padding=10)
        hist_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.history_text = scrolledtext.ScrolledText(
            hist_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg=self.colors["text_bg"],
            fg=self.colors["fg"],
            relief="flat",
            padx=12,
            pady=12,
        )
        self.history_text.pack(fill="both", expand=True)
        self.refresh_history()

    def setup_settings_tab(self):
        api_frame = ttk.LabelFrame(self.settings_tab, text="  API  ", padding=16)
        api_frame.pack(fill="x", padx=12, pady=12)

        inner = tk.Frame(api_frame, bg=self.colors["rail"])
        inner.pack(fill="x")

        tk.Label(inner, text="Anthropic API Key", bg=self.colors["rail"], fg=self.colors["fg"]).grid(
            row=0, column=0, sticky="w", pady=6
        )
        self.api_key_entry = tk.Entry(
            inner,
            width=48,
            show="●",
            bg=self.colors["text_bg"],
            fg=self.colors["fg"],
            insertbackground=self.colors["gold"],
            relief="flat",
            font=("Segoe UI", 10),
        )
        if self.config.get("api_key"):
            self.api_key_entry.insert(0, self.config["api_key"])
        self.api_key_entry.grid(row=0, column=1, padx=10, pady=6)
        ttk.Button(inner, text="Save", command=self.save_api_key, style="Accent.TButton").grid(
            row=0, column=2, padx=6
        )

        tk.Label(
            inner,
            text="Stored in config.json (gitignored). You can also set ANTHROPIC_API_KEY.",
            bg=self.colors["rail"],
            fg=self.colors["muted"],
            font=("Segoe UI", 8),
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 4))

        screen_frame = ttk.LabelFrame(self.settings_tab, text="  Screenshot Region  ", padding=16)
        screen_frame.pack(fill="x", padx=12, pady=(0, 12))

        screen_inner = tk.Frame(screen_frame, bg=self.colors["rail"])
        screen_inner.pack()

        defaults = self.bbox or (0, 0, 1920, 1080)
        labels = [("Left", defaults[0]), ("Top", defaults[1]), ("Right", defaults[2]), ("Bottom", defaults[3])]
        self.coord_entries = {}
        for i, (label, default) in enumerate(labels):
            row, col = divmod(i, 2)
            tk.Label(screen_inner, text=f"{label}:", bg=self.colors["rail"], fg=self.colors["fg"]).grid(
                row=row, column=col * 2, sticky="w", padx=(8, 4), pady=6
            )
            entry = tk.Entry(
                screen_inner,
                width=12,
                bg=self.colors["text_bg"],
                fg=self.colors["fg"],
                insertbackground=self.colors["gold"],
                relief="flat",
            )
            entry.insert(0, str(default))
            entry.grid(row=row, column=col * 2 + 1, padx=(0, 16), pady=6)
            self.coord_entries[label] = entry

        btn_row = tk.Frame(screen_frame, bg=self.colors["rail"])
        btn_row.pack(pady=10)
        ttk.Button(btn_row, text="Full Screen", command=self.use_full_screen).pack(side="left", padx=8)
        ttk.Button(btn_row, text="Save Region", command=self.save_coordinates, style="Accent.TButton").pack(
            side="left", padx=8
        )
        ttk.Button(btn_row, text="Pick on Screen", command=self.pick_region).pack(side="left", padx=8)

        info = ttk.LabelFrame(self.settings_tab, text="  Tips  ", padding=16)
        info.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        tip = (
            "1. Save your API key, then set a capture region around the poker table.\n"
            "2. Prefer Vision mode — it reads cards, pot, and stacks from the screenshot.\n"
            "3. Hotkey: Ctrl+Shift+A captures and analyzes (works from any tab).\n"
            "4. Manual input is best when you already know the exact spot.\n"
            "5. Local math (pot odds, made hand, SPR) appears instantly beside the AI advice."
        )
        tk.Label(
            info,
            text=tip,
            justify="left",
            bg=self.colors["rail"],
            fg=self.colors["fg"],
            font=("Segoe UI", 10),
        ).pack(anchor="w")

    def _bind_hotkeys(self):
        # Tk wants Shift capitalized and the key letter in the case you mean
        self.root.bind_all("<Control-Shift-A>", lambda e: self.capture_and_analyze())

    def _toggle_vision(self):
        self.config["use_vision"] = bool(self.vision_var.get())
        self.analyzer.config["use_vision"] = self.config["use_vision"]
        save_config(self.config)

    def _configure_result_tags(self):
        t = self.result_text
        t.tag_configure("header", foreground=self.colors["gold"], font=("Consolas", 11, "bold"))
        t.tag_configure("label", foreground=self.colors["muted"])
        t.tag_configure("value", foreground=self.colors["cream"])
        t.tag_configure("fold", foreground=self.colors["fold"], font=("Consolas", 12, "bold"))
        t.tag_configure("call", foreground=self.colors["call"], font=("Consolas", 12, "bold"))
        t.tag_configure("raise", foreground=self.colors["raise"], font=("Consolas", 12, "bold"))
        t.tag_configure("check", foreground=self.colors["check"], font=("Consolas", 12, "bold"))
        t.tag_configure("error", foreground=self.colors["error"])
        t.tag_configure("dim", foreground=self.colors["muted"], font=("Consolas", 9))

    def _set_math_placeholder(self):
        self.math_text.configure(state="normal")
        self.math_text.delete("1.0", tk.END)
        self.math_text.insert(
            "1.0",
            "Local readouts appear here after each analysis:\n\n"
            "• Hole / board\n"
            "• Made hand or preflop tier\n"
            "• Pot odds & required equity\n"
            "• Stack-to-pot ratio (SPR)",
        )
        self.math_text.configure(state="disabled")

    def _update_api_badge(self):
        ok = bool(self.analyzer.client)
        self.api_badge.configure(
            text="API ●" if ok else "API ○",
            fg=self.colors["success"] if ok else self.colors["muted"],
        )

    # ----- settings actions -----

    def save_api_key(self):
        key = self.api_key_entry.get().strip()
        if not key:
            messagebox.showwarning("API Key", "Please enter an API key.")
            return
        self.analyzer.set_api_key(key)
        self.config["api_key"] = key
        self._update_api_badge()
        self.status_var.set("API key saved")
        messagebox.showinfo("Saved", "API key saved to config.json")

    def use_full_screen(self):
        for label, value in (("Left", "0"), ("Top", "0"), ("Right", "1920"), ("Bottom", "1080")):
            self.coord_entries[label].delete(0, tk.END)
            self.coord_entries[label].insert(0, value)
        self.bbox = None
        self.config["bbox"] = None
        self.analyzer.config["bbox"] = None
        save_config(self.config)

    def save_coordinates(self):
        try:
            left = int(self.coord_entries["Left"].get())
            top = int(self.coord_entries["Top"].get())
            right = int(self.coord_entries["Right"].get())
            bottom = int(self.coord_entries["Bottom"].get())
            if right <= left or bottom <= top:
                raise ValueError("Right/Bottom must be greater than Left/Top")
            self.bbox = (left, top, right, bottom)
            self.config["bbox"] = list(self.bbox)
            self.analyzer.config["bbox"] = list(self.bbox)
            save_config(self.config)
            messagebox.showinfo("Saved", f"Capture region set to {self.bbox}")
        except ValueError as e:
            messagebox.showerror("Invalid region", str(e))

    def pick_region(self):
        """Fullscreen translucent overlay to drag-select a capture region."""
        overlay = tk.Toplevel(self.root)
        overlay.attributes("-fullscreen", True)
        overlay.attributes("-alpha", 0.3)
        overlay.configure(bg="black")
        overlay.attributes("-topmost", True)

        canvas = tk.Canvas(overlay, cursor="cross", bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        start = {"x": 0, "y": 0}
        rect = {"id": None}

        def on_press(event):
            start["x"], start["y"] = event.x, event.y
            if rect["id"]:
                canvas.delete(rect["id"])
            rect["id"] = canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline="#d4a017", width=2
            )

        def on_drag(event):
            if rect["id"]:
                canvas.coords(rect["id"], start["x"], start["y"], event.x, event.y)

        def on_release(event):
            x1, y1 = start["x"], start["y"]
            x2, y2 = event.x, event.y
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            overlay.destroy()
            if right - left < 20 or bottom - top < 20:
                self.status_var.set("Region too small — try again")
                return
            self.bbox = (left, top, right, bottom)
            self.config["bbox"] = list(self.bbox)
            self.analyzer.config["bbox"] = list(self.bbox)
            save_config(self.config)
            for label, value in zip(
                ("Left", "Top", "Right", "Bottom"), self.bbox
            ):
                if label in self.coord_entries:
                    self.coord_entries[label].delete(0, tk.END)
                    self.coord_entries[label].insert(0, str(value))
            self.status_var.set(f"Region set to {self.bbox}")

        def on_escape(_event):
            overlay.destroy()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        overlay.bind("<Escape>", on_escape)
        overlay.focus_force()

    # ----- analysis -----

    def capture_and_analyze(self):
        if self._analyzing:
            return
        if not self.analyzer.client:
            messagebox.showwarning("API Key", "Set your Anthropic API key in Settings first.")
            self.notebook.select(self.settings_tab)
            return

        self._analyzing = True
        self.status_var.set("Capturing & analyzing…")
        self.action_banner.configure(text="Thinking…", fg=self.colors["gold"], bg=self.colors["rail"])
        self.result_text.delete("1.0", tk.END)
        thread = threading.Thread(target=self._analyze_thread, daemon=True)
        thread.start()

    def _analyze_thread(self):
        try:
            # Brief delay so the hotkey UI / overlay isn't in the shot
            self.root.after(0, lambda: self.root.iconify())
            import time
            time.sleep(0.35)
            result = self.analyzer.analyze_screenshot(bbox=self.bbox)
            self.root.after(0, self.root.deiconify)
            self.root.after(0, self.display_result, result)
        except Exception as e:
            self.root.after(0, self.root.deiconify)
            self.root.after(0, self.show_error, str(e))
        finally:
            self.root.after(0, self._analysis_done)

    def _analysis_done(self):
        self._analyzing = False

    def show_error(self, error_msg: str):
        self.status_var.set("Error")
        self.action_banner.configure(text="Analysis failed", fg=self.colors["error"])
        messagebox.showerror("Error", f"Analysis failed:\n{error_msg}")

    def display_result(self, result: dict):
        self.status_var.set(f"Done · source: {result.get('source', '?')}")
        gs = result["game_state"]
        rec = result["recommendation"]
        local = result.get("local_summary") or summarize_situation(gs)

        self._render_local_math(local, gs)
        self._render_recommendation(gs, rec, result)
        self.refresh_history()

    def _render_local_math(self, local: dict, gs: dict):
        lines = []
        lines.append(f"Hole   {local.get('hole_display') or '—'}")
        lines.append(f"Board  {local.get('board_display') or '—'}")
        lines.append("")
        if local.get("made_hand") and "category" in local["made_hand"]:
            mh = local["made_hand"]
            lines.append(f"Made   {mh['category']}")
            lines.append(f"Best   {mh.get('display', '')}")
        elif local.get("preflop"):
            pf = local["preflop"]
            lines.append(f"Preflop {pf.get('label')} [{pf.get('tier')}]")
        else:
            lines.append("Hand   (need more cards)")
        lines.append("")
        odds = local.get("pot_odds") or {}
        lines.append(f"Odds   {odds.get('ratio', '—')}")
        if odds.get("required_equity_pct") is not None:
            lines.append(f"Need   {odds['required_equity_pct']}% equity")
        if odds.get("note"):
            lines.append(f"       {odds['note']}")
        lines.append("")
        spr = local.get("spr")
        lines.append(f"SPR    {spr if spr is not None else '—'}")
        if gs.get("position"):
            lines.append(f"Pos    {gs.get('position')}")
        if gs.get("betting_round"):
            lines.append(f"Street {gs.get('betting_round')}")

        self.math_text.configure(state="normal")
        self.math_text.delete("1.0", tk.END)
        self.math_text.insert("1.0", "\n".join(lines))
        self.math_text.configure(state="disabled")

    def _action_style(self, action: str):
        a = (action or "").strip().lower()
        if a == "fold":
            return "fold", self.colors["fold"], "FOLD"
        if a in ("call",):
            return "call", self.colors["call"], "CALL"
        if a in ("raise", "bet"):
            return "raise", self.colors["raise"], a.upper()
        if a == "check":
            return "check", self.colors["check"], "CHECK"
        return "value", self.colors["cream"], (action or "—").upper()

    def _render_recommendation(self, gs: dict, rec: dict, result: dict):
        self.result_text.delete("1.0", tk.END)

        if "error" in rec:
            self.action_banner.configure(text="Error", fg=self.colors["error"])
            self.result_text.insert(tk.END, rec["error"] + "\n", "error")
            return

        action = rec.get("action", "Unknown")
        tag, color, label = self._action_style(action)
        banner = label
        if rec.get("raise_amount") is not None and tag == "raise":
            banner = f"{label}  ${rec['raise_amount']}"
        conf = rec.get("confidence")
        if conf:
            banner = f"{banner}   ·   {conf} confidence"
        self.action_banner.configure(text=banner, fg=color, bg=self.colors["rail"])

        def line(label_t, value, value_tag="value"):
            self.result_text.insert(tk.END, f"{label_t:<16}", "label")
            self.result_text.insert(tk.END, f"{value}\n", value_tag)

        self.result_text.insert(tk.END, "GAME STATE\n", "header")
        line("Position", gs.get("position") or "Unknown")
        line("Cards", format_cards(gs.get("your_cards") or []))
        line("Board", format_cards(gs.get("community_cards") or []))
        pot = gs.get("pot_size")
        line("Pot", f"${pot}" if pot is not None else "Unknown")
        call = gs.get("current_bet")
        line("To call", f"${call}" if call is not None else "Unknown")
        stack = gs.get("your_stack")
        line("Stack", f"${stack}" if stack is not None else "Unknown")
        line("Street", gs.get("betting_round") or "Unknown")
        if gs.get("players_in_hand"):
            line("Players", gs.get("players_in_hand"))

        self.result_text.insert(tk.END, "\nADVICE\n", "header")
        self.result_text.insert(tk.END, f"{'Action':<16}", "label")
        self.result_text.insert(tk.END, f"{label}\n", tag)
        if rec.get("raise_amount") is not None:
            line("Size", f"${rec['raise_amount']}")
        if rec.get("sizing_note"):
            line("Sizing", rec["sizing_note"])
        self.result_text.insert(tk.END, "\n")
        self.result_text.insert(tk.END, "Reasoning\n", "header")
        self.result_text.insert(tk.END, f"{rec.get('reasoning', 'N/A')}\n\n", "value")
        if rec.get("pot_odds"):
            line("Pot odds", rec["pot_odds"])
        if rec.get("estimated_equity"):
            line("Equity est.", rec["estimated_equity"])

        self.result_text.insert(tk.END, f"\nsource: {result.get('source', '?')}\n", "dim")
        if gs.get("raw_ocr"):
            self.result_text.insert(tk.END, "\nOCR DEBUG\n", "header")
            self.result_text.insert(tk.END, gs["raw_ocr"][:2000] + "\n", "dim")

    def show_manual_input(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Manual Input")
        dialog.geometry("520x620")
        dialog.configure(bg=self.colors["bg"])
        dialog.transient(self.root)
        dialog.grab_set()

        header = tk.Frame(dialog, bg=self.colors["felt"], height=48)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text="Manual Hand Input",
            bg=self.colors["felt"],
            fg=self.colors["cream"],
            font=("Georgia", 14, "bold"),
        ).pack(pady=12)

        content = tk.Frame(dialog, bg=self.colors["bg"])
        content.pack(fill="both", expand=True, padx=20, pady=16)

        fields = [
            ("Position", "BTN"),
            ("Your cards (Ah Kd)", ""),
            ("Community cards", ""),
            ("Pot size", ""),
            ("Current bet / to call", ""),
            ("Your stack", ""),
            ("Players in hand", ""),
        ]
        entries = []
        for i, (label, default) in enumerate(fields):
            tk.Label(content, text=label, bg=self.colors["bg"], fg=self.colors["fg"], anchor="w").grid(
                row=i, column=0, sticky="w", pady=7, padx=(0, 10)
            )
            entry = tk.Entry(
                content,
                width=28,
                bg=self.colors["text_bg"],
                fg=self.colors["fg"],
                insertbackground=self.colors["gold"],
                relief="flat",
                font=("Segoe UI", 10),
            )
            entry.insert(0, default)
            entry.grid(row=i, column=1, pady=7, sticky="ew")
            entries.append(entry)

        tk.Label(content, text="Betting round", bg=self.colors["bg"], fg=self.colors["fg"]).grid(
            row=len(fields), column=0, sticky="w", pady=7
        )
        round_var = tk.StringVar(value="Flop")
        ttk.Combobox(
            content,
            textvariable=round_var,
            values=["Preflop", "Flop", "Turn", "River"],
            state="readonly",
            width=26,
        ).grid(row=len(fields), column=1, pady=7, sticky="ew")
        content.columnconfigure(1, weight=1)

        def submit():
            if not self.analyzer.client:
                messagebox.showwarning("API Key", "Set your API key in Settings first.")
                return
            try:
                manual_state = {
                    "position": entries[0].get().strip() or None,
                    "your_cards": parse_card_tokens(entries[1].get()),
                    "community_cards": parse_card_tokens(entries[2].get()),
                    "pot_size": float(entries[3].get()) if entries[3].get().strip() else None,
                    "current_bet": float(entries[4].get()) if entries[4].get().strip() else None,
                    "your_stack": float(entries[5].get()) if entries[5].get().strip() else None,
                    "players_in_hand": int(entries[6].get()) if entries[6].get().strip() else None,
                    "betting_round": round_var.get(),
                }
            except ValueError as e:
                messagebox.showerror("Invalid input", str(e))
                return

            dialog.destroy()
            if self._analyzing:
                return
            self._analyzing = True
            self.status_var.set("Analyzing manual hand…")
            self.action_banner.configure(text="Thinking…", fg=self.colors["gold"])

            def worker():
                try:
                    result = self.analyzer.analyze_screenshot(manual_input=manual_state)
                    self.root.after(0, self.display_result, result)
                except Exception as e:
                    self.root.after(0, self.show_error, str(e))
                finally:
                    self.root.after(0, self._analysis_done)

            threading.Thread(target=worker, daemon=True).start()

        btns = tk.Frame(dialog, bg=self.colors["bg"])
        btns.pack(pady=(0, 16))
        ttk.Button(btns, text="Analyze", command=submit, style="Accent.TButton").pack(side="left", padx=8)
        ttk.Button(btns, text="Cancel", command=dialog.destroy).pack(side="left", padx=8)

    # ----- history -----

    def refresh_history(self):
        self.history_text.delete("1.0", tk.END)
        history = self.analyzer.get_history(limit=50)
        stats = self.analyzer.history_stats()
        action_bits = " · ".join(f"{k} {v}" for k, v in sorted(stats["actions"].items()))
        self.stats_var.set(f"{stats['total']} hands" + (f"  |  {action_bits}" if action_bits else ""))

        if not history:
            self.history_text.insert("1.0", "No hands yet. Capture a spot or enter one manually.")
            return

        for i, hand in enumerate(reversed(history), 1):
            gs = hand.get("game_state", {})
            rec = hand.get("recommendation", {})
            local = hand.get("local_summary") or {}
            ts = (hand.get("timestamp") or "")[:19].replace("T", " ")
            num = len(history) - i + 1
            action = rec.get("action", "Unknown")
            made = (local.get("made_hand") or {}).get("category", "")

            block = (
                f"{'─' * 72}\n"
                f"#{num}  {ts}  ·  {hand.get('source', '?')}\n"
                f"  {gs.get('position', '?')}  |  "
                f"{format_cards(gs.get('your_cards') or [])}  |  "
                f"board {format_cards(gs.get('community_cards') or [])}\n"
                f"  pot ${gs.get('pot_size', '?')}  to call ${gs.get('current_bet', '?')}"
                f"  ·  {action}"
            )
            if rec.get("raise_amount") is not None:
                block += f" ${rec['raise_amount']}"
            if made:
                block += f"  ·  {made}"
            reason = (rec.get("reasoning") or "")[:140]
            if reason:
                block += f"\n  {reason}{'…' if len(rec.get('reasoning') or '') > 140 else ''}"
            block += "\n"
            self.history_text.insert(tk.END, block)

    def export_history(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if filepath:
            try:
                self.analyzer.export_history_csv(filepath)
                messagebox.showinfo("Exported", f"Saved to:\n{filepath}")
            except OSError as e:
                messagebox.showerror("Export failed", str(e))

    def clear_history(self):
        if messagebox.askyesno("Clear history", "Delete all saved hands? This cannot be undone."):
            self.analyzer.clear_history()
            self.refresh_history()
            messagebox.showinfo("Cleared", "Hand history cleared.")


def main():
    root = tk.Tk()
    try:
        # Slight DPI awareness on Windows
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    PokerAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
