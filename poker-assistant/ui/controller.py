"""
Controller — ties together Vision, Engine, Equity, and Overlay.

Runs a loop that:
  1. Fetches current table state
  2. Queries the CFR engine for the GTO action
  3. Computes real-time equity
  4. Updates the overlay window
"""

import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import requests
import yaml

# Allow imports from sibling packages
sys.path.insert(0, str(Path(__file__).parent.parent / "engine"))
sys.path.insert(0, str(Path(__file__).parent.parent / "vision"))
sys.path.insert(0, str(Path(__file__).parent))

from assistant_engine import AssistantEngine  # type: ignore[import-not-found]
from confidence_gate import confidence_block_reasons
from equity_service import (  # type: ignore[import-not-found]  # path dinamico via main.py
    calculate_equity,
)
from overlay import PokerOverlay
from state_extractor import get_extractor  # type: ignore[import-not-found]
from temporal_smoother import TemporalSmoother  # type: ignore[import-not-found]


class AssistantController:
    def __init__(
        self,
        mode: str = "manual",
        update_interval: float = 1.5,
        config_path: str = "config.yaml",
    ):
        self.mode = mode
        self.update_interval = update_interval
        self.config_path = config_path
        self.config = self._load_config()
        self.extractor = get_extractor(mode, config_path)
        self.engine = AssistantEngine()
        self.overlay = PokerOverlay()
        self._running = False
        self._thread: threading.Thread | None = None

        # Optional temporal smoothing: protects against single-frame misreads.
        temporal_cfg = self.config.get("vision", {}).get("temporal", {})
        self.temporal_smoother: TemporalSmoother | None = None
        if temporal_cfg.get("enabled"):
            try:
                self.temporal_smoother = TemporalSmoother(
                    window_size=int(temporal_cfg.get("window_size", 5)),
                    agreement_threshold=float(
                        temporal_cfg.get("agreement_threshold", 0.6)
                    ),
                    min_samples=int(temporal_cfg.get("min_samples", 3)),
                )
            except (TypeError, ValueError) as exc:
                print(
                    f"[controller] TemporalSmoother config non valida, disattivato: {exc}"
                )
                self.temporal_smoother = None

    def _load_config(self) -> dict:
        path = Path(self.config_path)
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except OSError:
            return {}

    def start(self):
        """Start the assistant loop in a background thread; overlay runs on main."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.overlay.run()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.overlay.close()

    def _loop(self):
        while self._running:
            try:
                self._tick()
            except (
                requests.RequestException,
                KeyError,
                IndexError,
                ValueError,
                TypeError,
                AttributeError,
                RuntimeError,
                OSError,
                cv2.error,
            ) as e:
                traceback.print_exc()
                self.overlay.update(status=f"Error: {e}")
            time.sleep(self.update_interval)

    def _tick(self):
        # ---- 1. Get state ----
        if self.mode == "manual":
            # In manual mode we expect the user to feed state via a shared queue
            # or we simply poll a static dict for demo purposes.
            # For real usage, manual mode is driven by CLI input in main.py.
            state = self._fetch_manual_state()
        else:
            state = self.extractor.extract()

        if state is None:
            return

        # Temporal smoothing sees every tick so single-frame misreads cannot
        # propagate straight into the engine/overlay.
        if self.temporal_smoother is not None:
            state = self.temporal_smoother.update(state)

        hole = state["hole"]
        board = state["board"]
        pot = state["pot"]
        to_call = state["to_call"]
        position = state["position"]
        stage = state["stage"]

        # FALLBACK heuristic: an inferred amount is never safe to play.
        if to_call == 0 and pot > 0 and stage == "preflop" and position == "SB":
            bb = state.get("big_blind", 2)
            if pot >= 2 * bb:
                to_call = bb - bb // 2  # = 1 for BB=2
                state["to_call"] = to_call
                state["to_call_source"] = "estimated"
                state.setdefault("estimated_fields", []).append("to_call")
                state.setdefault("uncertainty_reasons", []).append("to_call estimated")
                state.setdefault("confidence", {})["to_call"] = 0.0
                state["is_uncertain"] = True
                print(
                    f"[controller] FALLBACK: SB preflop, pot={pot} >= 2*BB -> "
                    f"to_call={to_call}"
                )

        block_reasons = confidence_block_reasons(state)
        if block_reasons:
            self.overlay.update(
                status=f"WAIT | {stage.upper()} | {position} | {block_reasons[0]}",
                hand=" ".join(hole) if hole else "NO CARDS",
                board=" ".join(board),
                equity=0.0,
                action="WAIT",
                sizing="",
            )
            return

        # Guard against empty or partial reads
        if len(hole) != 2:
            self.overlay.update(
                status=f"{stage.upper()} | {position} | Pot {pot}",
                hand="NO CARDS",
                board="",
                equity=0.0,
                action="--",
                sizing="",
            )
            return

        # Guard against duplicate cards (vision misread)
        all_cards = hole + board
        if len(set(all_cards)) != len(all_cards):
            dupes = [c for c in set(all_cards) if all_cards.count(c) > 1]
            self.overlay.update(
                status=f"Error: dup {dupes}",
                hand=" ".join(hole),
                board=" ".join(board),
                equity=0.0,
                action="--",
                sizing="",
            )
            return

        # ---- 2. Equity ----
        equity = calculate_equity(hole, board, n=1000)
        hand_str = " ".join(hole)
        board_str = " ".join(board) if board else ""

        # ---- 3. Engine recommendation ----
        # Build a minimal history for the engine
        history = self._build_history(state)
        rec = self.engine.recommend(
            hole=hole,
            board=board,
            history=history,
            pot=pot,
            stack=state.get("stack", 1000),
            big_blind=state.get("big_blind", 2),
            is_dealer=(position == "BTN"),
            to_call=to_call,
            position=position,
            effective_stack=state.get("effective_stack"),
            num_active=state.get("num_active", 2),
        )

        action = rec["action"]

        # Safety: never fold when we can check for free
        if to_call == 0 and action == "f":
            action = "k"

        # Pretty-print action
        if action.startswith("b"):
            sizing = action
            pretty = f"RAISE {action[1:]}"
        elif action == "c":
            pretty = "CALL"
            sizing = f"Call {to_call}"
        elif action == "f":
            pretty = "FOLD"
            sizing = ""
        elif action == "k":
            pretty = "CHECK"
            sizing = ""
        else:
            pretty = action.upper()
            sizing = ""

        # ---- 4. Update overlay ----
        self.overlay.update(
            equity=equity,
            action=pretty,
            sizing=sizing,
            hand=hand_str,
            board=board_str,
            status=f"{stage.upper()} | {position} | Pot €{pot / 100:.2f}",
        )

    def _build_history(self, state: dict) -> list:
        """
        Convert raw state into a minimal action history for the CFR engine.
        This is a simplified heuristic; a full implementation would track
        every action street-by-street.
        """
        # For now, assume a single pending decision:
        # if to_call > 0, opponent has bet.
        # If to_call == 0, we can check.
        to_call = state.get("to_call", 0)
        if to_call > 0:
            return [f"b{to_call}"]
        return []

    def _fetch_manual_state(self) -> dict | None:
        """Placeholder: in a real CLI flow, this would read from a queue."""
        return getattr(self, "_manual_state", None)

    def set_manual_state(self, state: dict):
        self._manual_state = state


if __name__ == "__main__":
    ctrl = AssistantController(mode="manual")
    # Demo state
    ctrl.set_manual_state(
        {
            "hole": ["As", "Kh"],
            "board": ["Qd", "Jh", "2c"],
            "pot": 120,
            "to_call": 20,
            "position": "BTN",
            "stage": "flop",
            "stack": 980,
            "big_blind": 2,
        }
    )
    ctrl.start()
