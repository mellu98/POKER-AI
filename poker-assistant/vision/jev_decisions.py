"""Jev Decisions — layer di decisioni testuali su OpenRouter (/api/alpha/decisions).

Jev (TypeSafe System One) è un modello text-only che ritorna scelte tipizzate
con probabilità e confidence. NON sostituisce il modello vision: fa da auditor
logico sullo stato estratto (coerenza stage/board, plausibilità numeri,
necessità di rilettura focalizzata).

Schema dell'endpoint (reverse-engineered, non ancora nei listati pubblici):

    POST /api/alpha/decisions
    {
      "model": "typesafe/jev-1.13",
      "state": <string | record | array>,
      "questions": {
        "<id>": {
          "type": "choice",
          "question": "...",
          "instructions": "...",
          "criteria": {"<scelta>": "<descrizione>", ...}
        }
      }
    }

    200 -> {"answers": {"<id>": {"choice": "...", "probabilities": {...},
                                 "confidence": 0.0-1.0}}, "usage": {...}}
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "typesafe/jev-1.13"
DEFAULT_API_URL = "https://openrouter.ai/api/alpha/decisions"


def _stage_from_board_count(n: int) -> str:
    """Mappa il numero di carte del board allo stage (0/3/4/5)."""
    return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(n, "invalid")


class JevClient:
    """Client minimale per l'endpoint decisions di OpenRouter."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        api_url: str = DEFAULT_API_URL,
        timeout: float = 15.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.api_url = api_url
        self.timeout = timeout

    def decide(self, state: Any, questions: dict) -> dict | None:
        """Esegue una chiamata decisions. Ritorna ``answers`` o None su errore."""
        if not self.api_key:
            print("[jev] OPENROUTER_API_KEY non configurata: skip")
            return None

        payload = {
            "model": self.model,
            "state": state,
            "questions": questions,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        t0 = time.time()
        try:
            resp = requests.post(
                self.api_url, headers=headers, json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            print(f"[jev] richiesta fallita ({exc})")
            return None
        dt = time.time() - t0

        if not resp.ok:
            print(f"[jev] HTTP {resp.status_code}: {resp.text[:200]}")
            return None

        try:
            data = resp.json()
        except (ValueError, KeyError):
            print(f"[jev] risposta non-JSON: {resp.text[:200]}")
            return None

        answers = data.get("answers")
        if not isinstance(answers, dict) or not answers:
            print(f"[jev] nessuna answer nella risposta: {data}")
            return None

        usage = data.get("usage", {})
        print(
            f"[jev] {dt:.2f}s — {len(answers)}/"
            f"{len(questions)} risposte, cost=${usage.get('cost', 0):.6f}"
        )
        return answers

    # ------------------------------------------------------------------ #
    #  Preset: audit di coerenza dello stato estratto
    # ------------------------------------------------------------------ #

    def consistency_check(self, state: dict) -> dict | None:
        """Audita lo stato estratto: coerenza stage/board, numeri plausibili,
        necessità di rilettura focalizzata delle carte."""
        # stato sintetico e leggibile: Jev non ha bisogno dei campi grezzi
        audit_state = {
            "hole": state.get("hole", []),
            "board": state.get("board", []),
            "board_count": len(state.get("board", [])),
            "reported_stage": state.get("stage"),
            "expected_stage": _stage_from_board_count(len(state.get("board", []))),
            "pot": state.get("pot"),
            "to_call": state.get("to_call"),
            "stack": state.get("stack"),
            "effective_stack": state.get("effective_stack"),
            "button_seat": state.get("button_seat"),
            "unknown_suits": sum("?" in str(c) for c in state.get("hole", [])),
        }

        questions = {
            "stage_consistent": {
                "type": "choice",
                "question": "Is the reported street consistent with the board card count?",
                "instructions": (
                    "0 board cards = preflop, 3 = flop, 4 = turn, 5 = river. "
                    "Compare reported_stage with expected_stage."
                ),
                "criteria": {
                    "yes": "reported_stage matches the board card count",
                    "no": "mismatch between stage and number of board cards",
                },
            },
            "numbers_plausible": {
                "type": "choice",
                "question": "Are pot, to_call and stacks mutually plausible?",
                "instructions": (
                    "Sanity checks: pot >= 0; to_call >= 0; to_call <= effective_stack; "
                    "stack > 0. A to_call larger than the effective stack is impossible."
                ),
                "criteria": {
                    "yes": "all numeric relationships are plausible",
                    "no": "at least one impossible or nonsensical relationship",
                },
            },
            "needs_refocus": {
                "type": "choice",
                "question": "Should the table be re-read with a focused verification pass?",
                "instructions": (
                    "Trigger a re-read when the hole cards contain unknown suits ('?') "
                    "or the numbers look off. Do not trigger for clean reads."
                ),
                "criteria": {
                    "yes": "unknown suits present or suspicious numbers: re-read",
                    "no": "state is clean and consistent: no re-read needed",
                },
            },
        }
        return self.decide(audit_state, questions)


class JevAuditor:
    """Wrapper asincrono: audita periodicamente lo stato senza bloccare il loop.

    Specchia il pattern di _maybe_refresh_position_async dell'estrattore:
    un thread in background aggiorna l'ultimo report, che extract() allega
    allo stato come ``jev_audit`` quando disponibile.
    """

    def __init__(
        self,
        client: JevClient,
        check_interval_seconds: float = 10.0,
    ) -> None:
        self.client = client
        self.check_interval = check_interval_seconds
        self.last_report: dict | None = None
        self.last_check_time: float = 0.0
        self._running = False
        self._lock = threading.Lock()

    def maybe_check_async(self, state: dict) -> None:
        """Lancia il check in background se scaduto l'intervallo."""
        now = time.time()
        with self._lock:
            if self._running:
                return
            if now - self.last_check_time < self.check_interval:
                return
            self._running = True

        snapshot = dict(state)

        def _worker() -> None:
            try:
                report = self.client.consistency_check(snapshot)
                if report is not None:
                    self.last_report = report
                    self.last_check_time = time.time()
            finally:
                with self._lock:
                    self._running = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def attach(self, state: dict) -> dict:
        """Allega l'ultimo report disponibile allo stato (non-blocking)."""
        if self.last_report is not None:
            state["jev_audit"] = self.last_report
        return state
