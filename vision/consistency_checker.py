"""
consistency_checker.py — Controlli rapidi di coerenza sullo stato estratto.

Scopo: decidere se lo stato locale è affidabile o se serve una verifica LLM.
I controlli devono essere velocissimi (solo logica, nessuna chiamata esterna).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ConsistencyReport:
    is_valid: bool
    reasons: List[str] = field(default_factory=list)
    needs_llm_fallback: bool = False


class ConsistencyChecker:
    """Controlli di coerenza per lo stato del tavolo."""

    def __init__(
        self,
        big_blind: int = 2,
        max_stack_variance_ratio: float = 0.20,
    ):
        self.big_blind = big_blind
        self.max_stack_variance_ratio = max_stack_variance_ratio

    def check(self, state: dict) -> ConsistencyReport:
        reasons: List[str] = []

        # 1. Carte obbligatorie
        hole = state.get("hole", [])
        if len(hole) != 2:
            reasons.append(f"hole cards count != 2: {len(hole)}")

        # 2. Board coerente con stage
        board = state.get("board", [])
        stage = state.get("stage", "preflop")
        expected_board = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
        if expected_board.get(stage) != len(board):
            reasons.append(f"board length {len(board)} inconsistent with stage {stage}")

        # 3. Duplicati
        all_cards = list(hole) + list(board)
        if len(all_cards) != len(set(all_cards)):
            reasons.append("duplicate cards detected")

        # 4. Pot e to_call non negativi
        pot = state.get("pot", 0)
        to_call = state.get("to_call", 0)
        if pot < 0:
            reasons.append("pot is negative")
        if to_call < 0:
            reasons.append("to_call is negative")

        # 5. Preflop: to_call=0 quando non sei in BB è sospetto se il pot è > 0
        position = state.get("position", "BTN")
        if stage == "preflop" and to_call == 0 and pot > 0 and position not in ("BB", "SB"):
            reasons.append("preflop to_call=0 but not blind and pot > 0")

        # 6. Numero giocatori attivi plausibile
        num_active = state.get("num_active", 0)
        if num_active < 2:
            reasons.append(f"only {num_active} active players")

        # 7. Stack hero: confronto tra ROI generico e stack seat
        hero_stack = state.get("stack")
        effective_stack = state.get("effective_stack")
        opponent_stacks = state.get("opponent_stacks", {})

        if hero_stack is not None and effective_stack is not None:
            if effective_stack > 0:
                ratio = abs(hero_stack - effective_stack) / effective_stack
                if ratio > self.max_stack_variance_ratio:
                    reasons.append(
                        f"hero stack {hero_stack} differs from effective {effective_stack} "
                        f"by {ratio:.0%}"
                    )

        # 8. Se manca il button seat, è un warning ma non blocca (fallback LLM)
        button_seat = state.get("button_seat")
        if button_seat is None:
            reasons.append("dealer button not detected locally")

        needs_llm = bool(reasons)
        # Alcuni reason sono warning leggeri: non richiedono LLM
        light_reasons = {"dealer button not detected locally"}
        if set(reasons) <= light_reasons:
            needs_llm = False

        return ConsistencyReport(
            is_valid=not reasons,
            reasons=reasons,
            needs_llm_fallback=needs_llm,
        )


def format_report(report: ConsistencyReport) -> str:
    if report.is_valid:
        return "[consistency] OK"
    reasons = " | ".join(report.reasons)
    flag = " [LLM FALLBACK]" if report.needs_llm_fallback else ""
    return f"[consistency] FAIL: {reasons}{flag}"


if __name__ == "__main__":
    checker = ConsistencyChecker(big_blind=2)
    sample = {
        "hole": ["As", "Kh"],
        "board": [],
        "pot": 0,
        "to_call": 0,
        "stack": 1000,
        "position": "BB",
        "stage": "preflop",
        "num_active": 6,
        "button_seat": 2,
        "effective_stack": 1000,
    }
    print(format_report(checker.check(sample)))
