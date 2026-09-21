"""Modello di stato tipizzato e validato per la pipeline poker.

Sostituisce il dict informale che viaggiava tra estrattore → controller →
engine con una dataclass congelata che si auto-valida alla costruzione:

- carte parse/validate via PokerKit (52-card deck, no duplicati)
- street coerente col numero di carte del board (0/3/4/5)
- sanity monetaria (pot >= 0, to_call <= effective_stack)
- azioni legali calcolate dal contesto economico
- fingerprint() per la macchina a stati (stabilità tra frame)
- to_dict() per retrocompatibilità con controller/overlay/engine esistenti

Ispirato a poker-vision-lab (MIT, shoujikes-eng) — adattato al nostro
output LLM (centesimi, non BB) e ai nostri campi.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from pokerkit import Card


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    WAIT = "wait"


_BOARD_COUNT_TO_STREET = {
    0: Street.PREFLOP,
    3: Street.FLOP,
    4: Street.TURN,
    5: Street.RIVER,
}


def _parse_card(code: str) -> str | None:
    """Parse e valida una carta ('As', 'Td') via PokerKit → notazione canonica."""
    if not isinstance(code, str) or len(code) != 2:
        return None
    try:
        cards = list(Card.parse(code))
    except (ValueError, TypeError, KeyError, IndexError):
        # carta malformata ('Zx', 'A', '') → PokerKit la rifiuta
        return None
    if len(cards) != 1:
        return None
    return repr(cards[0])


def _valid_cards_or_none(cards: list[str]) -> list[str]:
    """Valida una lista di carte: parse + nessun duplicato. [] se invalida."""
    parsed = []
    seen: set[str] = set()
    for c in cards:
        p = _parse_card(c)
        if p is None or p in seen:
            return []
        seen.add(p)
        parsed.append(p)
    return parsed


def _street_from_board(board: list[str]) -> Street:
    return _BOARD_COUNT_TO_STREET.get(len(board), Street.PREFLOP)


@dataclass(frozen=True)
class PokerState:
    """Stato del tavolo validato. Costruire con from_raw_dict()."""

    # --- carte (notazione 'As', 'Td', validate PokerKit) ---
    hero_cards: tuple[str, ...] = ()
    board: tuple[str, ...] = ()

    # --- contesto economico (centesimi) ---
    pot: int = 0
    to_call: int = 0
    hero_stack: int = 0
    effective_stack: int = 0
    big_blind: int = 2

    # --- contesto posizionale ---
    position: str = "BTN"
    button_seat: int | None = None
    num_active: int = 2

    # --- metadati ---
    street: Street = Street.PREFLOP
    confidence: float = 1.0
    field_confidence: dict[str, float] = field(default_factory=dict)
    validation_errors: tuple[str, ...] = ()
    captured_at: float = field(default_factory=time.time)
    raw: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    #  Costruzione
    # ------------------------------------------------------------------ #

    @classmethod
    def from_raw_dict(cls, raw: dict[str, Any], big_blind: int = 2) -> PokerState:
        """Costruisce uno stato validato dal dict prodotto dall'estrattore.

        Mai lancia: gli errori finiscono in validation_errors e i campi
        invalidati vengono degradati (carte → (), numeri → clamp).
        """
        errors: list[str] = []

        hero = _valid_cards_or_none(list(raw.get("hole", [])))
        if raw.get("hole") and not hero:
            errors.append(f"hero_cards invalide: {raw.get('hole')}")

        board = _valid_cards_or_none(list(raw.get("board", [])))
        if raw.get("board") and not board:
            errors.append(f"board invalido: {raw.get('board')}")

        # cross-duplicati (carta in hole e board insieme è impossibile)
        if hero and board and set(hero) & set(board):
            shared = sorted(set(hero) & set(board))
            errors.append(f"carte duplicate hole/board: {shared}")
            board = tuple(c for c in board if c not in set(hero))

        # numeri: interi sicuri con clamp
        def _int_field(name: str, default: int) -> int:
            v = raw.get(name, default)
            try:
                v = int(v)
            except (TypeError, ValueError):
                errors.append(f"{name} non numerico: {v!r}")
                return default
            if v < 0:
                errors.append(f"{name} negativo: {v}")
                return 0
            return v

        pot = _int_field("pot", 0)
        to_call = _int_field("to_call", 0)
        stack = _int_field("stack", 0)
        eff = _int_field("effective_stack", stack or 1000)
        bb = _int_field("big_blind", big_blind) or big_blind

        # clamp monetario: mai callare oltre lo stack effettivo
        if to_call > eff > 0:
            errors.append(f"to_call {to_call} > effective_stack {eff}: clamp")
            to_call = eff

        if stack > 0 and to_call > stack:
            errors.append(f"to_call {to_call} > stack {stack}: clamp")
            to_call = stack

        street = _street_from_board(list(board))
        reported = raw.get("stage")
        if reported and reported != street.value and len(board) in (0, 3, 4, 5):
            errors.append(
                f"stage segnalato {reported!r} ma board={len(board)} → {street.value}"
            )

        btn = raw.get("button_seat")
        if btn is not None and not isinstance(btn, int):
            errors.append(f"button_seat non int: {btn!r}")
            btn = None

        num_active = _int_field("num_active", 2) or 2

        conf = 1.0
        try:
            conf = float(raw.get("confidence", 1.0))
        except (TypeError, ValueError):
            errors.append("confidence non numerica")

        return cls(
            hero_cards=tuple(hero),
            board=tuple(board),
            pot=pot,
            to_call=to_call,
            hero_stack=stack,
            effective_stack=eff,
            big_blind=bb,
            position=str(raw.get("position", "BTN")),
            button_seat=btn,
            num_active=max(2, num_active),
            street=street,
            confidence=max(0.0, min(1.0, conf)),
            validation_errors=tuple(errors),
            raw=dict(raw),
        )

    # ------------------------------------------------------------------ #
    #  Proprietà derivate
    # ------------------------------------------------------------------ #

    @property
    def has_hand(self) -> bool:
        """True se l'hero ha 2 carte valide in mano."""
        return len(self.hero_cards) == 2

    @property
    def is_valid(self) -> bool:
        """True se non ci sono errori bloccanti e c'è una mano da giocare."""
        return self.has_hand and not self._blocking_errors

    @property
    def _blocking_errors(self) -> tuple[str, ...]:
        return tuple(
            e for e in self.validation_errors if "invalid" in e or "duplicate" in e
        )

    @property
    def legal_actions(self) -> tuple[ActionType, ...]:
        """Azioni legali calcolate dal contesto economico (non da PokerKit
        full-state: noi non abbiamo la storia completa delle azioni)."""
        if not self.has_hand:
            return (ActionType.WAIT,)
        if self.to_call > 0:
            actions: list[ActionType] = [ActionType.FOLD, ActionType.CALL]
            if self.effective_stack > self.to_call:
                actions.append(ActionType.RAISE)
            return tuple(actions)
        actions = [ActionType.CHECK]
        if self.effective_stack > self.big_blind:
            actions.append(ActionType.BET)
        return tuple(actions)

    @property
    def pot_bb(self) -> float:
        return round(self.pot / self.big_blind, 2) if self.big_blind else 0.0

    @property
    def effective_stack_bb(self) -> float:
        return (
            round(self.effective_stack / self.big_blind, 2) if self.big_blind else 0.0
        )

    @property
    def spr(self) -> float:
        """Stack-to-Pot Ratio: guida il sizing postflop."""
        return round(self.effective_stack / self.pot, 2) if self.pot > 0 else 99.0

    @property
    def hero_cards_str(self) -> str:
        return " ".join(self.hero_cards)

    @property
    def board_str(self) -> str:
        return " ".join(self.board)

    def fingerprint(self) -> tuple[Any, ...]:
        """Identità percettiva dello stato: per la FSM (stabilità tra frame)."""
        return (
            self.hero_cards,
            self.board,
            self.street,
            self.position,
            self.button_seat,
            self.num_active,
            self.pot,
            self.to_call,
            self.effective_stack,
        )

    # ------------------------------------------------------------------ #
    #  Retrocompatibilità: il dict che controller/overlay/engine si aspettano
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.raw)
        d.update(
            {
                "hole": list(self.hero_cards),
                "board": list(self.board),
                "stage": self.street.value,
                "pot": self.pot,
                "to_call": self.to_call,
                "stack": self.hero_stack,
                "effective_stack": self.effective_stack,
                "big_blind": self.big_blind,
                "position": self.position,
                "button_seat": self.button_seat,
                "num_active": self.num_active,
                # confidence_gate si aspetta un dict per campo:
                # il valore complessivo valido per tutti i campi validati
                "confidence": {
                    "hole": self.confidence,
                    "cards": self.confidence,
                    "to_call": self.confidence,
                    "position": self.confidence,
                    "stage": self.confidence,
                },
                "legal_actions": [a.value for a in self.legal_actions],
                "spr": self.spr,
                "pot_bb": self.pot_bb,
                "validation_errors": list(self.validation_errors),
                "is_valid": self.is_valid,
            }
        )
        return d

    def with_updates(self, **changes: Any) -> PokerState:
        """Copia con modifiche (le dataclass frozen si aggiornano così)."""
        return replace(self, **changes)
