"""Game state models — карты, рука, борд, игроки, текущая раздача."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# ---------- Constants ----------

RANKS = "23456789TJQKA"
SUITS = "shdc"  # spades, hearts, diamonds, clubs


# ---------- Card ----------


class Card(BaseModel):
    """Одна карта. Каноническая запись 'As', 'Td', '7c'."""

    code: str  # "As", "Td", "7h"

    @field_validator("code")
    @classmethod
    def _validate(cls, v: str) -> str:
        if len(v) != 2:
            raise ValueError(f"Card code must be 2 chars: {v}")
        rank, suit = v[0].upper(), v[1].lower()
        if rank not in RANKS:
            raise ValueError(f"Invalid rank '{rank}', expected one of {RANKS}")
        if suit not in SUITS:
            raise ValueError(f"Invalid suit '{suit}', expected one of {SUITS}")
        return rank + suit

    @property
    def rank(self) -> str:
        return self.code[0]

    @property
    def suit(self) -> str:
        return self.code[1]

    def __str__(self) -> str:
        return self.code

    def __hash__(self) -> int:
        return hash(self.code)


# ---------- Position ----------


class Position(str, Enum):
    """6-max позиции. Для 9-max добавь UTG1/UTG2/MP."""

    UTG = "UTG"
    MP = "MP"
    CO = "CO"
    BTN = "BTN"
    SB = "SB"
    BB = "BB"


# ---------- Street ----------


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


# ---------- Action ----------


class ActionKind(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALLIN = "allin"


class Action(BaseModel):
    player: Position
    kind: ActionKind
    amount_bb: float = 0.0  # bet/raise size in big blinds


# ---------- Player ----------


class Player(BaseModel):
    position: Position
    stack_bb: float = 100.0
    in_hand: bool = True
    is_hero: bool = False


# ---------- Game state ----------


class GameState(BaseModel):
    """Полное состояние раздачи на момент решения hero."""

    hero_cards: tuple[Card, Card]
    board: list[Card] = Field(default_factory=list)  # 0/3/4/5 карт
    players: list[Player]
    pot_bb: float = 1.5  # SB+BB по умолчанию
    to_call_bb: float = 0.0
    action_history: list[Action] = Field(default_factory=list)
    effective_stack_bb: float = 100.0

    @property
    def hero(self) -> Player:
        return next(p for p in self.players if p.is_hero)

    @property
    def street(self) -> Street:
        n = len(self.board)
        if n == 0:
            return Street.PREFLOP
        if n == 3:
            return Street.FLOP
        if n == 4:
            return Street.TURN
        if n == 5:
            return Street.RIVER
        raise ValueError(f"Invalid board length: {n}")

    @property
    def num_players_in_hand(self) -> int:
        return sum(1 for p in self.players if p.in_hand)

    @property
    def pot_odds(self) -> float:
        """% from pot needed to call (0..1). 0 if nothing to call."""
        if self.to_call_bb <= 0:
            return 0.0
        return self.to_call_bb / (self.pot_bb + self.to_call_bb)


# ---------- Helpers ----------


def parse_cards(s: str) -> list[Card]:
    """'AsKd' / 'As Kd' / 'AsKd 7h2c' → [Card, ...]."""
    cleaned = s.replace(" ", "").replace(",", "")
    if len(cleaned) % 2 != 0:
        raise ValueError(f"Card string length must be even: {s!r}")
    return [Card(code=cleaned[i : i + 2]) for i in range(0, len(cleaned), 2)]


def cards_to_str(cards: list[Card]) -> str:
    return " ".join(str(c) for c in cards)
