"""
seat_layout.py — Mappatura posizioni poker relative al dealer button.

Per tavoli 6-max e 9-max. Le posizioni sono calcolate in senso orario
a partire dal dealer button (BTN). Il modulo non dipende dal sito:
riceve solo:
  - table_type: "6max" | "9max"
  - button_seat: indice del seat che ha il bottone
  - hero_seat: indice del seat dell'eroe
  - seats: lista degli indici seat attivi (opzionale)

Esempio 9max con button al seat 3:
  seat 3 -> BTN
  seat 4 -> SB
  seat 5 -> BB
  seat 6 -> UTG
  seat 7 -> UTG+1
  seat 8 -> UTG+2
  seat 0 -> LJ
  seat 1 -> HJ
  seat 2 -> CO
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


# Posizioni in ordine orario a partire dal dealer button.
_POSITIONS_6MAX = ["BTN", "SB", "BB", "UTG", "HJ", "CO"]
_POSITIONS_9MAX = ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"]

# Nota: la lista 9max è ruotata in modo che BTN sia a offset 0 quando
# calcoliamo (i - button) % n. Equivalentemente:
_9MAX_CLOCKWISE = ["BTN", "SB", "BB", "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO"]

_VALID_POSITIONS = set(_POSITIONS_9MAX + _POSITIONS_6MAX)


@dataclass(frozen=True)
class SeatInfo:
    index: int
    position: str
    is_hero: bool
    is_button: bool


class SeatLayout:
    """Layout del tavolo con mappatura posizioni dal dealer button."""

    def __init__(self, table_type: str, button_seat: int, hero_seat: int):
        if table_type not in ("6max", "9max"):
            raise ValueError(f"Unsupported table_type: {table_type}")
        self.table_type = table_type
        self.max_seats = 6 if table_type == "6max" else 9
        self.button_seat = self._normalize(button_seat)
        self.hero_seat = self._normalize(hero_seat)
        self._order = _POSITIONS_6MAX if table_type == "6max" else _9MAX_CLOCKWISE

    def _normalize(self, seat: int) -> int:
        return int(seat) % self.max_seats

    def position_for(self, seat: int) -> str:
        """Restituisce la posizione poker del seat dato."""
        seat = self._normalize(seat)
        offset = (seat - self.button_seat) % self.max_seats
        return self._order[offset]

    def hero_position(self) -> str:
        return self.position_for(self.hero_seat)

    def seat_for_position(self, position: str) -> Optional[int]:
        """Restituisce l'indice seat che occupa una certa posizione."""
        position = position.upper()
        if position not in _VALID_POSITIONS:
            return None
        for seat in range(self.max_seats):
            if self.position_for(seat) == position:
                return seat
        return None

    def active_positions(self, active_seats: Sequence[int]) -> List[SeatInfo]:
        """Restituisce info posizione per tutti i seat attivi."""
        out: List[SeatInfo] = []
        for seat in active_seats:
            seat = self._normalize(seat)
            out.append(
                SeatInfo(
                    index=seat,
                    position=self.position_for(seat),
                    is_hero=(seat == self.hero_seat),
                    is_button=(seat == self.button_seat),
                )
            )
        return out

    def all_seats(self) -> List[SeatInfo]:
        """Restituisce info per tutti i seat del tavolo."""
        return self.active_positions(range(self.max_seats))

    def relative_position(self, seat: int) -> Dict[str, int]:
        """Distanze in posti di un seat da hero e da button."""
        seat = self._normalize(seat)
        return {
            "seats_from_button": (seat - self.button_seat) % self.max_seats,
            "seats_from_hero": (seat - self.hero_seat) % self.max_seats,
        }


def positions_order(table_type: str) -> List[str]:
    """Restituisce l'ordine delle posizioni a partire dal BTN."""
    if table_type == "6max":
        return list(_POSITIONS_6MAX)
    if table_type == "9max":
        return list(_9MAX_CLOCKWISE)
    raise ValueError(f"Unsupported table_type: {table_type}")


def is_blind(position: str) -> bool:
    return position.upper() in {"SB", "BB"}


def is_early_position(position: str) -> bool:
    return position.upper() in {"UTG", "UTG+1", "UTG+2"}


def is_late_position(position: str) -> bool:
    return position.upper() in {"CO", "BTN"}


if __name__ == "__main__":
    # Demo
    layout = SeatLayout("9max", button_seat=3, hero_seat=0)
    print("9max, button=3, hero=0")
    for info in layout.all_seats():
        marker = ""
        if info.is_hero:
            marker += " [HERO]"
        if info.is_button:
            marker += " [BUTTON]"
        print(f"  seat {info.index}: {info.position}{marker}")
    print(f"Hero position: {layout.hero_position()}")
