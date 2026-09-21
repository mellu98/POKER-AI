"""Preflop advisor — RFI (raise first in) charts for 100bb 6-max NLHE.

Источники базы: публичные GTO-чарты (GTOWizard, Upswing). Это приближение,
не точные solver-частоты — годится для домашней игры.
"""

from __future__ import annotations

from .game_state import Card, Position

RANK_ORDER = "23456789TJQKA"
RANK_VAL = {r: i for i, r in enumerate(RANK_ORDER)}


def hand_code(c1: Card, c2: Card) -> str:
    """Convert two cards to canonical hand notation: 'AA', 'AKs', 'AKo'."""
    r1, r2 = c1.rank, c2.rank
    if RANK_VAL[r1] < RANK_VAL[r2]:
        r1, r2 = r2, r1
        c1, c2 = c2, c1
    if r1 == r2:
        return r1 + r2
    suited = "s" if c1.suit == c2.suit else "o"
    return r1 + r2 + suited


# ---------- RFI ranges (100bb 6-max) ----------
#
# Каждая позиция — set рук, которые открываем raise first in.
# Размер открытия: 2.5bb (BTN), 3bb (CO/MP/UTG), 3bb SB.

_RFI_RANGES: dict[Position, set[str]] = {
    Position.UTG: {
        # pairs
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77",
        # suited
        "AKs", "AQs", "AJs", "ATs", "A9s", "A5s", "A4s",
        "KQs", "KJs", "KTs", "QJs", "QTs", "JTs", "T9s", "98s",
        # offsuit
        "AKo", "AQo", "AJo", "KQo",
    },
    Position.MP: {
        # all of UTG plus:
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "KQs", "KJs", "KTs", "K9s", "QJs", "QTs", "Q9s",
        "JTs", "J9s", "T9s", "T8s", "98s", "87s", "76s",
        "AKo", "AQo", "AJo", "ATo", "KQo", "KJo",
    },
    Position.CO: {
        # MP plus more
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s",
        "QJs", "QTs", "Q9s", "Q8s",
        "JTs", "J9s", "J8s",
        "T9s", "T8s", "T7s",
        "98s", "97s", "87s", "86s", "76s", "75s", "65s", "54s",
        "AKo", "AQo", "AJo", "ATo", "A9o",
        "KQo", "KJo", "KTo",
        "QJo", "QTo", "JTo",
    },
    Position.BTN: {
        # widest. All pairs, all suited Ax/Kx down to K2s, broad suited connectors/gappers
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s",
        "QJs", "QTs", "Q9s", "Q8s", "Q7s", "Q6s", "Q5s", "Q4s",
        "JTs", "J9s", "J8s", "J7s", "J6s",
        "T9s", "T8s", "T7s", "T6s",
        "98s", "97s", "96s", "87s", "86s", "85s", "76s", "75s", "74s", "65s", "64s", "54s", "53s", "43s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o", "A4o", "A3o", "A2o",
        "KQo", "KJo", "KTo", "K9o", "K8o",
        "QJo", "QTo", "Q9o", "Q8o",
        "JTo", "J9o", "J8o",
        "T9o", "T8o", "98o", "87o", "76o",
    },
    Position.SB: {
        # SB vs BB — wide raise-or-fold (упрощение: не делаем limp mix)
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s",
        "QJs", "QTs", "Q9s", "Q8s",
        "JTs", "J9s", "J8s",
        "T9s", "T8s", "T7s",
        "98s", "97s", "87s", "86s", "76s", "75s", "65s", "54s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o",
        "KQo", "KJo", "KTo", "K9o",
        "QJo", "QTo", "Q9o",
        "JTo", "J9o", "T9o",
    },
    Position.BB: set(),  # BB never RFIs (by definition)
}


def rfi_advice(position: Position, c1: Card, c2: Card) -> dict:
    """Совет для unopened pot: raise или fold?

    Returns: {'action': 'raise'|'fold', 'hand': 'AKs', 'size_bb': 2.5|3.0}
    """
    code = hand_code(c1, c2)
    in_range = code in _RFI_RANGES.get(position, set())
    size_bb = 2.5 if position == Position.BTN else 3.0

    if in_range:
        return {"action": "raise", "hand": code, "size_bb": size_bb}
    return {"action": "fold", "hand": code, "size_bb": 0.0}


def hand_in_range(position: Position, code: str) -> bool:
    return code in _RFI_RANGES.get(position, set())


def range_size_pct(position: Position) -> float:
    """% от всех 1326 комбо. Грубое приближение для UI."""
    # Считаем по hand-кодам с весами:
    # pair = 6 combos, suited = 4 combos, offsuit = 12 combos
    range_set = _RFI_RANGES.get(position, set())
    total_combos = 0
    for h in range_set:
        if len(h) == 2:  # pair
            total_combos += 6
        elif h.endswith("s"):
            total_combos += 4
        elif h.endswith("o"):
            total_combos += 12
    return total_combos / 1326
