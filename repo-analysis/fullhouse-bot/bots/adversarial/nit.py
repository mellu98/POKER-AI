"""Ultra-tight nit: only AA/KK/QQ/AK preflop, bets only set+ postflop.

Folds everything else, including top pair. The exploit is obvious: bluff the
river when this bot doesn't raise, since it will fold one-pair to any sizing.
"""

import random
from typing import Any

random.seed(11_003)

BOT_NAME = "Nit"

_PREMIUM_PAIRS = {"AA", "KK", "QQ"}


def _hand_class(hole: list[str]) -> str:
    r0, r1 = hole[0][0], hole[1][0]
    s0, s1 = hole[0][1], hole[1][1]
    if r0 == r1:
        return r0 + r1
    high, low = sorted([r0, r1], key="23456789TJQKA".index, reverse=True)
    suited = "s" if s0 == s1 else "o"
    return high + low + suited


def _is_set_or_better(hole: list[str], board: list[str]) -> bool:
    """Set (trips with pocket pair), full house, quads. Bare trips on a paired
    board don't count - we want a real monster."""
    if len(board) < 3:
        return False
    hole_ranks = [c[0] for c in hole]
    board_ranks = [c[0] for c in board]

    if hole_ranks[0] != hole_ranks[1]:
        return False
    # Pocket pair: need at least one matching board card for a set
    return hole_ranks[0] in board_ranks


def decide(state: dict[str, Any]) -> dict[str, Any]:
    street = state["street"]
    can_check = state["can_check"]
    stack = state["your_stack"]
    bet_this = state["your_bet_this_street"]
    min_r = state["min_raise_to"]
    pot = state["pot"]
    hole = state["your_cards"]

    if street == "preflop":
        cls = _hand_class(hole)
        is_premium = cls in _PREMIUM_PAIRS or cls in ("AKs", "AKo")
        if is_premium:
            raise_to = min(max(min_r, int(pot * 3)), stack + bet_this)
            if raise_to >= stack + bet_this:
                return {"action": "all_in"}
            return {"action": "raise", "amount": raise_to}
        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # Postflop: only sets+ continue
    if _is_set_or_better(hole, state["community_cards"]):
        raise_to = min(max(min_r, int(pot * 0.75) + state["current_bet"]),
                       stack + bet_this)
        if raise_to >= stack + bet_this:
            return {"action": "all_in"}
        return {"action": "raise", "amount": raise_to}

    if can_check:
        return {"action": "check"}
    return {"action": "fold"}
