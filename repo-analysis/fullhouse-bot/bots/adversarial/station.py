"""Calling station: peels with any pair, folds only to a shove with air.

Adversarial sparring partner for v3 exploit-deviation. Never raises so an
exploiter that value-bets thinner should print money against it. The one
hard fold case is a near-all-in bet when we have no pair and no draw, which
keeps the bot from incinerating its stack on pure air.
"""

import random
from typing import Any

random.seed(11_001)

BOT_NAME = "Station"


def _ranks_on_board(cards: list[str]) -> list[str]:
    return [c[0] for c in cards]


def _has_pair_or_better(hole: list[str], board: list[str]) -> bool:
    """Pocket pair, any pair with board, or two-pair+. Cheap rank-only check."""
    hole_ranks = [c[0] for c in hole]
    board_ranks = _ranks_on_board(board)

    if hole_ranks[0] == hole_ranks[1]:
        return True
    for r in hole_ranks:
        if r in board_ranks:
            return True
    return False


def _has_draw(hole: list[str], board: list[str]) -> bool:
    """Flush draw or open-ended straight draw, approximate."""
    if len(board) < 3:
        return False
    cards = hole + board
    suits = [c[1] for c in cards]
    for s in set(suits):
        if suits.count(s) == 4:
            return True

    rank_order = "23456789TJQKA"
    ranks = sorted({rank_order.index(c[0]) for c in cards})
    # Wheel: treat A as low too
    if 12 in ranks:
        ranks = sorted(set(ranks + [-1]))
    for i in range(len(ranks) - 3):
        window = ranks[i:i + 4]
        if window[-1] - window[0] == 3:
            return True
    return False


def decide(state: dict[str, Any]) -> dict[str, Any]:
    owed = state["amount_owed"]
    stack = state["your_stack"]
    hole = state["your_cards"]
    board = state["community_cards"]

    if state["can_check"]:
        return {"action": "check"}

    if owed == 0:
        return {"action": "check"}

    has_pair = _has_pair_or_better(hole, board)
    has_draw = _has_draw(hole, board)

    # Preflop with no board: any two cards count as "speculation worth the call"
    # if it's not a near-shove. This matches real stations who never fold preflop.
    if state["street"] == "preflop":
        if owed <= stack * 0.5:
            return {"action": "call"}
        # Big bet preflop: still call with anything decent
        ranks = sorted([c[0] for c in hole])
        high_card = ranks[-1] in "AKQJT" or ranks[0] == ranks[1]
        if high_card:
            return {"action": "call"}
        return {"action": "fold"}

    # Postflop: call any bet up to 50% of stack with pair+
    if has_pair and owed <= stack * 0.5:
        return {"action": "call"}

    # With a draw, call smaller bets
    if has_draw and owed <= stack * 0.33:
        return {"action": "call"}

    # Facing a shove with no pair and no draw -> fold
    if owed >= stack * 0.8:
        return {"action": "fold"}

    # Small bets get called with anything (true station behavior)
    if owed <= stack * 0.15:
        return {"action": "call"}

    return {"action": "fold"}
