"""Hyper-aggressive maniac: raises 60-80% of the time with random sizing.

Adversarial sparring partner. The intent is a bot that punishes passive play
but folds enough equity to lose to anyone who tightens up and value-bets thin.
Sizing is uniformly random between min-raise and 2x pot to stress-test our
sizing-classification heuristics.
"""

import random
from typing import Any

_rng = random.Random(11_002)

BOT_NAME = "Maniac"


def decide(state: dict[str, Any]) -> dict[str, Any]:
    stack = state["your_stack"]
    pot = state["pot"]
    min_r = state["min_raise_to"]
    bet_this = state["your_bet_this_street"]
    owed = state["amount_owed"]
    can_check = state["can_check"]

    max_raise = stack + bet_this  # all-in cap
    roll = _rng.random()

    # 70% raise, 20% call, 10% fold/check (midpoint of 60-80 / 30 / 10 spec)
    if roll < 0.70:
        upper = max(min_r, min(int(pot * 2), max_raise))
        if upper <= min_r:
            raise_to = min(max_raise, min_r)
        else:
            raise_to = _rng.randint(min_r, upper)
        raise_to = min(raise_to, max_raise)
        raise_to = max(raise_to, min_r)
        # If raising would put us all-in, just shove
        if raise_to >= max_raise:
            return {"action": "all_in"}
        return {"action": "raise", "amount": raise_to}

    if roll < 0.90:
        if can_check:
            return {"action": "check"}
        # Don't call into a near-shove just to look passive
        if owed >= stack * 0.9:
            return {"action": "fold"}
        return {"action": "call"}

    if can_check:
        return {"action": "check"}
    return {"action": "fold"}
