"""Random bot: chaotic, unpredictable player.

Makes random decisions weighted toward action. Occasionally shoves all-in
for maximum chaos. Not completely braindead — checks when free, doesn't
fold to zero-cost actions — but there's no hand-reading or strategy here.
"""

import random

_rng = random.Random(7777)


def decide(state: dict) -> dict:
    can_check = state.get("can_check", False)
    owed = state.get("amount_owed", 0)
    stack = state.get("your_stack", 0)
    min_raise = state.get("min_raise_to", 0)

    roll = _rng.random()

    # Free to see cards — usually check, sometimes raise for chaos
    if can_check and owed == 0:
        if roll < 0.55:
            return {"action": "check"}
        if roll < 0.70 and min_raise > 0 and stack > min_raise:
            size = _rng.randint(min_raise, max(min_raise, min(stack, min_raise * 3)))
            return {"action": "raise", "amount": size}
        if roll < 0.75 and stack > 0:
            return {"action": "all_in"}
        return {"action": "check"}

    # Facing a bet
    if roll < 0.40:
        return {"action": "call"}
    if roll < 0.60:
        return {"action": "fold"}
    if roll < 0.80 and min_raise > 0 and stack > min_raise:
        size = _rng.randint(min_raise, max(min_raise, min(stack, min_raise * 3)))
        return {"action": "raise", "amount": size}
    if roll < 0.88 and stack > 0:
        return {"action": "all_in"}
    return {"action": "call"}
