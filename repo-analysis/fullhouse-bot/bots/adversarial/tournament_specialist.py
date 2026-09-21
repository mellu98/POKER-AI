"""Tournament specialist: stack-aware, ICM-flavoured aggression.

Three regimes by effective stack depth:
  * push/fold under 15bb (Nash-ish jam range from each position)
  * 15-30bb: tight raise-or-fold, jam over opens with mid-strength
  * 30bb+: normal TAG ranges with stack-pressure adjustments

When this bot has a big stack, it widens (bullies). When facing a short stack,
it tightens to isolate (avoids race-flips with a stack that can call).
"""

from __future__ import annotations

import hashlib
from typing import Any

import eval7

BOT_NAME = "TournamentSpecialist"
BOT_AVATAR = "emoji_events"

BIG_BLIND = 100
_RANK_ORDER = "23456789TJQKA"
_RANK_VAL = {r: i for i, r in enumerate(_RANK_ORDER)}


def _hand_class(hole: list[str]) -> str:
    r0, r1 = hole[0][0], hole[1][0]
    s0, s1 = hole[0][1], hole[1][1]
    if r0 == r1:
        return r0 + r1
    if _RANK_VAL[r0] < _RANK_VAL[r1]:
        r0, r1 = r1, r0
        s0, s1 = s1, s0
    return r0 + r1 + ("s" if s0 == s1 else "o")


def _det_random(*parts: Any) -> float:
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / (1 << 64)


# Push/fold jam ranges by effective bb. Wider as stack shrinks.
_JAM_10BB: frozenset[str] = frozenset({
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    "AKs", "AKo", "AQs", "AQo", "AJs", "AJo", "ATs", "ATo", "A9s", "A8s", "A7s",
    "A5s", "A4s", "A3s", "A2s",
    "KQs", "KQo", "KJs", "KJo", "KTs", "K9s",
    "QJs", "QTs", "JTs", "T9s", "98s",
})

_JAM_15BB: frozenset[str] = frozenset({
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55",
    "AKs", "AKo", "AQs", "AQo", "AJs", "AJo", "ATs",
    "KQs", "KQo", "KJs",
    "QJs", "JTs",
})

# Short stack call-jam (vs opens).
_CALL_JAM_15BB: frozenset[str] = frozenset({
    "AA", "KK", "QQ", "JJ", "TT", "99",
    "AKs", "AKo", "AQs", "AQo", "AJs",
    "KQs",
})


def _eff_bb(state: dict) -> float:
    """Effective stack in BB for current player vs smallest opponent stack."""
    my_stack = state["your_stack"]
    me = state["seat_to_act"]
    opp_stacks = [
        p.get("stack", 0)
        for p in state.get("players") or []
        if p.get("seat") != me and not p.get("is_folded")
    ]
    if not opp_stacks:
        return my_stack / BIG_BLIND
    return min(my_stack, max(opp_stacks)) / BIG_BLIND


def _avg_opp_bb(state: dict) -> float:
    me = state["seat_to_act"]
    opp = [
        p.get("stack", 0)
        for p in state.get("players") or []
        if p.get("seat") != me and not p.get("is_folded")
    ]
    if not opp:
        return 100.0
    return (sum(opp) / len(opp)) / BIG_BLIND


def _i_am_chip_leader(state: dict) -> bool:
    my_stack = state["your_stack"]
    me = state["seat_to_act"]
    opp_max = max(
        (p.get("stack", 0) for p in state.get("players") or [] if p.get("seat") != me),
        default=0,
    )
    return my_stack > opp_max * 1.3


def _facing_short_stack(state: dict) -> bool:
    me = state["seat_to_act"]
    opp = [
        p.get("stack", 0)
        for p in state.get("players") or []
        if p.get("seat") != me and not p.get("is_folded")
    ]
    if not opp:
        return False
    return min(opp) / BIG_BLIND < 15


def _count_raises(state: dict) -> int:
    n = 0
    for entry in state.get("action_log") or []:
        if entry.get("action") in ("raise", "all_in"):
            n += 1
    return n


def _was_aggressor(state: dict) -> bool:
    log = state.get("action_log") or []
    me = state["seat_to_act"]
    last = None
    for entry in log:
        if entry.get("action") in ("raise", "all_in"):
            last = entry.get("seat")
    return last == me


def _cap_raise(state: dict, raise_to: int) -> int:
    my_stack = state["your_stack"]
    my_bet = state["your_bet_this_street"]
    min_to = state["min_raise_to"]
    max_to = my_stack + my_bet
    return max(min_to, min(raise_to, max_to))


def _raise_to(state: dict, target: int) -> dict:
    target = _cap_raise(state, target)
    my_stack = state["your_stack"]
    my_bet = state["your_bet_this_street"]
    if target >= my_stack + my_bet:
        return {"action": "all_in"}
    return {"action": "raise", "amount": target}


def _hand_strength(hole: list[str], board: list[str]) -> str:
    if not board:
        return "preflop"
    cards = [eval7.Card(c) for c in hole + board]
    score = eval7.evaluate(cards)
    htype = str(eval7.handtype(score))
    if htype in ("Two Pair", "Three of a Kind", "Straight", "Flush",
                 "Full House", "Four of a Kind", "Straight Flush"):
        return "strong"
    h_ranks = [_RANK_VAL[c[0]] for c in hole]
    b_ranks = sorted({_RANK_VAL[c[0]] for c in board})
    top = b_ranks[-1] if b_ranks else 0
    if htype == "Pair":
        if h_ranks[0] == h_ranks[1] and h_ranks[0] > top:
            return "strong"
        if top in h_ranks:
            return "medium"
        return "medium"
    suits = [c[1] for c in hole]
    b_suits = [c[1] for c in board]
    has_fd = suits[0] == suits[1] and b_suits.count(suits[0]) >= 2
    all_ranks = sorted(set(h_ranks) | {_RANK_VAL[c[0]] for c in board})
    has_oesd = any(
        len(set(range(lo, lo + 5)) & set(all_ranks)) >= 4 for lo in range(11)
    )
    if has_fd or has_oesd:
        return "draw"
    if max(h_ranks) >= 12:
        return "overcard"
    return "air"


def _decide_push_fold(state: dict, eff_bb: float) -> dict:
    """Sub-15bb push/fold regime."""
    hole = state["your_cards"]
    hclass = _hand_class(hole)
    raises = _count_raises(state)
    can_check = state.get("can_check", False)
    owed = state["amount_owed"]
    stack = state["your_stack"]

    jam_range = _JAM_10BB if eff_bb <= 10 else _JAM_15BB

    if raises == 0:
        if hclass in jam_range:
            return {"action": "all_in"}
        return {"action": "check"} if can_check else {"action": "fold"}

    if raises == 1:
        # Facing an open at short depth: call-jam with strong hands.
        if hclass in _CALL_JAM_15BB:
            return {"action": "all_in"}
        # Pocket pairs / suited broadway under 10bb: also jam.
        if eff_bb <= 12 and hclass in jam_range and owed <= stack * 0.30:
            return {"action": "all_in"}
        return {"action": "check"} if can_check else {"action": "fold"}

    # Raised pots beyond a single jam: only premiums.
    if hclass in ("AA", "KK", "QQ", "AKs", "AKo"):
        return {"action": "all_in"}
    return {"action": "check"} if can_check else {"action": "fold"}


def _decide_medium_stack(state: dict, eff_bb: float) -> dict:
    """15-30bb: tight, raise-or-jam over opens."""
    hole = state["your_cards"]
    hclass = _hand_class(hole)
    raises = _count_raises(state)
    can_check = state.get("can_check", False)
    cur_bet = state.get("current_bet", BIG_BLIND)
    min_to = state["min_raise_to"]
    stack = state["your_stack"]

    open_range = frozenset({
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44",
        "AKs", "AKo", "AQs", "AQo", "AJs", "AJo", "ATs", "ATo",
        "A9s", "A5s", "A4s",
        "KQs", "KQo", "KJs", "KTs",
        "QJs", "QTs", "JTs", "T9s", "98s",
    })

    jam_over_open = frozenset({
        "AA", "KK", "QQ", "JJ", "TT", "99",
        "AKs", "AKo", "AQs", "AQo", "AJs",
        "KQs",
    })

    if raises == 0:
        if hclass in open_range:
            target = max(int(2.2 * BIG_BLIND), min_to)
            return _raise_to(state, target)
        return {"action": "check"} if can_check else {"action": "fold"}

    if raises == 1:
        if hclass in jam_over_open:
            return {"action": "all_in"}
        if hclass in open_range and state["amount_owed"] <= stack * 0.12:
            return {"action": "call"}
        return {"action": "check"} if can_check else {"action": "fold"}

    if hclass in ("AA", "KK", "QQ", "AKs"):
        return {"action": "all_in"}
    return {"action": "check"} if can_check else {"action": "fold"}


def _decide_deep_preflop(state: dict, eff_bb: float) -> dict:
    """30bb+: TAG ranges, widened if chip leader, tightened vs short stacks."""
    hole = state["your_cards"]
    hclass = _hand_class(hole)
    raises = _count_raises(state)
    can_check = state.get("can_check", False)
    cur_bet = state.get("current_bet", BIG_BLIND)
    min_to = state["min_raise_to"]
    stack = state["your_stack"]

    is_leader = _i_am_chip_leader(state)
    facing_short = _facing_short_stack(state)

    open_range = frozenset({
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55",
        "AKs", "AKo", "AQs", "AQo", "AJs", "AJo", "ATs",
        "KQs", "KQo", "KJs", "KTs",
        "QJs", "QTs", "JTs", "T9s", "98s", "87s", "76s",
    })

    if is_leader and not facing_short:
        open_range = open_range | frozenset({
            "44", "33", "22",
            "A9s", "A8s", "A7s", "A5s", "A4s", "A3s", "A2s",
            "K9s", "Q9s", "J9s", "T8s", "97s", "86s", "65s", "54s",
            "ATo", "KJo", "QJo",
        })

    three_bet = frozenset({
        "AA", "KK", "QQ", "JJ",
        "AKs", "AKo", "AQs", "AQo", "AJs",
        "A5s", "A4s",
    })
    if facing_short:
        # vs short stack: only big hands 3-bet (no light pressure on a shove-able stack).
        three_bet = frozenset({"AA", "KK", "QQ", "AKs", "AKo"})

    call_3b = frozenset({
        "TT", "99", "88", "77",
        "AJs", "ATs", "AQo",
        "KQs", "KJs", "QJs", "JTs", "T9s",
    })

    if raises == 0:
        if hclass in open_range:
            target = max(int(2.5 * BIG_BLIND), min_to)
            return _raise_to(state, target)
        return {"action": "check"} if can_check else {"action": "fold"}

    if raises == 1:
        if hclass in three_bet:
            target = max(int(cur_bet * 3.0), min_to)
            return _raise_to(state, target)
        if hclass in open_range and hclass in call_3b:
            if state["amount_owed"] <= stack * 0.12:
                return {"action": "call"}
        return {"action": "check"} if can_check else {"action": "fold"}

    if raises == 2:
        if hclass in ("AA", "KK", "AKs"):
            return {"action": "all_in"}
        if hclass in ("QQ", "JJ", "AKo") and state["amount_owed"] <= stack * 0.5:
            return {"action": "call"}
        if hclass in call_3b and state["amount_owed"] <= stack * 0.20:
            return {"action": "call"}
        return {"action": "check"} if can_check else {"action": "fold"}

    if hclass in ("AA", "KK", "AKs"):
        return {"action": "all_in"}
    return {"action": "check"} if can_check else {"action": "fold"}


def _decide_postflop(state: dict, eff_bb: float) -> dict:
    hole = state["your_cards"]
    board = state["community_cards"]
    pot = max(1, state["pot"])
    stack = state["your_stack"]
    owed = state["amount_owed"]
    can_check = state.get("can_check", False)
    street = state["street"]
    my_bet = state["your_bet_this_street"]

    strength = _hand_strength(hole, board)
    pfa = _was_aggressor(state)
    is_leader = _i_am_chip_leader(state)
    facing_short = _facing_short_stack(state)
    bluff_roll = _det_random(tuple(hole), tuple(board), street, "ts")

    def bet(frac: float) -> dict:
        target = my_bet + owed + int((pot + owed) * frac)
        return _raise_to(state, target)

    # Short-stack postflop: jam-or-fold with strong hands, give up otherwise.
    if eff_bb < 8:
        if not can_check and owed > 0:
            if strength in ("strong",) or (strength == "medium" and owed <= stack * 0.3):
                return {"action": "all_in"}
            return {"action": "fold"}
        if strength in ("strong", "medium"):
            return {"action": "all_in"}
        return {"action": "check"}

    # Facing a bet.
    if not can_check and owed > 0:
        call_eq_needed = owed / (pot + owed) if (pot + owed) else 1.0
        if strength == "strong":
            if street in ("flop", "turn") and bluff_roll < 0.30:
                return bet(0.7)
            return {"action": "call"}
        if strength == "medium":
            if call_eq_needed <= 0.30:
                return {"action": "call"}
            return {"action": "fold"}
        if strength == "draw":
            if call_eq_needed <= 0.28:
                return {"action": "call"}
            return {"action": "fold"}
        return {"action": "fold"}

    # Apply big-stack pressure: cbet more.
    cbet_freq = 0.65
    if is_leader and not facing_short:
        cbet_freq = 0.80
    if facing_short:
        cbet_freq = 0.50  # tighten vs short opponents

    if pfa and street == "flop":
        if strength == "strong":
            return bet(0.6)
        if strength in ("medium", "draw") and bluff_roll < cbet_freq:
            return bet(0.5)
        if strength in ("overcard", "air") and bluff_roll < cbet_freq * 0.7:
            return bet(0.4)
        return {"action": "check"}

    if pfa and street == "turn":
        if strength == "strong":
            return bet(0.7)
        if strength == "draw" and bluff_roll < 0.50:
            return bet(0.6)
        return {"action": "check"}

    if pfa and street == "river":
        if strength == "strong":
            return bet(0.75)
        if strength == "overcard" and bluff_roll < 0.20 and is_leader:
            return bet(0.6)
        return {"action": "check"}

    if not pfa:
        if strength == "strong" and bluff_roll < 0.50:
            return bet(0.5)
    return {"action": "check"}


def decide(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("type") == "warmup":
        return {"ok": True}
    try:
        eff_bb = _eff_bb(state)
        if state.get("street") == "preflop":
            if eff_bb <= 15:
                return _decide_push_fold(state, eff_bb)
            if eff_bb <= 30:
                return _decide_medium_stack(state, eff_bb)
            return _decide_deep_preflop(state, eff_bb)
        return _decide_postflop(state, eff_bb)
    except Exception:
        if state.get("can_check"):
            return {"action": "check"}
        return {"action": "fold"}
