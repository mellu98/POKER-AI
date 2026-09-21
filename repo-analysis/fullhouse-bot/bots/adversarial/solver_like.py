"""Solver-like adversary: balanced ranges, mixed value/bluff frequencies,
pot-fraction sizings (33% / 67% / 100%). No opponent adaptation.

The threat model is a competitor running a CFR blueprint with no exploit
layer. Hands are split into value, semi-bluff, and bluff buckets, and
each bucket picks a sizing/frequency from a fixed mixed strategy. Mix
selection is deterministic per-hand (hashed) so this bot looks identical
to any profiler across sessions.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

import eval7

BOT_NAME = "SolverLike"
BOT_AVATAR = "psychology"

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


# ~22% open range, GTO-style 6-max RFI averaged across positions.
_OPEN_RANGE: frozenset[str] = frozenset({
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    "AKs", "AKo", "AQs", "AQo", "AJs", "AJo", "ATs", "ATo",
    "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KQs", "KQo", "KJs", "KJo", "KTs", "KTo", "K9s",
    "QJs", "QJo", "QTs", "Q9s",
    "JTs", "J9s",
    "T9s", "T8s",
    "98s", "87s", "76s", "65s", "54s",
})

# ~9% 3-bet range, polar: premiums + suited-A blockers + suited connector mix.
_THREE_BET_RANGE: frozenset[str] = frozenset({
    "AA", "KK", "QQ", "JJ",
    "AKs", "AKo", "AQs", "AQo", "AJs",
    "KQs",
    "A5s", "A4s", "A3s",
    "76s", "65s",
})

# Continue vs 3-bet: pairs + broadways that flop well.
_CALL_3BET: frozenset[str] = frozenset({
    "TT", "99", "88", "77", "66", "55",
    "AJs", "ATs", "AQo", "AJo",
    "KQs", "KJs", "KQo",
    "QJs", "JTs", "T9s", "98s", "87s",
})

# 4-bet for value or strong bluff blocker.
_FOUR_BET: frozenset[str] = frozenset({"AA", "KK", "AKs", "A5s"})


def _det_random(*parts: Any) -> float:
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / (1 << 64)


def _board_texture(board: list[str]) -> str:
    if len(board) < 3:
        return "preflop"
    suits = [c[1] for c in board]
    ranks = sorted(_RANK_VAL[c[0]] for c in board)
    flush_2 = max(suits.count(s) for s in set(suits)) >= 2
    flush_3 = max(suits.count(s) for s in set(suits)) >= 3
    span = ranks[-1] - ranks[0]
    paired = len(set(ranks)) < len(ranks)
    if paired:
        return "paired"
    if flush_3 or (flush_2 and span <= 4):
        return "wet"
    if span <= 4 and not flush_2:
        return "semi"
    if flush_2 or span <= 6:
        return "semi"
    return "dry"


def _hand_strength(hole: list[str], board: list[str]) -> str:
    """Bucket: value / medium / draw / air."""
    if not board:
        return "preflop"
    cards = [eval7.Card(c) for c in hole + board]
    score = eval7.evaluate(cards)
    htype = str(eval7.handtype(score))
    if htype in ("Straight", "Flush", "Full House", "Four of a Kind",
                 "Straight Flush", "Three of a Kind", "Two Pair"):
        return "value"

    h_ranks = [_RANK_VAL[c[0]] for c in hole]
    b_ranks = sorted({_RANK_VAL[c[0]] for c in board})
    top = b_ranks[-1] if b_ranks else 0

    if htype == "Pair":
        # Overpair / top pair good kicker = value; lower pairs medium.
        if h_ranks[0] == h_ranks[1] and h_ranks[0] > top:
            return "value"
        if top in h_ranks:
            kicker = max(h for h in h_ranks if h != top) if h_ranks[0] != h_ranks[1] else h_ranks[0]
            return "value" if kicker >= 10 else "medium"
        return "medium"

    # High card: flush draw / OESD / gutshot+overcard = draw.
    suits = [c[1] for c in hole]
    b_suits = [c[1] for c in board]
    has_fd = suits[0] == suits[1] and b_suits.count(suits[0]) >= 2
    all_ranks = sorted(set(h_ranks) | {_RANK_VAL[c[0]] for c in board})
    has_oesd = False
    for lo in range(0, 11):
        window = set(range(lo, lo + 5))
        if len(window & set(all_ranks)) >= 4:
            has_oesd = True
            break
    if has_fd or has_oesd:
        return "draw"
    if max(h_ranks) >= 12:
        return "draw"  # ace-high backdoors as semi-bluff
    return "air"


def _count_raises(state: dict) -> int:
    n = 0
    for entry in state.get("action_log") or []:
        if entry.get("action") in ("raise", "all_in"):
            n += 1
    return n


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


def _pot_frac_target(state: dict, frac: float) -> int:
    pot = max(1, state["pot"])
    my_bet = state["your_bet_this_street"]
    owed = state["amount_owed"]
    # When facing a bet, "pot fraction" sizes on top of the call amount.
    return my_bet + owed + int((pot + owed) * frac)


def _pick_sizing(roll: float) -> float:
    """Mixed sizing: 33% / 67% / 100% pot at roughly 40/35/25 frequency."""
    if roll < 0.40:
        return 0.33
    if roll < 0.75:
        return 0.67
    return 1.00


def _decide_preflop(state: dict) -> dict:
    hole = state["your_cards"]
    hclass = _hand_class(hole)
    raises = _count_raises(state)
    can_check = state.get("can_check", False)
    cur_bet = state.get("current_bet", 100)
    min_to = state["min_raise_to"]
    stack = state["your_stack"]

    in_open = hclass in _OPEN_RANGE
    in_3bet = hclass in _THREE_BET_RANGE
    in_call_3b = hclass in _CALL_3BET
    in_4bet = hclass in _FOUR_BET

    mix_roll = _det_random(tuple(hole), "pf", raises)

    if raises == 0:
        if in_open:
            # 80% raise, 20% limp/check (looks balanced).
            if mix_roll < 0.80 or not can_check:
                target = max(int(2.5 * 100), min_to)
                return _raise_to(state, target)
            if can_check:
                return {"action": "check"}
            return {"action": "call"}
        return {"action": "check"} if can_check else {"action": "fold"}

    if raises == 1:
        if in_3bet:
            target = max(int(cur_bet * 3.0), min_to)
            return _raise_to(state, target)
        if in_open:
            # Flat the open with the middle of the range.
            if state["amount_owed"] <= stack * 0.10:
                return {"action": "call"}
            return {"action": "fold"}
        return {"action": "check"} if can_check else {"action": "fold"}

    if raises == 2:
        if in_4bet:
            target = max(int(cur_bet * 2.4), min_to)
            return _raise_to(state, target)
        if in_call_3b and state["amount_owed"] <= stack * 0.20:
            return {"action": "call"}
        return {"action": "check"} if can_check else {"action": "fold"}

    # 4-bet+ pots: only AA/KK/AKs jam.
    if hclass in ("AA", "KK", "AKs"):
        return {"action": "all_in"}
    return {"action": "check"} if can_check else {"action": "fold"}


def _decide_postflop(state: dict) -> dict:
    hole = state["your_cards"]
    board = state["community_cards"]
    pot = max(1, state["pot"])
    stack = state["your_stack"]
    owed = state["amount_owed"]
    can_check = state.get("can_check", False)
    street = state["street"]

    strength = _hand_strength(hole, board)
    texture = _board_texture(board)
    size_roll = _det_random(tuple(hole), tuple(board), street, "size")
    bluff_roll = _det_random(tuple(hole), tuple(board), street, "bluff")

    # Facing a bet.
    if not can_check and owed > 0:
        call_eq_needed = owed / (pot + owed) if (pot + owed) else 1.0
        # Continue frequencies per bucket — keep MDF roughly honest.
        if strength == "value":
            # Mix raise vs call: 30% raise for value on flop/turn.
            if street in ("flop", "turn") and size_roll < 0.30:
                target = _pot_frac_target(state, _pick_sizing(size_roll * 3.3))
                return _raise_to(state, target)
            return {"action": "call"}
        if strength == "medium":
            # Call if pot odds are reasonable.
            if call_eq_needed <= 0.33:
                return {"action": "call"}
            return {"action": "fold"}
        if strength == "draw":
            if call_eq_needed <= 0.28:
                return {"action": "call"}
            return {"action": "fold"}
        # Air: occasional float on flop only with small bets.
        if street == "flop" and call_eq_needed <= 0.18 and bluff_roll < 0.30:
            return {"action": "call"}
        return {"action": "fold"}

    # Checked to / first to act. Decide bet vs check.
    # C-bet flop ~70% on dry, ~55% on semi, ~40% on wet.
    cbet_freq = {"dry": 0.70, "semi": 0.55, "wet": 0.40, "paired": 0.65}.get(texture, 0.55)

    if strength == "value":
        frac = _pick_sizing(size_roll)
        # Value bet polar on wet boards.
        if texture == "wet" and street in ("turn", "river"):
            frac = max(frac, 0.67)
        target = _pot_frac_target(state, frac)
        return _raise_to(state, target)

    if strength == "medium":
        # Thin value / protection bet ~50% on flop, mostly check turn/river.
        if street == "flop" and bluff_roll < 0.50:
            target = _pot_frac_target(state, 0.33)
            return _raise_to(state, target)
        return {"action": "check"}

    if strength == "draw":
        # Semi-bluff cbet at cbet_freq on flop, ~45% barrel on turn.
        if street == "flop" and bluff_roll < cbet_freq:
            frac = _pick_sizing(size_roll)
            target = _pot_frac_target(state, frac)
            return _raise_to(state, target)
        if street == "turn" and bluff_roll < 0.45:
            target = _pot_frac_target(state, 0.67)
            return _raise_to(state, target)
        return {"action": "check"}

    # Air: pure bluff at lower freq, polar sizing.
    if street == "flop" and bluff_roll < cbet_freq * 0.5:
        target = _pot_frac_target(state, 0.33)
        return _raise_to(state, target)
    if street == "river" and bluff_roll < 0.20:
        # Polar river bluff with blockers.
        if max(_RANK_VAL[c[0]] for c in hole) >= 12:
            target = _pot_frac_target(state, 0.67)
            return _raise_to(state, target)
    return {"action": "check"}


def decide(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("type") == "warmup":
        return {"ok": True}
    try:
        if state.get("street") == "preflop":
            return _decide_preflop(state)
        return _decide_postflop(state)
    except Exception:
        if state.get("can_check"):
            return {"action": "check"}
        return {"action": "fold"}
