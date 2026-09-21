"""Numba-compiled 7-card poker hand evaluator.

Cards are integers 0-51: rank = card >> 2 (0=deuce .. 12=ace), suit = card & 3.
evaluate7() returns an int32 score where higher = stronger hand.
Only comparison ordering matters, not absolute values.

Zero heap allocations - all state is in scalar locals and bit-packed integers.
"""

from __future__ import annotations

import numba as nb

_i32 = nb.int32
_i64 = nb.int64


@nb.njit(_i32(_i32, _i32), cache=True)
def _top_n_bits(mask, n):
    """Extract top n set bits from a 13-bit mask, pack into 4-bit nibbles."""
    result = _i32(0)
    shift = (n - 1) * 4
    for bit in range(12, -1, -1):
        if mask & (1 << bit):
            result |= _i32(bit) << shift
            shift -= 4
            n -= 1
            if n == 0:
                break
    return result


@nb.njit(_i32(_i32), cache=True)
def _find_straight_high(mask):
    """Find highest straight in a 13-bit rank bitmask. Returns -1 if none."""
    for high in range(12, 3, -1):
        needed = _i32(0x1F) << (high - 4)
        if (mask & needed) == needed:
            return _i32(high)
    if (mask & 0x100F) == 0x100F:
        return _i32(3)
    return _i32(-1)


@nb.njit(_i32(nb.int8[:], _i32), cache=True)
def _evaluate_n(cards, n):
    """Evaluate best 5-card hand from n cards (5 <= n <= 7). Higher = better."""
    # Also track suit rank masks and suit counts as scalars.
    rc = _i64(0)  # rank counts packed
    sc0 = _i32(0)
    sc1 = _i32(0)
    sc2 = _i32(0)
    sc3 = _i32(0)
    sm0 = _i32(0)
    sm1 = _i32(0)
    sm2 = _i32(0)
    sm3 = _i32(0)

    for i in range(n):
        c = _i32(cards[i])
        r = c >> 2
        s = c & 3
        rc += _i64(1) << (r * 4)
        bit = _i32(1) << r
        if s == 0:
            sc0 += 1
            sm0 |= bit
        elif s == 1:
            sc1 += 1
            sm1 |= bit
        elif s == 2:
            sc2 += 1
            sm2 |= bit
        else:
            sc3 += 1
            sm3 |= bit

    # Flush detection
    flush_mask = _i32(-1)  # -1 = no flush
    if sc0 >= 5:
        flush_mask = sm0
    elif sc1 >= 5:
        flush_mask = sm1
    elif sc2 >= 5:
        flush_mask = sm2
    elif sc3 >= 5:
        flush_mask = sm3

    # Ranks present bitmask
    ranks_present = _i32(0)
    for r in range(13):
        if (rc >> (r * 4)) & 0xF:
            ranks_present |= _i32(1) << r

    # Straight detection
    straight_high = _find_straight_high(ranks_present)

    sf_high = _i32(-1)
    if flush_mask >= 0:
        sf_high = _find_straight_high(flush_mask)

    # Straight flush
    if sf_high >= 0:
        return (_i32(8) << 24) | (sf_high << 16)

    # Scan rank counts for quads/trips/pairs (high to low)
    quad_rank = _i32(-1)
    trips_rank = _i32(-1)
    trips_rank2 = _i32(-1)
    pair_rank = _i32(-1)
    pair_rank2 = _i32(-1)

    for r in range(12, -1, -1):
        cnt = _i32((rc >> (r * 4)) & 0xF)
        if cnt == 4:
            quad_rank = r
        elif cnt == 3:
            if trips_rank < 0:
                trips_rank = r
            else:
                trips_rank2 = r
        elif cnt == 2:
            if pair_rank < 0:
                pair_rank = r
            elif pair_rank2 < 0:
                pair_rank2 = r

    # Four of a kind
    if quad_rank >= 0:
        kicker_mask = ranks_present & ~(_i32(1) << quad_rank)
        return (_i32(7) << 24) | (quad_rank << 16) | _top_n_bits(kicker_mask, 1)

    # Full house
    if trips_rank >= 0:
        fh_pair = _i32(-1)
        if trips_rank2 >= 0:
            fh_pair = trips_rank2
        elif pair_rank >= 0:
            fh_pair = pair_rank
        if fh_pair >= 0:
            return (_i32(6) << 24) | (trips_rank << 16) | (fh_pair << 12)

    # Flush
    if flush_mask >= 0:
        return (_i32(5) << 24) | _top_n_bits(flush_mask, 5)

    # Straight
    if straight_high >= 0:
        return (_i32(4) << 24) | (straight_high << 16)

    # Three of a kind
    if trips_rank >= 0:
        kicker_mask = ranks_present & ~(_i32(1) << trips_rank)
        return (_i32(3) << 24) | (trips_rank << 16) | _top_n_bits(kicker_mask, 2)

    # Two pair
    if pair_rank >= 0 and pair_rank2 >= 0:
        kicker_mask = ranks_present & ~(
            (_i32(1) << pair_rank) | (_i32(1) << pair_rank2)
        )
        return (
            (_i32(2) << 24)
            | (pair_rank << 16)
            | (pair_rank2 << 12)
            | _top_n_bits(kicker_mask, 1)
        )

    # One pair
    if pair_rank >= 0:
        kicker_mask = ranks_present & ~(_i32(1) << pair_rank)
        return (_i32(1) << 24) | (pair_rank << 16) | _top_n_bits(kicker_mask, 3)

    # High card
    return _top_n_bits(ranks_present, 5)


@nb.njit(_i32(nb.int8[:]), cache=True)
def evaluate7(cards):
    """Evaluate a 7-card hand. Convenience wrapper."""
    return _evaluate_n(cards, 7)


@nb.njit(_i32(nb.int8[:], _i32), cache=True)
def evaluate(cards, n):
    """Evaluate best 5-card hand from n cards (5 <= n <= 7)."""
    return _evaluate_n(cards, n)


# Card conversion helpers

_RANK_ORDER = "23456789TJQKA"
_SUIT_ORDER = "shdc"
_RANK_MAP = {r: i for i, r in enumerate(_RANK_ORDER)}
_SUIT_MAP = {s: i for i, s in enumerate(_SUIT_ORDER)}


def card_str_to_int(s: str) -> int:
    return _RANK_MAP[s[0]] * 4 + _SUIT_MAP[s[1]]


def card_int_to_str(c: int) -> str:
    return _RANK_ORDER[c >> 2] + _SUIT_ORDER[c & 3]
