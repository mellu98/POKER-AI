"""
Preflop Chart Lookup — GTO ranges for real-time preflop decisions.

Replaces the under-trained CFR preflop models with well-established GTO ranges.
Ranges are expressed in compact notation and expanded at import time.
"""
from typing import List, Dict, Tuple

RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]


def _expand(patterns: List[str]) -> List[str]:
    """Expand compact range patterns into explicit hand strings.

    Supported patterns:
        "A2s+"   -> A2s, A3s, ..., AKs
        "K9s+"   -> K9s, KTs, KJs, KQs
        "KJo+"   -> KJo, KQo
        "AQo+"   -> AQo, AKo
        "77+"    -> 77, 88, ..., AA
        "AA"     -> AA
        "AKs"    -> AKs
    """
    result = []
    for p in patterns:
        if len(p) >= 3 and p.endswith("+"):
            core = p[:-1]
            if len(core) == 2 and core[0] == core[1]:
                # Pair range  e.g. "77+"
                idx = RANKS.index(core[0])
                for i in range(idx, -1, -1):
                    result.append(RANKS[i] * 2)
            elif len(core) == 3:
                high, low, suit = core[0], core[1], core[2]
                idx_high = RANKS.index(high)
                idx_low = RANKS.index(low)
                # low must be worse than high (higher index)
                for i in range(idx_low, idx_high, -1):
                    result.append(high + RANKS[i] + suit)
            else:
                result.append(core)
        else:
            result.append(p)
    return result


# --------------------------------------------------------------------------- #
#  GTO Ranges — 6-max, 100bb, simplified from Pio / GTO+ baseline charts
# --------------------------------------------------------------------------- #

_OPEN_RANGES: Dict[str, Dict[str, List[str]]] = {
    "UTG": {
        "raise": _expand(["77+", "A9s+", "KTs+", "QJs", "JTs", "T9s", "98s", "87s", "AQo+"]),
    },
    "MP": {
        "raise": _expand(["66+", "A2s+", "K9s+", "Q9s+", "J9s+", "T8s+", "98s", "87s", "76s", "65s", "AJo+", "KQo"]),
    },
    "CO": {
        "raise": _expand(["22+", "A2s+", "K9s+", "Q9s+", "J9s+", "T8s+", "98s", "87s", "76s", "65s", "54s", "ATo+", "KJo+", "QJo"]),
    },
    "BTN": {
        "raise": _expand(["22+", "A2s+", "K2s+", "Q2s+", "J6s+", "T6s+", "96s+", "86s+", "76s", "65s", "54s", "A2o+", "K9o+", "Q9o+", "J9o+", "T9o", "98o"]),
    },
    "SB": {
        "raise": _expand(["22+", "A2s+", "K2s+", "Q5s+", "J7s+", "T7s+", "96s+", "86s+", "76s", "65s", "54s", "A7o+", "K9o+", "Q9o+", "J9o+", "T9o", "98o"]),
    },
}

# When facing a raise (vs 3bet / call decision)
_VS_RAISE_RANGES: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "UTG": {
        "three_bet": _expand(["QQ+", "AKs", "AKo"]),
        "call": _expand(["JJ", "TT", "99", "88", "77", "AQs", "AJs", "ATs", "KQs", "QJs", "JTs", "T9s", "98s", "87s", "AQo"]),
    },
    "MP": {
        "three_bet": _expand(["QQ+", "AKs", "AKo", "AQs", "AJs"]),
        "call": _expand(["JJ", "TT", "99", "88", "77", "66", "ATs+", "KQs", "QJs", "JTs", "T9s", "98s", "87s", "76s", "65s", "AQo", "KQo"]),
    },
    "CO": {
        "three_bet": _expand(["JJ+", "AKs", "AKo", "AQs", "AJs", "ATs", "KQs", "A5s", "A4s"]),
        "call": _expand(["22-TT", "A2s-A9s", "K9s-KJs", "Q9s-QTs", "J9s", "T8s", "98s", "87s", "76s", "65s", "54s", "ATo", "KJo", "QJo"]),
    },
    "BTN": {
        "three_bet": _expand(["JJ+", "AKs", "AKo", "AQs", "AJs", "ATs", "KQs", "A5s", "A4s", "A3s", "A2s"]),
        "call": _expand(["22-TT", "K2s-KJs", "Q2s-QTs", "J6s-J9s", "T6s-T9s", "96s-98s", "86s-87s", "76s", "65s", "54s", "A2o-ATo", "K9o-KJo", "Q9o-QJo", "J9o", "T9o", "98o"]),
    },
    "SB": {
        "three_bet": _expand(["TT+", "AKs", "AKo", "AQs", "AJs", "ATs", "KQs", "A5s-A2s"]),
        "call": _expand(["22-99", "K2s-KJs", "Q5s-QTs", "J7s-J9s", "T7s-T9s", "96s-98s", "86s-87s", "76s", "65s", "54s", "A7o-ATo", "K9o-KJo", "Q9o-QJo", "J9o", "T9o", "98o"]),
    },
    "BB": {
        "three_bet": _expand(["TT+", "AKs", "AKo", "AQs", "AJs", "ATs", "KQs", "A5s", "A4s", "A3s", "A2s"]),
        "call": _expand(["22-99", "A2s-A9s", "K2s-KJs", "Q2s-QTs", "J6s-J9s", "T6s-T9s", "96s-98s", "86s-87s", "76s", "65s", "54s", "A2o-ATo", "K9o-KJo", "Q9o-QJo", "J9o", "T9o", "98o"]),
    },
}


def _normalize_hand(card1: str, card2: str) -> str:
    """Convert 'As' 'Kh' into chart notation 'AKo'."""
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    idx1 = RANKS.index(r1)
    idx2 = RANKS.index(r2)
    if idx1 == idx2:
        return r1 + r2
    if idx1 > idx2:
        r1, r2 = r2, r1
    suited = "s" if s1 == s2 else "o"
    return r1 + r2 + suited


def _get_situation(position: str, history: List[str]) -> Tuple[str, str]:
    """Return (chart_position, situation_key) from game context."""
    pos = position.upper()

    # Mappa posizioni 9-max sui range 6-max disponibili
    pos_map = {
        "UTG+1": "MP",
        "UTG+2": "MP",
        "LJ": "MP",
        "HJ": "CO",
    }
    pos = pos_map.get(pos, pos)

    # Determine if we are facing a raise or opening
    has_bet = any(h.startswith("b") for h in history)
    if has_bet:
        return pos, "vs_raise"
    return pos, "open"


def lookup(
    hole: List[str],
    position: str,
    history: List[str],
    pot: float = 0,
    stack: float = 1000,
    big_blind: float = 2.0,
) -> Dict:
    """
    Return a preflop recommendation from GTO chart lookup.

    Returns same shape as AssistantEngine.recommend:
        {
            "action": str,
            "strategy": dict,
            "infoset_key": str,
            "stage": "preflop",
        }
    """
    hand = _normalize_hand(hole[0], hole[1])
    pos, situation = _get_situation(position, history)

    # Fallback to BTN if unknown position
    if pos not in _OPEN_RANGES:
        pos = "BTN"

    if situation == "open":
        chart = _OPEN_RANGES[pos]
        if hand in chart.get("raise", []):
            action = "bMIN"
            strategy = {"bMIN": 1.0}
        else:
            action = "f"
            strategy = {"f": 1.0}
    else:
        chart = _VS_RAISE_RANGES.get(pos, _VS_RAISE_RANGES["BB"])
        if hand in chart.get("three_bet", []):
            action = "bMAX"
            strategy = {"bMAX": 1.0}
        elif hand in chart.get("call", []):
            action = "c"
            strategy = {"c": 1.0}
        else:
            action = "f"
            strategy = {"f": 1.0}

    return {
        "action": action,
        "strategy": strategy,
        "infoset_key": f"{pos}_{situation}_{hand}",
        "stage": "preflop",
    }


# --------------------------------------------------------------------------- #
#  Quick sanity test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    test_hands = [
        (["As", "Ah"], "BTN", [], "open"),
        (["7d", "2c"], "UTG", [], "open"),
        (["As", "Kh"], "CO", ["b10"], "vs_raise"),
        (["Qd", "Jh"], "BB", ["b10"], "vs_raise"),
    ]
    for hole, pos, hist, sit in test_hands:
        rec = lookup(hole, pos, hist)
        print(f"{sit:10s} | {pos:3s} | {_normalize_hand(hole[0], hole[1]):4s} -> {rec['action']:5s} | {rec['strategy']}")
