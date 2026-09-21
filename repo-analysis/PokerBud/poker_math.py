"""
Local poker math: card normalization, hand categories, pot odds, and simple equity hints.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Optional

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_VALUES = {r: i for i, r in enumerate(RANKS, start=2)}
SUIT_SYMBOLS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
SUIT_FROM_SYMBOL = {"♠": "s", "♥": "h", "♦": "d", "♣": "c", "♤": "s", "♡": "h", "♢": "d", "♧": "c"}


def normalize_card(raw: str) -> Optional[str]:
    """Normalize a card string to Rank+suit form (e.g. 'Ah', 'Td')."""
    if not raw:
        return None
    text = raw.strip().replace("10", "T").upper()
    if len(text) < 2:
        return None

    rank = text[0]
    if rank not in RANKS:
        return None

    suit_char = text[1].lower()
    if suit_char in SUIT_FROM_SYMBOL:
        suit_char = SUIT_FROM_SYMBOL[suit_char]
    if suit_char not in SUITS:
        return None

    return f"{rank}{suit_char}"


def normalize_cards(cards: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for card in cards:
        normalized = normalize_card(card)
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def format_card(card: str) -> str:
    normalized = normalize_card(card)
    if not normalized:
        return str(card)
    return f"{normalized[0]}{SUIT_SYMBOLS[normalized[1]]}"


def format_cards(cards: Iterable[str]) -> str:
    formatted = [format_card(c) for c in cards]
    return "  ".join(formatted) if formatted else "—"


def parse_card_tokens(text: str) -> list[str]:
    """Extract normalized cards from free text like 'Ah Kd' or 'A♥ K♦'."""
    if not text:
        return []
    import re

    pattern = re.compile(r"(10|[AKQJT2-9])\s*([shdcSHDC♠♥♦♣♤♡♢♧])")
    found = []
    for rank, suit in pattern.findall(text):
        card = normalize_card(f"{rank}{suit}")
        if card:
            found.append(card)
    # Also accept whitespace-separated tokens
    if not found:
        for token in text.replace(",", " ").split():
            card = normalize_card(token)
            if card:
                found.append(card)
    return normalize_cards(found)


def _hand_rank_tuple(cards: list[str]) -> tuple:
    """
    Rank a 5-card hand. Higher tuple wins.
    Categories: 8 straight flush … 0 high card.
    """
    ranks = sorted((RANK_VALUES[c[0]] for c in cards), reverse=True)
    suits = [c[1] for c in cards]
    counts: dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    by_count = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    count_values = [c for _, c in by_count]
    unique_ranks = sorted(counts.keys(), reverse=True)

    is_flush = len(set(suits)) == 1

    # Straight (wheel A-2-3-4-5)
    is_straight = False
    straight_high = 0
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        elif unique_ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            straight_high = 5

    if is_straight and is_flush:
        return (8, straight_high)
    if count_values == [4, 1]:
        return (7, by_count[0][0], by_count[1][0])
    if count_values == [3, 2]:
        return (6, by_count[0][0], by_count[1][0])
    if is_flush:
        return (5, *ranks)
    if is_straight:
        return (4, straight_high)
    if count_values == [3, 1, 1]:
        kickers = sorted([r for r, c in counts.items() if c == 1], reverse=True)
        return (3, by_count[0][0], *kickers)
    if count_values == [2, 2, 1]:
        pair_ranks = sorted([r for r, c in counts.items() if c == 2], reverse=True)
        kicker = [r for r, c in counts.items() if c == 1][0]
        return (2, pair_ranks[0], pair_ranks[1], kicker)
    if count_values == [2, 1, 1, 1]:
        kickers = sorted([r for r, c in counts.items() if c == 1], reverse=True)
        return (1, by_count[0][0], *kickers)
    return (0, *ranks)


CATEGORY_NAMES = {
    8: "Straight Flush",
    7: "Four of a Kind",
    6: "Full House",
    5: "Flush",
    4: "Straight",
    3: "Three of a Kind",
    2: "Two Pair",
    1: "One Pair",
    0: "High Card",
}


def best_five_card_hand(hole: list[str], board: list[str]) -> Optional[dict]:
    """Evaluate best 5-card hand from hole + board (needs ≥5 total cards)."""
    hole = normalize_cards(hole)
    board = normalize_cards(board)
    all_cards = hole + board
    if len(all_cards) < 5:
        return None
    if len(set(all_cards)) != len(all_cards):
        return {"error": "Duplicate cards detected"}

    best = None
    best_combo = None
    for combo in combinations(all_cards, 5):
        score = _hand_rank_tuple(list(combo))
        if best is None or score > best:
            best = score
            best_combo = combo

    return {
        "category": CATEGORY_NAMES[best[0]],
        "category_rank": best[0],
        "score": best,
        "cards": list(best_combo),
        "display": format_cards(best_combo),
    }


def preflop_hand_strength(hole: list[str]) -> Optional[dict]:
    """Rough preflop tier for two hole cards."""
    cards = normalize_cards(hole)
    if len(cards) != 2:
        return None

    r1, r2 = RANK_VALUES[cards[0][0]], RANK_VALUES[cards[1][0]]
    suited = cards[0][1] == cards[1][1]
    high, low = max(r1, r2), min(r1, r2)
    gap = high - low
    pair = high == low

    if pair and high >= 10:
        tier, label = "premium", "Premium pair"
    elif pair and high >= 7:
        tier, label = "strong", "Medium pair"
    elif pair:
        tier, label = "playable", "Small pair"
    elif high == 14 and low >= 12:
        tier, label = "premium", "Big Broadway"
    elif high == 14 and low >= 10 and suited:
        tier, label = "strong", "Suited Ace"
    elif high == 14 and low >= 10:
        tier, label = "strong", "Ace-x Broadway"
    elif high >= 12 and low >= 10 and suited:
        tier, label = "strong", "Suited Broadway"
    elif high >= 11 and low >= 10:
        tier, label = "playable", "Broadway"
    elif suited and gap <= 2 and high >= 9:
        tier, label = "playable", "Suited connector"
    elif high == 14 and suited:
        tier, label = "playable", "Suited Ace"
    else:
        tier, label = "weak", "Marginal / fold-ish"

    return {
        "tier": tier,
        "label": label,
        "suited": suited,
        "pair": pair,
        "display": format_cards(cards),
    }


def pot_odds(pot: float, to_call: float) -> Optional[dict]:
    """Return pot odds as ratio and required equity percent."""
    if pot is None or to_call is None:
        return None
    try:
        pot_f = float(pot)
        call_f = float(to_call)
    except (TypeError, ValueError):
        return None
    if call_f <= 0:
        return {"ratio": "—", "required_equity_pct": 0.0, "note": "Nothing to call"}
    if pot_f < 0:
        return None

    total = pot_f + call_f
    required = call_f / total * 100
    # Express as X:1 (pot:call style commonly shown as call needs 1/(pot/call+1))
    ratio = f"{pot_f / call_f:.2f}:1"
    return {
        "ratio": ratio,
        "required_equity_pct": round(required, 1),
        "note": f"Need ~{required:.0f}% equity to break even on a call",
    }


def stack_to_pot_ratio(stack: float, pot: float) -> Optional[float]:
    try:
        s, p = float(stack), float(pot)
    except (TypeError, ValueError):
        return None
    if p <= 0:
        return None
    return round(s / p, 2)


def summarize_situation(game_state: dict) -> dict:
    """Local quick-read summary attached to analysis results."""
    hole = normalize_cards(game_state.get("your_cards") or [])
    board = normalize_cards(game_state.get("community_cards") or [])
    pot = game_state.get("pot_size")
    to_call = game_state.get("current_bet")
    stack = game_state.get("your_stack")

    summary: dict = {
        "hole_display": format_cards(hole),
        "board_display": format_cards(board),
        "pot_odds": pot_odds(pot, to_call) if to_call else pot_odds(pot, 0),
        "spr": stack_to_pot_ratio(stack, pot) if stack is not None and pot is not None else None,
    }

    if len(hole) == 2 and not board:
        summary["made_hand"] = None
        summary["preflop"] = preflop_hand_strength(hole)
    elif len(hole) + len(board) >= 5:
        summary["preflop"] = None
        summary["made_hand"] = best_five_card_hand(hole, board)
    else:
        summary["preflop"] = preflop_hand_strength(hole) if len(hole) == 2 else None
        summary["made_hand"] = None

    return summary
