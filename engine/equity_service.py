"""
Equity Service — real-time hand equity calculation.

Uses Monte-Carlo simulation against random opponent hands.
Caches results for instant repeated lookups.
"""
import random
from functools import lru_cache
from typing import List, Tuple

from fast_evaluator import Deck, get_player_score
from phevaluator import evaluate_cards

try:
    from treys import Evaluator, Card as TreysCard
    _treys_eval = Evaluator()
    _HAS_TREYS = True
except Exception:
    _HAS_TREYS = False


def _to_tuple(cards: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(cards))


@lru_cache(maxsize=4096)
def _cached_equity(hole: Tuple[str, ...], board: Tuple[str, ...], n: int) -> float:
    """Monte-Carlo equity vs one random opponent."""
    hole = list(hole)
    board = list(board)
    used = set(hole + board)
    deck = Deck(excluded_cards=list(used))

    wins = 0.0
    for _ in range(n):
        random.shuffle(deck)
        opp = deck[:2]
        # Complete board to 5 cards if needed
        remaining = deck[2:]
        full_board = board + remaining[: 5 - len(board)]

        our_score = get_player_score(hole, full_board)
        opp_score = get_player_score(opp, full_board)

        if our_score < opp_score:
            wins += 1.0
        elif our_score == opp_score:
            wins += 0.5

    return wins / n if n > 0 else 0.5


def calculate_equity(hole: List[str], board: List[str], n: int = 2000) -> float:
    """
    Return equity (0.0 to 1.0) of hole cards vs a random opponent hand.
    """
    return _cached_equity(_to_tuple(hole), _to_tuple(board), n)


def calculate_equity_vs_range(
    hole: List[str],
    board: List[str],
    opponent_range: List[List[str]],
    n: int = 500,
) -> float:
    """
    Return equity vs a specific range of opponent hands.
    opponent_range: list of [card1, card2] combinations.
    """
    if not opponent_range:
        return calculate_equity(hole, board, n)

    board = list(board)
    used = set(hole + board)
    deck = Deck(excluded_cards=list(used))

    wins = 0.0
    total = 0
    for opp_hole in opponent_range:
        if set(opp_hole) & used:
            continue
        for _ in range(n):
            random.shuffle(deck)
            remaining = [c for c in deck if c not in opp_hole]
            full_board = board + remaining[: 5 - len(board)]

            our_score = get_player_score(hole, full_board)
            opp_score = get_player_score(opp_hole, full_board)

            if our_score < opp_score:
                wins += 1.0
            elif our_score == opp_score:
                wins += 0.5
            total += 1

    return wins / total if total > 0 else 0.5


def get_hand_strength_class(hole: List[str], board: List[str]) -> str:
    """
    Human-readable hand class, e.g. 'Two Pair', 'Flush'.
    Falls back to 'Unknown' if treys is not available.
    """
    if not _HAS_TREYS:
        return "Unknown"
    if len(board) < 3:
        return "Preflop"

    def to_treys(card: str) -> int:
        return TreysCard.new(card)

    treys_hole = [to_treys(c) for c in hole]
    treys_board = [to_treys(c) for c in board]
    rank = _treys_eval.evaluate(treys_board, treys_hole)
    return _treys_eval.class_to_string(_treys_eval.get_rank_class(rank))


def benchmark():
    """Quick performance benchmark."""
    import time

    hands = [
        (["As", "Kh"], ["Qd", "Jh", "2c"]),
        (["7d", "2c"], []),
        (["Qh", "Qd"], ["Qs", "Jd", "Tc"]),
        (["Ad", "Ac"], ["As", "Kd", "3h", "7c"]),
    ]

    start = time.perf_counter()
    for hole, board in hands:
        calculate_equity(hole, board, n=2000)
    elapsed = time.perf_counter() - start

    print(f"Benchmark: {len(hands)} equities x 2000 samples in {elapsed:.3f}s")
    print(f"Average per hand: {elapsed / len(hands):.3f}s")

    # Cache hit test
    start = time.perf_counter()
    for _ in range(100):
        calculate_equity(["As", "Kh"], ["Qd", "Jh", "2c"], n=2000)
    elapsed2 = time.perf_counter() - start
    print(f"100 cached lookups in {elapsed2:.4f}s")


if __name__ == "__main__":
    eq = calculate_equity(["As", "Kh"], ["Qd", "Jh", "2c"], n=5000)
    print(f"Equity AsKh on QdJh2c: {eq:.2%}")
    print(f"Hand strength: {get_hand_strength_class(['As','Kh'], ['Qd','Jh','2c'])}")
    print()
    benchmark()
