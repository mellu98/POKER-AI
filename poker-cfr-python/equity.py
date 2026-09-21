"""
Equity calculator for Texas Hold'em.
Computes exact equity percentile of a hand on a given board
by enumerating all possible opponent hole cards from the remaining deck.
"""
import json
import os
import random
from itertools import combinations
from functools import lru_cache
from typing import List, Tuple

from texas_engine import Card, Deck
from evaluator import evaluate_seven

PREFLOP_CACHE_PATH = 'preflop_equity_cache.json'


def _hole_key(hole: List[Card]) -> str:
    c1, c2 = sorted(hole, key=lambda c: (c.rank, c.suit))
    return f"{c1}{c2}"


def _load_preflop_cache() -> dict:
    if os.path.exists(PREFLOP_CACHE_PATH):
        with open(PREFLOP_CACHE_PATH, 'r') as f:
            return json.load(f)
    return {}


def _save_preflop_cache(cache: dict) -> None:
    with open(PREFLOP_CACHE_PATH, 'w') as f:
        json.dump(cache, f)


def _compute_preflop_equity(hole: List[Card], num_samples: int = 500) -> float:
    """Monte Carlo equity for preflop hole cards vs random opponent."""
    hole_set = set(hole)
    deck = [c for c in Deck().cards if c not in hole_set]
    wins = 0.0
    total = 0
    for _ in range(num_samples):
        opp = random.sample(deck, 2)
        remaining = [c for c in deck if c not in opp]
        board = random.sample(remaining, 5)
        our_rank = evaluate_seven(hole, board)
        opp_rank = evaluate_seven(opp, board)
        total += 1
        if our_rank < opp_rank:
            wins += 1.0
        elif our_rank == opp_rank:
            wins += 0.5
    return wins / total if total > 0 else 0.5


def preflop_equity_bucket(hole: List[Card], num_buckets: int = 10) -> int:
    """
    Map preflop hole cards to an equity bucket.
    Uses a persistent JSON cache so the expensive Monte Carlo
    calculation is done only once per unique starting hand.
    """
    cache = _load_preflop_cache()
    key = _hole_key(hole)
    if key not in cache:
        eq = _compute_preflop_equity(hole, num_samples=100)
        cache[key] = eq
        _save_preflop_cache(cache)
    else:
        eq = cache[key]
    eq = min(eq, 0.9999)
    return int(eq * num_buckets)


@lru_cache(maxsize=None)
def _hand_equity_percentile(hole: Tuple[Card, ...], board: Tuple[Card, ...]) -> float:
    """
    Return the percentile (0.0 to 1.0) of the given hole cards on the given board.
    1.0 means the nuts, 0.0 means the worst possible hand.
    Equity is computed against a uniform random distribution of opponent hole cards.
    """
    our_rank = evaluate_seven(list(hole), list(board))

    # Build remaining deck
    used = set(hole + board)
    remaining = [c for c in Deck().cards if c not in used]

    total = 0
    wins = 0.0
    for opp_hole in combinations(remaining, 2):
        opp_rank = evaluate_seven(list(opp_hole), list(board))
        total += 1
        if our_rank < opp_rank:
            wins += 1.0
        elif our_rank == opp_rank:
            wins += 0.5

    return wins / total if total > 0 else 0.5


def hand_equity_percentile(hole: List[Card], board: List[Card]) -> float:
    return _hand_equity_percentile(tuple(hole), tuple(board))


def equity_bucket(hole: List[Card], board: List[Card], num_buckets: int = 10) -> int:
    """
    Map the exact equity to a discrete bucket index (0 .. num_buckets-1).
    """
    eq = _hand_equity_percentile(tuple(hole), tuple(board))
    # Clamp to [0, 0.9999] to avoid num_buckets edge case
    eq = min(eq, 0.9999)
    return int(eq * num_buckets)
