"""
Hand evaluator using the treys library.
"""
from typing import List
from treys import Evaluator as TreysEvaluator, Card as TreysCard
from texas_engine import Card


_evaluator = TreysEvaluator()


def evaluate_seven(hole: List[Card], board: List[Card]) -> int:
    """
    Evaluate the best 5-card hand from 7 cards.
    Returns a rank (lower is better). Royal flush = 1.
    """
    treys_hole = [c.to_treys() for c in hole]
    treys_board = [c.to_treys() for c in board]
    return _evaluator.evaluate(treys_board, treys_hole)


def evaluate_class(hole: List[Card], board: List[Card]) -> str:
    """Return the human-readable hand class (e.g. 'Straight Flush', 'Two Pair')."""
    treys_hole = [c.to_treys() for c in hole]
    treys_board = [c.to_treys() for c in board]
    rank = _evaluator.evaluate(treys_board, treys_hole)
    return _evaluator.class_to_string(_evaluator.get_rank_class(rank))
