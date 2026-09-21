"""Monte Carlo equity calculator on top of treys.

Hero hand vs N random opponents OR vs a specified range.
"""

from __future__ import annotations

import random
from typing import Iterable

from treys import Card as TCard
from treys import Deck, Evaluator

from .game_state import Card

_EVAL = Evaluator()


def _to_treys(card: Card) -> int:
    # treys uses: rank in '23456789TJQKA', suit in 'shdc'
    return TCard.new(card.code)


def _to_treys_list(cards: Iterable[Card]) -> list[int]:
    return [_to_treys(c) for c in cards]


def equity_vs_random(
    hero: tuple[Card, Card],
    board: list[Card],
    n_opponents: int = 1,
    n_iterations: int = 5000,
    seed: int | None = None,
) -> dict:
    """Возвращает {'win': float, 'tie': float, 'lose': float, 'equity': float}.

    equity = win + tie / (n_opponents + 1)
    """
    if seed is not None:
        random.seed(seed)

    hero_t = _to_treys_list(list(hero))
    board_t = _to_treys_list(board)
    used = set(hero_t + board_t)

    wins = ties = 0

    for _ in range(n_iterations):
        # build a fresh deck minus known cards
        deck = Deck()
        deck.cards = [c for c in deck.cards if c not in used]
        random.shuffle(deck.cards)

        # deal opponents
        opp_hands = []
        idx = 0
        for _ in range(n_opponents):
            opp_hands.append(deck.cards[idx : idx + 2])
            idx += 2

        # complete board to 5 cards
        need = 5 - len(board_t)
        runout = deck.cards[idx : idx + need]
        full_board = board_t + runout

        hero_score = _EVAL.evaluate(full_board, hero_t)
        opp_scores = [_EVAL.evaluate(full_board, h) for h in opp_hands]

        best_opp = min(opp_scores)  # treys: lower = better
        if hero_score < best_opp:
            wins += 1
        elif hero_score == best_opp:
            ties += 1
        # else: loss

    losses = n_iterations - wins - ties
    win_p = wins / n_iterations
    tie_p = ties / n_iterations
    lose_p = losses / n_iterations
    equity = win_p + tie_p / (n_opponents + 1)

    return {
        "win": win_p,
        "tie": tie_p,
        "lose": lose_p,
        "equity": equity,
        "iterations": n_iterations,
    }


def hand_strength_summary(
    hero: tuple[Card, Card],
    board: list[Card],
    n_opponents: int = 1,
    n_iterations: int = 5000,
) -> str:
    """Краткая текстовая сводка для UI."""
    r = equity_vs_random(hero, board, n_opponents, n_iterations)
    return (
        f"vs {n_opponents} random: equity {r['equity']:.1%} "
        f"(win {r['win']:.1%} / tie {r['tie']:.1%} / lose {r['lose']:.1%})"
    )
