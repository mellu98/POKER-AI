"""
Multi-street Texas Hold'em rules.
4 streets: preflop, flop, turn, river.
Each street has a Kuhn-like betting round:
  - if no bet yet: actions are Check (k) or Bet (b)
  - if a bet was made: actions are Fold (f) or Call (c)
Bet size is fixed at 1 chip.
Max 1 bet + 1 call per street (no re-raise).
"""
from typing import List, Tuple

from texas_engine import Card, Deck, deal_hand


def deal_progressive_hand() -> Tuple[List[Card], List[Card], List[List[Card]]]:
    """Deal a hand and split the board by street."""
    p0, p1, board = deal_hand()
    board_by_street = [
        [],          # preflop
        board[:3],   # flop
        board[:4],   # turn
        board[:5],   # river
    ]
    return p0, p1, board_by_street


def has_bet(street_history: str) -> bool:
    return 'b' in street_history


def available_actions(street_history: str) -> List[str]:
    """Return the two legal actions for the current decision."""
    if not has_bet(street_history):
        return ['k', 'b']  # check, bet
    return ['f', 'c']      # fold, call


def current_player(street_history: str) -> int:
    return len(street_history) % 2


def is_street_terminal(street_history: str) -> bool:
    """A street ends when both players agree (checks/calls) or someone folds."""
    if 'f' in street_history:
        return True
    if street_history == 'kk':
        return True
    if street_history in ('bc', 'kbc'):
        return True
    return False


def is_terminal(history_by_street: Tuple[str, ...]) -> bool:
    """Hand is over if someone folded, or river street ended."""
    for h in history_by_street:
        if 'f' in h:
            return True
    if len(history_by_street) >= 4:
        if is_street_terminal(history_by_street[3]):
            return True
    return False


def _player_investment_in_street(street_history: str, player: int) -> int:
    """How many chips the player put in this street (bet or call)."""
    invest = 0
    for i, action in enumerate(street_history):
        if i % 2 != player:
            continue
        if action in ('b', 'c'):
            invest += 1
    return invest


def investment(history_by_street: Tuple[str, ...], player: int) -> int:
    """Total chips invested by player in the hand (ante + bets + calls)."""
    total = 1  # ante
    for h in history_by_street:
        total += _player_investment_in_street(h, player)
    return total


def who_folded(history_by_street: Tuple[str, ...]) -> int:
    """Return the player index who folded, or None."""
    for h in history_by_street:
        if 'f' in h:
            idx = h.index('f')
            return idx % 2
    return None


def payoff(cards: Tuple[List[Card], List[Card], List[Card]],
           history_by_street: Tuple[str, ...],
           player: int) -> float:
    """
    Net payoff for `player`.
    Winner receives the opponent's total investment.
    Loser loses their own total investment.
    """
    p0_hole, p1_hole, board = cards
    folder = who_folded(history_by_street)

    if folder is not None:
        if folder == player:
            return -investment(history_by_street, player)
        return investment(history_by_street, 1 - player)

    # Showdown
    from evaluator import evaluate_seven
    p0_rank = evaluate_seven(p0_hole, board)
    p1_rank = evaluate_seven(p1_hole, board)

    if p0_rank < p1_rank:
        winner = 0
    elif p0_rank > p1_rank:
        winner = 1
    else:
        return 0.0

    if winner == player:
        return investment(history_by_street, 1 - player)
    return -investment(history_by_street, player)
