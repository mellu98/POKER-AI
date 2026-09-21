"""
Texas Hold'em engine: 52-card deck, hole cards, board cards.
"""
import random
from itertools import combinations
from typing import List, Tuple


class Card:
    """A standard playing card with rank (2-14) and suit (0-3)."""

    SUITS = ['s', 'h', 'd', 'c']  # spades, hearts, diamonds, clubs
    RANKS = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}

    def __init__(self, rank: int, suit: int):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        if self.rank == 10:
            r = 'T'
        else:
            r = self.RANKS.get(self.rank, str(self.rank))
        return f"{r}{self.SUITS[self.suit]}"

    def __eq__(self, other):
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    def to_treys(self) -> int:
        """Convert to treys integer representation."""
        from treys import Card as TreysCard
        return TreysCard.new(str(self))


class Deck:
    """Standard 52-card deck."""

    def __init__(self):
        self.cards: List[Card] = [Card(r, s) for s in range(4) for r in range(2, 15)]

    def shuffle(self):
        random.shuffle(self.cards)

    def remove(self, cards: List[Card]):
        """Remove specific cards from the deck."""
        self.cards = [c for c in self.cards if c not in cards]

    def deal(self, num: int = 1) -> List[Card]:
        dealt = self.cards[:num]
        self.cards = self.cards[num:]
        return dealt

    def remaining(self) -> List[Card]:
        return self.cards[:]

    def __repr__(self):
        return f"Deck({len(self.cards)} cards)"


def deal_hand() -> Tuple[List[Card], List[Card], List[Card]]:
    """
    Deal a complete heads-up hand.
    Returns: (player0_hole, player1_hole, board)
    """
    deck = Deck()
    deck.shuffle()
    p0 = deck.deal(2)
    p1 = deck.deal(2)
    board = deck.deal(5)
    return p0, p1, board
