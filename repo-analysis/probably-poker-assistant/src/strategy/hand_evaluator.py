"""
Poker hand evaluation.
Identifies hand types and calculates hand strength.
"""
from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
from src.utils.logger import logger

class HandType(Enum):
    """Poker hand types."""
    ROYAL_FLUSH = 10
    STRAIGHT_FLUSH = 9
    FOUR_OF_A_KIND = 8
    FULL_HOUSE = 7
    FLUSH = 6
    STRAIGHT = 5
    THREE_OF_A_KIND = 4
    TWO_PAIR = 3
    PAIR = 2
    HIGH_CARD = 1

@dataclass
class HandEvaluation:
    """Result of hand evaluation."""
    hand_type: HandType
    hand_strength: float  # 0-1 scale
    description: str
    best_five: List[str]  # Best 5-card combination
    kickers: List[str]
    
    def __str__(self):
        return f"{self.hand_type.name} - {self.description}"

class HandEvaluator:
    """Evaluate poker hands."""
    
    def __init__(self):
        """Initialize hand evaluator."""
        self.rank_values = {
            'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
            '9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
            '4': 4, '3': 3, '2': 2
        }
    
    def parse_card(self, card: str) -> Tuple[int, str]:
        """
        Parse card string into rank and suit.
        
        Args:
            card: Card string like "Ah" or "Kd"
        
        Returns:
            Tuple of (rank_value, suit)
        """
        rank = card[0]
        suit = card[1]
        return (self.rank_values[rank], suit)
    
    def evaluate(self, hole_cards: List[str], 
                community_cards: List[str]) -> HandEvaluation:
        """
        Evaluate best 5-card hand.
        
        Args:
            hole_cards: Player's 2 hole cards
            community_cards: Community cards (0-5)
        
        Returns:
            HandEvaluation object
        """
        all_cards = hole_cards + community_cards
        
        if len(all_cards) < 2:
            return self._create_default_evaluation()
        
        # Parse cards
        parsed = [self.parse_card(card) for card in all_cards]
        
        # Check for each hand type (highest to lowest)
        result = (
            self._check_royal_flush(parsed) or
            self._check_straight_flush(parsed) or
            self._check_four_of_kind(parsed) or
            self._check_full_house(parsed) or
            self._check_flush(parsed) or
            self._check_straight(parsed) or
            self._check_three_of_kind(parsed) or
            self._check_two_pair(parsed) or
            self._check_pair(parsed) or
            self._check_high_card(parsed)
        )
        
        logger.info(f"Hand evaluation: {result}")
        return result
    
    def _create_default_evaluation(self) -> HandEvaluation:
        """Create default evaluation for invalid hands."""
        return HandEvaluation(
            hand_type=HandType.HIGH_CARD,
            hand_strength=0.0,
            description="No valid hand",
            best_five=[],
            kickers=[]
        )
    
    def _check_flush(self, cards: List[Tuple[int, str]]) -> Optional[HandEvaluation]:
        """Check for flush."""
        suits = [s for r, s in cards]
        suit_counts = Counter(suits)
        
        for suit, count in suit_counts.items():
            if count >= 5:
                flush_cards = [r for r, s in cards if s == suit]
                flush_cards.sort(reverse=True)
                best_five = flush_cards[:5]
                
                # Calculate strength (better flush = higher cards)
                strength = 0.6 + (sum(best_five[:3]) / (14*3)) * 0.1
                
                return HandEvaluation(
                    hand_type=HandType.FLUSH,
                    hand_strength=strength,
                    description=f"Flush, {self._rank_name(best_five[0])} high",
                    best_five=[f"{self._rank_name(r)}{suit}" for r in best_five],
                    kickers=[]
                )
        return None
    
    def _check_straight(self, cards: List[Tuple[int, str]]) -> Optional[HandEvaluation]:
        """Check for straight."""
        ranks = sorted(set([r for r, s in cards]), reverse=True)
        
        # Check for ace-low straight (A-2-3-4-5)
        if 14 in ranks:
            ranks.append(1)
        
        for i in range(len(ranks) - 4):
            if ranks[i] - ranks[i+4] == 4:
                high_card = ranks[i]
                strength = 0.5 + (high_card / 14) * 0.05
                
                return HandEvaluation(
                    hand_type=HandType.STRAIGHT,
                    hand_strength=strength,
                    description=f"Straight, {self._rank_name(high_card)} high",
                    best_five=[],
                    kickers=[]
                )
        return None
    
    def _check_straight_flush(self, cards: List[Tuple[int, str]]) -> Optional[HandEvaluation]:
        """Check for straight flush."""
        suits = set([s for r, s in cards])
        
        for suit in suits:
            suited_cards = [(r, s) for r, s in cards if s == suit]
            if len(suited_cards) < 5:
                continue
            straight = self._check_straight(suited_cards)
            if straight:
                return HandEvaluation(
                    hand_type=HandType.STRAIGHT_FLUSH,
                    hand_strength=0.9,
                    description="Straight Flush",
                    best_five=[],
                    kickers=[]
                )
        return None
    
    def _check_royal_flush(self, cards: List[Tuple[int, str]]) -> Optional[HandEvaluation]:
        """Check for royal flush."""
        straight_flush = self._check_straight_flush(cards)
        if straight_flush:
            ranks = [r for r, s in cards]
            if 14 in ranks and 13 in ranks:  # Has Ace and King
                return HandEvaluation(
                    hand_type=HandType.ROYAL_FLUSH,
                    hand_strength=1.0,
                    description="Royal Flush",
                    best_five=[],
                    kickers=[]
                )
        return None
    
    def _check_four_of_kind(self, cards: List[Tuple[int, str]]) -> Optional[HandEvaluation]:
        """Check for four of a kind."""
        ranks = [r for r, s in cards]
        rank_counts = Counter(ranks)
        
        for rank, count in rank_counts.items():
            if count == 4:
                strength = 0.8 + (rank / 14) * 0.05
                return HandEvaluation(
                    hand_type=HandType.FOUR_OF_A_KIND,
                    hand_strength=strength,
                    description=f"Four {self._rank_name(rank)}",
                    best_five=[],
                    kickers=[]
                )
        return None
    
    def _check_full_house(self, cards: List[Tuple[int, str]]) -> Optional[HandEvaluation]:
        """Check for full house."""
        ranks = [r for r, s in cards]
        rank_counts = Counter(ranks)
        
        trips = [r for r, c in rank_counts.items() if c >= 3]
        pairs = [r for r, c in rank_counts.items() if c >= 2]
        
        if trips and len(pairs) >= 2:
            trip_rank = max(trips)
            pair_rank = max([p for p in pairs if p != trip_rank])
            strength = 0.7 + (trip_rank / 14) * 0.05
            
            return HandEvaluation(
                hand_type=HandType.FULL_HOUSE,
                hand_strength=strength,
                description=f"{self._rank_name(trip_rank)} full of {self._rank_name(pair_rank)}",
                best_five=[],
                kickers=[]
            )
        return None
    
    def _check_three_of_kind(self, cards: List[Tuple[int, str]]) -> Optional[HandEvaluation]:
        """Check for three of a kind."""
        ranks = [r for r, s in cards]
        rank_counts = Counter(ranks)
        
        trips = sorted([r for r, c in rank_counts.items() if c >= 3], reverse=True)
        if trips:
            rank = trips[0]
            strength = 0.4 + (rank / 14) * 0.05
            return HandEvaluation(
                hand_type=HandType.THREE_OF_A_KIND,
                hand_strength=strength,
                description=f"Three {self._rank_name(rank)}",
                best_five=[],
                kickers=[]
            )
        return None
    
    def _check_two_pair(self, cards: List[Tuple[int, str]]) -> Optional[HandEvaluation]:
        """Check for two pair."""
        ranks = [r for r, s in cards]
        rank_counts = Counter(ranks)
        
        pairs = sorted([r for r, c in rank_counts.items() if c >= 2], reverse=True)
        
        if len(pairs) >= 2:
            high_pair = pairs[0]
            low_pair = pairs[1]
            strength = 0.3 + ((high_pair + low_pair) / (14*2)) * 0.05
            
            return HandEvaluation(
                hand_type=HandType.TWO_PAIR,
                hand_strength=strength,
                description=f"Two Pair, {self._rank_name(high_pair)} and {self._rank_name(low_pair)}",
                best_five=[],
                kickers=[]
            )
        return None
    
    def _check_pair(self, cards: List[Tuple[int, str]]) -> Optional[HandEvaluation]:
        """Check for pair."""
        ranks = [r for r, s in cards]
        rank_counts = Counter(ranks)
        
        pairs = sorted([r for r, c in rank_counts.items() if c >= 2], reverse=True)
        if pairs:
            rank = pairs[0]
            strength = 0.2 + (rank / 14) * 0.05
            return HandEvaluation(
                hand_type=HandType.PAIR,
                hand_strength=strength,
                description=f"Pair of {self._rank_name(rank)}",
                best_five=[],
                kickers=[]
            )
        return None
    
    def _check_high_card(self, cards: List[Tuple[int, str]]) -> HandEvaluation:
        """High card (always matches)."""
        ranks = sorted([r for r, s in cards], reverse=True)
        best_five = ranks[:5]
        
        # Calculate granular strength using base-15 for the top 5 cards
        # This ensures correct tie-breaking
        score = 0
        for i, rank in enumerate(best_five):
            weight = 15 ** (4 - i)
            score += rank * weight
            
        max_score = 14 * (15**4) + 13 * (15**3) + 12 * (15**2) + 11 * 15 + 10
        norm_score = score / max_score
        
        strength = norm_score * 0.18  # Increased range slightly to 0-0.18 (Pair is 0.2)
        
        return HandEvaluation(
            hand_type=HandType.HIGH_CARD,
            hand_strength=strength,
            description=f"{self._rank_name(best_five[0])} high",
            best_five=[],
            kickers=[]
        )
    
    def _rank_name(self, rank: int) -> str:
        """Convert rank value to name."""
        rank_names = {
            14: 'Ace', 13: 'King', 12: 'Queen', 11: 'Jack', 10: 'Ten',
            9: 'Nine', 8: 'Eight', 7: 'Seven', 6: 'Six', 5: 'Five',
            4: 'Four', 3: 'Three', 2: 'Two', 1: 'Ace'
        }
        return rank_names.get(rank, str(rank))
