"""
Board Texture Analyzer

Analyzes community cards to determine board texture characteristics
for postflop strategy decisions.

Texture Categories:
- Dry: Disconnected, rainbow, low draw potential (e.g., A72r)
- Medium: Some draws possible, moderate connectivity (e.g., KT4ss)
- Wet: High connectivity, flush/straight draws (e.g., JT9ss)
- Dynamic: Paired boards, many turn/river changes possible
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import Counter


RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
RANK_VALUES = {r: 14 - i for i, r in enumerate(RANKS)}  # A=14, K=13, ..., 2=2


@dataclass
class BoardTexture:
    """Complete analysis of board texture."""
    # Basic characteristics
    is_paired: bool
    is_trips: bool
    is_monotone: bool          # 3+ same suit
    is_two_tone: bool          # Exactly 2 suits
    is_rainbow: bool           # 3 different suits (on flop)

    # Draw possibilities
    flush_possible: bool       # 3+ same suit on board
    flush_draw_possible: bool  # 2 same suit (can make flush with 2 cards)
    straight_possible: bool    # Board allows made straight
    straight_draw_possible: bool  # OESD or gutshot possible

    # Texture metrics
    connectivity: float        # 0.0 (A72r) to 1.0 (JT9)
    high_card: str            # Highest card on board
    high_card_value: int

    # Classification
    texture_category: str      # "dry", "medium", "wet", "dynamic"
    texture_score: float       # 0.0 (very dry) to 1.0 (very wet)

    # Draw counts
    num_flush_cards: int       # Cards of most common suit
    num_straight_cards: int    # Cards that could make straight

    def __str__(self) -> str:
        return f"{self.texture_category.upper()} ({self.texture_score:.2f}): {self.high_card}-high"


class BoardAnalyzer:
    """
    Analyzes poker board textures for strategic decisions.

    Usage:
        analyzer = BoardAnalyzer()
        texture = analyzer.analyze(['Ah', 'Kd', '2c'])
        print(texture.texture_category)  # "dry"
    """

    def analyze(self, community_cards: List[str]) -> BoardTexture:
        """
        Analyze community cards and return complete texture analysis.

        Args:
            community_cards: List of cards like ['Ah', 'Kd', '2c']

        Returns:
            BoardTexture with all analysis fields populated
        """
        if not community_cards:
            return self._empty_texture()

        # Parse cards
        ranks = [card[0] for card in community_cards]
        suits = [card[1] for card in community_cards]
        rank_values = [RANK_VALUES[r] for r in ranks]

        # Basic characteristics
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        is_paired = any(c >= 2 for c in rank_counts.values())
        is_trips = any(c >= 3 for c in rank_counts.values())

        num_suits = len(suit_counts)
        most_common_suit_count = max(suit_counts.values())

        is_monotone = most_common_suit_count >= 3
        is_two_tone = num_suits == 2 and len(community_cards) >= 3
        is_rainbow = num_suits >= 3

        # Flush analysis
        flush_possible = most_common_suit_count >= 3
        flush_draw_possible = most_common_suit_count >= 2

        # Straight analysis
        straight_possible, straight_draw_possible = self._analyze_straights(rank_values)

        # Connectivity (gaps between sorted ranks)
        connectivity = self._calculate_connectivity(rank_values)

        # High card
        high_card_value = max(rank_values)
        high_card = RANKS[14 - high_card_value]

        # Calculate texture score
        texture_score = self._calculate_texture_score(
            is_paired=is_paired,
            is_monotone=is_monotone,
            connectivity=connectivity,
            flush_draw_possible=flush_draw_possible,
            straight_draw_possible=straight_draw_possible,
            high_card_value=high_card_value
        )

        # Classify texture
        texture_category = self._classify_texture(
            texture_score=texture_score,
            is_paired=is_paired,
            is_trips=is_trips
        )

        return BoardTexture(
            is_paired=is_paired,
            is_trips=is_trips,
            is_monotone=is_monotone,
            is_two_tone=is_two_tone,
            is_rainbow=is_rainbow,
            flush_possible=flush_possible,
            flush_draw_possible=flush_draw_possible,
            straight_possible=straight_possible,
            straight_draw_possible=straight_draw_possible,
            connectivity=connectivity,
            high_card=high_card,
            high_card_value=high_card_value,
            texture_category=texture_category,
            texture_score=texture_score,
            num_flush_cards=most_common_suit_count,
            num_straight_cards=len(set(rank_values))
        )

    def _empty_texture(self) -> BoardTexture:
        """Return texture for preflop (no community cards)."""
        return BoardTexture(
            is_paired=False,
            is_trips=False,
            is_monotone=False,
            is_two_tone=False,
            is_rainbow=False,
            flush_possible=False,
            flush_draw_possible=False,
            straight_possible=False,
            straight_draw_possible=False,
            connectivity=0.0,
            high_card="",
            high_card_value=0,
            texture_category="preflop",
            texture_score=0.0,
            num_flush_cards=0,
            num_straight_cards=0
        )

    def _analyze_straights(self, rank_values: List[int]) -> Tuple[bool, bool]:
        """
        Analyze straight possibilities on the board.

        Returns:
            (straight_possible, straight_draw_possible)
        """
        if len(rank_values) < 3:
            return False, False

        sorted_vals = sorted(set(rank_values), reverse=True)

        # Check for made straight (5 consecutive)
        straight_possible = False
        if len(sorted_vals) >= 5:
            for i in range(len(sorted_vals) - 4):
                if sorted_vals[i] - sorted_vals[i + 4] == 4:
                    straight_possible = True
                    break
            # Check wheel (A2345)
            if set([14, 5, 4, 3, 2]).issubset(set(sorted_vals)):
                straight_possible = True

        # Check for straight draw (3+ cards within 5 ranks)
        straight_draw_possible = False
        for i, high in enumerate(sorted_vals):
            low = high - 4
            cards_in_range = sum(1 for v in sorted_vals if low <= v <= high)
            if cards_in_range >= 3:
                straight_draw_possible = True
                break

        # Also check wheel draw
        wheel_ranks = [14, 5, 4, 3, 2]
        wheel_count = sum(1 for v in sorted_vals if v in wheel_ranks)
        if wheel_count >= 3:
            straight_draw_possible = True

        return straight_possible, straight_draw_possible

    def _calculate_connectivity(self, rank_values: List[int]) -> float:
        """
        Calculate board connectivity score.

        High connectivity: cards close together (JT9 = 1.0)
        Low connectivity: cards far apart (A72 = 0.0)

        Returns:
            Float from 0.0 to 1.0
        """
        if len(rank_values) < 2:
            return 0.0

        sorted_vals = sorted(set(rank_values), reverse=True)

        # Calculate total gaps
        total_gap = 0
        for i in range(len(sorted_vals) - 1):
            gap = sorted_vals[i] - sorted_vals[i + 1] - 1
            total_gap += gap

        # Maximum possible gap for this many cards
        max_gap = (len(sorted_vals) - 1) * 11  # Worst case: A..2

        if max_gap == 0:
            return 1.0

        # Invert so lower gaps = higher connectivity
        connectivity = 1.0 - (total_gap / max_gap)

        # Boost for consecutive cards
        consecutive_count = 0
        for i in range(len(sorted_vals) - 1):
            if sorted_vals[i] - sorted_vals[i + 1] == 1:
                consecutive_count += 1

        connectivity = min(1.0, connectivity + consecutive_count * 0.1)

        return round(connectivity, 2)

    def _calculate_texture_score(self,
                                  is_paired: bool,
                                  is_monotone: bool,
                                  connectivity: float,
                                  flush_draw_possible: bool,
                                  straight_draw_possible: bool,
                                  high_card_value: int) -> float:
        """
        Calculate overall texture wetness score.

        0.0 = Very dry (A72 rainbow)
        1.0 = Very wet (JT9 monotone)
        """
        score = 0.0

        # Connectivity contributes most
        score += connectivity * 0.4

        # Flush draws
        if is_monotone:
            score += 0.25
        elif flush_draw_possible:
            score += 0.15

        # Straight draws
        if straight_draw_possible:
            score += 0.15

        # Paired boards are dynamic (texture can change a lot)
        if is_paired:
            score += 0.1

        # Middle cards (6-J) are wetter than A-high or low boards
        if 6 <= high_card_value <= 11:
            score += 0.1

        return min(1.0, round(score, 2))

    def _classify_texture(self,
                          texture_score: float,
                          is_paired: bool,
                          is_trips: bool) -> str:
        """
        Classify texture into category.

        Categories:
        - dry: Low connectivity, rainbow, few draws
        - medium: Moderate connectivity or two-tone
        - wet: High connectivity, flush possible, many draws
        - dynamic: Paired board that can change significantly
        """
        if is_trips:
            return "dynamic"

        if is_paired:
            # Paired boards are dynamic regardless of other factors
            return "dynamic"

        if texture_score >= 0.65:
            return "wet"
        elif texture_score >= 0.35:
            return "medium"
        else:
            return "dry"

    def get_draw_outs(self, board: List[str], hand: List[str]) -> Dict[str, int]:
        """
        Calculate number of outs for common draws.

        Args:
            board: Community cards
            hand: Player's hole cards

        Returns:
            Dict with draw types and their outs count
        """
        all_cards = board + hand
        ranks = [card[0] for card in all_cards]
        suits = [card[1] for card in all_cards]

        suit_counts = Counter(suits)
        outs = {}

        # Flush draw outs
        for suit, count in suit_counts.items():
            if count == 4:
                outs['flush'] = 9  # 13 - 4 already seen
                break
            elif count == 3:
                # Backdoor flush
                outs['backdoor_flush'] = 10

        # Straight draw outs (simplified)
        rank_values = sorted([RANK_VALUES[r] for r in ranks])
        unique_vals = sorted(set(rank_values))

        # Check for OESD (8 outs)
        for i in range(len(unique_vals) - 3):
            window = unique_vals[i:i+4]
            if window[-1] - window[0] == 3:
                # 4 consecutive
                if window[0] > 2 and window[-1] < 14:
                    outs['oesd'] = 8
                else:
                    outs['gutshot'] = 4
                break

        return outs

    def get_best_possible_hand(self, board: List[str]) -> str:
        """
        Determine the nuts (best possible hand) on this board.

        Args:
            board: Community cards

        Returns:
            String description of the nuts
        """
        if len(board) < 3:
            return "unknown"

        texture = self.analyze(board)

        if texture.is_trips:
            return "quads"

        if texture.is_paired:
            return "full_house"

        if texture.flush_possible:
            return "flush"

        if texture.straight_possible:
            return "straight"

        # Unpaired board
        return "set"
