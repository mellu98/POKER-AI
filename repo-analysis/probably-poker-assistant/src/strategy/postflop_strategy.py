"""
Postflop Strategy Module

Provides GTO-approximation postflop decisions based on:
- Board texture analysis
- Hand strength and draw potential
- Position (in position vs out of position)
- Bet sizing and pot geometry

Key Concepts:
- Dry boards: Bet large, high frequency
- Wet boards: Bet smaller, lower frequency (or check)
- Draw hands: Consider implied odds and fold equity
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

from src.strategy.board_analyzer import BoardAnalyzer, BoardTexture
from src.strategy.hand_evaluator import HandEvaluator, HandEvaluation, HandType
from src.utils.logger import logger


class PostflopAction(Enum):
    """Possible postflop actions."""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


class HandCategory(Enum):
    """Categories of made hands and draws."""
    MONSTER = "monster"        # Full house+, nut flush, nut straight
    STRONG = "strong"          # Overpair+, top pair top kicker
    MEDIUM = "medium"          # Top pair weak kicker, second pair
    WEAK = "weak"              # Third pair, weak pair
    DRAW = "draw"              # Flush draw, OESD
    AIR = "air"                # No pair, no draw


@dataclass
class PostflopDecision:
    """Result of postflop strategy calculation."""
    action: PostflopAction
    sizing_pct: Optional[float]  # Bet size as % of pot (e.g., 0.66 = 66%)
    frequency: float              # How often to take this action (0-1)
    confidence: float             # Confidence in decision (0-1)
    hand_category: HandCategory
    reasoning: str


class PostflopStrategy:
    """
    GTO-approximation postflop strategy engine.

    Uses board texture and hand strength to determine optimal actions.
    """

    def __init__(self):
        """Initialize postflop strategy components."""
        self.board_analyzer = BoardAnalyzer()
        self.hand_evaluator = HandEvaluator()

    def categorize_hand(self,
                        hand: List[str],
                        board: List[str],
                        hand_eval: HandEvaluation) -> HandCategory:
        """
        Categorize hand strength for strategy selection.

        Args:
            hand: Hole cards
            board: Community cards
            hand_eval: Pre-computed hand evaluation

        Returns:
            HandCategory enum value
        """
        strength = hand_eval.hand_strength
        hand_type = hand_eval.hand_type

        # Monster hands (full house+)
        if hand_type in [HandType.ROYAL_FLUSH, HandType.STRAIGHT_FLUSH,
                         HandType.FOUR_OF_A_KIND, HandType.FULL_HOUSE]:
            return HandCategory.MONSTER

        # Strong hands (flush, straight, set, two pair, strong overpair)
        if hand_type in [HandType.FLUSH, HandType.STRAIGHT, HandType.THREE_OF_A_KIND]:
            return HandCategory.STRONG

        if hand_type == HandType.TWO_PAIR:
            if strength > 0.5:
                return HandCategory.STRONG
            return HandCategory.MEDIUM

        # Pairs
        if hand_type == HandType.PAIR:
            if strength > 0.45:
                return HandCategory.STRONG  # Overpair or top pair good kicker
            elif strength > 0.30:
                return HandCategory.MEDIUM  # Top pair, middle pair
            else:
                return HandCategory.WEAK    # Bottom pair

        # High card - check for draws
        if self._has_strong_draw(hand, board):
            return HandCategory.DRAW

        return HandCategory.AIR

    def _has_strong_draw(self, hand: List[str], board: List[str]) -> bool:
        """Check if hand has flush draw or OESD."""
        if len(board) < 3:
            return False

        all_cards = hand + board
        suits = [card[1] for card in all_cards]
        ranks = [card[0] for card in all_cards]

        # Flush draw (4 to a flush)
        from collections import Counter
        suit_counts = Counter(suits)
        if any(count >= 4 for count in suit_counts.values()):
            return True

        # OESD (4 consecutive ranks possible)
        rank_order = "AKQJT98765432"
        rank_values = sorted([rank_order.index(r) for r in ranks])

        for i in range(len(rank_values) - 3):
            window = rank_values[i:i+4]
            if window[-1] - window[0] <= 4:
                return True

        return False

    def get_cbet_action(self,
                        hand: List[str],
                        board: List[str],
                        is_in_position: bool,
                        pot_size: float,
                        num_opponents: int = 1,
                        is_preflop_raiser: bool = True) -> PostflopDecision:
        """
        Determine continuation bet strategy.

        Args:
            hand: Hole cards
            board: Community cards (flop)
            is_in_position: True if we act last
            pot_size: Current pot size
            num_opponents: Number of opponents in hand
            is_preflop_raiser: True if we were the preflop aggressor

        Returns:
            PostflopDecision with action and sizing
        """
        # Analyze board texture
        texture = self.board_analyzer.analyze(board)

        # Evaluate our hand
        hand_eval = self.hand_evaluator.evaluate(hand, board)
        hand_cat = self.categorize_hand(hand, board, hand_eval)

        logger.debug(f"C-bet analysis: {texture.texture_category} board, {hand_cat.value} hand")

        # Determine c-bet strategy based on texture
        if texture.texture_category == "dry":
            return self._cbet_dry_board(hand_cat, texture, is_in_position, num_opponents)
        elif texture.texture_category == "wet":
            return self._cbet_wet_board(hand_cat, texture, is_in_position, num_opponents)
        elif texture.texture_category == "dynamic":
            return self._cbet_dynamic_board(hand_cat, texture, is_in_position, num_opponents)
        else:  # medium
            return self._cbet_medium_board(hand_cat, texture, is_in_position, num_opponents)

    def _cbet_dry_board(self,
                        hand_cat: HandCategory,
                        texture: BoardTexture,
                        is_in_position: bool,
                        num_opponents: int) -> PostflopDecision:
        """
        C-bet strategy on dry boards (A72r, K83r, etc.)

        Strategy: Bet large, high frequency. Villain has few draws.
        """
        # Vs single opponent on dry board: high frequency c-bet
        if num_opponents == 1:
            if hand_cat in [HandCategory.MONSTER, HandCategory.STRONG]:
                return PostflopDecision(
                    action=PostflopAction.BET,
                    sizing_pct=0.75,
                    frequency=1.0,
                    confidence=0.9,
                    hand_category=hand_cat,
                    reasoning="Value bet on dry board"
                )
            elif hand_cat == HandCategory.MEDIUM:
                return PostflopDecision(
                    action=PostflopAction.BET,
                    sizing_pct=0.66,
                    frequency=0.8,
                    confidence=0.75,
                    hand_category=hand_cat,
                    reasoning="Value/protection on dry board"
                )
            elif hand_cat == HandCategory.WEAK:
                return PostflopDecision(
                    action=PostflopAction.BET,
                    sizing_pct=0.50,
                    frequency=0.5,
                    confidence=0.6,
                    hand_category=hand_cat,
                    reasoning="Thin value/protection on dry board"
                )
            else:  # AIR or DRAW
                # Bluff on dry board
                return PostflopDecision(
                    action=PostflopAction.BET,
                    sizing_pct=0.66,
                    frequency=0.4,
                    confidence=0.55,
                    hand_category=hand_cat,
                    reasoning="Bluff c-bet on dry board (high fold equity)"
                )
        else:
            # Multiway: tighten up
            if hand_cat in [HandCategory.MONSTER, HandCategory.STRONG]:
                return PostflopDecision(
                    action=PostflopAction.BET,
                    sizing_pct=0.75,
                    frequency=0.9,
                    confidence=0.85,
                    hand_category=hand_cat,
                    reasoning="Value bet multiway on dry board"
                )
            else:
                return PostflopDecision(
                    action=PostflopAction.CHECK,
                    sizing_pct=None,
                    frequency=0.8,
                    confidence=0.7,
                    hand_category=hand_cat,
                    reasoning="Check multiway without strong hand"
                )

    def _cbet_wet_board(self,
                        hand_cat: HandCategory,
                        texture: BoardTexture,
                        is_in_position: bool,
                        num_opponents: int) -> PostflopDecision:
        """
        C-bet strategy on wet boards (JT9ss, 876dd, etc.)

        Strategy: Bet smaller or check more often. Many draws available.
        """
        if hand_cat in [HandCategory.MONSTER, HandCategory.STRONG]:
            # Still bet for value, but sizing can be smaller
            return PostflopDecision(
                action=PostflopAction.BET,
                sizing_pct=0.50,  # Smaller sizing on wet boards
                frequency=0.85,
                confidence=0.85,
                hand_category=hand_cat,
                reasoning="Value bet on wet board (smaller sizing)"
            )
        elif hand_cat == HandCategory.DRAW:
            # Semi-bluff with draws
            if is_in_position:
                return PostflopDecision(
                    action=PostflopAction.BET,
                    sizing_pct=0.40,
                    frequency=0.6,
                    confidence=0.65,
                    hand_category=hand_cat,
                    reasoning="Semi-bluff with draw in position"
                )
            else:
                return PostflopDecision(
                    action=PostflopAction.CHECK,
                    sizing_pct=None,
                    frequency=0.7,
                    confidence=0.6,
                    hand_category=hand_cat,
                    reasoning="Check draw OOP to control pot"
                )
        elif hand_cat == HandCategory.MEDIUM:
            # Check more often with medium hands on wet boards
            return PostflopDecision(
                action=PostflopAction.CHECK,
                sizing_pct=None,
                frequency=0.6,
                confidence=0.6,
                hand_category=hand_cat,
                reasoning="Check medium hand on wet board"
            )
        else:
            # Don't bluff as much on wet boards
            return PostflopDecision(
                action=PostflopAction.CHECK,
                sizing_pct=None,
                frequency=0.75,
                confidence=0.65,
                hand_category=hand_cat,
                reasoning="Check air on wet board (low fold equity)"
            )

    def _cbet_medium_board(self,
                           hand_cat: HandCategory,
                           texture: BoardTexture,
                           is_in_position: bool,
                           num_opponents: int) -> PostflopDecision:
        """
        C-bet strategy on medium texture boards.

        Strategy: Balanced approach between dry and wet.
        """
        if hand_cat in [HandCategory.MONSTER, HandCategory.STRONG]:
            return PostflopDecision(
                action=PostflopAction.BET,
                sizing_pct=0.66,
                frequency=0.9,
                confidence=0.85,
                hand_category=hand_cat,
                reasoning="Value bet on medium texture"
            )
        elif hand_cat == HandCategory.MEDIUM:
            return PostflopDecision(
                action=PostflopAction.BET,
                sizing_pct=0.50,
                frequency=0.6,
                confidence=0.65,
                hand_category=hand_cat,
                reasoning="Mixed strategy with medium hand"
            )
        elif hand_cat == HandCategory.DRAW:
            return PostflopDecision(
                action=PostflopAction.BET,
                sizing_pct=0.50,
                frequency=0.55,
                confidence=0.6,
                hand_category=hand_cat,
                reasoning="Semi-bluff with draw"
            )
        else:
            return PostflopDecision(
                action=PostflopAction.CHECK,
                sizing_pct=None,
                frequency=0.6,
                confidence=0.55,
                hand_category=hand_cat,
                reasoning="Check weak hand on medium board"
            )

    def _cbet_dynamic_board(self,
                            hand_cat: HandCategory,
                            texture: BoardTexture,
                            is_in_position: bool,
                            num_opponents: int) -> PostflopDecision:
        """
        C-bet strategy on dynamic (paired) boards.

        Strategy: Polarized - bet strong or check more often.
        """
        if hand_cat in [HandCategory.MONSTER, HandCategory.STRONG]:
            return PostflopDecision(
                action=PostflopAction.BET,
                sizing_pct=0.33,  # Small sizing on paired boards
                frequency=0.9,
                confidence=0.85,
                hand_category=hand_cat,
                reasoning="Value bet on paired board"
            )
        elif hand_cat == HandCategory.MEDIUM:
            return PostflopDecision(
                action=PostflopAction.CHECK,
                sizing_pct=None,
                frequency=0.7,
                confidence=0.65,
                hand_category=hand_cat,
                reasoning="Check medium hand on paired board"
            )
        else:
            # Check most air/draws on paired boards
            return PostflopDecision(
                action=PostflopAction.CHECK,
                sizing_pct=None,
                frequency=0.8,
                confidence=0.6,
                hand_category=hand_cat,
                reasoning="Check on paired board"
            )

    def get_facing_bet_action(self,
                              hand: List[str],
                              board: List[str],
                              equity: float,
                              pot_odds: float,
                              bet_size_pct: float,
                              is_in_position: bool = True) -> PostflopDecision:
        """
        Determine action when facing a bet.

        Args:
            hand: Hole cards
            board: Community cards
            equity: Our equity vs villain's range (0-100)
            pot_odds: Ratio we're getting (e.g., 3.0 means 3:1)
            bet_size_pct: Bet size as % of pot
            is_in_position: True if we act last

        Returns:
            PostflopDecision
        """
        texture = self.board_analyzer.analyze(board)
        hand_eval = self.hand_evaluator.evaluate(hand, board)
        hand_cat = self.categorize_hand(hand, board, hand_eval)

        # Calculate required equity to call
        required_equity = 100 / (pot_odds + 1)

        # Equity margin
        equity_margin = equity - required_equity

        logger.debug(f"Facing bet: equity={equity:.1f}%, required={required_equity:.1f}%, margin={equity_margin:.1f}%")

        # Strong hands - consider raising for value
        if hand_cat in [HandCategory.MONSTER, HandCategory.STRONG]:
            if equity > 60:
                return PostflopDecision(
                    action=PostflopAction.RAISE,
                    sizing_pct=2.5 * bet_size_pct,  # ~2.5x raise
                    frequency=0.7,
                    confidence=0.8,
                    hand_category=hand_cat,
                    reasoning=f"Raise for value with {hand_eval.description}"
                )
            else:
                return PostflopDecision(
                    action=PostflopAction.CALL,
                    sizing_pct=None,
                    frequency=1.0,
                    confidence=0.85,
                    hand_category=hand_cat,
                    reasoning=f"Call with {hand_eval.description}"
                )

        # Medium hands - call if profitable
        if hand_cat == HandCategory.MEDIUM:
            if equity_margin > 5:
                return PostflopDecision(
                    action=PostflopAction.CALL,
                    sizing_pct=None,
                    frequency=0.9,
                    confidence=0.7,
                    hand_category=hand_cat,
                    reasoning=f"Profitable call: {equity:.0f}% > {required_equity:.0f}% required"
                )
            elif equity_margin > -5:
                return PostflopDecision(
                    action=PostflopAction.CALL,
                    sizing_pct=None,
                    frequency=0.5,
                    confidence=0.5,
                    hand_category=hand_cat,
                    reasoning="Marginal call at breakeven"
                )
            else:
                return PostflopDecision(
                    action=PostflopAction.FOLD,
                    sizing_pct=None,
                    frequency=0.8,
                    confidence=0.65,
                    hand_category=hand_cat,
                    reasoning=f"Fold: {equity:.0f}% < {required_equity:.0f}% required"
                )

        # Draws - consider implied odds
        if hand_cat == HandCategory.DRAW:
            # Add implied odds bonus for draws
            implied_equity = equity + 10  # Rough implied odds adjustment

            if implied_equity > required_equity:
                return PostflopDecision(
                    action=PostflopAction.CALL,
                    sizing_pct=None,
                    frequency=0.85,
                    confidence=0.65,
                    hand_category=hand_cat,
                    reasoning=f"Call with draw: {equity:.0f}% + implied odds"
                )
            else:
                return PostflopDecision(
                    action=PostflopAction.FOLD,
                    sizing_pct=None,
                    frequency=0.7,
                    confidence=0.6,
                    hand_category=hand_cat,
                    reasoning="Fold draw without odds"
                )

        # Weak hands and air
        if equity_margin > 10:
            # Surprising equity - maybe float?
            if is_in_position:
                return PostflopDecision(
                    action=PostflopAction.CALL,
                    sizing_pct=None,
                    frequency=0.4,
                    confidence=0.45,
                    hand_category=hand_cat,
                    reasoning="Float in position with equity"
                )

        return PostflopDecision(
            action=PostflopAction.FOLD,
            sizing_pct=None,
            frequency=0.9,
            confidence=0.75,
            hand_category=hand_cat,
            reasoning="Fold weak hand vs bet"
        )
