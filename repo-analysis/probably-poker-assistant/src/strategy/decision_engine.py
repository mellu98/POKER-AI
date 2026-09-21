"""
Main strategic decision engine.
Combines GTO preflop lookup, hand evaluation, equity, and pot odds to suggest actions.

Architecture:
- Preflop: Uses PreflopStrategy for GTO-based decisions
- Postflop: Uses hand evaluation + equity + pot odds heuristics
"""
from typing import Optional, List
from dataclasses import dataclass

from src.strategy.hand_evaluator import HandEvaluator, HandEvaluation
from src.strategy.equity_calculator import EquityCalculator
from src.strategy.pot_odds import PotOddsCalculator
from src.strategy.preflop_strategy import PreflopStrategy, Position, PreflopAction
from src.strategy.postflop_strategy import PostflopStrategy, PostflopAction
from src.strategy.board_analyzer import BoardAnalyzer
from src.detection.game_state import GameState, BettingRound
from src.utils.logger import logger
from src.utils.config_loader import config_loader


@dataclass
class ActionFrequencies:
    """GTO action frequencies for mixed strategies."""
    fold: float = 0.0  # 0-100 percentage
    call: float = 0.0
    raise_: float = 0.0  # Using raise_ since raise is a keyword
    check: float = 0.0
    bet: float = 0.0

    def as_dict(self):
        """Return as dictionary with clean names."""
        return {
            'fold': self.fold,
            'call': self.call,
            'raise': self.raise_,
            'check': self.check,
            'bet': self.bet
        }


@dataclass
class Decision:
    """Strategic decision output."""
    action: str  # fold, call, raise, check
    amount_bb: Optional[float]  # Bet size in big blinds
    amount_chips: Optional[float]  # Bet size in chips
    confidence: float  # 0-1 confidence score
    reasoning: List[str]  # Explanation for the decision
    hand_evaluation: HandEvaluation
    equity: float  # Win probability percentage
    pot_odds: Optional[float]  # Pot odds ratio
    source: str = "unknown"  # gto_solver, heuristic, or mixed
    action_frequencies: Optional[ActionFrequencies] = None  # GTO frequencies
    spr: Optional[float] = None  # Stack-to-pot ratio
    position_advantage: bool = True  # Whether we have position

    def __str__(self):
        action_str = self.action.upper()
        if self.amount_bb:
            action_str += f" {self.amount_bb:.1f}BB"
        return f"{action_str} ({self.confidence*100:.0f}% conf, {self.source})"


class DecisionEngine:
    """
    Main poker decision engine using GTO lookup + heuristics.

    Preflop: GTO ranges from database/preflop/
    Postflop: Hand strength + equity + pot odds heuristics
    """

    def __init__(self):
        """Initialize decision engine with all strategy components."""
        # Core evaluators
        self.hand_evaluator = HandEvaluator()
        self.equity_calc = EquityCalculator()
        self.pot_odds_calc = PotOddsCalculator()
        self.board_analyzer = BoardAnalyzer()

        # GTO preflop strategy
        self.preflop_strategy = PreflopStrategy()

        # GTO-approximation postflop strategy
        self.postflop_strategy = PostflopStrategy()

        # Load configuration
        try:
            self.config = config_loader.load('settings.json')
        except Exception:
            self.config = {}

        self.style = self.config.get('strategy', {}).get('style', 'tight_aggressive')

        # Big blind size for chip calculations (default 1.0, can be updated)
        self.bb_size = 1.0

        logger.info(f"DecisionEngine initialized with style: {self.style}")

    def set_bb_size(self, bb_size: float):
        """Set big blind size for accurate chip calculations."""
        self.bb_size = bb_size
        logger.debug(f"BB size set to {bb_size}")

    def decide(self, game_state: GameState) -> Decision:
        """
        Make strategic decision based on current game state.

        Args:
            game_state: Current game state with cards, pot, stack, etc.

        Returns:
            Decision object with action, sizing, confidence, and reasoning
        """
        logger.info(f"Deciding for {game_state.betting_round.value}: {game_state.hole_cards}")

        # Validate we have hole cards
        if len(game_state.hole_cards) < 2:
            return self._create_error_decision("No hole cards detected")

        # Evaluate hand
        hand_eval = self.hand_evaluator.evaluate(
            game_state.hole_cards,
            game_state.community_cards
        )

        # Calculate equity (reduce iterations preflop for speed)
        iterations = 500 if game_state.betting_round == BettingRound.PREFLOP else 1000
        equity = self.equity_calc.calculate_equity(
            game_state.hole_cards,
            game_state.community_cards,
            num_opponents=max(1, game_state.num_opponents),
            iterations=iterations
        )

        # Calculate pot odds if facing a bet
        pot_odds = None
        if game_state.current_bet and game_state.current_bet > 0:
            pot_odds = self.pot_odds_calc.calculate_pot_odds(
                game_state.pot_size,
                game_state.current_bet
            )

        # Route to appropriate decision method
        if game_state.betting_round == BettingRound.PREFLOP:
            decision = self._decide_preflop_gto(game_state, hand_eval, equity, pot_odds)
        else:
            decision = self._decide_postflop(game_state, hand_eval, equity, pot_odds)

        logger.info(f"Decision: {decision}")
        return decision

    def _decide_preflop_gto(self,
                           game_state: GameState,
                           hand_eval: HandEvaluation,
                           equity: float,
                           pot_odds: Optional[float]) -> Decision:
        """
        Make preflop decision using GTO lookup.

        Uses PreflopStrategy for position-aware GTO decisions.
        """
        # Determine our position (default to BTN if unknown)
        position = self._get_position(game_state)

        # Determine if we're facing a raise
        facing_raise = game_state.current_bet and game_state.current_bet > self.bb_size

        if facing_raise:
            # Facing a raise - use vs_raise logic
            # Assume raiser is one position earlier (simplified)
            raiser_position = self._estimate_raiser_position(position)
            preflop_decision = self.preflop_strategy.get_vs_raise_action(
                hand=game_state.hole_cards,
                our_position=position,
                raiser_position=raiser_position,
                raise_size_bb=game_state.current_bet / self.bb_size if self.bb_size > 0 else 2.5,
                stack_bb=game_state.stack_size / self.bb_size if self.bb_size > 0 else 100
            )
        else:
            # No raise - opening decision
            preflop_decision = self.preflop_strategy.get_open_action(
                hand=game_state.hole_cards,
                position=position,
                stack_bb=game_state.stack_size / self.bb_size if self.bb_size > 0 else 100
            )

        # Convert PreflopDecision to Decision
        action = preflop_decision.action.value
        amount_bb = preflop_decision.sizing_bb

        # Build reasoning
        reasoning = [preflop_decision.reasoning]
        if preflop_decision.frequency < 1.0:
            reasoning.append(f"Mixed strategy: {preflop_decision.frequency*100:.0f}% frequency")
        reasoning.append(f"Position: {position.value}")
        reasoning.append(f"Equity: {equity:.1f}%")

        # Calculate action frequencies for GTO display
        freq = preflop_decision.frequency
        action_freqs = ActionFrequencies()
        if action == "fold":
            action_freqs.fold = freq * 100
            action_freqs.raise_ = (1 - freq) * 100 * 0.3  # Estimate
            action_freqs.call = (1 - freq) * 100 * 0.7
        elif action == "raise":
            action_freqs.raise_ = freq * 100
            action_freqs.fold = (1 - freq) * 100 * 0.6
            action_freqs.call = (1 - freq) * 100 * 0.4
        elif action == "call":
            action_freqs.call = freq * 100
            action_freqs.fold = (1 - freq) * 100 * 0.5
            action_freqs.raise_ = (1 - freq) * 100 * 0.5

        # Calculate SPR
        spr = None
        if game_state.pot_size and game_state.pot_size > 0 and game_state.stack_size:
            spr = game_state.stack_size / game_state.pot_size

        return Decision(
            action=action,
            amount_bb=amount_bb,
            amount_chips=amount_bb * self.bb_size if amount_bb else None,
            confidence=preflop_decision.confidence,
            reasoning=reasoning,
            hand_evaluation=hand_eval,
            equity=equity,
            pot_odds=pot_odds,
            source=preflop_decision.source,
            action_frequencies=action_freqs,
            spr=spr,
            position_advantage=position in [Position.BTN, Position.CO]
        )

    def _decide_postflop(self,
                        game_state: GameState,
                        hand_eval: HandEvaluation,
                        equity: float,
                        pot_odds: Optional[float]) -> Decision:
        """
        Make postflop decision using board texture + GTO-approximation strategy.

        Uses PostflopStrategy for texture-aware decisions:
        - Dry boards: Bet large, high frequency
        - Wet boards: Bet smaller, check more
        - Dynamic boards: Polarized strategy
        """
        reasoning = []
        source = "postflop_gto"

        # Analyze board texture
        board_texture = self.board_analyzer.analyze(game_state.community_cards)
        reasoning.append(f"Board: {board_texture.texture_category} ({board_texture.high_card}-high)")

        # Check if we can check (no bet to call)
        can_check = game_state.current_bet == 0 or game_state.current_bet is None

        if can_check:
            # We can bet or check - use c-bet strategy
            postflop_decision = self.postflop_strategy.get_cbet_action(
                hand=game_state.hole_cards,
                board=game_state.community_cards,
                is_in_position=True,  # Assume IP for now
                pot_size=game_state.pot_size,
                num_opponents=max(1, game_state.num_opponents)
            )

            action = postflop_decision.action.value
            if postflop_decision.sizing_pct:
                amount_bb = (game_state.pot_size * postflop_decision.sizing_pct) / self.bb_size
            else:
                amount_bb = None

            confidence = postflop_decision.confidence
            reasoning.append(f"Hand: {postflop_decision.hand_category.value}")
            reasoning.append(postflop_decision.reasoning)
            reasoning.append(f"Equity: {equity:.1f}%")

        else:
            # Facing a bet - use facing bet strategy
            bet_size_pct = game_state.current_bet / game_state.pot_size if game_state.pot_size > 0 else 0.5

            postflop_decision = self.postflop_strategy.get_facing_bet_action(
                hand=game_state.hole_cards,
                board=game_state.community_cards,
                equity=equity,
                pot_odds=pot_odds or 2.0,
                bet_size_pct=bet_size_pct,
                is_in_position=True
            )

            action = postflop_decision.action.value
            if postflop_decision.action == PostflopAction.RAISE and postflop_decision.sizing_pct:
                amount_bb = (game_state.pot_size * postflop_decision.sizing_pct) / self.bb_size
            else:
                amount_bb = None

            confidence = postflop_decision.confidence
            reasoning.append(f"Hand: {postflop_decision.hand_category.value}")
            reasoning.append(postflop_decision.reasoning)
            if pot_odds:
                reasoning.append(f"Pot odds: {pot_odds:.1f}:1")
            reasoning.append(f"Equity: {equity:.1f}%")

        # Calculate action frequencies based on texture and hand strength
        action_freqs = ActionFrequencies()
        if can_check:
            # In position betting spot
            if postflop_decision.confidence > 0.7:
                action_freqs.bet = postflop_decision.confidence * 100
                action_freqs.check = (1 - postflop_decision.confidence) * 100
            else:
                action_freqs.check = 60  # Default check frequency
                action_freqs.bet = 40
        else:
            # Facing a bet
            if action == "fold":
                action_freqs.fold = postflop_decision.confidence * 100
                action_freqs.call = (1 - postflop_decision.confidence) * 50
                action_freqs.raise_ = (1 - postflop_decision.confidence) * 50
            elif action == "call":
                action_freqs.call = postflop_decision.confidence * 100
                action_freqs.fold = (1 - postflop_decision.confidence) * 60
                action_freqs.raise_ = (1 - postflop_decision.confidence) * 40
            elif action == "raise":
                action_freqs.raise_ = postflop_decision.confidence * 100
                action_freqs.call = (1 - postflop_decision.confidence) * 60
                action_freqs.fold = (1 - postflop_decision.confidence) * 40

        # Calculate SPR
        spr = None
        if game_state.pot_size and game_state.pot_size > 0 and game_state.stack_size:
            spr = game_state.stack_size / game_state.pot_size

        return Decision(
            action=action,
            amount_bb=amount_bb,
            amount_chips=amount_bb * self.bb_size if amount_bb else None,
            confidence=confidence,
            reasoning=reasoning,
            hand_evaluation=hand_eval,
            equity=equity,
            pot_odds=pot_odds,
            source=source,
            action_frequencies=action_freqs,
            spr=spr,
            position_advantage=True  # Assuming IP for now
        )

    def _get_position(self, game_state: GameState) -> Position:
        """
        Determine player position from game state.

        Falls back to BTN if position unknown (most common assumption).
        """
        if game_state.position:
            # Map from game_state Position enum to strategy Position enum
            pos_map = {
                "UTG": Position.UTG,
                "UTG+1": Position.MP,
                "MP": Position.MP,
                "MP+1": Position.CO,
                "CO": Position.CO,
                "BTN": Position.BTN,
                "SB": Position.SB,
                "BB": Position.BB,
            }
            pos_str = game_state.position.value if hasattr(game_state.position, 'value') else str(game_state.position)
            return pos_map.get(pos_str, Position.BTN)

        # Default to BTN (optimistic assumption for opens)
        return Position.BTN

    def _estimate_raiser_position(self, our_position: Position) -> Position:
        """
        Estimate the raiser's position based on our position.

        Simple heuristic: assume raiser is ~2 positions earlier.
        """
        position_order = [Position.UTG, Position.MP, Position.CO, Position.BTN, Position.SB, Position.BB]

        try:
            our_idx = position_order.index(our_position)
            # Raiser is typically earlier position
            raiser_idx = max(0, our_idx - 2)
            return position_order[raiser_idx]
        except ValueError:
            return Position.MP  # Default assumption

    def _create_error_decision(self, error_msg: str) -> Decision:
        """Create a default fold decision when error occurs."""
        from src.strategy.hand_evaluator import HandType

        return Decision(
            action="fold",
            amount_bb=None,
            amount_chips=None,
            confidence=0.0,
            reasoning=[f"Error: {error_msg}"],
            hand_evaluation=HandEvaluation(
                hand_type=HandType.HIGH_CARD,
                hand_strength=0.0,
                description="Unknown",
                best_five=[],
                kickers=[]
            ),
            equity=0.0,
            pot_odds=None,
            source="error"
        )
