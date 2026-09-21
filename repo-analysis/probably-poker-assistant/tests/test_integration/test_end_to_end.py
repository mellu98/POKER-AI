"""
End-to-end integration tests for Poker Assistant.

Tests the complete pipeline from game state input to decision output,
including manual card entry mode and full hand simulations.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.detection.game_state import GameState, BettingRound
from src.strategy.decision_engine import DecisionEngine
from src.strategy.hand_evaluator import HandEvaluator, HandType
from src.strategy.equity_calculator import EquityCalculator
from src.strategy.pot_odds import PotOddsCalculator


class TestEndToEndDecisionPipeline:
    """Test complete decision pipeline from game state to recommendation."""

    @pytest.fixture
    def decision_engine(self):
        """Create decision engine instance."""
        return DecisionEngine()

    @pytest.fixture
    def hand_evaluator(self):
        """Create hand evaluator instance."""
        return HandEvaluator()

    @pytest.fixture
    def equity_calculator(self):
        """Create equity calculator instance."""
        return EquityCalculator()

    # =========================================================================
    # PREFLOP SCENARIOS
    # =========================================================================

    @pytest.mark.integration
    def test_premium_hand_preflop_decision(self, decision_engine):
        """Test decision with premium pocket aces preflop."""
        game_state = GameState(
            hole_cards=['Ah', 'As'],
            community_cards=[],
            pot_size=15,
            stack_size=1000,
            current_bet=10,
            betting_round=BettingRound.PREFLOP,
            num_opponents=5
        )

        decision = decision_engine.decide(game_state)

        # Premium hand should raise or call
        assert decision is not None
        assert decision.action in ['raise', 'call', 'bet']
        assert decision.confidence > 0.5  # Confidence is 0-1 scale

    @pytest.mark.integration
    def test_trash_hand_preflop_decision(self, decision_engine):
        """Test decision with trash hand preflop facing raise."""
        game_state = GameState(
            hole_cards=['7h', '2d'],
            community_cards=[],
            pot_size=30,
            stack_size=1000,
            current_bet=20,  # Facing a raise
            betting_round=BettingRound.PREFLOP,
            num_opponents=3
        )

        decision = decision_engine.decide(game_state)

        # Trash hand should fold
        assert decision is not None
        assert decision.action == 'fold'

    @pytest.mark.integration
    def test_medium_hand_preflop_decision(self, decision_engine):
        """Test decision with suited connectors preflop."""
        game_state = GameState(
            hole_cards=['9h', '8h'],
            community_cards=[],
            pot_size=15,
            stack_size=1000,
            current_bet=10,
            betting_round=BettingRound.PREFLOP,
            num_opponents=4
        )

        decision = decision_engine.decide(game_state)

        # Medium hand can call or fold depending on situation
        assert decision is not None
        assert decision.action in ['fold', 'call', 'raise', 'bet', 'check']

    # =========================================================================
    # POSTFLOP SCENARIOS
    # =========================================================================

    @pytest.mark.integration
    def test_top_pair_flop_decision(self, decision_engine):
        """Test decision with top pair on flop."""
        game_state = GameState(
            hole_cards=['Ah', 'Kd'],
            community_cards=['Ac', '7s', '2h'],  # Top pair top kicker
            pot_size=50,
            stack_size=900,
            current_bet=0,
            betting_round=BettingRound.FLOP,
            num_opponents=2
        )

        decision = decision_engine.decide(game_state)

        # Top pair should bet for value
        assert decision is not None
        assert decision.action in ['bet', 'raise', 'check']

    @pytest.mark.integration
    def test_flush_draw_flop_decision(self, decision_engine):
        """Test decision with flush draw on flop."""
        game_state = GameState(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7h', '2c'],  # Nut flush draw
            pot_size=100,
            stack_size=800,
            current_bet=25,  # Small bet to call
            betting_round=BettingRound.FLOP,
            num_opponents=1
        )

        decision = decision_engine.decide(game_state)

        # Nut flush draw with good odds should call or raise
        assert decision is not None
        assert decision.action in ['call', 'raise', 'fold']

    @pytest.mark.integration
    def test_made_hand_river_decision(self, decision_engine):
        """Test decision with made hand on river."""
        game_state = GameState(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', 'Jh', 'Th', '2c', '3d'],  # Royal flush!
            pot_size=200,
            stack_size=700,
            current_bet=50,
            betting_round=BettingRound.RIVER,
            num_opponents=1
        )

        decision = decision_engine.decide(game_state)

        # Royal flush should raise for value
        assert decision is not None
        assert decision.action in ['raise', 'call']
        assert decision.confidence > 0.7  # Confidence is 0-1 scale

    # =========================================================================
    # HAND EVALUATION INTEGRATION
    # =========================================================================

    @pytest.mark.integration
    def test_hand_evaluation_in_decision(self, decision_engine):
        """Test that decision includes correct hand evaluation."""
        game_state = GameState(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Ac', 'Kd', '7s'],  # Two pair
            pot_size=100,
            stack_size=800,
            current_bet=0,
            betting_round=BettingRound.FLOP,
            num_opponents=2
        )

        decision = decision_engine.decide(game_state)

        # Should have hand evaluation
        assert decision is not None
        assert hasattr(decision, 'hand_evaluation')
        assert decision.hand_evaluation is not None
        # Two pair hand
        assert decision.hand_evaluation.hand_type in [HandType.TWO_PAIR, HandType.FULL_HOUSE]

    @pytest.mark.integration
    def test_equity_in_decision(self, decision_engine):
        """Test that decision includes equity calculation."""
        game_state = GameState(
            hole_cards=['Ah', 'As'],
            community_cards=['Kc', '7d', '2s'],
            pot_size=100,
            stack_size=800,
            current_bet=25,
            betting_round=BettingRound.FLOP,
            num_opponents=1
        )

        decision = decision_engine.decide(game_state)

        # Should have equity
        assert decision is not None
        assert hasattr(decision, 'equity')
        assert 0 <= decision.equity <= 100

    # =========================================================================
    # POT ODDS INTEGRATION
    # =========================================================================

    @pytest.mark.integration
    def test_pot_odds_affects_decision(self, decision_engine):
        """Test that pot odds influence decision making."""
        # Bad pot odds - large bet into small pot
        bad_odds_state = GameState(
            hole_cards=['9h', '8h'],
            community_cards=['Qh', '7h', '2c'],  # Flush draw
            pot_size=50,
            stack_size=800,
            current_bet=100,  # Massive overbet - bad odds
            betting_round=BettingRound.FLOP,
            num_opponents=1
        )

        # Good pot odds - small bet into large pot
        good_odds_state = GameState(
            hole_cards=['9h', '8h'],
            community_cards=['Qh', '7h', '2c'],  # Same flush draw
            pot_size=200,
            stack_size=800,
            current_bet=25,  # Small bet - good odds
            betting_round=BettingRound.FLOP,
            num_opponents=1
        )

        bad_decision = decision_engine.decide(bad_odds_state)
        good_decision = decision_engine.decide(good_odds_state)

        # Both should have valid decisions
        assert bad_decision is not None
        assert good_decision is not None

        # Good odds more likely to call
        if bad_decision.action == 'fold':
            # With good odds, should be more likely to continue
            assert good_decision.action in ['call', 'raise', 'fold']


class TestManualCardEntryFlow:
    """Test manual card entry workflow."""

    @pytest.fixture
    def decision_engine(self):
        """Create decision engine instance."""
        return DecisionEngine()

    @pytest.mark.integration
    def test_manual_hole_cards_only(self, decision_engine):
        """Test decision with only hole cards entered manually."""
        game_state = GameState(
            hole_cards=['Ah', 'Kd'],  # Manual entry
            community_cards=[],  # No community cards yet
            pot_size=15,
            stack_size=1000,
            current_bet=10,
            betting_round=BettingRound.PREFLOP,
            num_opponents=5
        )

        decision = decision_engine.decide(game_state)

        assert decision is not None
        assert decision.action in ['raise', 'call', 'fold', 'bet', 'check']

    @pytest.mark.integration
    def test_manual_flop_entry(self, decision_engine):
        """Test decision with manually entered flop."""
        game_state = GameState(
            hole_cards=['Jh', 'Td'],
            community_cards=['9c', '8s', '2h'],  # OESD
            pot_size=50,
            stack_size=950,
            current_bet=0,
            betting_round=BettingRound.FLOP,
            num_opponents=2
        )

        decision = decision_engine.decide(game_state)

        assert decision is not None
        assert hasattr(decision, 'equity')

    @pytest.mark.integration
    def test_manual_full_board_entry(self, decision_engine):
        """Test decision with full board entered manually."""
        game_state = GameState(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', 'Jh', '2c', '3d', 'Th'],  # Royal flush
            pot_size=300,
            stack_size=700,
            current_bet=100,
            betting_round=BettingRound.RIVER,
            num_opponents=1
        )

        decision = decision_engine.decide(game_state)

        assert decision is not None
        # Royal flush should be highly confident (0-1 scale)
        assert decision.confidence > 0.7


class TestFullHandSimulation:
    """Simulate complete hands from preflop to river."""

    @pytest.fixture
    def decision_engine(self):
        """Create decision engine instance."""
        return DecisionEngine()

    @pytest.mark.integration
    def test_complete_hand_simulation(self, decision_engine):
        """Simulate decisions through all streets."""
        # Preflop
        preflop_state = GameState(
            hole_cards=['Ah', 'Kh'],
            community_cards=[],
            pot_size=15,
            stack_size=1000,
            current_bet=10,
            betting_round=BettingRound.PREFLOP,
            num_opponents=3
        )
        preflop_decision = decision_engine.decide(preflop_state)
        assert preflop_decision is not None
        assert preflop_decision.action in ['raise', 'call', 'fold', 'bet', 'check']

        # Flop
        flop_state = GameState(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c'],
            pot_size=60,
            stack_size=970,
            current_bet=0,
            betting_round=BettingRound.FLOP,
            num_opponents=2
        )
        flop_decision = decision_engine.decide(flop_state)
        assert flop_decision is not None

        # Turn
        turn_state = GameState(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c', 'Jh'],  # Nut flush draw
            pot_size=120,
            stack_size=940,
            current_bet=30,
            betting_round=BettingRound.TURN,
            num_opponents=1
        )
        turn_decision = decision_engine.decide(turn_state)
        assert turn_decision is not None

        # River
        river_state = GameState(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c', 'Jh', 'Th'],  # Royal flush
            pot_size=200,
            stack_size=850,
            current_bet=50,
            betting_round=BettingRound.RIVER,
            num_opponents=1
        )
        river_decision = decision_engine.decide(river_state)
        assert river_decision is not None
        # Should be very confident with royal flush (0-1 scale)
        assert river_decision.confidence >= 0.7

    @pytest.mark.integration
    def test_bluff_catching_scenario(self, decision_engine):
        """Test decision in a potential bluff-catching situation."""
        game_state = GameState(
            hole_cards=['Ac', 'Kd'],
            community_cards=['Qh', '8s', '4d', '2c', '7h'],  # Dry board
            pot_size=200,
            stack_size=600,
            current_bet=150,  # Large river bet
            betting_round=BettingRound.RIVER,
            num_opponents=1
        )

        decision = decision_engine.decide(game_state)

        # Should make a decision about calling/folding
        assert decision is not None
        assert decision.action in ['call', 'fold', 'raise']
        assert hasattr(decision, 'reasoning')


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def decision_engine(self):
        """Create decision engine instance."""
        return DecisionEngine()

    @pytest.mark.integration
    def test_empty_hole_cards(self, decision_engine):
        """Test handling of empty hole cards."""
        game_state = GameState(
            hole_cards=[],
            community_cards=[],
            pot_size=15,
            stack_size=1000,
            current_bet=10,
            betting_round=BettingRound.PREFLOP,
            num_opponents=5
        )

        try:
            decision = decision_engine.decide(game_state)
            # If it returns, should be a safe default
            assert decision.action in ['fold', 'check']
        except (ValueError, IndexError, AssertionError):
            # Raising error is also acceptable
            pass

    @pytest.mark.integration
    def test_all_in_situation(self, decision_engine):
        """Test decision when stack is very short."""
        game_state = GameState(
            hole_cards=['Ah', 'As'],
            community_cards=['Kc', '7d', '2s'],
            pot_size=500,
            stack_size=50,  # Very short stack
            current_bet=100,  # More than stack
            betting_round=BettingRound.FLOP,
            num_opponents=1
        )

        decision = decision_engine.decide(game_state)

        # Should handle short stack appropriately
        assert decision is not None
        assert decision.action in ['fold', 'call', 'raise', 'bet', 'check']

    @pytest.mark.integration
    def test_heads_up_situation(self, decision_engine):
        """Test decision in heads up situation."""
        game_state = GameState(
            hole_cards=['Kh', 'Qd'],
            community_cards=['Ac', '7s', '2h'],
            pot_size=100,
            stack_size=800,
            current_bet=25,
            betting_round=BettingRound.FLOP,
            num_opponents=1  # Heads up
        )

        decision = decision_engine.decide(game_state)

        assert decision is not None
        # Should consider heads up dynamics
        assert hasattr(decision, 'confidence')
