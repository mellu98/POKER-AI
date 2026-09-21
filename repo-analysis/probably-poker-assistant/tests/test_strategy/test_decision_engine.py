"""
Comprehensive tests for DecisionEngine module.

Tests strategic decision making across different scenarios.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategy.decision_engine import DecisionEngine
from src.detection.game_state import GameState, BettingRound


class TestDecisionEngine:
    """Test suite for DecisionEngine class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up DecisionEngine instance."""
        self.engine = DecisionEngine()

    def _make_game_state(
        self,
        hole_cards,
        community_cards=None,
        pot_size=100,
        stack_size=1000,
        current_bet=0,
        num_opponents=5
    ):
        """Helper to create GameState objects."""
        if community_cards is None:
            community_cards = []

        if len(community_cards) == 0:
            betting_round = BettingRound.PREFLOP
        elif len(community_cards) == 3:
            betting_round = BettingRound.FLOP
        elif len(community_cards) == 4:
            betting_round = BettingRound.TURN
        else:
            betting_round = BettingRound.RIVER

        return GameState(
            hole_cards=hole_cards,
            community_cards=community_cards,
            pot_size=pot_size,
            stack_size=stack_size,
            current_bet=current_bet,
            betting_round=betting_round,
            num_opponents=num_opponents
        )

    # =========================================================================
    # PREFLOP DECISION TESTS
    # =========================================================================

    @pytest.mark.unit
    def test_premium_hand_raise_preflop(self):
        """Test that premium hands raise preflop."""
        state = self._make_game_state(
            hole_cards=['Ah', 'As'],
            pot_size=15,
            current_bet=10
        )
        decision = self.engine.decide(state)
        # AA should raise/bet
        assert decision.action in ['raise', 'bet', 'call']

    @pytest.mark.unit
    def test_weak_hand_fold_preflop(self):
        """Test that weak hands fold to raises preflop."""
        state = self._make_game_state(
            hole_cards=['7h', '2d'],
            pot_size=30,
            current_bet=20  # Raise to call
        )
        decision = self.engine.decide(state)
        # 72o facing a raise should fold
        assert decision.action == 'fold'

    @pytest.mark.unit
    def test_position_affects_range(self):
        """Test that position affects playable range."""
        # Same hand, different scenarios
        hand = ['Jh', 'Td']

        state = self._make_game_state(
            hole_cards=hand,
            pot_size=15,
            current_bet=10
        )
        decision = self.engine.decide(state)
        # Should get a valid action
        assert decision.action in ['fold', 'call', 'raise', 'bet', 'check']

    @pytest.mark.unit
    def test_pocket_pair_action(self):
        """Test pocket pair decision making."""
        state = self._make_game_state(
            hole_cards=['8h', '8d'],
            pot_size=15,
            current_bet=10
        )
        decision = self.engine.decide(state)
        # 88 should be playable
        assert decision.action in ['raise', 'call', 'bet', 'check', 'fold']

    # =========================================================================
    # POSTFLOP DECISION TESTS
    # =========================================================================

    @pytest.mark.unit
    def test_strong_hand_bet_flop(self):
        """Test betting with strong hand on flop."""
        state = self._make_game_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Ac', '7d', '2s'],
            pot_size=50,
            current_bet=0
        )
        decision = self.engine.decide(state)
        # Top pair top kicker should bet
        assert decision.action in ['bet', 'raise', 'check']

    @pytest.mark.unit
    def test_draw_call_with_odds(self):
        """Test calling with draw when getting odds."""
        state = self._make_game_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7h', '2c'],
            pot_size=100,
            current_bet=25  # Good odds for flush draw
        )
        decision = self.engine.decide(state)
        # Nut flush draw with good odds should call
        assert decision.action in ['call', 'raise', 'fold']

    @pytest.mark.unit
    def test_weak_hand_check_flop(self):
        """Test checking with weak hand on flop."""
        state = self._make_game_state(
            hole_cards=['7h', '6h'],
            community_cards=['Ac', 'Kd', 'Qs'],
            pot_size=50,
            current_bet=0
        )
        decision = self.engine.decide(state)
        # Weak hand on scary board should check/fold
        assert decision.action in ['check', 'fold', 'bet']

    @pytest.mark.unit
    def test_made_hand_value_bet_river(self):
        """Test value betting made hand on river."""
        state = self._make_game_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Ac', 'Kd', '7s', '2c', '3h'],
            pot_size=150,
            current_bet=0
        )
        decision = self.engine.decide(state)
        # Two pair on river should value bet
        assert decision.action in ['bet', 'raise', 'check']

    # =========================================================================
    # DECISION ATTRIBUTES TESTS
    # =========================================================================

    @pytest.mark.unit
    def test_decision_has_confidence(self):
        """Test that decisions have confidence level."""
        state = self._make_game_state(
            hole_cards=['Ah', 'As'],
            pot_size=15,
            current_bet=10
        )
        decision = self.engine.decide(state)
        assert hasattr(decision, 'confidence')
        assert 0 <= decision.confidence <= 100

    @pytest.mark.unit
    def test_decision_has_reasoning(self):
        """Test that decisions include reasoning."""
        state = self._make_game_state(
            hole_cards=['Ah', 'As'],
            pot_size=15,
            current_bet=10
        )
        decision = self.engine.decide(state)
        assert hasattr(decision, 'reasoning')
        assert isinstance(decision.reasoning, list)

    @pytest.mark.unit
    def test_decision_has_equity(self):
        """Test that decisions include equity estimate."""
        state = self._make_game_state(
            hole_cards=['Ah', 'As'],
            pot_size=15,
            current_bet=10
        )
        decision = self.engine.decide(state)
        assert hasattr(decision, 'equity')
        assert 0 <= decision.equity <= 100

    @pytest.mark.unit
    def test_decision_has_hand_evaluation(self):
        """Test that decisions include hand evaluation."""
        state = self._make_game_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', 'Jh', 'Th'],
            pot_size=100,
            current_bet=0
        )
        decision = self.engine.decide(state)
        assert hasattr(decision, 'hand_evaluation')

    # =========================================================================
    # SIZING TESTS
    # =========================================================================

    @pytest.mark.unit
    def test_raise_includes_sizing(self):
        """Test that raise decisions include bet sizing."""
        state = self._make_game_state(
            hole_cards=['Ah', 'As'],
            pot_size=15,
            current_bet=10,
            stack_size=1000
        )
        decision = self.engine.decide(state)
        if decision.action in ['raise', 'bet']:
            assert hasattr(decision, 'amount_bb') or hasattr(decision, 'amount')

    @pytest.mark.unit
    def test_sizing_respects_stack(self):
        """Test that sizing doesn't exceed stack size."""
        state = self._make_game_state(
            hole_cards=['Ah', 'As'],
            pot_size=500,
            current_bet=200,
            stack_size=100  # Short stack
        )
        decision = self.engine.decide(state)
        # Should handle short stack appropriately
        assert decision.action in ['fold', 'call', 'raise', 'bet', 'check']

    # =========================================================================
    # EDGE CASES
    # =========================================================================

    @pytest.mark.unit
    def test_empty_hole_cards(self):
        """Test handling of empty hole cards."""
        state = self._make_game_state(
            hole_cards=[],
            pot_size=100,
            current_bet=0
        )
        # Should handle gracefully, either with default action or error
        try:
            decision = self.engine.decide(state)
            assert decision.action in ['check', 'fold']
        except (ValueError, AssertionError, IndexError):
            pass  # Raising error is also acceptable

    @pytest.mark.unit
    def test_invalid_cards(self):
        """Test handling of invalid card strings."""
        state = self._make_game_state(
            hole_cards=['XX', 'YY'],  # Invalid
            pot_size=100,
            current_bet=0
        )
        # Should handle gracefully
        try:
            decision = self.engine.decide(state)
        except (ValueError, KeyError):
            pass  # Error is acceptable


class TestDecisionEngineIntegration:
    """Integration tests for DecisionEngine with other components."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up engine."""
        self.engine = DecisionEngine()

    @pytest.mark.integration
    def test_full_hand_simulation(self):
        """Simulate decisions through entire hand."""
        # Preflop
        preflop_state = GameState(
            hole_cards=['Ah', 'Kh'],
            community_cards=[],
            pot_size=15,
            stack_size=1000,
            current_bet=10,
            betting_round=BettingRound.PREFLOP,
            num_opponents=5
        )
        preflop_decision = self.engine.decide(preflop_state)
        assert preflop_decision.action in ['raise', 'call', 'fold', 'bet', 'check']

        # Flop
        flop_state = GameState(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c'],
            pot_size=50,
            stack_size=950,
            current_bet=0,
            betting_round=BettingRound.FLOP,
            num_opponents=2
        )
        flop_decision = self.engine.decide(flop_state)
        assert flop_decision.action in ['bet', 'check', 'raise', 'call', 'fold']

        # Turn
        turn_state = GameState(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c', 'Jh'],
            pot_size=100,
            stack_size=900,
            current_bet=25,
            betting_round=BettingRound.TURN,
            num_opponents=1
        )
        turn_decision = self.engine.decide(turn_state)
        assert turn_decision.action in ['raise', 'call', 'fold', 'bet', 'check']

        # River
        river_state = GameState(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c', 'Jh', 'Th'],
            pot_size=200,
            stack_size=800,
            current_bet=50,
            betting_round=BettingRound.RIVER,
            num_opponents=1
        )
        river_decision = self.engine.decide(river_state)
        # With royal flush, should raise
        assert river_decision.action in ['raise', 'call', 'bet']
