"""
Comprehensive tests for GameState and GameStateTracker modules.

Tests game state management and betting round detection.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.detection.game_state import GameState, GameStateTracker, BettingRound


class TestBettingRound:
    """Test suite for BettingRound enum."""

    @pytest.mark.unit
    def test_betting_round_values(self):
        """Test all betting round values exist."""
        assert BettingRound.PREFLOP is not None
        assert BettingRound.FLOP is not None
        assert BettingRound.TURN is not None
        assert BettingRound.RIVER is not None

    @pytest.mark.unit
    def test_betting_round_string_values(self):
        """Test betting rounds have string values."""
        assert BettingRound.PREFLOP.value == "preflop"
        assert BettingRound.FLOP.value == "flop"
        assert BettingRound.TURN.value == "turn"
        assert BettingRound.RIVER.value == "river"


class TestGameState:
    """Test suite for GameState dataclass."""

    @pytest.mark.unit
    def test_gamestate_creation(self):
        """Test GameState creation with all fields."""
        state = GameState(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c'],
            pot_size=100,
            stack_size=1000,
            current_bet=25,
            betting_round=BettingRound.FLOP,
            num_opponents=2
        )
        assert state.hole_cards == ['Ah', 'Kh']
        assert state.community_cards == ['Qh', '7d', '2c']
        assert state.pot_size == 100
        assert state.stack_size == 1000
        assert state.current_bet == 25
        assert state.betting_round == BettingRound.FLOP
        assert state.num_opponents == 2

    @pytest.mark.unit
    def test_gamestate_default_values(self):
        """Test GameState with default values."""
        state = GameState(
            hole_cards=['Ah', 'Kh'],
            community_cards=[],
            pot_size=15,
            stack_size=1000,
            current_bet=10,
            betting_round=BettingRound.PREFLOP
        )
        assert len(state.community_cards) == 0
        assert state.num_opponents == 1  # Default

    @pytest.mark.unit
    def test_gamestate_empty_hole_cards(self):
        """Test GameState with empty hole cards."""
        state = GameState(
            hole_cards=[],
            community_cards=[],
            pot_size=15,
            stack_size=1000,
            current_bet=10,
            betting_round=BettingRound.PREFLOP
        )
        assert state.hole_cards == []

    @pytest.mark.unit
    def test_gamestate_str(self):
        """Test GameState string representation."""
        state = GameState(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c'],
            pot_size=100,
            stack_size=1000,
            current_bet=25,
            betting_round=BettingRound.FLOP
        )
        str_repr = str(state)
        assert 'flop' in str_repr
        assert 'Ah' in str_repr


class TestGameStateTracker:
    """Test suite for GameStateTracker class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up GameStateTracker instance."""
        self.tracker = GameStateTracker()

    # =========================================================================
    # BETTING ROUND DETECTION TESTS
    # =========================================================================

    @pytest.mark.unit
    def test_preflop_detection(self):
        """Test preflop detection with no community cards."""
        state = self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=[],
            pot_size=15,
            stack_size=1000,
            current_bet=10
        )
        assert state.betting_round == BettingRound.PREFLOP

    @pytest.mark.unit
    def test_flop_detection(self):
        """Test flop detection with 3 community cards."""
        state = self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c'],
            pot_size=50,
            stack_size=950,
            current_bet=0
        )
        assert state.betting_round == BettingRound.FLOP

    @pytest.mark.unit
    def test_turn_detection(self):
        """Test turn detection with 4 community cards."""
        state = self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c', 'Jh'],
            pot_size=100,
            stack_size=900,
            current_bet=25
        )
        assert state.betting_round == BettingRound.TURN

    @pytest.mark.unit
    def test_river_detection(self):
        """Test river detection with 5 community cards."""
        state = self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c', 'Jh', 'Th'],
            pot_size=200,
            stack_size=800,
            current_bet=50
        )
        assert state.betting_round == BettingRound.RIVER

    # =========================================================================
    # STATE TRANSITION TESTS
    # =========================================================================

    @pytest.mark.unit
    def test_state_transition_preflop_to_flop(self):
        """Test state transition from preflop to flop."""
        # Preflop
        state1 = self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=[],
            pot_size=15,
            stack_size=1000,
            current_bet=10
        )
        assert state1.betting_round == BettingRound.PREFLOP

        # Flop
        state2 = self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c'],
            pot_size=50,
            stack_size=950,
            current_bet=0
        )
        assert state2.betting_round == BettingRound.FLOP

    @pytest.mark.unit
    def test_state_transition_flop_to_turn(self):
        """Test state transition from flop to turn."""
        # Flop
        state1 = self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c'],
            pot_size=50,
            stack_size=950,
            current_bet=0
        )

        # Turn
        state2 = self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c', 'Jh'],
            pot_size=100,
            stack_size=900,
            current_bet=25
        )
        assert state2.betting_round == BettingRound.TURN

    @pytest.mark.unit
    def test_state_transition_full_hand(self):
        """Test complete hand state transitions."""
        rounds = []

        # Preflop
        state = self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=[],
            pot_size=15,
            stack_size=1000,
            current_bet=10
        )
        rounds.append(state.betting_round)

        # Flop
        state = self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c'],
            pot_size=50,
            stack_size=950,
            current_bet=0
        )
        rounds.append(state.betting_round)

        # Turn
        state = self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c', 'Jh'],
            pot_size=100,
            stack_size=900,
            current_bet=25
        )
        rounds.append(state.betting_round)

        # River
        state = self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c', 'Jh', 'Th'],
            pot_size=200,
            stack_size=800,
            current_bet=50
        )
        rounds.append(state.betting_round)

        expected = [BettingRound.PREFLOP, BettingRound.FLOP, BettingRound.TURN, BettingRound.RIVER]
        assert rounds == expected

    # =========================================================================
    # NEW HAND DETECTION TESTS
    # =========================================================================

    @pytest.mark.unit
    def test_new_hand_detection(self):
        """Test detection of new hand starting."""
        # Complete a hand to river
        self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c', 'Jh', 'Th'],
            pot_size=200,
            stack_size=800,
            current_bet=50
        )

        # Start new hand (back to preflop with new hole cards)
        state = self.tracker.update_state(
            hole_cards=['9d', '8d'],  # New hole cards
            community_cards=[],
            pot_size=15,
            stack_size=1200,  # Different stack
            current_bet=10
        )
        assert state.betting_round == BettingRound.PREFLOP

    # =========================================================================
    # EDGE CASES
    # =========================================================================

    @pytest.mark.unit
    def test_none_community_cards(self):
        """Test handling of None community cards."""
        state = self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=None,
            pot_size=100,
            stack_size=1000,
            current_bet=0
        )
        # Should handle gracefully - None treated as empty
        assert state is not None

    @pytest.mark.unit
    def test_invalid_community_card_count(self):
        """Test handling of invalid community card count."""
        # 1 community card (invalid - should be 0, 3, 4, or 5)
        state = self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh'],
            pot_size=50,
            stack_size=950,
            current_bet=0
        )
        # Should still create state
        assert state is not None

    @pytest.mark.unit
    def test_partial_state_update(self):
        """Test updating state with partial information."""
        # First update with full info
        self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c'],
            pot_size=100,
            stack_size=900,
            current_bet=25
        )

        # Second update with only some fields changed
        state = self.tracker.update_state(
            hole_cards=['Ah', 'Kh'],
            community_cards=['Qh', '7d', '2c'],
            pot_size=150,  # Pot increased
            stack_size=875,  # Stack decreased
            current_bet=0  # New betting round
        )
        assert state.pot_size == 150

    @pytest.mark.unit
    def test_determine_betting_round_method(self):
        """Test the determine_betting_round method directly."""
        assert self.tracker.determine_betting_round([]) == BettingRound.PREFLOP
        assert self.tracker.determine_betting_round(['Ah', 'Kh', 'Qh']) == BettingRound.FLOP
        assert self.tracker.determine_betting_round(['Ah', 'Kh', 'Qh', 'Jh']) == BettingRound.TURN
        assert self.tracker.determine_betting_round(['Ah', 'Kh', 'Qh', 'Jh', 'Th']) == BettingRound.RIVER
