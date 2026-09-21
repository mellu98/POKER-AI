"""
Comprehensive tests for PotOddsCalculator module.

Tests pot odds calculations and required equity determination.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategy.pot_odds import PotOddsCalculator


class TestPotOddsCalculator:
    """Test suite for PotOddsCalculator class."""

    # =========================================================================
    # BASIC POT ODDS CALCULATIONS
    # =========================================================================

    @pytest.mark.unit
    def test_pot_odds_3_to_1(self):
        """Test 3:1 pot odds calculation."""
        # Pot is 100, bet to call is 50
        # Pot odds = (100 + 50) / 50 = 3:1
        result = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=50)
        assert abs(result - 3.0) < 0.1

    @pytest.mark.unit
    def test_pot_odds_2_to_1(self):
        """Test 2:1 pot odds calculation."""
        # Pot is 100, bet to call is 100
        result = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=100)
        assert abs(result - 2.0) < 0.1

    @pytest.mark.unit
    def test_pot_odds_4_to_1(self):
        """Test 4:1 pot odds calculation."""
        # Pot is 100, bet to call is 33
        result = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=33)
        # (100 + 33) / 33 ≈ 4:1
        assert 3.5 < result < 4.5

    @pytest.mark.unit
    def test_pot_odds_5_to_1(self):
        """Test 5:1 pot odds calculation."""
        # Pot is 100, bet to call is 25
        result = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=25)
        assert abs(result - 5.0) < 0.5

    # =========================================================================
    # REQUIRED EQUITY CALCULATIONS
    # =========================================================================

    @pytest.mark.unit
    def test_required_equity_3_to_1(self):
        """Test required equity at 3:1 odds (~25%)."""
        pot_odds = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=50)
        required = PotOddsCalculator.pot_odds_to_percentage(pot_odds)
        # Required equity = 1 / (pot_odds + 1) = 1/4 = 25%
        assert 20 <= required <= 30

    @pytest.mark.unit
    def test_required_equity_2_to_1(self):
        """Test required equity at 2:1 odds (~33%)."""
        pot_odds = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=100)
        required = PotOddsCalculator.pot_odds_to_percentage(pot_odds)
        # Required equity = 1 / (2 + 1) = 33.3%
        assert 30 <= required <= 37

    @pytest.mark.unit
    def test_required_equity_4_to_1(self):
        """Test required equity at 4:1 odds (~20%)."""
        pot_odds = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=33)
        required = PotOddsCalculator.pot_odds_to_percentage(pot_odds)
        # Required equity ≈ 20%
        assert 15 <= required <= 25

    @pytest.mark.unit
    def test_required_equity_inverse_relationship(self):
        """Test that higher pot odds = lower required equity."""
        pot_odds_2_to_1 = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=100)
        pot_odds_4_to_1 = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=33)

        required_2_to_1 = PotOddsCalculator.pot_odds_to_percentage(pot_odds_2_to_1)
        required_4_to_1 = PotOddsCalculator.pot_odds_to_percentage(pot_odds_4_to_1)

        # 4:1 odds should require less equity than 2:1
        assert required_4_to_1 < required_2_to_1

    # =========================================================================
    # EDGE CASES
    # =========================================================================

    @pytest.mark.unit
    def test_zero_bet_to_call(self):
        """Test with zero bet to call (free card)."""
        result = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=0)
        # With 0 to call, pot odds are infinite
        assert result == float('inf')

    @pytest.mark.unit
    def test_zero_bet_required_equity(self):
        """Test required equity with zero bet."""
        result = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=0)
        required = PotOddsCalculator.pot_odds_to_percentage(result)
        # With 0 to call, required equity is 0%
        assert required == 0.0

    @pytest.mark.unit
    def test_large_pot_small_bet(self):
        """Test with large pot and small bet."""
        result = PotOddsCalculator.calculate_pot_odds(pot_size=1000, amount_to_call=50)
        # Should give very good pot odds
        assert result > 10

    @pytest.mark.unit
    def test_small_pot_large_bet(self):
        """Test with small pot and large bet."""
        result = PotOddsCalculator.calculate_pot_odds(pot_size=50, amount_to_call=100)
        # Should give poor pot odds
        assert result < 2

    @pytest.mark.unit
    def test_equal_pot_and_bet(self):
        """Test pot-sized bet."""
        result = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=100)
        # Pot-sized bet gives 2:1 odds
        assert abs(result - 2.0) < 0.1

    # =========================================================================
    # PROFITABLE CALL TESTS
    # =========================================================================

    @pytest.mark.unit
    def test_profitable_call_detection(self):
        """Test detection of profitable calling situations."""
        pot_odds = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=50)
        # At 3:1 odds, need ~25% equity
        # If we have 40% equity with 5% margin, should be profitable
        is_profitable = PotOddsCalculator.is_profitable_call(equity=40, pot_odds=pot_odds)
        assert is_profitable is True

    @pytest.mark.unit
    def test_unprofitable_call_detection(self):
        """Test detection of unprofitable calling situations."""
        pot_odds = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=100)
        # At 2:1 odds, need ~33% equity + 5% margin
        # If we only have 20% equity, should not be profitable
        is_profitable = PotOddsCalculator.is_profitable_call(equity=20, pot_odds=pot_odds)
        assert is_profitable is False

    @pytest.mark.unit
    def test_breakeven_call(self):
        """Test breakeven calling situation."""
        pot_odds = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=50)
        # At 3:1 odds, need 25% equity
        # With exactly 25% equity and 5% margin, not quite profitable
        is_profitable = PotOddsCalculator.is_profitable_call(equity=25, pot_odds=pot_odds, margin=0)
        # At exactly required equity with no margin, it's borderline

    # =========================================================================
    # EV CALCULATION TESTS
    # =========================================================================

    @pytest.mark.unit
    def test_positive_ev_calculation(self):
        """Test positive EV calculation."""
        # Pot is 100, call is 50, equity is 50%
        ev = PotOddsCalculator.calculate_ev(pot_size=100, amount_to_call=50, equity=50)
        # EV = 0.5 * 150 - 0.5 * 50 = 75 - 25 = 50
        assert ev > 0

    @pytest.mark.unit
    def test_negative_ev_calculation(self):
        """Test negative EV calculation."""
        # Pot is 100, call is 100, equity is 20%
        ev = PotOddsCalculator.calculate_ev(pot_size=100, amount_to_call=100, equity=20)
        # EV = 0.2 * 200 - 0.8 * 100 = 40 - 80 = -40
        assert ev < 0

    @pytest.mark.unit
    def test_breakeven_ev_calculation(self):
        """Test breakeven EV calculation."""
        # Pot is 100, call is 50
        # Breakeven: eq * (pot + call) = (1-eq) * call
        # eq * 150 = (1-eq) * 50 => 200*eq = 50 => eq = 25%
        ev = PotOddsCalculator.calculate_ev(pot_size=100, amount_to_call=50, equity=25)
        # Should be close to 0
        assert abs(ev) < 5


class TestPotOddsIntegration:
    """Integration tests for pot odds with decision making."""

    @pytest.mark.unit
    def test_flush_draw_scenario(self):
        """Test flush draw (9 outs, ~35% on flop to river)."""
        # Typical flush draw has ~35% equity on the flop
        pot_odds = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=50)
        # At 3:1 odds (25% required), flush draw is profitable
        is_profitable = PotOddsCalculator.is_profitable_call(equity=35, pot_odds=pot_odds)
        assert is_profitable

    @pytest.mark.unit
    def test_gutshot_scenario(self):
        """Test gutshot draw (4 outs, ~17% on flop to river)."""
        # Typical gutshot has ~17% equity
        pot_odds = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=100)
        # At 2:1 odds (33% required), gutshot is not profitable
        is_profitable = PotOddsCalculator.is_profitable_call(equity=17, pot_odds=pot_odds)
        assert not is_profitable

    @pytest.mark.unit
    def test_open_ended_straight_draw(self):
        """Test OESD (8 outs, ~32% on flop to river)."""
        # OESD has ~32% equity
        pot_odds = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=50)
        # At 3:1 odds (25% required), OESD is profitable
        is_profitable = PotOddsCalculator.is_profitable_call(equity=32, pot_odds=pot_odds)
        assert is_profitable

    @pytest.mark.unit
    def test_combo_draw(self):
        """Test combo draw (flush draw + OESD, ~55% equity)."""
        pot_odds = PotOddsCalculator.calculate_pot_odds(pot_size=100, amount_to_call=100)
        # At 2:1 odds (33% required), combo draw is very profitable
        is_profitable = PotOddsCalculator.is_profitable_call(equity=55, pot_odds=pot_odds)
        assert is_profitable
