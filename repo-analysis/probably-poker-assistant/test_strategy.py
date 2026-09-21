"""
Test hand evaluation and strategy engine.
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.strategy.hand_evaluator import HandEvaluator, HandType
from src.strategy.equity_calculator import EquityCalculator
from src.strategy.pot_odds import PotOddsCalculator
from src.strategy.decision_engine import DecisionEngine
from src.detection.game_state import GameState, BettingRound
from src.utils.logger import logger

def test_hand_evaluator():
    """Test hand evaluation."""
    logger.info("="*50)
    logger.info("Testing Hand Evaluator")
    logger.info("="*50)
    
    evaluator = HandEvaluator()
    
    # Test cases: (hole, community, expected_description)
    test_hands = [
        (["Ah", "Kh"], ["Qh", "Jh", "Th"], "Royal Flush"),
        (["As", "Ad"], ["Ac", "Ah", "Kd"], "Four Ace"),
        (["Kh", "Kd"], ["Ks", "2h", "2d"], "King"),
        (["7h", "8h"], ["9h", "Th", "2h"], "Flush"),
        (["9c", "8d"], ["7h", "6s", "5c"], "Straight"),
        (["Qs", "Qh"], ["Qd", "5c", "3h"], "Three Queen"),
        (["Jh", "Jd"], ["9h", "9c", "2s"], "Two Pair"),
        (["Ah", "Kd"], ["Ac", "7h", "3c"], "Pair of Ace"),
    ]
    
    for hole, community, expected in test_hands:
        result = evaluator.evaluate(hole, community)
        logger.info(f"{hole} + {community} = {result.description}")
        assert expected.lower() in result.description.lower(), f"Expected {expected}, got {result.description}"
    
    logger.info("✓ All hand evaluation tests passed")
    return True

def test_equity_calculator():
    """Test equity calculation."""
    logger.info("\n" + "="*50)
    logger.info("Testing Equity Calculator")
    logger.info("="*50)
    
    calc = EquityCalculator()
    
    # Test pocket aces vs random hand
    equity = calc.calculate_equity(["Ah", "Ad"], [], num_opponents=1, iterations=100)
    logger.info(f"AA preflop equity: {equity:.1f}%")
    assert equity > 70, "AA should have >70% equity"
    
    # Test top pair
    equity = calc.calculate_equity(["Ah", "Kd"], ["Ac", "7h", "3c"], num_opponents=1, iterations=100)
    logger.info(f"Top pair equity: {equity:.1f}%")
    assert equity > 50, "Top pair should have >50% equity"
    
    logger.info("✓ Equity calculation tests passed")
    return True

def test_decision_engine():
    """Test decision engine."""
    logger.info("\n" + "="*50)
    logger.info("Testing Decision Engine")
    logger.info("="*50)
    
    engine = DecisionEngine()
    
    # Create dummy game state
    gs = GameState(
        hole_cards=["Ah", "Ad"],
        community_cards=[],
        pot_size=10.0,
        stack_size=100.0,
        current_bet=0.0,
        betting_round=BettingRound.PREFLOP
    )
    
    decision = engine.decide(gs)
    logger.info(f"Decision with AA: {decision}")
    assert decision.action == "raise"
    
    # Test fold scenario
    gs_bad = GameState(
        hole_cards=["7c", "2h"],
        community_cards=[],
        pot_size=20.0,
        stack_size=90.0,
        current_bet=10.0,
        betting_round=BettingRound.PREFLOP,
        num_opponents=3
    )
    decision_bad = engine.decide(gs_bad)
    logger.info(f"Decision with 72o vs 10 chip bet: {decision_bad}")
    assert decision_bad.action == "fold"
    
    logger.info("✓ Decision engine tests passed")
    return True

if __name__ == "__main__":
    try:
        test_hand_evaluator()
        test_equity_calculator()
        test_decision_engine()
        logger.info("\n" + "="*50)
        logger.info("✓ ALL STRATEGY TESTS PASSED")
        logger.info("="*50)
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
