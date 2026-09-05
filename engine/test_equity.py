"""
Unit tests for Equity Service.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from equity_service import calculate_equity, get_hand_strength_class


def test_equity_range():
    eq = calculate_equity(["As", "Ah"], [], n=1000)
    assert 0.8 < eq < 0.95, f"AA equity should be ~85%, got {eq:.2%}"
    print(f"[PASS] AA preflop equity: {eq:.2%}")


def test_equity_flop():
    eq = calculate_equity(["As", "Kh"], ["Qd", "Jh", "2c"], n=1000)
    assert 0.5 < eq < 0.75, f"AKo on QJ2 equity should be ~60%, got {eq:.2%}"
    print(f"[PASS] AKo on QJ2 equity: {eq:.2%}")


def test_hand_strength():
    cls = get_hand_strength_class(["As", "Ah"], ["Ad", "Ac", "Kc"])
    assert "Four of a Kind" in cls or "Four" in cls
    print(f"[PASS] Hand strength AA on AAK: {cls}")


def test_cache():
    eq1 = calculate_equity(["7d", "2c"], ["3h", "4s", "5d"], n=500)
    eq2 = calculate_equity(["7d", "2c"], ["3h", "4s", "5d"], n=500)
    assert eq1 == eq2, "Cache should return identical result"
    print("[PASS] Cache consistency")


if __name__ == "__main__":
    test_equity_range()
    test_equity_flop()
    test_hand_strength()
    test_cache()
    print("\n=== ALL EQUITY TESTS PASSED ===")
