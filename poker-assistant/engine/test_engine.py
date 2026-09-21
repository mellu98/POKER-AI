"""
Unit tests for the Assistant Engine.
"""
import sys
from pathlib import Path

# Ensure we can import from this directory
sys.path.insert(0, str(Path(__file__).parent))

from assistant_engine import AssistantEngine


def test_loads_without_crash():
    engine = AssistantEngine()
    assert engine.preflop_infosets is not None
    assert engine.postflop_infosets is not None
    print("[PASS] Models loaded successfully")


def test_preflop_recommendation():
    engine = AssistantEngine()
    # Scenario: opponent bet 10, we must decide (fold/call/raise)
    rec = engine.recommend(
        hole=["As", "Kh"],
        board=[],
        history=["b10"],
        pot=24,
        stack=988,
        big_blind=2,
    )
    assert rec["stage"] == "preflop"
    assert "action" in rec
    assert "strategy" in rec
    assert "infoset_key" in rec
    print(f"[PASS] Preflop recommendation: {rec['action']} | strategy: {rec['strategy']}")


def test_postflop_recommendation():
    engine = AssistantEngine()
    # Scenario: flop, opponent bet 20, we must decide
    rec = engine.recommend(
        hole=["As", "Kh"],
        board=["Qd", "Jh", "2c"],
        history=["b20"],
        pot=64,
        stack=968,
        big_blind=2,
    )
    assert rec["stage"] == "postflop"
    assert "action" in rec
    assert "strategy" in rec
    print(f"[PASS] Postflop recommendation: {rec['action']} | strategy: {rec['strategy']}")


def test_multiple_random_hands():
    """Stress-test: 100 random lookups should not crash."""
    engine = AssistantEngine()
    import random

    holes = [
        ["As", "Kh"], ["7d", "2c"], ["Qh", "Qd"], ["Jc", "Jd"],
        ["9s", "9h"], ["Ad", "Ac"], ["Ks", "Qs"], ["Th", "Tc"],
    ]
    boards = [
        [],
        ["Qd", "Jh", "2c"],
        ["As", "Kd", "3h", "7c"],
        ["2s", "3d", "4h", "5c", "6d"],
    ]
    # Only use histories that do NOT end the betting round
    histories = [
        [],
        ["k"],
        ["b10"],
        ["k", "b20"],
        ["b30"],
    ]

    for i in range(100):
        hole = random.choice(holes)
        # Ensure board does not overlap with hole cards
        all_cards = [f"{r}{s}" for s in "shcd" for r in "23456789TJQKA"]
        available = [c for c in all_cards if c not in hole]
        board = random.choice(boards)
        if board:
            board = random.sample(available, len(board))
        history = random.choice(histories)
        rec = engine.recommend(
            hole=hole,
            board=board,
            history=history,
            pot=random.randint(10, 500),
            stack=random.randint(100, 1000),
            big_blind=2,
        )
        assert rec["action"] is not None
        assert rec["strategy"] is not None

    print("[PASS] 100 random lookups completed without crash")


if __name__ == "__main__":
    test_loads_without_crash()
    test_preflop_recommendation()
    test_postflop_recommendation()
    test_multiple_random_hands()
    print("\n=== ALL TESTS PASSED ===")
