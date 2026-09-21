"""
End-to-end integration test (no GUI).
Verifies that engine + equity + state parsing work together.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "engine"))
sys.path.insert(0, str(Path(__file__).parent / "vision"))
sys.path.insert(0, str(Path(__file__).parent / "ui"))

from engine.assistant_engine import AssistantEngine
from engine.equity_service import calculate_equity, get_hand_strength_class


def test_pipeline():
    engine = AssistantEngine()

    states = [
        {
            "hole": ["As", "Kh"],
            "board": [],
            "pot": 24,
            "to_call": 20,
            "position": "BTN",
            "stage": "preflop",
            "stack": 988,
            "big_blind": 2,
        },
        {
            "hole": ["Qh", "Qd"],
            "board": ["Qs", "Jd", "Tc"],
            "pot": 120,
            "to_call": 0,
            "position": "BTN",
            "stage": "flop",
            "stack": 900,
            "big_blind": 2,
        },
    ]

    for state in states:
        hole = state["hole"]
        board = state["board"]
        pot = state["pot"]
        to_call = state["to_call"]
        stage = state["stage"]

        # Equity
        eq = calculate_equity(hole, board, n=1000)
        strength = get_hand_strength_class(hole, board)

        # Engine recommendation
        history = [f"b{to_call}"] if to_call > 0 else []
        rec = engine.recommend(
            hole=hole,
            board=board,
            history=history,
            pot=pot,
            stack=state["stack"],
            big_blind=state["big_blind"],
            is_dealer=(state["position"] == "BTN"),
        )

        print(f"Stage: {stage:8s} | Hand: {' '.join(hole):6s} | Board: {' '.join(board) or '-':11s}")
        print(f"  Equity: {eq:6.1%} | Strength: {strength}")
        print(f"  Action: {rec['action']:6s} | Strategy: {rec['strategy']}")
        print()

    print("=== END-TO-END TEST PASSED ===")


if __name__ == "__main__":
    test_pipeline()
