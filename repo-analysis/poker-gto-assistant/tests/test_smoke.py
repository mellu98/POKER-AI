"""Smoke-тест: импорты + базовый сценарий."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_agent.adviser import advise
from poker_agent.equity import equity_vs_random
from poker_agent.game_state import (
    Card,
    GameState,
    Player,
    Position,
    parse_cards,
)
from poker_agent.preflop import hand_code, rfi_advice


def test_card_parsing():
    cards = parse_cards("AsKs")
    assert len(cards) == 2
    assert cards[0].code == "As"
    assert cards[1].code == "Ks"


def test_hand_code():
    a, k = parse_cards("AhKh")
    assert hand_code(a, k) == "AKs"
    a, k = parse_cards("AhKd")
    assert hand_code(a, k) == "AKo"
    a1, a2 = parse_cards("AhAd")
    assert hand_code(a1, a2) == "AA"


def test_rfi_advice_btn_strong():
    a, k = parse_cards("AhKh")
    rec = rfi_advice(Position.BTN, a, k)
    assert rec["action"] == "raise"
    assert rec["size_bb"] == 2.5


def test_rfi_advice_utg_trash():
    c1, c2 = parse_cards("7h2c")
    rec = rfi_advice(Position.UTG, c1, c2)
    assert rec["action"] == "fold"


def test_equity_aces_vs_random():
    aa = (Card(code="As"), Card(code="Ac"))
    eq = equity_vs_random(aa, [], n_opponents=1, n_iterations=1000, seed=42)
    # AA heads-up должно быть ~85%
    assert eq["equity"] > 0.80, f"AA heads-up equity слишком низкое: {eq['equity']:.3f}"


def test_full_preflop_scenario():
    state = GameState(
        hero_cards=tuple(parse_cards("AhKh")),
        board=[],
        players=[
            Player(position=p, stack_bb=100, in_hand=True, is_hero=(p == Position.BTN))
            for p in [Position.UTG, Position.MP, Position.CO, Position.BTN, Position.SB, Position.BB]
        ],
        pot_bb=1.5,
        to_call_bb=0,
        effective_stack_bb=100,
    )
    rec = advise(state)
    assert rec["phase"] == "RFI"
    assert rec["action"] == "raise"
    assert rec["hand_code"] == "AKs"


def test_postflop_scenario():
    state = GameState(
        hero_cards=tuple(parse_cards("AhKh")),
        board=parse_cards("Qh7s2c"),
        players=[
            Player(position=p, stack_bb=100, in_hand=True, is_hero=(p == Position.BTN))
            for p in [Position.BTN, Position.BB]
        ],
        pot_bb=6.0,
        to_call_bb=4.0,
        effective_stack_bb=97,
    )
    rec = advise(state)
    assert rec["phase"] == "postflop"
    assert "equity_vs_random" in rec
    assert 0 <= rec["equity_vs_random"] <= 1


if __name__ == "__main__":
    print("Запускаю smoke tests...")
    test_card_parsing()
    print("  ✓ card parsing")
    test_hand_code()
    print("  ✓ hand code")
    test_rfi_advice_btn_strong()
    print("  ✓ RFI BTN AKs = raise")
    test_rfi_advice_utg_trash()
    print("  ✓ RFI UTG 72o = fold")
    test_equity_aces_vs_random()
    print("  ✓ equity AA vs random")
    test_full_preflop_scenario()
    print("  ✓ full preflop scenario")
    test_postflop_scenario()
    print("  ✓ postflop scenario")
    print("\nALL OK")
