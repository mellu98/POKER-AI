"""Quick sanity checks for poker_math."""

from poker_math import (
    best_five_card_hand,
    format_cards,
    normalize_card,
    parse_card_tokens,
    pot_odds,
    preflop_hand_strength,
)


def test_normalize_and_parse():
    assert normalize_card("ah") == "Ah"
    assert normalize_card("10s") == "Ts"
    assert normalize_card("K♥") == "Kh"
    assert parse_card_tokens("Ah Kd Jh") == ["Ah", "Kd", "Jh"]
    assert format_cards(["Ah", "Kd"]) == "A♥  K♦"


def test_preflop():
    premium = preflop_hand_strength(["As", "Ah"])
    assert premium["tier"] == "premium"
    ak = preflop_hand_strength(["Ah", "Kd"])
    assert ak["tier"] == "premium"


def test_made_hand():
    # Pair of kings on board
    result = best_five_card_hand(["Ah", "Kd"], ["Kh", "Qc", "2d", "7s", "9c"])
    assert result["category"] == "One Pair"
    # Flush
    flush = best_five_card_hand(["Ah", "2h"], ["Kh", "Qh", "7h", "3c", "9d"])
    assert flush["category"] == "Flush"


def test_pot_odds():
    odds = pot_odds(100, 25)
    assert odds["required_equity_pct"] == 20.0
    assert pot_odds(50, 0)["required_equity_pct"] == 0.0


if __name__ == "__main__":
    test_normalize_and_parse()
    test_preflop()
    test_made_hand()
    test_pot_odds()
    print("All poker_math checks passed.")
