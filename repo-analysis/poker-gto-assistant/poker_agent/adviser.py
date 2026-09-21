"""Adviser — роутинг между preflop chart, equity calc и solver."""

from __future__ import annotations

from .equity import equity_vs_random
from .game_state import GameState, Position, Street
from .preflop import hand_code, range_size_pct, rfi_advice
from .solver import is_available as solver_available
from .solver import solve_postflop


def advise(state: GameState) -> dict:
    """Главная точка входа. Возвращает dict со всеми вычислениями для UI."""
    result: dict = {
        "street": state.street.value,
        "hand_code": hand_code(*state.hero_cards),
        "pot_odds": state.pot_odds,
    }

    if state.street == Street.PREFLOP:
        result.update(_preflop_advice(state))
    else:
        result.update(_postflop_advice(state))

    return result


def _preflop_advice(state: GameState) -> dict:
    """Префлоп: если pot невозможно открыт — RFI chart, иначе пока заглушка."""
    hero_pos = state.hero.position

    # Был ли уже raise до hero?
    raises_before_hero = sum(
        1
        for a in state.action_history
        if a.kind.value in ("bet", "raise")
        and a.player != hero_pos
    )

    if raises_before_hero == 0:
        rfi = rfi_advice(hero_pos, *state.hero_cards)
        rfi["range_pct"] = range_size_pct(hero_pos)
        rfi["reasoning"] = (
            f"RFI {hero_pos.value}: рейзим {rfi['range_pct']:.0%} рук. "
            f"{rfi['hand']} {'IN range' if rfi['action'] == 'raise' else 'OUT of range'}."
        )
        return {"phase": "RFI", **rfi}

    # facing raise — пока заглушка
    return {
        "phase": "facing_raise",
        "action": "TBD (M2: 3bet/call/fold charts)",
        "reasoning": "Префлоп vs open — чарты будут в M2.",
    }


def _postflop_advice(state: GameState) -> dict:
    """Постфлоп: equity + solver (если установлен)."""
    n_villains = state.num_players_in_hand - 1
    eq = equity_vs_random(
        state.hero_cards,
        state.board,
        n_opponents=max(1, n_villains),
        n_iterations=3000,
    )

    out: dict = {
        "phase": "postflop",
        "equity_vs_random": eq["equity"],
        "win_pct": eq["win"],
        "tie_pct": eq["tie"],
        "lose_pct": eq["lose"],
        "reasoning": (
            f"vs {n_villains} random: equity {eq['equity']:.1%}. "
            f"Pot odds {state.pot_odds:.1%} — "
            f"{'+EV call' if eq['equity'] > state.pot_odds else 'не хватает на call'}."
        ),
    }

    if solver_available():
        sr = solve_postflop(state)
        out["solver"] = {
            "available": True,
            "best_action": sr.best_action,
            "all_actions": sr.actions,
        }
    else:
        out["solver"] = {"available": False, "msg": "TexasSolver не установлен (см. solver.install_instructions())"}

    return out
