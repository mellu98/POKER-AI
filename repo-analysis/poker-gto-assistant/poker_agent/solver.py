"""TexasSolver wrapper — постфлоп GTO через subprocess.

TexasSolver: https://github.com/bupticybee/TexasSolver
Установка: скачай бинарь под macOS, положи путь в TEXAS_SOLVER_BIN.

Эта обёртка — STUB на M1: API готов, реальный вызов закомментирован.
Подключим когда фронтенд заработает на ручных решениях.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .game_state import GameState

TEXAS_SOLVER_BIN = os.getenv("TEXAS_SOLVER_BIN", "/Applications/TexasSolver/console_solver")


@dataclass
class SolverResult:
    """Решение солвера для текущего spot-а."""

    actions: list[dict]  # [{'action': 'check', 'freq': 0.55, 'ev': 12.3}, ...]
    raw_output: str = ""

    @property
    def best_action(self) -> dict | None:
        if not self.actions:
            return None
        return max(self.actions, key=lambda a: a.get("freq", 0))


def is_available() -> bool:
    """Проверить доступен ли бинарь TexasSolver."""
    return Path(TEXAS_SOLVER_BIN).exists()


def _build_config(state: GameState, oop_range: str, ip_range: str) -> str:
    """Сгенерировать config для TexasSolver console mode.

    TexasSolver принимает текстовый сценарий с командами:
        set_pot, set_effective_stack, set_board, set_range_ip, set_range_oop,
        set_bet_sizes, build_tree, start_solve, dump_result, ...

    Это упрощённая заготовка — реальный config зависит от стрита и линии.
    """
    board_str = "".join(c.code for c in state.board)
    cmds = [
        f"set_pot {state.pot_bb}",
        f"set_effective_stack {state.effective_stack_bb}",
        f"set_board {board_str}",
        f"set_range_oop {oop_range}",
        f"set_range_ip {ip_range}",
        "set_bet_sizes oop,flop,bet,50",
        "set_bet_sizes ip,flop,bet,50",
        "set_bet_sizes oop,flop,raise,60",
        "set_bet_sizes ip,flop,raise,60",
        "set_allin_threshold 0.67",
        "set_raise_limit 3",
        "build_tree",
        "set_thread_num 8",
        "set_accuracy 0.5",
        "set_max_iteration 200",
        "start_solve",
        "dump_result strategy.json",
    ]
    return "\n".join(cmds) + "\n"


def solve_postflop(
    state: GameState,
    oop_range: str = "AA-22,AKs-A2s,AKo-A2o",  # placeholder
    ip_range: str = "AA-22,AKs-A2s,AKo-A2o",
    timeout_sec: int = 60,
) -> SolverResult:
    """Запустить TexasSolver и получить стратегию для текущего spot-а.

    TODO M2: дописать парсер strategy.json под наш SolverResult.
    """
    if not is_available():
        # graceful degradation — UI должен показать что solver недоступен
        return SolverResult(actions=[], raw_output="solver_not_installed")

    config = _build_config(state, oop_range, ip_range)

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "scenario.txt"
        cfg_path.write_text(config)

        try:
            result = subprocess.run(
                [TEXAS_SOLVER_BIN, str(cfg_path)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return SolverResult(actions=[], raw_output="timeout")

        # TODO M2: распарсить strategy.json в Path(tmp) / "strategy.json"
        return SolverResult(actions=[], raw_output=result.stdout)


def install_instructions() -> str:
    return (
        "TexasSolver не установлен. Поставь:\n"
        "  1. https://github.com/bupticybee/TexasSolver/releases — скачай macOS-бинарь\n"
        "  2. Распакуй в /Applications/TexasSolver/\n"
        "  3. export TEXAS_SOLVER_BIN=/Applications/TexasSolver/console_solver\n"
    )
