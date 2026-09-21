"""In-process match runner used by local sims and diagnostics."""

from __future__ import annotations

import importlib.util
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = REPO_ROOT / "engine_vendored"

if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from engine.game import PokerEngine, STARTING_STACK

MATCH_LOG_MAX_ENTRIES = 200

_DECIDE_CACHE: dict[str, Callable[[dict], dict]] = {}


def _resolve_bot_path(bot_path: str | Path) -> Path:
    path = Path(bot_path).resolve()
    if path.is_dir():
        path = path / "bot.py"
    return path


def clear_decide_cache(bot_path: str | Path | None = None) -> None:
    """Clear decide cache for one bot or for every cached bot."""
    if bot_path is None:
        _DECIDE_CACHE.clear()
        return
    _DECIDE_CACHE.pop(str(_resolve_bot_path(bot_path)), None)


def load_decide(bot_path: str | Path, *, cache: bool = False) -> Callable[[dict], dict]:
    """Load a bot's decide() function from a file path or bot directory."""
    path = _resolve_bot_path(bot_path)
    key = str(path)

    if cache and key in _DECIDE_CACHE:
        return _DECIDE_CACHE[key]

    if not path.is_file():
        raise FileNotFoundError(f"bot.py not found at {path}")

    mod_name = f"fhbot_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "decide"):
        raise AttributeError(f"{path} has no decide()")

    decide = module.decide
    if cache:
        _DECIDE_CACHE[key] = decide
    return decide


def run_match(
    bot_paths: dict[str, str | Path],
    n_hands: int = 400,
    seed: int | None = None,
    *,
    cache_bots: bool = False,
    cache_paths: set[str] | None = None,
) -> dict:
    """Run one match and return result fields matching the upstream harness."""
    bot_ids = list(bot_paths.keys())
    n_bots = len(bot_ids)
    if not (2 <= n_bots <= 9):
        raise ValueError(f"need 2-9 bots, got {n_bots}")

    if cache_paths is None and cache_bots:
        cache_paths = {str(_resolve_bot_path(path)) for path in bot_paths.values()}
    elif cache_paths is not None:
        cache_paths = {str(_resolve_bot_path(path)) for path in cache_paths}

    deciders: dict[str, Callable[[dict], dict]] = {}
    errors: dict[str, list[str]] = {bot_id: [] for bot_id in bot_ids}
    for bot_id, path in bot_paths.items():
        resolved = str(_resolve_bot_path(path))
        use_cache = cache_paths is not None and resolved in cache_paths
        try:
            deciders[bot_id] = load_decide(path, cache=use_cache)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "bot %s failed to load, substituting fold: %s",
                bot_id,
                exc,
            )
            errors[bot_id].append(f"load_failed: {exc}")
            deciders[bot_id] = _fold_fn

    stacks = {bot_id: STARTING_STACK for bot_id in bot_ids}
    match_action_log: list[dict] = []
    dealer = 0
    match_id = "ip_" + uuid.uuid4().hex[:8]
    start = time.time()

    last_hand_num = -1
    for hand_num in range(n_hands):
        last_hand_num = hand_num
        alive = [bot_id for bot_id in bot_ids if stacks[bot_id] > 0]
        if len(alive) < 2:
            break

        hand_id = f"{match_id}_h{hand_num:04d}"
        hand_seed = (seed * 1000003 + hand_num) if seed is not None else None
        engine = PokerEngine(
            hand_id=hand_id,
            bot_ids=alive,
            dealer_seat=dealer % len(alive),
            starting_stacks={bot_id: stacks[bot_id] for bot_id in alive},
            seed=hand_seed,
        )
        _play_hand(engine, deciders, alive, match_action_log, hand_num, errors)

        for bot_id in alive:
            stacks[bot_id] = engine.players[alive.index(bot_id)].stack
        dealer += 1

    return {
        "match_id": match_id,
        "bot_ids": bot_ids,
        "seed": seed,
        "n_hands": min(n_hands, last_hand_num + 1 if n_hands else 0),
        "duration_s": round(time.time() - start, 3),
        "final_stacks": stacks,
        "chip_delta": {bot_id: stacks[bot_id] - STARTING_STACK for bot_id in bot_ids},
        "bot_errors": errors,
    }


def _fold_fn(_state: dict) -> dict:
    return {"action": "fold"}


def _inject_log(state: dict, match_log: list[dict]) -> dict:
    if state.get("type") == "action_request":
        state["match_action_log"] = match_log[-MATCH_LOG_MAX_ENTRIES:]
    return state


def _play_hand(
    engine: PokerEngine,
    deciders: dict[str, Callable[[dict], dict]],
    active_bots: list[str],
    match_action_log: list[dict],
    hand_num: int,
    errors: dict[str, list[str]],
) -> dict:
    state = _inject_log(engine.start_hand(), match_action_log)
    steps = 0
    while state.get("type") == "action_request":
        seat = state["seat_to_act"]
        bot_id = active_bots[seat]
        try:
            action = deciders[bot_id](state)
            if not isinstance(action, dict) or "action" not in action:
                raise ValueError("decide must return dict with 'action'")
        except Exception as exc:
            errors[bot_id].append(f"hand{hand_num}: {exc!r}")
            action = {"action": "fold"}

        match_action_log.append(
            {
                "hand_num": hand_num,
                "seat": seat,
                "bot_id": bot_id,
                "action": action.get("action"),
                "amount": action.get("amount"),
            }
        )
        state = _inject_log(engine.apply_action(seat, action), match_action_log)
        steps += 1
        if steps > 1000:
            raise RuntimeError(f"hand exceeded 1000 steps: {engine.hand_id}")
    return state
