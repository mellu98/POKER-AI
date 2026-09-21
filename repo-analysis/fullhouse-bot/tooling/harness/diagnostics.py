"""Diagnostic tracing and leak analysis for the bench harness."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from bot.features import infer_dealer

from .engine import _fold_fn, _play_hand, load_decide

REPO_ROOT = Path(__file__).resolve().parents[2]

from engine.game import PokerEngine, STARTING_STACK

BB = 100
POSITIONS_6MAX = ("BTN", "SB", "BB", "LJ", "HJ", "CO")
POSITIONS_BY_COUNT = {
    2: ("BTN", "BB"),
    3: ("BTN", "SB", "BB"),
    4: ("BTN", "SB", "BB", "CO"),
    5: ("BTN", "SB", "BB", "HJ", "CO"),
}

_FORK_CTX = mp.get_context("fork")


def _position(state: dict) -> str:
    n_players = len(state.get("players", []))
    seat = state["seat_to_act"]
    dealer = infer_dealer(state, max(n_players, 2))
    if n_players in POSITIONS_BY_COUNT:
        return POSITIONS_BY_COUNT[n_players][(seat - dealer) % n_players]
    if n_players >= 6:
        return POSITIONS_6MAX[(seat - dealer) % n_players]
    offset = (seat - dealer) % n_players
    if offset == 0:
        return "BTN"
    if offset == n_players - 1:
        return "BB"
    if offset == n_players - 2:
        return "SB"
    return "CO"


class DecisionTracer:
    """Callable wrapper around decide() that captures per-decision traces."""

    def __init__(self, decide_fn):
        self._decide = decide_fn
        self.hand_traces: dict[str, list[dict]] = {}

    def __call__(self, state: dict) -> dict:
        result = self._decide(state)
        hand_id = state.get("hand_id", "")
        trace = {
            "street": state.get("street", "preflop"),
            "position": _position(state),
            "pot": state.get("pot", 0),
            "amount_owed": state.get("amount_owed", 0),
            "your_stack": state.get("your_stack", 0),
            "your_cards": list(state.get("your_cards", [])),
            "community_cards": list(state.get("community_cards", [])),
            "can_check": state.get("can_check", False),
            "action": result.get("action"),
            "amount": result.get("amount"),
        }
        self.hand_traces.setdefault(hand_id, []).append(trace)
        return result

    def reset(self) -> None:
        self.hand_traces = {}


def _diagnose_one(args: tuple) -> dict:
    our_path, opp_path, opp_name, seed, match_idx, n_hands = args
    tracer = DecisionTracer(load_decide(our_path))

    try:
        opp_decide = load_decide(opp_path)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "opponent %s failed to load, substituting fold: %s",
            opp_name,
            exc,
        )
        opp_decide = _fold_fn

    bot_ids = ["ours", opp_name]
    deciders = {"ours": tracer, opp_name: opp_decide}
    stacks: dict[str, int] = {"ours": STARTING_STACK, opp_name: STARTING_STACK}
    match_action_log: list[dict] = []
    errors: dict[str, list[str]] = {"ours": [], opp_name: []}
    dealer = 0
    match_id = f"diag_{match_idx:03d}"

    hand_deltas: dict[str, int] = {}
    for hand_num in range(n_hands):
        alive = [bot_id for bot_id in bot_ids if stacks[bot_id] > 0]
        if len(alive) < 2:
            break

        hand_seed = (seed * 1000003 + hand_num) if seed is not None else None
        hand_id = f"{match_id}_h{hand_num:04d}"
        stack_before = stacks.get("ours", 0)

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

        stack_after = stacks.get("ours", 0)
        hand_deltas[hand_id] = stack_after - stack_before
        dealer += 1

    return {
        "match_result": {
            "match_id": match_id,
            "seed": seed,
            "chip_delta": {
                bot_id: stacks[bot_id] - STARTING_STACK for bot_id in bot_ids
            },
            "bot_errors": errors,
        },
        "hand_traces": dict(tracer.hand_traces),
        "hand_deltas": hand_deltas,
    }


@dataclass
class GroupStats:
    n_decisions: int = 0
    n_hands: int = 0
    total_delta: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)

    @property
    def bb100(self) -> float:
        if self.n_hands == 0:
            return 0.0
        return (self.total_delta / self.n_hands) / BB * 100

    def action_pct(self, action: str) -> float:
        if self.n_decisions == 0:
            return 0.0
        return self.action_counts.get(action, 0) / self.n_decisions * 100


class LeakReport:
    def __init__(
        self,
        all_traces: dict[str, list[dict]],
        hand_deltas: dict[str, int],
        match_results: list[dict],
        opponent: str,
        n_matches: int,
        hands_per_match: int,
        seed: int,
    ):
        self.all_traces = all_traces
        self.hand_deltas = hand_deltas
        self.match_results = match_results
        self.opponent = opponent
        self.n_matches = n_matches
        self.hands_per_match = hands_per_match
        self.seed = seed

    def _by_group(self, key_fn) -> dict[str, GroupStats]:
        groups: dict[str, GroupStats] = {}
        hand_groups: dict[str, set[str]] = {}

        for hand_id, decisions in self.all_traces.items():
            hand_groups[hand_id] = set()
            for decision in decisions:
                key = key_fn(decision)
                if key is None:
                    continue
                group = groups.setdefault(key, GroupStats())
                group.n_decisions += 1
                action = decision.get("action", "unknown")
                group.action_counts[action] = group.action_counts.get(action, 0) + 1
                hand_groups[hand_id].add(key)

        for hand_id, keys in hand_groups.items():
            delta = self.hand_deltas.get(hand_id, 0)
            for key in keys:
                group = groups[key]
                group.n_hands += 1
                group.total_delta += delta

        return groups

    def _overall_stats(self) -> dict[str, Any]:
        deltas = list(self.hand_deltas.values())
        n = len(deltas)
        if n == 0:
            return {"n": 0, "bb100": 0.0, "ci": 0.0}
        total = sum(deltas)
        mean = total / n
        bb100 = mean / BB * 100
        sd = statistics.stdev(deltas) if n > 1 else 0.0
        stderr = sd / math.sqrt(n)
        ci = 1.96 * stderr / BB * 100
        return {"n": n, "bb100": round(bb100, 2), "ci": round(ci, 2)}

    def _top_losses(self, n: int = 10) -> list[tuple[str, int, list[dict]]]:
        sorted_hands = sorted(self.hand_deltas.items(), key=lambda item: item[1])
        return [
            (hand_id, delta, self.all_traces.get(hand_id, []))
            for hand_id, delta in sorted_hands[:n]
        ]

    def _leak_tag(self, stats: GroupStats) -> str:
        if stats.bb100 < -10 and stats.n_decisions >= 20:
            return " <- LEAK"
        return ""

    def print_report(self) -> None:
        overall = self._overall_stats()
        print(f"\n=== LEAK REPORT vs {self.opponent} ===")
        print(
            f"hands: {overall['n']} ({self.n_matches} matches x {self.hands_per_match}), seed={self.seed}"
        )
        print(f"overall: {overall['bb100']:+.2f} bb/100 +/- {overall['ci']:.2f}")
        print(
            "note: per-group bb/100 sums > overall (a hand contributes to every group it touches)"
        )

        by_street = self._by_group(lambda decision: decision.get("street", "preflop"))
        print("\nBY STREET:")
        for street in ("preflop", "flop", "turn", "river"):
            group = by_street.get(street)
            if group:
                print(
                    f"  {street:<10s} {group.bb100:+8.2f} bb/100  ({group.n_decisions} decisions){self._leak_tag(group)}"
                )

        by_pos = self._by_group(lambda decision: decision.get("position"))
        print("\nBY POSITION:")
        pos_parts = []
        for pos in ("BTN", "CO", "HJ", "LJ", "SB", "BB"):
            group = by_pos.get(pos)
            if group:
                pos_parts.append(f"  {pos}: {group.bb100:+.2f}")
        print("  ".join(pos_parts))

        losses = self._top_losses(10)
        if losses:
            print("\nTOP 10 BIGGEST LOSSES:")
            for hand_id, delta, decisions in losses:
                if not decisions:
                    print(f"  Hand {hand_id}: delta: {delta:+d} (no trace)")
                    continue
                cards = " ".join(decisions[0].get("your_cards", []))
                board_cards: list[str] = []
                for decision in decisions:
                    for card in decision.get("community_cards", []):
                        if card not in board_cards:
                            board_cards.append(card)
                board = " ".join(board_cards) if board_cards else "-"
                pos = decisions[0].get("position", "?")
                print(
                    f"  Hand {hand_id}: {cards} | Board: {board} | {pos} | delta: {delta:+d}"
                )
                for decision in decisions:
                    action = decision.get("action", "?")
                    amount = decision.get("amount")
                    street = decision.get("street", "?")
                    amount_str = f" {amount}" if amount is not None else ""
                    print(f"    {street}: {action}{amount_str}")

    def to_dict(self) -> dict:
        overall = self._overall_stats()
        by_street = self._by_group(lambda decision: decision.get("street", "preflop"))
        by_position = self._by_group(lambda decision: decision.get("position"))
        return {
            "opponent": self.opponent,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "overall": overall,
            "by_street": {
                key: {
                    "bb100": round(group.bb100, 2),
                    "n_decisions": group.n_decisions,
                    "n_hands": group.n_hands,
                    "action_counts": group.action_counts,
                }
                for key, group in by_street.items()
            },
            "by_position": {
                key: {
                    "bb100": round(group.bb100, 2),
                    "n_decisions": group.n_decisions,
                    "n_hands": group.n_hands,
                    "action_counts": group.action_counts,
                }
                for key, group in by_position.items()
            },
            "top_losses": [
                {"hand_id": hand_id, "delta": delta, "decisions": decisions}
                for hand_id, delta, decisions in self._top_losses(10)
            ],
            "match_results": self.match_results,
        }


def run_diagnose(
    our_path: str | Path,
    opp_name: str,
    opp_path: str | Path,
    *,
    n_matches: int,
    n_hands: int,
    seed: int,
    procs: int,
) -> int:
    jobs = [
        (str(our_path), str(opp_path), opp_name, seed, idx, n_hands)
        for idx in range(n_matches)
    ]

    start = time.time()
    all_traces: dict[str, list[dict]] = {}
    hand_deltas: dict[str, int] = {}
    match_results: list[dict] = []
    with _FORK_CTX.Pool(procs) as pool:
        for result in pool.imap_unordered(_diagnose_one, jobs):
            match_results.append(result["match_result"])
            all_traces.update(result["hand_traces"])
            hand_deltas.update(result["hand_deltas"])

    report = LeakReport(
        all_traces,
        hand_deltas,
        match_results,
        opponent=opp_name,
        n_matches=n_matches,
        hands_per_match=n_hands,
        seed=seed,
    )
    report.print_report()

    out_dir = REPO_ROOT / "sims" / "diagnose"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir / f"diagnose_{opp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)

    duration = time.time() - start
    print(f"\nwrote {out_path}")
    print(f"duration: {duration:.1f}s")
    return 0
