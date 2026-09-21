"""Benchmark, smoke, and diagnose commands for the local harness."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import os
import statistics
import tempfile
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
SIMS_DIR = REPO_ROOT / "sims"
DEFAULT_BOT = REPO_ROOT / "bot" / "bot.py"

BB = 100
STARTING_STACK = 10_000
DEFAULT_HANDS_PER_MATCH = 400

_FORK_CTX = mp.get_context("fork")

from .bot_registry import ALL_BOTS, EXTRA_BOTS, TIER2_ARCHETYPES
from .engine import clear_decide_cache, load_decide, run_match


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=str(path.parent),
            delete=False,
            encoding="utf-8",
        ) as tmp:
            json.dump(payload, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _load_json(path: Path) -> dict:
    try:
        with path.open() as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"failed to read JSON from {path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object in {path}, got {type(data).__name__}")
    return data


def _rotate_pick(pool: list[str], n: int, offset: int) -> list[str]:
    size = len(pool)
    return [pool[(offset + i) % size] for i in range(min(n, size))]


def _compute_stats(deltas: list[float], hands_per_match: int) -> dict:
    n = len(deltas)
    if n == 0:
        return {
            "n": 0,
            "mean_chips": 0.0,
            "stderr": 0.0,
            "ci95": (0.0, 0.0),
            "bb100": 0.0,
            "bb100_ci95": (0.0, 0.0),
            "win_rate": 0.0,
        }
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    stderr = sd / math.sqrt(n)
    ci = 1.96 * stderr
    hands_factor = hands_per_match / 100
    bb100 = mean / hands_factor / BB
    bb100_ci = ci / hands_factor / BB
    return {
        "n": n,
        "mean_chips": round(mean, 1),
        "stderr": round(stderr, 1),
        "ci95": (round(mean - ci, 1), round(mean + ci, 1)),
        "bb100": round(bb100, 2),
        "bb100_ci95": (round(bb100 - bb100_ci, 2), round(bb100 + bb100_ci, 2)),
        "win_rate": round(sum(1 for delta in deltas if delta > 0) / n, 3),
    }


def _finish_position(final_stacks: dict[str, int], our_id: str) -> int:
    sorted_ids = sorted(
        final_stacks, key=lambda bot_id: final_stacks[bot_id], reverse=True
    )
    return sorted_ids.index(our_id) + 1


_WARMUP_STATE: dict = {
    "type": "action_request",
    "your_cards": ["As", "Kd"],
    "community_cards": [],
    "pot": 150,
    "your_stack": 9900,
    "amount_owed": 100,
    "can_check": False,
    "seat_to_act": 0,
    "street": "preflop",
    "current_bet": 100,
    "min_raise_to": 200,
    "your_bet_this_street": 0,
    "dealer": 0,
    "action_log": [],
    "players": [{"seat": 0, "stack": 9900, "is_folded": False, "is_all_in": False}],
}


def _warmup_bots(
    bot_paths: list[str],
    *,
    strict_paths: set[str] | None = None,
    cache_paths: set[str] | None = None,
) -> None:
    strict_paths = strict_paths or set()
    cache_paths = cache_paths or set()
    failures: list[tuple[str, Exception]] = []
    for path in bot_paths:
        try:
            decide = load_decide(path, cache=path in cache_paths)
            decide(_WARMUP_STATE)
        except Exception as exc:
            if path in strict_paths:
                raise RuntimeError(f"warmup failed for required bot {path}") from exc
            failures.append((path, exc))
    for path, exc in failures:
        print(f"warmup skipped for {path}: {exc}", flush=True)


@contextlib.contextmanager
def _override_bot_data_dir(data_dir: Path | None, candidate_path: str | Path | None):
    """Point the candidate at a staged data dir. Opponents read their own data, so
    this only affects our bot."""
    previous = os.environ.get("BOT_DATA_DIR")
    if data_dir is not None:
        os.environ["BOT_DATA_DIR"] = str(data_dir)
    if candidate_path is not None:
        clear_decide_cache(candidate_path)
    try:
        yield
    finally:
        if candidate_path is not None:
            clear_decide_cache(candidate_path)
        if previous is None:
            os.environ.pop("BOT_DATA_DIR", None)
        else:
            os.environ["BOT_DATA_DIR"] = previous


def _run_6max_one(args: tuple) -> dict:
    our_path, opp_paths_dict, seed, match_idx, n_hands, cache_paths = args
    bot_paths = {"ours": our_path, **opp_paths_dict}
    result = run_match(
        bot_paths,
        n_hands=n_hands,
        seed=seed + match_idx,
        cache_paths=cache_paths,
    )
    our_delta = result["chip_delta"]["ours"]
    finish = _finish_position(result["final_stacks"], "ours")

    opp_hand0_errors = 0
    our_errors = len(result["bot_errors"].get("ours", []))
    for bot_id, errors in result["bot_errors"].items():
        if bot_id != "ours":
            opp_hand0_errors += sum(
                1 for err in errors if "hand0:" in err or "load_failed:" in err
            )
    total_errors = sum(len(errors) for errors in result["bot_errors"].values())
    real_errors = total_errors - opp_hand0_errors
    if opp_hand0_errors > 0:
        import logging

        logging.getLogger(__name__).warning(
            "match %d: %d opponent errors filtered (hand0/load_failed)",
            match_idx,
            opp_hand0_errors,
        )

    return {
        "seed": seed + match_idx,
        "match_idx": match_idx,
        "chip_delta": our_delta,
        "finish": finish,
        "duration_s": result["duration_s"],
        "opponents": list(opp_paths_dict.keys()),
        "errors": our_errors,
        "total_real_errors": real_errors,
        "bot_errors": {
            bot_id: errs for bot_id, errs in result["bot_errors"].items() if errs
        },
    }


def run_benchmark(
    *,
    bot_path: str | Path = DEFAULT_BOT,
    matches: int = 30,
    hands: int | None = None,
    procs: int | None = None,
    seed: int = 42,
    fast: bool = False,
    out_path: Path | None = None,
    latest_path: Path | None = None,
    data_dir: Path | None = None,
    include_match_details: bool = True,
    progress: bool = True,
    include_extra: bool = False,
) -> dict:
    our_path = str(Path(bot_path).resolve())
    fast = bool(fast)

    if fast:
        bot_pool = TIER2_ARCHETYPES
        n_hands = hands or 200
        pool_label = "adversarial-only"
    else:
        bot_pool = {**ALL_BOTS, **EXTRA_BOTS} if include_extra else ALL_BOTS
        n_hands = hands or DEFAULT_HANDS_PER_MATCH
        pool_label = "all tiers + extra" if include_extra else "all tiers"

    max_procs = min(8, max(1, (mp.cpu_count() or 2) - 1))
    n_procs = procs or (max_procs if fast else min(4, max_procs))

    all_names = list(bot_pool.keys())
    jobs: list[tuple] = []
    opponent_cache_paths = {str(Path(path).resolve()) for path in bot_pool.values()}
    cache_paths = opponent_cache_paths | {our_path}
    for idx in range(matches):
        picked = _rotate_pick(all_names, 5, offset=idx)
        opp_dict = {name: bot_pool[name] for name in picked}
        jobs.append((our_path, opp_dict, seed, idx, n_hands, cache_paths))

    all_paths = list({our_path} | set(bot_pool.values()))
    with _override_bot_data_dir(data_dir, our_path):
        if progress:
            print(f"warming up {len(all_paths)} bots...", flush=True)
        warmup_start = time.time()
        _warmup_bots(all_paths, strict_paths={our_path}, cache_paths=cache_paths)
        warmup_dur = time.time() - warmup_start
        if progress:
            print(f"warmup: {warmup_dur:.1f}s", flush=True)

        start = time.time()
        results: list[dict] = []
        with _FORK_CTX.Pool(n_procs) as pool:
            for idx, result in enumerate(pool.imap_unordered(_run_6max_one, jobs)):
                results.append(result)
                if progress and (idx + 1) % max(10, matches // 5) == 0:
                    elapsed = time.time() - start
                    print(f"  [{idx + 1}/{matches}] {elapsed:.1f}s", flush=True)

    duration = time.time() - start
    results.sort(key=lambda row: row["match_idx"])

    deltas = [row["chip_delta"] for row in results]
    stats = _compute_stats(deltas, n_hands)
    bust_count = sum(1 for delta in deltas if delta == -STARTING_STACK)
    bust_rate = bust_count / matches if matches else 0.0
    finishes = [row["finish"] for row in results]
    avg_finish = statistics.fmean(finishes) if finishes else 0.0
    win_count = sum(1 for delta in deltas if delta > 0)

    previous = (
        _load_json(latest_path) if latest_path and latest_path.is_file() else None
    )

    iso_ts = datetime.now().isoformat(timespec="seconds")
    payload = {
        "candidate": our_path,
        "timestamp": iso_ts,
        "seed": seed,
        "n_matches": matches,
        "procs": n_procs,
        "hands_per_match": n_hands,
        "format": "6-max",
        "fast_mode": fast,
        "duration_s": round(duration, 1),
        "warmup_s": round(warmup_dur, 1),
        "pool_label": pool_label,
        "overall": {
            **stats,
            "ci95": list(stats["ci95"]),
            "bb100_ci95": list(stats["bb100_ci95"]),
        },
        "bust_rate": round(bust_rate, 3),
        "bust_count": bust_count,
        "avg_finish": round(avg_finish, 2),
        "win_count": win_count,
    }
    if include_match_details:
        payload["matches"] = [
            {
                "seed": row["seed"],
                "opponents": row["opponents"],
                "chip_delta": row["chip_delta"],
                "finish": row["finish"],
                "duration_s": row["duration_s"],
                "errors": row["errors"],
                "total_real_errors": row["total_real_errors"],
                "bot_errors": row["bot_errors"],
            }
            for row in results
        ]
    if previous:
        prev_bb100 = previous.get("overall", {}).get("bb100")
        if prev_bb100 is not None:
            payload["delta_vs_previous"] = {
                "timestamp": previous.get("timestamp"),
                "bb100": round(stats["bb100"] - prev_bb100, 2),
                "previous_bb100": prev_bb100,
            }

    if out_path is not None:
        _write_json_atomic(out_path, payload)
    if latest_path is not None:
        _write_json_atomic(latest_path, payload)

    return payload


def print_benchmark_report(payload: dict, *, include_matches: bool = True) -> None:
    stats = payload["overall"]
    lo, hi = stats["bb100_ci95"]
    matches = payload["n_matches"]
    procs = payload.get("procs")
    pool_label = payload.get("pool_label", "all tiers")
    if procs is None:
        procs = "?"

    print("\n=== BENCH RESULTS ===")
    print(f"candidate: {payload['candidate']}")
    print(f"timestamp: {payload['timestamp']}")
    print(f"format: 6-max, {payload['hands_per_match']} hands/match, {pool_label}")
    print(f"matches: {matches} ({procs} procs, seed={payload['seed']})")
    print(
        f"duration: {payload['duration_s']:.1f}s (warmup: {payload['warmup_s']:.1f}s)"
    )
    print()
    print("OVERALL:")
    print(
        f"  bb/100:      {stats['bb100']:+.2f} +/- {stats['bb100'] - lo:.2f}  [95% CI: {lo:+.2f} to {hi:+.2f}]"
    )
    win_rate = stats["win_rate"] * 100
    win_count = payload.get("win_count")
    if win_count is None:
        win_count = sum(
            1 for row in payload.get("matches", []) if row["chip_delta"] > 0
        )
    bust_count = payload.get("bust_count")
    if bust_count is None:
        bust_count = sum(
            1
            for row in payload.get("matches", [])
            if row["chip_delta"] == -STARTING_STACK
        )
    print(
        f"  win_rate:    {win_rate:.1f}%  ({win_count}/{matches} matches net positive)"
    )
    print(
        f"  bust_rate:   {payload['bust_rate'] * 100:.1f}%  ({bust_count}/{matches} matches busted out)"
    )
    print(f"  avg_finish:  {payload['avg_finish']:.1f} / 6")

    delta = payload.get("delta_vs_previous")
    if delta:
        prev_ts = (delta.get("timestamp") or "unknown")[:10]
        print(f"\nDELTA vs previous ({prev_ts}):")
        print(
            f"  bb/100:  {stats['bb100']:+.2f} vs {delta['previous_bb100']:+.2f}  "
            f"(delta {delta['bb100']:+.2f})"
        )

    if include_matches and "matches" in payload:
        print("\nPER-MATCH:")
        for idx, row in enumerate(payload["matches"], start=1):
            opp_str = ", ".join(row["opponents"])
            print(
                f"  #{idx:02d}  seed={row['seed']:<6d}  delta={row['chip_delta']:+5d}  "
                f"finish={row['finish']}/6  opponents: {opp_str}"
            )


def _run_smoke(args: argparse.Namespace) -> int:
    payload = run_benchmark(
        bot_path=args.bot,
        matches=5,
        hands=args.hands or DEFAULT_HANDS_PER_MATCH,
        procs=args.procs,
        seed=args.seed,
        fast=False,
        out_path=None,
        latest_path=None,
        include_match_details=True,
    )
    stats = payload["overall"]
    total_our_errors = sum(row["errors"] for row in payload["matches"])
    lo, hi = stats["bb100_ci95"]
    print("\n=== SMOKE TEST ===")
    print(f"candidate: {payload['candidate']}")
    print(
        f"matches: 5 x {payload['hands_per_match']} hands, 6-max, seed={payload['seed']}"
    )
    print(
        f"duration: {payload['duration_s']:.1f}s (warmup: {payload['warmup_s']:.1f}s)"
    )
    print()
    print("RESULT:")
    print(
        f"  bb/100:     {stats['bb100']:+.2f} +/- {stats['bb100'] - lo:.2f}  [95% CI: {lo:+.2f} to {hi:+.2f}]"
    )
    print(f"  win_rate:   {stats['win_rate'] * 100:.1f}%")
    print(f"  errors:     {total_our_errors}")
    passed = stats["bb100"] >= -50 and total_our_errors == 0
    print(f"\n{'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def _run_bench(args: argparse.Namespace) -> int:
    latest_path = SIMS_DIR / "bench_latest.json" if args.update_latest else None
    out_path = (
        Path(args.out).resolve()
        if args.out
        else SIMS_DIR / f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    payload = run_benchmark(
        bot_path=args.bot,
        matches=args.matches,
        hands=args.hands,
        procs=args.procs,
        seed=args.seed,
        fast=args.fast,
        out_path=out_path,
        latest_path=latest_path,
        include_match_details=not args.summary_only,
        include_extra=args.include_extra,
        data_dir=Path(args.data_dir) if args.data_dir else None,
    )
    print_benchmark_report(payload, include_matches=not args.summary_only)
    print(f"\nwrote {out_path}")
    return 0


def _run_diagnose(args: argparse.Namespace) -> int:
    from .diagnostics import run_diagnose

    pool = {**ALL_BOTS, **EXTRA_BOTS}
    if args.opponent not in pool:
        print(f"Unknown opponent: {args.opponent!r}")
        print(f"Available: {', '.join(sorted(pool))}")
        return 1
    return run_diagnose(
        args.bot,
        args.opponent,
        pool[args.opponent],
        n_matches=args.matches,
        n_hands=args.hands or DEFAULT_HANDS_PER_MATCH,
        seed=args.seed,
        procs=args.procs,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Poker bot test harness: smoke, bench, diagnose"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    for name, help_text in [
        ("smoke", "Quick sanity check"),
        ("bench", "Full benchmark"),
        ("diagnose", "Deep leak analysis vs one opponent"),
    ]:
        subparser = sub.add_parser(name, help=help_text)
        subparser.add_argument("--bot", default=str(DEFAULT_BOT))
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument("--procs", type=int, default=None)
        subparser.add_argument(
            "--hands",
            type=int,
            default=None,
            help="Hands per match (default: 400, or 200 with --fast)",
        )
        if name == "bench":
            subparser.add_argument("--matches", type=int, default=30)
            subparser.add_argument(
                "--out", default=None, help="Optional explicit JSON output path"
            )
            subparser.add_argument(
                "--fast",
                action="store_true",
                help="Fast mode: adversarial-only opponents, 200 hands, max procs",
            )
            subparser.add_argument(
                "--include-extra",
                action="store_true",
                help="Add the extra/ bots to the pool",
            )
            subparser.add_argument(
                "--data-dir",
                default=None,
                help="Stage the candidate from this dir (its deep_cfr_model*.npz)",
            )
            subparser.add_argument(
                "--summary-only",
                action="store_true",
                help="Skip verbose per-match console output",
            )
            subparser.add_argument(
                "--no-update-latest",
                dest="update_latest",
                action="store_false",
                help="Do not refresh sims/bench_latest.json",
            )
            subparser.set_defaults(update_latest=True)
        if name == "diagnose":
            subparser.add_argument("opponent")
            subparser.add_argument("--matches", type=int, default=10)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    fast = getattr(args, "fast", False)
    max_procs = min(8, max(1, (mp.cpu_count() or 2) - 1))
    args.procs = args.procs or (max_procs if fast else min(4, max_procs))

    if args.mode == "smoke":
        return _run_smoke(args)
    if args.mode == "bench":
        return _run_bench(args)
    if args.mode == "diagnose":
        return _run_diagnose(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
