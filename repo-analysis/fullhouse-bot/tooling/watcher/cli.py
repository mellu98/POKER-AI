"""CLI entrypoint for the training watcher."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from .runner import GPU_CKPT_BASE, GPU_HOST, LOCAL_CKPT_BASE, TrainingWatcher
from .utils import write_json_atomic

_DETACHED_ENV = "FULLHOUSE_WATCHER_DETACHED"


def _detach_process(argv: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env[_DETACHED_ENV] = "1"
    child_args = [arg for arg in argv[1:] if arg != "--detach"]
    with log_path.open("a", encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tooling.watcher.cli", *child_args],
            cwd=str(Path(__file__).resolve().parents[2]),
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    print(f"Detached watcher pid={proc.pid}. Logs: {log_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pull, bench, and plot Deep CFR checkpoints during training"
    )
    parser.add_argument("version", help="Training version (e.g. v12, v13)")
    parser.add_argument(
        "--bot", default=str(Path(__file__).resolve().parents[2] / "bot" / "bot.py")
    )
    parser.add_argument(
        "--matches", type=int, default=1000, help="Single-checkpoint benchmark matches"
    )
    parser.add_argument("--ensemble-matches", type=int, default=200)
    parser.add_argument("--ensemble-only", action="store_true")
    parser.add_argument("--once", action="store_true", help="Single pass, no polling")
    parser.add_argument("--poll", type=int, default=60, help="Seconds between polls")
    parser.add_argument(
        "--idle-limit",
        type=int,
        default=0,
        help="Idle polls before exit (0 = run forever)",
    )
    parser.add_argument(
        "--pull-only",
        action="store_true",
        help="Copy available remote checkpoints, update watcher state, and exit",
    )
    parser.add_argument(
        "--max-benches-per-cycle",
        type=int,
        default=None,
        help="Max checkpoint benches before returning to polling",
    )
    parser.add_argument(
        "--ensemble-every",
        type=int,
        default=None,
        help="Run ensemble benches every N completed checkpoint results",
    )
    parser.add_argument("--ensemble-top-k", type=int, default=6)
    parser.add_argument("--ensemble-max-models", type=int, default=4)
    parser.add_argument(
        "--strategy-every",
        type=int,
        default=0,
        help="Run strategy analysis every N completed checkpoint results",
    )
    parser.add_argument(
        "--gpu-host", default=os.environ.get("FULLHOUSE_GPU_HOST", GPU_HOST)
    )
    parser.add_argument(
        "--gpu-ckpt-base",
        default=os.environ.get("FULLHOUSE_GPU_CKPT_BASE", GPU_CKPT_BASE),
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Compatibility flag; continuous watcher mode detaches by default",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Keep the continuous watcher attached to the terminal",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_benches_per_cycle is None:
        args.max_benches_per_cycle = 0 if args.once else 1
    if args.ensemble_every is None:
        args.ensemble_every = 1 if args.once else 0

    out_dir = Path(__file__).resolve().parents[2] / "sims" / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "watcher.log"

    should_detach = os.environ.get(_DETACHED_ENV) != "1" and (
        args.detach
        or (
            not args.foreground
            and not args.once
            and not args.ensemble_only
            and not args.pull_only
        )
    )
    if should_detach:
        return _detach_process(sys.argv, log_file)

    watcher = TrainingWatcher(args)
    watcher.acquire()
    logger = watcher.logger
    try:
        if watcher.existing_results:
            logger.log(f"loaded {len(watcher.existing_results)} existing results")

        if args.pull_only:
            checkpoints, new = watcher.refresh_checkpoints()
            logger.log(
                f"pull-only complete: {new} new checkpoints, {len(checkpoints)} local total"
            )
            return 0

        if args.ensemble_only:
            checkpoints, _ = watcher.refresh_checkpoints()
            checkpoints = sorted(
                (LOCAL_CKPT_BASE / f"deep_cfr_{args.version}").glob("iter_*.npz")
            )
            if checkpoints:
                ens, err = watcher.bench_ensemble(checkpoints)
                if ens["bb100"] is not None:
                    ens["n_models"] = len(checkpoints)
                    write_json_atomic(watcher.out_dir / "ensemble_result.json", ens)
                    logger.log(
                        f"ensemble-only: {ens['bb100']:+.2f} bb/100 across {len(checkpoints)} checkpoints"
                    )
                elif err:
                    logger.log(f"ensemble-only failed: {err}", level="WARN")
            return 0

        idle_polls = 0
        while True:
            n_new = watcher.run_cycle()
            if args.once:
                break

            if n_new == 0:
                idle_polls += 1
                if args.idle_limit > 0 and idle_polls >= args.idle_limit:
                    logger.log(
                        f"idle limit reached after {idle_polls} polls; stopping watcher"
                    )
                    break
                logger.log(
                    f"no new checkpoints; sleeping {args.poll}s (idle polls={idle_polls})"
                )
            else:
                idle_polls = 0
            time.sleep(args.poll)

        logger.log(f"all outputs in {watcher.out_dir}/")
        return 0
    finally:
        watcher.close()


if __name__ == "__main__":
    raise SystemExit(main())
