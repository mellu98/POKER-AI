"""Watcher runtime and state machine."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import numpy as np
from tooling.harness.benchmarks import run_benchmark
from tooling.runtime_data import staged_runtime_data_dir

from .analysis import (
    analyze_strategies,
    bench_summary,
    build_ensemble_candidates,
    load_existing_ensemble_result,
    persist_ensemble_leaderboard,
    plot_results,
)
from .state import (
    benchmarked_checkpoints,
    load_jobs,
    load_results,
    load_state,
    mark_job_done,
    mark_job_failed,
    mark_job_running,
    persist,
    sync_jobs,
)
from .sync import pull_checkpoints
from .utils import (
    WatcherLogger,
    acquire_lock,
    checkpoint_iter,
    load_json,
    now_ts,
    write_json_atomic,
)

REPO = Path(__file__).resolve().parents[2]
GPU_HOST = "gpu-exec"
GPU_CKPT_BASE = "~/advit/fullhouse-bot/checkpoints"
LOCAL_CKPT_BASE = REPO / "checkpoints"
SIMS_DIR = REPO / "sims"


class TrainingWatcher:
    def __init__(self, args):
        self.args = args
        self.out_dir = SIMS_DIR / args.version
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.out_dir / "results.json"
        self.state_file = self.out_dir / "watcher_state.json"
        self.jobs_file = self.out_dir / "watcher_jobs.json"
        self.summary_file = self.out_dir / "watcher_summary.json"
        self.log_file = self.out_dir / "watcher.log"
        self.lock_file = self.out_dir / "watcher.lock"
        self.logger = WatcherLogger(self.log_file)
        self.existing_results = load_results(self.results_file)
        self.state = load_state(
            self.state_file,
            version=args.version,
            existing_results=self.existing_results,
        )
        self.jobs = load_jobs(self.jobs_file, existing_results=self.existing_results)
        self.lock_fp = None

    def acquire(self) -> None:
        meta = {
            "pid": os.getpid(),
            "version": self.args.version,
            "started_at": now_ts(),
            "argv": sys.argv[1:],
        }
        self.lock_fp = acquire_lock(self.lock_file, meta)

    def close(self) -> None:
        if self.lock_fp is not None:
            self.lock_fp.close()
            self.lock_fp = None

    def persist(self, ensemble_result: dict | None = None) -> None:
        queued = [row for row in self.jobs.values() if row.get("status") == "queued"]
        self.state["queue_depth"] = len(queued)
        persist(
            results_file=self.results_file,
            state_file=self.state_file,
            jobs_file=self.jobs_file,
            summary_file=self.summary_file,
            state=self.state,
            jobs=self.jobs,
            existing_results=self.existing_results,
            ensemble_result=ensemble_result,
        )

    def refresh_checkpoints(self) -> tuple[list[Path], int]:
        checkpoints, new, pull_err = pull_checkpoints(
            version=self.args.version,
            gpu_host=self.args.gpu_host,
            gpu_ckpt_base=self.args.gpu_ckpt_base,
            local_ckpt_base=LOCAL_CKPT_BASE,
        )
        self.state["last_poll_at"] = now_ts()
        self.state["last_pull_error"] = pull_err
        if checkpoints:
            self.state["last_seen_checkpoint_iter"] = max(
                checkpoint_iter(ckpt) for ckpt in checkpoints
            )
        self.jobs = sync_jobs(self.jobs, checkpoints, self.existing_results)
        self.persist()
        if pull_err:
            self.logger.log(f"pull warning: {pull_err}", level="WARN")
        elif new > 0:
            self.logger.log(
                f"pulled {new} new checkpoints ({len(checkpoints)} local total)"
            )
        return checkpoints, new

    @contextlib.contextmanager
    def bench_slot(self, label: str, out_path: Path):
        bench_lock = SIMS_DIR / "bench.lock"
        meta = {
            "pid": os.getpid(),
            "label": label,
            "out": str(out_path),
            "started_at": now_ts(),
        }
        self.logger.log(f"{label}: waiting for global bench lock")
        bench_lock_fp = acquire_lock(bench_lock, meta, blocking=True)
        self.logger.log(f"{label}: acquired global bench lock")
        try:
            yield
        finally:
            bench_lock_fp.close()
            self.logger.log(f"{label}: released global bench lock")

    def bench_checkpoint(self, ckpt: Path) -> tuple[dict, str | None]:
        iteration = checkpoint_iter(ckpt)
        out_path = self.out_dir / f"bench_iter_{iteration:05d}.json"
        label = f"iter {iteration}"
        mark_job_running(self.jobs, iteration=iteration, out_path=out_path)
        self.state["running_iter"] = iteration
        self.persist()

        with self.bench_slot(label, out_path):
            try:
                with staged_runtime_data_dir([ckpt]) as data_dir:
                    payload = run_benchmark(
                        bot_path=self.args.bot,
                        matches=self.args.matches,
                        hands=None,
                        procs=None,
                        seed=42,
                        fast=False,
                        out_path=out_path,
                        latest_path=None,
                        data_dir=data_dir,
                        include_match_details=True,
                        progress=False,
                    )
            except Exception as exc:
                error = str(exc)
                mark_job_failed(
                    self.jobs, iteration=iteration, error=error, out_path=out_path
                )
                self.state["running_iter"] = None
                self.persist()
                return {"bb100": None}, error

        summary = bench_summary(payload)
        row = {**summary, "iter": iteration}
        self.existing_results[iteration] = row
        mark_job_done(
            self.jobs, iteration=iteration, summary=summary, out_path=out_path
        )
        self.state["last_benched_iter"] = iteration
        self.state["last_bench_ok_at"] = now_ts()
        self.state["running_iter"] = None
        self.persist()
        self.logger.log(
            f"{label}: bb/100={summary['bb100']:+.2f} ci=({summary['ci_lo']:+.1f}, {summary['ci_hi']:+.1f}) "
            f"win={summary['win'] * 100:.0f}% bust={summary['bust'] * 100:.0f}%"
        )
        return summary, None

    def bench_ensemble(
        self, checkpoints: list[Path], *, out_name: str = "bench_ensemble.json"
    ) -> tuple[dict, str | None]:
        out_path = (
            self.out_dir / out_name
            if out_name == "bench_ensemble.json"
            else self.out_dir / "ensemble_candidates" / out_name
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        label = f"ensemble {out_name}"
        with self.bench_slot(label, out_path):
            try:
                with staged_runtime_data_dir(checkpoints) as data_dir:
                    payload = run_benchmark(
                        bot_path=self.args.bot,
                        matches=self.args.ensemble_matches,
                        hands=None,
                        procs=None,
                        seed=42,
                        fast=False,
                        out_path=out_path,
                        latest_path=None,
                        data_dir=data_dir,
                        include_match_details=True,
                        progress=False,
                    )
            except Exception as exc:
                return {"bb100": None}, str(exc)
        summary = bench_summary(payload)
        self.logger.log(
            f"{label}: bb/100={summary['bb100']:+.2f} ci=({summary['ci_lo']:+.1f}, {summary['ci_hi']:+.1f}) "
            f"models={len(checkpoints)}"
        )
        return summary, None

    def bench_ensemble_candidates(
        self, checkpoints: list[Path], results: list[dict]
    ) -> list[dict]:
        candidates = build_ensemble_candidates(
            checkpoints,
            results,
            top_k=self.args.ensemble_top_k,
            max_models=self.args.ensemble_max_models,
        )
        if not candidates:
            return []

        ckpt_by_iter = {checkpoint_iter(ckpt): ckpt for ckpt in checkpoints}
        leaderboard: list[dict] = []
        for candidate in candidates:
            iters = candidate["iters"]
            out_name = "ens_" + "_".join(f"{it:05d}" for it in iters) + ".json"
            out_path = self.out_dir / "ensemble_candidates" / out_name
            if out_path.exists():
                data = load_json(out_path)
                if not isinstance(data, dict):
                    summary, err = (
                        {"bb100": None},
                        f"stored ensemble JSON at {out_path} is not an object",
                    )
                else:
                    summary, err = bench_summary(data), None
            else:
                summary, err = self.bench_ensemble(
                    [ckpt_by_iter[it] for it in iters], out_name=out_name
                )
            if err:
                self.logger.log(f"ensemble {iters}: {err}", level="WARN")
                continue
            summary = {
                **summary,
                "iters": iters,
                "n_models": len(iters),
                "reason": candidate["reason"],
            }
            leaderboard.append(summary)
        leaderboard.sort(key=lambda row: row.get("bb100", float("-inf")), reverse=True)
        persist_ensemble_leaderboard(self.out_dir, leaderboard)
        return leaderboard

    def run_cycle(self) -> int:
        checkpoints, _ = self.refresh_checkpoints()
        if not checkpoints:
            return 0

        pending = [
            ckpt
            for ckpt in checkpoints
            if checkpoint_iter(ckpt) not in self.existing_results
        ]
        if self.args.max_benches_per_cycle > 0:
            pending = pending[: self.args.max_benches_per_cycle]

        n_new = 0
        for ckpt in pending:
            summary, err = self.bench_checkpoint(ckpt)
            if summary["bb100"] is None:
                self.logger.log(
                    f"iter {checkpoint_iter(ckpt)} failed: {err}", level="WARN"
                )
                continue
            n_new += 1
            checkpoints, _ = self.refresh_checkpoints()

        if not self.existing_results:
            return n_new

        results = sorted(self.existing_results.values(), key=lambda row: row["iter"])
        ensemble_result = None
        if n_new > 0:
            bbs = [row["bb100"] for row in results]
            best = max(results, key=lambda row: row["bb100"])
            self.logger.log(
                f"results={len(results)} mean={np.mean(bbs):+.1f} std={np.std(bbs):.1f} best=iter {best['iter']} ({best['bb100']:+.2f})"
            )

            analysis_checkpoints = benchmarked_checkpoints(
                checkpoints, self.existing_results
            )
            run_ensembles = (
                self.args.ensemble_every > 0
                and len(results) % self.args.ensemble_every == 0
            )
            if run_ensembles and analysis_checkpoints:
                ensemble_result, err = self.bench_ensemble(analysis_checkpoints)
                if ensemble_result["bb100"] is not None:
                    ensemble_result["n_models"] = len(analysis_checkpoints)
                    write_json_atomic(
                        self.out_dir / "ensemble_result.json", ensemble_result
                    )
                elif err:
                    self.logger.log(f"ensemble failed: {err}", level="WARN")
                    ensemble_result = None

                leaderboard = self.bench_ensemble_candidates(
                    analysis_checkpoints, results
                )
                if leaderboard:
                    best_combo = leaderboard[0]
                    self.logger.log(
                        f"best combo {best_combo['iters']} -> {best_combo['bb100']:+.2f} bb/100"
                    )
            else:
                ensemble_result = load_existing_ensemble_result(self.out_dir)

            plot_results(results, self.args.version, self.out_dir, ensemble_result)
            if (
                self.args.strategy_every > 0
                and len(results) % self.args.strategy_every == 0
            ):
                try:
                    for line in analyze_strategies(analysis_checkpoints):
                        self.logger.log(line)
                except Exception as exc:
                    self.logger.log(f"strategy analysis failed: {exc}", level="WARN")

        self.persist(ensemble_result=ensemble_result)
        return n_new
