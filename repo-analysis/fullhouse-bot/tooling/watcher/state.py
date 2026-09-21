"""Persistence for watcher state, results, and job ledger."""

from __future__ import annotations

from pathlib import Path

from .utils import checkpoint_iter, load_json, now_ts, write_json_atomic


def load_results(results_file: Path) -> dict[int, dict]:
    if not results_file.exists():
        return {}
    loaded = load_json(results_file)
    if not isinstance(loaded, list):
        raise RuntimeError(
            f"expected list in {results_file}, got {type(loaded).__name__}"
        )
    return {row["iter"]: row for row in loaded}


def load_state(
    state_file: Path, *, version: str, existing_results: dict[int, dict]
) -> dict:
    state = {
        "version": version,
        "last_poll_at": None,
        "last_pull_error": None,
        "last_seen_checkpoint_iter": (
            max(existing_results) if existing_results else None
        ),
        "last_benched_iter": max(existing_results) if existing_results else None,
        "last_bench_ok_at": None,
        "n_results": len(existing_results),
        "queue_depth": 0,
        "running_iter": None,
    }
    if state_file.exists():
        loaded = load_json(state_file)
        if not isinstance(loaded, dict):
            raise RuntimeError(
                f"expected dict in {state_file}, got {type(loaded).__name__}"
            )
        state.update(loaded)
    return state


def load_jobs(jobs_file: Path, *, existing_results: dict[int, dict]) -> dict[int, dict]:
    jobs: dict[int, dict] = {}
    if jobs_file.exists():
        loaded = load_json(jobs_file)
        if isinstance(loaded, list):
            for row in loaded:
                jobs[row["iter"]] = row
        elif isinstance(loaded, dict):
            for key, row in loaded.items():
                jobs[int(key)] = row
        else:
            raise RuntimeError(
                f"expected list or dict in {jobs_file}, got {type(loaded).__name__}"
            )

    for it, result in existing_results.items():
        jobs[it] = {
            **jobs.get(it, {}),
            "iter": it,
            "status": "done",
            "attempts": jobs.get(it, {}).get("attempts", 1),
            "bb100": result.get("bb100"),
            "error": None,
            "updated_at": now_ts(),
        }

    for job in jobs.values():
        if job.get("status") == "running":
            job["status"] = "queued"
            job["error"] = "previous process exited while benchmark was running"
            job["updated_at"] = now_ts()
    return jobs


def sync_jobs(
    jobs: dict[int, dict], checkpoints: list[Path], existing_results: dict[int, dict]
) -> dict[int, dict]:
    now = now_ts()
    known_iters = {checkpoint_iter(path) for path in checkpoints}
    for ckpt in checkpoints:
        it = checkpoint_iter(ckpt)
        if it in existing_results:
            jobs[it] = {
                **jobs.get(it, {}),
                "iter": it,
                "checkpoint": str(ckpt),
                "status": "done",
                "bb100": existing_results[it].get("bb100"),
                "error": None,
                "updated_at": now,
            }
            continue
        row = jobs.get(it)
        if row is None:
            jobs[it] = {
                "iter": it,
                "checkpoint": str(ckpt),
                "status": "queued",
                "attempts": 0,
                "bb100": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
        else:
            row["checkpoint"] = str(ckpt)
            if row.get("status") == "done":
                row["bb100"] = existing_results.get(it, {}).get("bb100")
            row["updated_at"] = now

    for it, row in list(jobs.items()):
        if row.get("status") != "done" and it not in known_iters:
            row["status"] = "missing"
            row["updated_at"] = now
    return jobs


def benchmarked_checkpoints(
    checkpoints: list[Path], existing_results: dict[int, dict]
) -> list[Path]:
    return [ckpt for ckpt in checkpoints if checkpoint_iter(ckpt) in existing_results]


def mark_job_running(jobs: dict[int, dict], *, iteration: int, out_path: Path) -> None:
    row = jobs.setdefault(iteration, {"iter": iteration})
    row["status"] = "running"
    row["attempts"] = row.get("attempts", 0) + 1
    row["bench_output"] = str(out_path)
    row["error"] = None
    row["updated_at"] = now_ts()


def mark_job_done(
    jobs: dict[int, dict], *, iteration: int, summary: dict, out_path: Path
) -> None:
    row = jobs.setdefault(iteration, {"iter": iteration})
    row["status"] = "done"
    row["bench_output"] = str(out_path)
    row["bb100"] = summary.get("bb100")
    row["ci_lo"] = summary.get("ci_lo")
    row["ci_hi"] = summary.get("ci_hi")
    row["error"] = None
    row["updated_at"] = now_ts()


def mark_job_failed(
    jobs: dict[int, dict], *, iteration: int, error: str, out_path: Path
) -> None:
    row = jobs.setdefault(iteration, {"iter": iteration})
    row["status"] = "failed"
    row["bench_output"] = str(out_path)
    row["error"] = error
    row["updated_at"] = now_ts()


def persist(
    *,
    results_file: Path,
    state_file: Path,
    jobs_file: Path,
    summary_file: Path,
    state: dict,
    jobs: dict[int, dict],
    existing_results: dict[int, dict],
    ensemble_result: dict | None = None,
) -> None:
    if existing_results:
        write_json_atomic(
            results_file, sorted(existing_results.values(), key=lambda row: row["iter"])
        )
        state["n_results"] = len(existing_results)
    else:
        state["n_results"] = 0

    write_json_atomic(state_file, state)
    write_json_atomic(jobs_file, sorted(jobs.values(), key=lambda row: row["iter"]))

    queued = [row for row in jobs.values() if row.get("status") == "queued"]
    failed = [row for row in jobs.values() if row.get("status") == "failed"]
    done = [row for row in jobs.values() if row.get("status") == "done"]
    summary = {
        "version": state["version"],
        "updated_at": now_ts(),
        "state": state,
        "queue_depth": len(queued),
        "failed_iters": [row["iter"] for row in failed],
        "done_iters": [row["iter"] for row in done],
        "best_result": (
            max(existing_results.values(), key=lambda row: row["bb100"])
            if existing_results
            else None
        ),
        "ensemble_result": ensemble_result,
    }
    write_json_atomic(summary_file, summary)
