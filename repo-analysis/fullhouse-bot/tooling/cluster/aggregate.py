"""Rebuild sims/<config>/results.json curves from per-iter bench files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tooling.cluster.common import REPO_ROOT, bench_iter_of, checkpoints_dir
from tooling.watcher.analysis import bench_summary
from tooling.watcher.utils import write_json_atomic


def benched_configs(sims_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in sims_root.iterdir()
        if path.is_dir() and next(path.glob("bench_iter_*.json"), None) is not None
    )


def expected_count(repo: Path, config: str) -> int:
    ckpt_dir = checkpoints_dir(repo, config)
    return len(list(ckpt_dir.glob("iter_*.npz"))) if ckpt_dir.is_dir() else 0


def rows_for(config_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(config_dir.glob("bench_iter_*.json")):
        iteration = bench_iter_of(path)
        if iteration is None:
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            rows.append({**bench_summary(payload), "iter": iteration})
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            print(f"  skip {path.name}: {exc}", file=sys.stderr)
    rows.sort(key=lambda row: row["iter"])
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild results.json curves from bench files"
    )
    parser.add_argument("--sims-root", default=str(REPO_ROOT / "sims"))
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)

    sims_root = Path(args.sims_root).resolve()
    repo = Path(args.repo).resolve()
    configs = args.configs or benched_configs(sims_root)
    total_done = total_expected = 0

    for config in configs:
        config_dir = sims_root / config
        rows = rows_for(config_dir) if config_dir.is_dir() else []
        if rows and not args.status:
            write_json_atomic(config_dir / "results.json", rows)
        expected = expected_count(repo, config)
        total_done += len(rows)
        total_expected += expected
        best = max(rows, key=lambda row: row["bb100"], default=None)
        best_str = f"best iter {best['iter']} {best['bb100']:+.2f}" if best else "-"
        print(f"  {config:<12} {len(rows):>2}/{expected or '?':<3} {best_str}")

    pct = f"{100 * total_done / total_expected:.0f}%" if total_expected else "?"
    print(f"total: {total_done}/{total_expected or '?'} ({pct})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
