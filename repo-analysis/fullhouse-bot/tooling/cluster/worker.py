"""Benchmark one checkpoint into sims/<config>/bench_iter_<NNNNN>.json."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tooling.cluster.common import (
    REPO_ROOT,
    bench_result_path,
    checkpoint_path,
    result_is_complete,
)

STATIC_RUNTIME = "preflop_equity.npz"


def tree_for(config: str, repo: Path, tree_root: Path) -> Path:
    candidates = [tree_root / f"{repo.name}-{config}"]
    match = re.match(r"(v\d+)", config)
    if match:
        candidates.append(tree_root / f"{repo.name}-{match.group(1)}")
    for path in candidates:
        if path.is_dir():
            return path
    return repo


def stage_dir() -> Path:
    scratch = os.environ.get("_CONDOR_SCRATCH_DIR")
    base = scratch if scratch and Path(scratch).is_dir() else None
    return Path(tempfile.mkdtemp(prefix="fh_stage_", dir=base))


def run_one(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    tree_root = Path(args.tree_root).resolve()
    sims_root = Path(args.sims_root).resolve()
    tree = tree_for(args.config, repo, tree_root)

    suffix = "_extra" if args.include_extra else ""
    out_path = bench_result_path(sims_root, args.config, args.iter, suffix=suffix)
    if result_is_complete(out_path):
        print(f"skip {args.config} iter {args.iter}: {out_path} exists")
        return 0

    ckpt = checkpoint_path(repo, args.config, args.iter)
    equity = tree / "data" / STATIC_RUNTIME
    bench = tree / "scripts" / "bench.py"
    for required in (ckpt, equity, bench):
        if not required.is_file():
            print(f"missing {required}", file=sys.stderr)
            return 1

    staged = stage_dir()
    try:
        shutil.copy2(equity, staged / STATIC_RUNTIME)
        shutil.copy2(ckpt, staged / "deep_cfr_model.npz")
        python_exe = args.python or sys.executable or "/usr/bin/python3"
        cmd = [
            python_exe,
            "scripts/bench.py",
            "bench",
            "--bot",
            "bot/bot.py",
            "--seed",
            str(args.seed),
            "--hands",
            str(args.hands),
            "--matches",
            str(args.matches),
            "--procs",
            str(args.procs),
            "--data-dir",
            str(staged),
            "--out",
            str(out_path),
            "--no-update-latest",
        ]
        if args.include_extra:
            cmd.append("--include-extra")
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        returncode = subprocess.run(cmd, cwd=tree, env=env).returncode
        if returncode != 0:
            print(f"bench failed rc={returncode}", file=sys.stderr)
            return returncode or 1
    finally:
        shutil.rmtree(staged, ignore_errors=True)

    if not result_is_complete(out_path):
        print(f"no valid result at {out_path}", file=sys.stderr)
        return 1
    with out_path.open(encoding="utf-8") as handle:
        bb100 = json.load(handle)["overall"]["bb100"]
    print(f"done {args.config} iter {args.iter}: bb/100={bb100:+.2f} -> {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one checkpoint into shared sims/"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--matches", type=int, default=2000)
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--tree-root", default=str(REPO_ROOT.parent))
    parser.add_argument("--sims-root", default=None)
    parser.add_argument(
        "--python", default=None, help="interpreter for bench.py (e.g. the NFS venv)"
    )
    parser.add_argument(
        "--include-extra",
        action="store_true",
        help="add held-out bots/extra/ (champion check)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sims_root is None:
        args.sims_root = str(Path(args.repo).resolve() / "sims")
    return run_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
