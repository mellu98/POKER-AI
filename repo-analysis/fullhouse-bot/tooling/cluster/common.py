from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONDOR_BIN = Path("/usr/local/condor/release/bin")
DEFAULT_FAMILIES = "yew|oak|ash|poplar|rowan|vertex|willow|maple|beech|cedar|curve|gpu"


def condor_tool(name: str) -> str:
    candidate = CONDOR_BIN / name
    return str(candidate) if candidate.is_file() else name


def checkpoints_dir(repo: Path, config: str) -> Path:
    return repo / "checkpoints" / f"deep_cfr_{config}"


def checkpoint_path(repo: Path, config: str, iteration: int) -> Path:
    return checkpoints_dir(repo, config) / f"iter_{iteration:05d}.npz"


def discover_configs(repo: Path) -> list[str]:
    base = repo / "checkpoints"
    return sorted(
        path.name[len("deep_cfr_") :]
        for path in base.glob("deep_cfr_*")
        if path.is_dir()
    )


def checkpoint_iters(repo: Path, config: str) -> list[int]:
    iterations: list[int] = []
    for path in checkpoints_dir(repo, config).glob("iter_*.npz"):
        try:
            iterations.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(iterations)


def bench_result_path(
    sims_root: Path,
    config: str,
    iteration: int,
    *,
    suffix: str = "",
) -> Path:
    return sims_root / config / f"bench_iter_{iteration:05d}{suffix}.json"


def bench_iter_of(path: Path) -> int | None:
    try:
        return int(path.stem.split("_")[2])
    except (IndexError, ValueError):
        return None


def result_is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(data, dict) and data.get("overall", {}).get("bb100") is not None
