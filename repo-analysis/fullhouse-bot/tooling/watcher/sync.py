"""Checkpoint syncing helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def pull_checkpoints(
    *,
    version: str,
    gpu_host: str,
    gpu_ckpt_base: str,
    local_ckpt_base: Path,
) -> tuple[list[Path], int, str | None]:
    local_dir = local_ckpt_base / f"deep_cfr_{version}"
    local_dir.mkdir(parents=True, exist_ok=True)
    remote = f"{gpu_host}:{gpu_ckpt_base}/deep_cfr_{version}/iter_*.npz"

    existing = set(path.name for path in local_dir.glob("iter_*.npz"))
    proc = subprocess.run(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            remote,
            str(local_dir) + "/",
        ],
        capture_output=True,
        text=True,
    )
    all_local = sorted(local_dir.glob("iter_*.npz"))
    new = len(all_local) - len(existing)
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        benign = (
            "No such file or directory",
            "not a regular file",
            "No match",
            "not found",
        )
        if not any(msg in stderr for msg in benign):
            return all_local, new, stderr or f"scp exited {proc.returncode}"
    return all_local, new, None
