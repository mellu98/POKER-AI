"""Shared runtime data contract for packaging and local benchmark staging."""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
STATIC_RUNTIME_FILES = ("preflop_equity.npz",)
MODEL_GLOB = "deep_cfr_model*.npz"


def iter_static_runtime_files(data_dir: Path = DATA_DIR) -> Iterator[Path]:
    for name in STATIC_RUNTIME_FILES:
        path = data_dir / name
        if path.exists():
            yield path


def iter_model_runtime_files(data_dir: Path = DATA_DIR) -> Iterator[Path]:
    yield from sorted(data_dir.glob(MODEL_GLOB))


def iter_packaged_runtime_files(data_dir: Path = DATA_DIR) -> Iterator[Path]:
    yield from iter_static_runtime_files(data_dir)
    yield from iter_model_runtime_files(data_dir)


def staged_model_name(index: int, total: int) -> str:
    return "deep_cfr_model.npz" if total == 1 else f"deep_cfr_model_{index:02d}.npz"


@contextlib.contextmanager
def staged_runtime_data_dir(models: Sequence[Path], *, base_data_dir: Path = DATA_DIR):
    with tempfile.TemporaryDirectory(prefix="fh_runtime_data_") as tmp:
        tmp_dir = Path(tmp)
        for path in iter_static_runtime_files(base_data_dir):
            shutil.copy2(path, tmp_dir / path.name)
        for idx, src in enumerate(models):
            shutil.copy2(src, tmp_dir / staged_model_name(idx, len(models)))
        yield tmp_dir
