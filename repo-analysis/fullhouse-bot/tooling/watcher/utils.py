"""Shared utilities for the training watcher."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path


def now_ts() -> int:
    return int(time.time())


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_json(path: Path) -> dict | list:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"failed to read JSON from {path}") from exc


def checkpoint_iter(path: Path) -> int:
    return int(path.stem.split("_")[1])


class WatcherLogger:
    def __init__(self, path: Path):
        self.path = path

    def log(self, message: str, *, level: str = "INFO", echo: bool = True) -> None:
        line = f"{iso_now()} [{level}] {message}"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if echo and not os.environ.get("FULLHOUSE_WATCHER_DETACHED"):
            print(line, flush=True)

    def exception(self, message: str) -> None:
        self.log(message, level="ERROR")


def acquire_lock(lock_path: Path, meta: dict, *, blocking: bool = False) -> object:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fp = lock_path.open("a+", encoding="utf-8")
    try:
        mode = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock_fp.fileno(), mode)
    except BlockingIOError:
        lock_fp.seek(0)
        holder = lock_fp.read().strip() or "unknown"
        raise RuntimeError(f"watcher already running: {holder}")
    lock_fp.seek(0)
    lock_fp.truncate()
    json.dump(meta, lock_fp)
    lock_fp.write("\n")
    lock_fp.flush()
    os.fsync(lock_fp.fileno())
    return lock_fp
