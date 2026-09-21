from __future__ import annotations

from _bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from tooling.cluster.worker import main


if __name__ == "__main__":
    raise SystemExit(main())
