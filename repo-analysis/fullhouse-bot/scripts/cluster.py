from __future__ import annotations

import sys
from _bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from tooling.cluster import aggregate, submit, sync

COMMANDS = {
    "aggregate": aggregate.main,
    "submit": submit.main,
    "sync": sync.main,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: cluster.py {submit|aggregate|sync} [...]")
        return 0

    command = argv[0]
    handler = COMMANDS.get(command)
    if handler is None:
        print(f"unknown subcommand: {command}", file=sys.stderr)
        print("usage: cluster.py {submit|aggregate|sync} [...]", file=sys.stderr)
        return 2
    return handler(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
