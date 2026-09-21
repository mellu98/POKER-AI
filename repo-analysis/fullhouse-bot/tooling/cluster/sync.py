"""Pull cluster bench results into the local sims/ tree and rebuild curves."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tooling.cluster.aggregate import main as aggregate_main
from tooling.cluster.common import REPO_ROOT, discover_configs


def pull_config(
    host: str, remote_sims: str, config: str, local_sims: Path, dry_run: bool
) -> int:
    dest = local_sims / config
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-az", "-c", "-i", "-e", "ssh -o BatchMode=yes"]
    if dry_run:
        cmd.append("--dry-run")
    cmd += [f"{host}:{remote_sims}/{config}/bench_iter_*.json", f"{dest}/"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.returncode == 23 or "No such file" in proc.stderr:
            print(f"  {config:<12} no remote files yet")
            return 0
        print(
            f"  {config:<12} rsync rc={proc.returncode}: {proc.stderr.strip()}",
            file=sys.stderr,
        )
        return proc.returncode
    changed = [
        line
        for line in proc.stdout.splitlines()
        if line and not line.startswith(("sent", "total"))
    ]
    print(
        f"  {config:<12} {len(changed)} files {'(would change)' if dry_run else 'updated'}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync cluster sims/ back to local and rebuild curves"
    )
    parser.add_argument("--host", default="yew11")
    parser.add_argument("--remote-sims", default="fullhouse-bot/sims")
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--local-sims", default=str(REPO_ROOT / "sims"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    local_sims = Path(args.local_sims).resolve()
    configs = args.configs or discover_configs(REPO_ROOT)
    print(f"pulling {len(configs)} configs from {args.host}:{args.remote_sims}")
    returncode = 0
    for config in configs:
        returncode |= pull_config(
            args.host, args.remote_sims, config, local_sims, args.dry_run
        )

    if args.dry_run:
        return returncode
    print("rebuilding local results.json:")
    return returncode | aggregate_main(
        ["--sims-root", str(local_sims), "--configs", *configs]
    )


if __name__ == "__main__":
    raise SystemExit(main())
