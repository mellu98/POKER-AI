"""Derive checkpoint jobs from the filesystem and submit them to HTCondor."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from tooling.cluster.common import (
    DEFAULT_FAMILIES,
    REPO_ROOT,
    bench_result_path,
    checkpoint_iters,
    condor_tool,
    discover_configs,
    result_is_complete,
)


def derive_jobs(
    repo: Path,
    sims_root: Path,
    configs: list[str],
    only_iters: set[int] | None,
) -> tuple[list[tuple[str, int]], int]:
    jobs: list[tuple[str, int]] = []
    done = 0
    for config in configs:
        for iteration in checkpoint_iters(repo, config):
            if only_iters is not None and iteration not in only_iters:
                continue
            if result_is_complete(bench_result_path(sims_root, config, iteration)):
                done += 1
            else:
                jobs.append((config, iteration))
    return jobs, done


def render_sub(
    args, repo: Path, tree_root: Path, sims_root: Path, joblist: Path
) -> str:
    worker = repo / "scripts" / "cluster_worker.py"
    condor_dir = sims_root / "_condor"
    job_args = (
        f"{worker} --config $(config) --iter $(iter) "
        f"--matches {args.matches} --procs {args.procs} --hands {args.hands} "
        f"--seed {args.seed} --repo {repo} --tree-root {tree_root} --sims-root {sims_root} "
        f"--python {args.python}"
        f"{' --include-extra' if args.include_extra else ''}"
    )
    family_clause = (
        f' && regexp("^({args.families})[0-9]+", Machine)' if args.families else ""
    )
    requirements = (
        f'(TARGET.Arch == "X86_64") && (TARGET.OpSys == "LINUX"){family_clause}'
    )
    return f"""\
universe              = vanilla
executable            = {args.python}
transfer_executable   = false
should_transfer_files = NO
getenv                = false
initialdir            = {repo}
arguments             = {job_args}
request_cpus          = {args.procs}
request_memory        = {args.request_memory}
requirements          = {requirements}
on_exit_remove        = (ExitCode == 0)
periodic_hold         = (JobStatus == 2) && (NumJobStarts > {args.max_starts})
notification          = never
log                   = {condor_dir}/jobs.log
output                = {condor_dir}/out/$(config)_$(iter).out
error                 = {condor_dir}/err/$(config)_$(iter).err
queue config,iter from {joblist}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Submit checkpoint benchmarks to HTCondor"
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--smoke", nargs=2, metavar=("CONFIG", "ITER"))
    parser.add_argument("--matches", type=int, default=2000)
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-memory", type=int, default=6000)
    parser.add_argument("--max-starts", type=int, default=5)
    parser.add_argument(
        "--families", default=DEFAULT_FAMILIES, help='machine-name regexp; "" for any'
    )
    parser.add_argument(
        "--python", default="/usr/bin/python3", help="job interpreter (e.g. NFS venv)"
    )
    parser.add_argument(
        "--include-extra",
        action="store_true",
        help="bench against held-out bots/extra/",
    )
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--iters", type=int, nargs="+", default=None)
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--tree-root", default=None)
    parser.add_argument("--sims-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.all and not args.smoke:
        parser.error("pass --all or --smoke CONFIG ITER")

    repo = Path(args.repo).resolve()
    tree_root = Path(args.tree_root).resolve() if args.tree_root else repo.parent
    sims_root = Path(args.sims_root).resolve() if args.sims_root else repo / "sims"
    configs = args.configs or discover_configs(repo)

    if args.smoke:
        config, iteration = args.smoke[0], int(args.smoke[1])
        jobs, done, configs = [(config, iteration)], 0, [config]
    else:
        only_iters = set(args.iters) if args.iters else None
        jobs, done = derive_jobs(repo, sims_root, configs, only_iters)

    condor_dir = sims_root / "_condor"
    (condor_dir / "out").mkdir(parents=True, exist_ok=True)
    (condor_dir / "err").mkdir(parents=True, exist_ok=True)
    for config in configs:
        (sims_root / config).mkdir(parents=True, exist_ok=True)

    if not jobs:
        print(f"nothing to submit ({done} already complete)")
        return 0

    suffix = "smoke" if args.smoke else "all"
    joblist = condor_dir / f"joblist_{suffix}.txt"
    joblist.write_text(
        "".join(f"{config} {iteration}\n" for config, iteration in jobs),
        encoding="utf-8",
    )
    sub_path = condor_dir / f"bench_{suffix}.sub"
    sub_path.write_text(
        render_sub(args, repo, tree_root, sims_root, joblist), encoding="utf-8"
    )

    print(
        f"{len(jobs)} jobs queued, {done} already complete "
        f"(matches={args.matches} procs={args.procs})"
    )
    print(f"  {sub_path}")
    if args.dry_run:
        print("dry-run: not submitting")
        return 0
    return subprocess.run([condor_tool("condor_submit"), str(sub_path)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
