"""Build the submission zip from bot runtime files and data assets."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from tooling.runtime_data import DATA_DIR, iter_packaged_runtime_files

ROOT = Path(__file__).resolve().parents[2]
BOT_DIR = ROOT / "bot"
BUILD_DIR = ROOT / "build"
DEFAULT_OUT = ROOT / "submission" / "bot.zip"

MODULE_ORDER = ["features.py", "deep_cfr_lookup.py", "bot.py"]

MAX_BOT_PY = 5 * 1024 * 1024
MAX_DATA = 200 * 1024 * 1024
MAX_TOTAL = 250 * 1024 * 1024

_INTERNAL_IMPORT = re.compile(
    r"^\s*from\s+\.\w+\s+import\s+.*$|"
    r"^\s*from\s+bot[\.\w]*\s+import\s+.*$|"
    r"^\s*import\s+bot[\.\w]+.*$"
)
_FUTURE_IMPORT = re.compile(r"^\s*from\s+__future__\s+import\s+.*$")
_IF_MAIN = re.compile(r'^if\s+__name__\s*==\s*["\']__main__["\']\s*:')


def _concat_modules() -> str:
    seen_imports: set[str] = set()
    header_imports: list[str] = ["from __future__ import annotations\n"]
    module_blocks: list[str] = []

    for filename in MODULE_ORDER:
        path = BOT_DIR / filename
        if not path.exists():
            print(f"  WARN: {path} not found, skipping", file=sys.stderr)
            continue

        lines = path.read_text().splitlines(keepends=True)
        filtered: list[str] = []
        skip_main_block = False
        main_indent = 0
        skip_paren_import = False

        for line in lines:
            stripped = line.rstrip()
            if skip_paren_import:
                if ")" in stripped:
                    skip_paren_import = False
                continue
            if skip_main_block:
                indent = len(line) - len(line.lstrip())
                if indent > main_indent or stripped == "":
                    continue
                skip_main_block = False
            if _IF_MAIN.match(stripped):
                skip_main_block = True
                main_indent = len(line) - len(line.lstrip())
                continue
            if _FUTURE_IMPORT.match(stripped):
                continue
            if _INTERNAL_IMPORT.match(stripped):
                if "(" in stripped and ")" not in stripped:
                    skip_paren_import = True
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                canon = stripped.strip()
                if canon in seen_imports:
                    continue
                seen_imports.add(canon)
                header_imports.append(line)
                continue
            filtered.append(line)

        block = "".join(filtered).strip()
        if block:
            module_blocks.append(f"\n# {filename}\n\n{block}\n")

    return "".join(header_imports) + "\n" + "".join(module_blocks)


def _gather_data() -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    seen_arcs: set[str] = set()
    for path in iter_packaged_runtime_files(DATA_DIR):
        arc = f"data/{path.name}"
        if arc not in seen_arcs:
            out.append((path, arc))
            seen_arcs.add(arc)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args(argv)

    print("concatenating bot/ modules...")
    bot_py_text = _concat_modules()

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    bot_py_path = BUILD_DIR / "bot.py"
    bot_py_path.write_text(bot_py_text)
    bot_size = bot_py_path.stat().st_size
    print(f"  bot.py: {bot_size / 1024:.1f} KB ({bot_py_text.count(chr(10))} lines)")

    if bot_size > MAX_BOT_PY:
        print(
            f"FAIL: bot.py is {bot_size / 1e6:.1f} MB (limit {MAX_BOT_PY / 1e6:.0f} MB)",
            file=sys.stderr,
        )
        return 2

    data_files = _gather_data()
    data_size = sum(path.stat().st_size for path, _ in data_files)
    if data_size > MAX_DATA:
        print(
            f"FAIL: data/ is {data_size / 1e6:.1f} MB (limit {MAX_DATA / 1e6:.0f} MB)",
            file=sys.stderr,
        )
        return 2

    total_size = bot_size + data_size
    if total_size > MAX_TOTAL:
        print(
            f"FAIL: total {total_size / 1e6:.1f} MB (limit {MAX_TOTAL / 1e6:.0f} MB)",
            file=sys.stderr,
        )
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bot.py", bot_py_text)
        for host, arc in data_files:
            archive.write(host, arc)

    zip_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out} ({zip_mb:.1f} MB)")
    print(f"  {'bot.py':40s} {bot_size / 1e6:7.3f} MB")
    for host, arc in data_files:
        print(f"  {arc:40s} {host.stat().st_size / 1e6:7.3f} MB")

    if args.skip_validate:
        return 0

    validator = ROOT / "engine_vendored" / "sandbox" / "validator.py"
    if not validator.exists():
        print("WARN: validator not found, skipping validation")
        return 0

    print("\nrunning upstream validator...")
    result = subprocess.run(
        [sys.executable, str(validator), str(args.out)], capture_output=True, text=True
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
