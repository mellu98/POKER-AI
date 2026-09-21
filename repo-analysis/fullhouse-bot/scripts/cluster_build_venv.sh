#!/usr/bin/env bash

# Usage: scripts/cluster_build_venv.sh [venv_path]
set -euo pipefail

VENV="${1:-$HOME/fh-bench-venv}"
export PATH="$HOME/.local/bin:$PATH"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

uv python install 3.12
uv venv "$VENV" --python 3.12 --relocatable
uv pip install --python "$VENV" eval7 "numpy==1.26.4" "scipy==1.11.4" "scikit-learn==1.8.0" joblib

"$VENV/bin/python" - <<'PY'
import eval7, numpy, scipy, sklearn, joblib
royal = [eval7.Card(s) for s in ("As", "Ks", "Qs", "Js", "Ts")]
print("ENV OK:", "numpy", numpy.__version__, "| eval7 royal-flush score =", eval7.evaluate(royal))
PY
echo "venv: $VENV"
