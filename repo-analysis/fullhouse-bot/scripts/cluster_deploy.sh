#!/usr/bin/env bash

# Usage: scripts/cluster_deploy.sh [ssh_host] [config ...]
set -euo pipefail

HOST="${1:-yew11}"; shift || true
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
SSH="ssh -o BatchMode=yes"

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  mapfile -t CONFIGS < <(ls -d "$LOCAL"/checkpoints/deep_cfr_* | sed 's#.*/deep_cfr_##')
fi

echo "== rsync code -> $HOST:fullhouse-bot =="
rsync -az -e "$SSH" \
  --exclude='.git' --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='*.pt' --exclude='build' --exclude='.DS_Store' --exclude='.pytest_cache' \
  --exclude='.ruff_cache' --exclude='checkpoints' --exclude='sims' \
  --exclude='data/hand_eval_lut.npz' \
  "$LOCAL/" "$HOST:fullhouse-bot/"

echo "== rsync ${#CONFIGS[@]} checkpoint sets =="
$SSH "$HOST" 'mkdir -p fullhouse-bot/checkpoints'
for c in "${CONFIGS[@]}"; do
  rsync -az -e "$SSH" "$LOCAL/checkpoints/deep_cfr_$c/" "$HOST:fullhouse-bot/checkpoints/deep_cfr_$c/"
done

echo "deploy done."
