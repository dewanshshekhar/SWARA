#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/root/data/repo/gongjunmin/miniconda3/bin/activate}"
CONDA_ENV_NAME="${EMPATH_CONDA_ENV:-empath_v15_train}"

HOST="${EMPATH_API_HOST:-0.0.0.0}"
PORT="${EMPATH_API_PORT:-8001}"
LOG_LEVEL="${EMPATH_API_LOG_LEVEL:-debug}"

cd "$ROOT_DIR"

# Temporarily turn off nounset to avoid unbound variable errors in the conda activate.d script
set +u
# shellcheck disable=SC1090
source "$CONDA_ACTIVATE" "$CONDA_ENV_NAME"
set -u

# NOTE: api_server uses in-memory queue/task storage and requires workers=1.
nohup python -m uvicorn empath.api_server:app \
	--host "0.0.0.0" \
	--port "8001" \
	--workers 1 \
	--log-level "$LOG_LEVEL" > server.log 2>&1 &
echo "Server started in background with PID $!. Logs in server.log"