#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

HOST="${DASHBOARD_HOST:-0.0.0.0}"
PORT="${DASHBOARD_PORT:-8080}"

echo "Starting Autonomy Learning Dashboard on ${HOST}:${PORT}"
echo "nuScenes dataroot: ${NUSCENES_DATAROOT:-/workspace/data/nuscenes}"
exec python dashboard_server.py --host "$HOST" --port "$PORT"
