#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${AUTONOMY_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export PERSISTENT_ROOT="${PERSISTENT_ROOT:-/workspace}"
export VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
export NUSCENES_DATAROOT="${NUSCENES_DATAROOT:-$PERSISTENT_ROOT/data/nuscenes}"
export AUTONOMY_OUTPUT_DIR="${AUTONOMY_OUTPUT_DIR:-$PERSISTENT_ROOT/autonomy-outputs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PERSISTENT_ROOT/.cache}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8888}"
source "$VENV_DIR/bin/activate"
# Default to authenticated access when starting the production dashboard.
if [[ -z "${DASHBOARD_TOKEN:-}" && "${DASHBOARD_TRUST_PROXY_AUTH:-0}" != 1 ]]; then
  export DASHBOARD_TOKEN_FILE="${DASHBOARD_TOKEN_FILE:-$PERSISTENT_ROOT/.autonomy/dashboard.token}"
  python - <<'PY'
import os, secrets
from pathlib import Path
p=Path(os.environ['DASHBOARD_TOKEN_FILE'])
p.parent.mkdir(parents=True,exist_ok=True)
if not p.exists():
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f: f.write(secrets.token_urlsafe(32))
os.chmod(p,0o600)
print('Dashboard authentication enabled; token file:',p)
PY
fi
python scripts/prepare_port.py
exec python dashboard_server.py --host "$DASHBOARD_HOST" --port "$DASHBOARD_PORT"
