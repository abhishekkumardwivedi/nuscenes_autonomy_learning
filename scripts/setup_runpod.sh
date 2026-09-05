#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${AUTONOMY_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PERSISTENT_ROOT:-/workspace}/.cache/pip}"
# Refuse to reuse a venv across incompatible Python minor versions.
if [[ -f "$VENV_DIR/pyvenv.cfg" ]]; then
  "$PYTHON_BIN" - "$VENV_DIR/pyvenv.cfg" <<'PY'
from pathlib import Path
import re, sys
text=Path(sys.argv[1]).read_text()
match=re.search(r'^version\s*=\s*(\d+)\.(\d+)',text,re.M)
if match and tuple(map(int,match.groups())) != sys.version_info[:2]:
    raise SystemExit('Python minor version changed. Use the original image or select a NEW VENV_DIR; do not mutate the old environment.')
PY
else
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python scripts/environment_check.py fingerprint
set +e
python scripts/environment_check.py ready
ready=$?
set -e
if [[ "$ready" == 0 ]]; then exit 0; fi
if [[ "$ready" != 3 ]]; then
  echo 'Environment validation failed. Repair the image/runtime or select a NEW VENV_DIR.' >&2
  exit "$ready"
fi
constraints=$(mktemp)
trap 'rm -f "$constraints"' EXIT
python - "$constraints" <<'PY'
from pathlib import Path
import sys, torch, torchvision
Path(sys.argv[1]).write_text(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\n')
PY
python -m pip install -c "$constraints" -r requirements.txt
python scripts/validate_repo.py
python smoke_test.py
python scripts/environment_check.py stamp
