#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[1/6] Checking Python"
"$PYTHON_BIN" - <<'PY'
import sys
print("Python:", sys.version)
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required by the dashboard; Python 3.12 is preferred for a clean nuScenes environment.")
PY

echo "[2/6] Creating/reusing virtual environment with RunPod system packages"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "[3/6] Verifying CUDA-enabled PyTorch before installing anything else"
if ! python - <<'PY'
try:
    import torch, torchvision
except Exception as exc:
    raise SystemExit(f"PyTorch/torchvision import failed: {exc}")
print("torch      :", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA build  :", torch.version.cuda)
print("CUDA usable :", torch.cuda.is_available())
if torch.version.cuda is None:
    raise SystemExit("This environment has a CPU-only PyTorch build. Use a RunPod PyTorch/CUDA template or install the matching CUDA wheel first.")
PY
then
  cat <<'EOF'

PyTorch check failed.
Do not let pip silently replace RunPod's CUDA build with a generic wheel.
Choose a RunPod PyTorch/CUDA image (recommended), then rerun this script.
EOF
  exit 2
fi

echo "[4/6] Installing nuScenes + dashboard dependencies"
pip install -r requirements.txt

echo "[5/6] Running repository validation"
python scripts/validate_repo.py
python smoke_test.py

echo "[6/6] Setup complete"
cat <<EOF

Activate later with:
  source $VENV_DIR/bin/activate

Start dashboard with:
  ./scripts/run_dashboard.sh

Default URL is served on port 8080. Expose HTTP port 8080 in RunPod.
EOF
