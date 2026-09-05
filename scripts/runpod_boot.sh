#!/usr/bin/env bash
# Set this as the RunPod container startup command; keep it in the foreground.
set -euo pipefail
export PERSISTENT_ROOT="${PERSISTENT_ROOT:-/workspace}"
export AUTONOMY_REPO_DIR="${AUTONOMY_REPO_DIR:-$PERSISTENT_ROOT/autonomy-learning-dashboard}"
export AUTONOMY_LOG_DIR="${AUTONOMY_LOG_DIR:-$PERSISTENT_ROOT/autonomy-logs}"
mkdir -p "$AUTONOMY_LOG_DIR"
exec > >(tee -a "$AUTONOMY_LOG_DIR/startup.log") 2>&1
echo "[$(date -Is)] Autonomy boot"
exec 9>"$AUTONOMY_LOG_DIR/boot.lock"
flock -n 9 || { echo 'Another boot/dashboard process owns the startup lock'; exit 1; }
if [[ ! -d "$AUTONOMY_REPO_DIR/.git" ]]; then
  : "${AUTONOMY_REPO_URL:?Set AUTONOMY_REPO_URL for first clone}"
  git clone "$AUTONOMY_REPO_URL" "$AUTONOMY_REPO_DIR"
fi
cd "$AUTONOMY_REPO_DIR"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
# Automatic pulls are opt-in; avoid surprise code changes on each disposable pod.
if [[ "${AUTONOMY_GIT_PULL:-0}" == 1 ]]; then git pull --ff-only; fi
./scripts/setup_runpod.sh
exec ./scripts/start_dashboard.sh
