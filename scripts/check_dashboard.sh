#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ -d .venv ]]; then source .venv/bin/activate; fi
python scripts/validate_repo.py
python - <<'PY'
from fastapi.testclient import TestClient
from dashboard.app import app
with TestClient(app) as c:
    assert c.get('/api/health').status_code == 200
    assert len(c.get('/api/stages').json()) == 21
    assert c.get('/api/stage/9/code').status_code == 200
    assert c.get('/').status_code == 200
print('Dashboard API smoke check passed.')
PY
