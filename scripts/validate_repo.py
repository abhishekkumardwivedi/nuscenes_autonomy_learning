from __future__ import annotations

import py_compile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.stage_metadata import STAGES  # noqa: E402

assert len(STAGES) == 21, f"Expected 21 stages, found {len(STAGES)}"
assert [s.number for s in STAGES] == list(range(21)), "Stage numbering must be 00..20"

for stage in STAGES:
    for rel in stage.code_files:
        path = ROOT / rel
        assert path.exists(), f"Stage {stage.number:02d} references missing code file: {rel}"

for path in ROOT.rglob("*.py"):
    if ".venv" in path.parts:
        continue
    py_compile.compile(str(path), doraise=True)

for rel in [
    "dashboard/static/index.html",
    "dashboard/static/styles.css",
    "dashboard/static/app.js",
    "dashboard_server.py",
    "scripts/setup_runpod.sh",
    "scripts/run_dashboard.sh",
]:
    assert (ROOT / rel).exists(), f"Missing dashboard/deployment file: {rel}"

print("Repository validation passed: stage registry, source files and Python syntax are consistent.")
