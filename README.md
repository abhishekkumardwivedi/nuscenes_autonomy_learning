# Autonomy Learning Dashboard

An educational nuScenes Stage 00–20 pipeline with a browser dashboard. Individual
stage Python files, code viewing, learning logs and artifacts remain central.
Some model heads are untrained teaching examples; this is not a production driving stack.

## Normal workflow

Create CUDA RunPod with persistent storage → bootstrap → authenticated port 8888
dashboard → select scene/stage → Run To → inspect Visual / Logs / Tensor / Code /
Hardware → move between stages → use Console when needed.

- Adjustable panel divider, fullscreen/header double-click, visual zoom/pan/Fit.
- Compact controls with progress, elapsed time, cancellation, tracebacks and retry.
- Left: Visual, Inputs, Outputs, Code, Graphs, Artifacts, Console.
- Right: Overview, Logs, Tensor Info, Parameters, Hardware, Profiler.
- Authenticated Linux PTY over the dashboard WSS port; reconnect, resize, Ctrl+C.
- NVML/psutil telemetry, rolling graphs, per-stage JSON profiles and comparison.
- Basic/Learning/Detailed bounded on-device tensor inspection.
- Scene playback, pause, previous/next, timeline, speeds and frame artifact caching.
- Same-port HTTPS/WSS works without UDP WebRTC.

## RunPod startup

```bash
bash /workspace/autonomy-learning-dashboard/scripts/runpod_boot.sh
```

Configure that as the template startup command and expose HTTP 8888. Read
[DEPLOY_RUNPOD.md](DEPLOY_RUNPOD.md) for first clone, persistent authentication,
paths, compatible images and troubleshooting. A readiness marker avoids repeated
pip installs. Jupyter remains installed but 8888 is reserved for this dashboard.
Persistent storage holds data, environment, model caches, settings, outputs and
logs. Processes and in-memory tensor caches disappear with the disposable Pod.

## Architecture

- `stages/`: Stage 00–20 educational modules; Stage 01 adds explicit replay padding.
- `dashboard/runner.py`: execution, caching, status and cancellation.
- `dashboard/hardware.py`, `profiling.py`: sampling and stage resource measurement.
- `dashboard/playback.py`: scene catalog, sequential frame orchestration/cache.
- `dashboard/terminal.py`, `security.py`: PTY and shared access gate.
- `dashboard/extensions.py`, `static/extensions.js`: infrastructure API/UI.
- `scripts/`: environment validation, bootstrap and launch.

Validate with `python -m unittest discover -s tests -v`, `python smoke_test.py`,
and `./scripts/check_dashboard.sh`. See [START_HERE.md](START_HERE.md) and
[STAGE_MAP.md](STAGE_MAP.md). Metrics are measured, sampled or explicitly N/A;
first-pass playback is compute-paced, not guaranteed real-time inference.
