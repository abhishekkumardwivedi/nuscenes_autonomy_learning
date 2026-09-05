# Codex RunPod maintenance contract

Preserve Stage 00–20 and educational modules. Infrastructure orchestrates them.
Read DEPLOY_RUNPOD.md and .env.example. Never commit .env, secrets, tokens,
datasets, generated outputs, weights or environments.

Use scripts/runpod_boot.sh. It checks the persistent environment, avoids needless
installs, preserves image torch, reserves 8888 and starts the protected dashboard.
Future Pod/template startup must be configured in the RunPod control plane;
repository file edits cannot do that alone.

Fix recurring setup issues durably in scripts, requirements or documentation.
Never silently reinstall CUDA/torch or delete an incompatible venv: use the original
image or a new VENV_DIR. Keep startup logs, token, settings, caches and data persistent.

Validate compilation, existing smoke tests and tests/test_dashboard.py. On Linux/CUDA
check shared HTTP/WSS auth, PTY prompt/resize/Ctrl+C/reconnect, hardware, persisted
profiles, stage errors/cancellation, static artifacts and scene playback. Use the
single HTTPS/WSS 8888 endpoint; UDP WebRTC is optional. A proxy hostname alone
does not justify a public unauthenticated terminal.

Commit durable fixes after verification. Report actual evidence and limitations:
NVML samples may miss peaks, missing counters are N/A, scene-boundary padding is
illustrative, and replay is compute-paced rather than guaranteed real-time inference.
