# Codex instructions for RunPod deployment

Use this file as the deployment contract when Codex pulls this repository onto RunPod.

## Goal

Deploy the Autonomy Learning Dashboard reproducibly on RunPod and keep every environment/setup fix in Git so the next fresh pod needs only the documented setup command.

## Procedure

1. Clone/pull the repo under `/workspace`.
2. Read `DEPLOY_RUNPOD.md`, `.env.example`, and `scripts/setup_runpod.sh` before making environment changes.
3. Run:
   ```bash
   ./scripts/setup_runpod.sh
   ```
4. If nuScenes Mini is absent, download it to `/workspace/data/nuscenes` using the repository script. **Never commit dataset content.**
5. Create `.env` from `.env.example` for local deployment values. **Never commit `.env` or secrets.**
6. Validate:
   ```bash
   ./scripts/check_dashboard.sh
   python smoke_test.py
   ```
7. Launch:
   ```bash
   ./scripts/run_dashboard.sh
   ```
8. Verify `/api/health`, the dashboard page, WebSocket status, stage code view, and at least Stage 00. If nuScenes exists, verify Stage 01 as well.

## If deployment needs a fix

Do not apply an undocumented one-off shell workaround and stop there. Instead:

- make the smallest durable change in `scripts/setup_runpod.sh`, `scripts/run_dashboard.sh`, requirements, `.env.example`, or documentation;
- keep CUDA-specific PyTorch installation explicit—do not let generic pip requirements silently replace the RunPod torch build;
- rerun `./scripts/check_dashboard.sh` and the relevant smoke test;
- inspect `git diff` and ensure no dataset, outputs, checkpoints, credentials, endpoint tokens, or `.env` are staged;
- commit the durable deployment fix with a clear commit message, for example:
  `fix(runpod): preserve CUDA torch during dashboard setup`.

If a problem is specific to one temporary RunPod host and cannot be made portable, document it in the deployment notes rather than hard-coding host-specific values.

## Networking rule

The dashboard must remain useful with only RunPod HTTP port 8080 exposed. WebRTC is an enhancement; HTTP/WebSocket fallback is mandatory. If direct WebRTC needs TURN/extra network configuration, keep it optional and configurable through environment variables.
