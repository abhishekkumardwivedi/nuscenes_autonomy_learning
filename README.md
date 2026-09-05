# Autonomy Learning Dashboard — nuScenes → Temporal BEV → Planning

A **teaching-first, browser-controlled autonomy stack**. The project exposes each major transformation instead of hiding everything behind one training command.

## One dashboard, 21 stages

```text
00 Foundation                11 Tracking
01 Dataset                   12 Occupancy
02 Sensor preprocessing      13 Agent prediction
03 Calibration / geometry    14 HD-map context
04 Camera encoder            15 World model
05 Camera → BEV              16 Behavior planning
06 Radar → BEV               17 Motion planning
07 Spatial fusion            18 Vehicle control
08 Ego motion                19 Safety supervision
09 Temporal BEV              20 Closed-loop boundary
10 Detection
```

Stages 00–15 are primarily exercised on **nuScenes Mini**. Serious planner/closed-loop work later moves to **nuPlan/CARLA**, while the same dashboard and stage interfaces remain useful.

## Why the dashboard exists

The local browser shows, for the selected stage:

- live/saved visual output,
- conceptual inputs and outputs,
- exact Python source code,
- structured logs,
- tensor shape/dtype/min/max/mean/std/memory summaries,
- saved images/JSON artifacts,
- progress, warnings, failures and connection state.

You can switch to any stage at any time without rerunning it.

## Run on RunPod

```bash
./scripts/setup_runpod.sh
cp .env.example .env
./scripts/run_dashboard.sh
```

Expose HTTP port **8080**. See `DEPLOY_RUNPOD.md`.

## Networking

- **WebSocket**: progress, logs, tensor/runtime events and automatic reconnection.
- **WebRTC**: preferred live visual stream when ICE connectivity is available.
- **HTTP visual fallback**: always available through the same port, so WebRTC is never a deployment blocker.

The WebRTC implementation supports optional STUN/TURN settings from `.env`.

## CLI still works

```bash
python main.py --list-stages
python main.py --dataroot /workspace/data/nuscenes --stop-after temporal --history 4 --verbose 3
```

## Teaching rule

Randomly initialized model outputs are never presented as if they were learned autonomy. Where training does not yet exist, the project deliberately shows real nuScenes ground truth and/or transparent deterministic baselines next to the learnable module.

## Repository checks

```bash
python scripts/validate_repo.py
./scripts/check_dashboard.sh
python smoke_test.py
```

## Deployment changes

`CODEX_RUNPOD.md` instructs Codex to turn any reusable RunPod deployment workaround into a repository script/config/documentation change, validate it, and commit the durable fix—without committing datasets, outputs, checkpoints or secrets.
