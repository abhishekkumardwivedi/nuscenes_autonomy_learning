# RunPod deployment

This repo is designed so the **browser is local** and all compute/data stay on RunPod.

## 1. RunPod prerequisites

Use a RunPod image that already contains a CUDA-enabled PyTorch + torchvision build. Keep the repo and dataset under `/workspace` if you want them to survive pod restarts.

Expose **HTTP port 8080** in the RunPod pod settings. The dashboard, REST API and WebSocket all use this one port.

WebRTC is optional and automatically falls back to HTTP snapshots if ICE cannot traverse the RunPod network. For reliable WebRTC across restrictive NAT/proxies, configure a TURN server in `.env` using `TURN_URL`, `TURN_USERNAME`, and `TURN_PASSWORD`.

## 2. Clone and set up

```bash
cd /workspace
git clone <YOUR_REPO_URL> autonomy-learning
cd autonomy-learning
./scripts/setup_runpod.sh
```

The setup script deliberately **does not reinstall torch/torchvision**. It first verifies that the pod has a CUDA build so pip cannot accidentally replace it with a mismatched wheel.

## 3. Configure

```bash
cp .env.example .env
```

At minimum check:

```bash
NUSCENES_DATAROOT=/workspace/data/nuscenes
DASHBOARD_PORT=8080
AUTONOMY_DEVICE=auto
```

## 4. Download nuScenes Mini (if needed)

```bash
./download_nuscenes_mini.sh /workspace/data/nuscenes
```

Expected metadata directory:

```text
/workspace/data/nuscenes/v1.0-mini
```

## 5. Start the dashboard

```bash
./scripts/run_dashboard.sh
```

Open the RunPod HTTP endpoint for port 8080 in your local browser.

To keep the dashboard running after disconnecting SSH:

```bash
mkdir -p outputs
nohup ./scripts/run_dashboard.sh > outputs/dashboard.log 2>&1 < /dev/null &
echo $! > outputs/dashboard.pid
```

This survives SSH disconnects, but rerun it after a pod restart. Read
`outputs/dashboard.log` for startup errors. If you choose another exposed HTTP
port, set `DASHBOARD_PORT` in `.env`. Port 8888 is commonly occupied by Jupyter;
stop Jupyter only if you intend to replace it with this dashboard.

## 6. What connection indicators mean

- **Backend connected**: REST + WebSocket are alive; stage control/logging works.
- **WebRTC connected**: live visual track is available.
- **WebRTC fallback**: model execution is still fine; the visual panel refreshes through HTTP on port 8080.
- If the browser loses the WebSocket, it automatically reconnects and reloads server state.

## 7. Validate after any deployment fix

```bash
./scripts/check_dashboard.sh
```

For CUDA/pipeline validation:

```bash
python smoke_test.py
```

Do not commit dataset files, generated outputs, model checkpoints, `.env`, credentials, or `.venv`.
