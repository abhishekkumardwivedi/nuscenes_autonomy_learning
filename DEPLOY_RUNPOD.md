# Disposable RunPod deployment

Use a CUDA-enabled PyTorch/torchvision image, preferably Python 3.12. Mount the
same network volume at `/workspace`. Compute instances and running processes are
disposable; repository, venv, dataset, settings, caches, logs and outputs persist.

## One startup command

Configure this as the Pod/template **container startup command** and expose HTTP 8888:

```bash
bash /workspace/autonomy-learning-dashboard/scripts/runpod_boot.sh
```

Replace the image's Jupyter launcher rather than appending this behind a supervisor
that continually restarts Jupyter. The script stops only Jupyter listeners on 8888;
it refuses to kill unrelated port owners. Jupyter stays installed and may be launched
separately on 8889 with its own authentication. Entry points differ across images:
the template setting is a RunPod control-plane action, not something repository
file edits can configure for future Pods.

First-time volume initialization through SSH:

```bash
cd /workspace
git clone https://github.com/abhishekkumardwivedi/nuscenes_autonomy_learning.git autonomy-learning-dashboard
cd autonomy-learning-dashboard
cp .env.example .env
./scripts/runpod_boot.sh
```

A separately provisioned copy of the boot script can also clone the repository
when `AUTONOMY_REPO_URL` and `AUTONOMY_REPO_DIR` are supplied. No pod IDs or public
URLs are stored in configuration. Open the current Pod's 8888 HTTP service link.

For an existing SSH session, instead of container startup:

```bash
mkdir -p /workspace/autonomy-logs
nohup ./scripts/runpod_boot.sh > /workspace/autonomy-logs/launcher.log 2>&1 < /dev/null &
```

The foreground boot process holds a `flock` to prevent duplicate launches and
logs to `$AUTONOMY_LOG_DIR/startup.log`. `nohup` survives SSH disconnects, not pod
destruction; configure the template startup command for automatic future boots.

## Persistent paths and environment

| Purpose | Default |
|---|---|
| Repository / venv | `/workspace/autonomy-learning-dashboard`, `.venv` |
| Dataset | `/workspace/data/nuscenes` |
| Outputs and profiles | `/workspace/autonomy-outputs` |
| Frame artifacts | output root `/playback/<configuration-key>/scene*/frame*` |
| Startup logs | `/workspace/autonomy-logs/startup.log` |
| Pip / torch / HF caches | `/workspace/.cache/` |
| Dashboard token | `/workspace/.autonomy/dashboard.token` |
| Settings | `/workspace/autonomy-config.json` |

All paths are configurable in `.env.example`. Never commit `.env`. Download data
once with `./download_nuscenes_mini.sh /workspace/data/nuscenes`; extraction uses
`--no-same-owner` for network volumes.

The setup script uses a system-site-packages venv and constrains torch/torchvision
to the image versions during pip installation. It tests CUDA arithmetic and
torchvision NMS, hashes requirements and saves `.venv/.setup.json` after validation.
A matching marker skips pip. Python minor/torch/torchvision/CUDA-build mismatch
fails with instructions: use the original image or a **new VENV_DIR**. It never
deletes an environment or silently replaces torch. Requirements changes install
only needed changes. `ALLOW_CPU=1` is for CPU development.

Settings are saved to `AUTONOMY_CONFIG_FILE`, defaulting to the path above. Saved
settings override initial environment pipeline defaults. Deliberately rename that
JSON to return to `.env` defaults. Transient playback paths are not user settings.

## Access and Console

A RunPod proxy hostname alone is **not authentication**. The production launcher
creates a random persistent token file with mode 600 and protects HTTP and WSS
with the same login. Read that file through trusted SSH, then enter its value at
`/login`. Never put the token in Git, a public URL, screenshots or startup logs.
The session cookie is HttpOnly, SameSite and Secure behind HTTPS; bearer auth is
also supported for trusted API clients. Rotate the file and restart to revoke
existing sessions. This is shared single-user access, not a multi-tenant system.

Console is an interactive bash PTY with the server user's permissions, usually
root. It runs over `/ws/console` on 8888, with no separate shell port. Only trusted
operators should get dashboard access. Cross-origin mutations/WSS are rejected.

`DASHBOARD_TRUST_PROXY_AUTH=1` is an explicit opt-in only when an actual
authenticating proxy protects **every** HTTP/WSS route and direct access. It is
off by default. Without token auth or that protected proxy, Console is disabled.
`DASHBOARD_CONSOLE=0` disables it even with authentication.

Console supports history, resize, Ctrl+C, scrollback, streaming output and tools
such as top. htop must be installed in the image if desired. Four sessions maximum,
512 KB replay per session. Reconnecting reuses the shell while the server lives;
use `exit` to close it. Browser reconnect cannot revive a shell after server/pod
restart. Checkpoint training and use a job supervisor for durable long jobs.

## Hardware, profiles and playback

One psutil/NVML sampler serves all browsers at `HARDWARE_INTERVAL` (default 1s,
minimum 0.5s). Rolling graphs retain 120 seconds. NVML is device-wide; torch memory
belongs to this dashboard process, not a separate Console training process.
Process CPU includes other dashboard threads and may exceed 100% across cores.
Network rates are container-visible aggregate traffic, not stage attribution.

Each attempted stage writes `profile.json` beside `summary.json`, including failures
and cancellation. CUDA synchronizes at boundaries for meaningful elapsed time;
this has some overhead. Torch peaks reset per stage. NVML peaks are sampled and
can miss spikes; short stages without samples show N/A. Output memory counts
unique backing storage in stage-written tensors/arrays, not model parameters.
Graphs compare profiles in the current run/frame; saved results carry runtime state.

Basic tensor inspection does no statistics. Learning samples at most 65,536
elements; Detailed at most 1,048,576. Computation stays on the tensor device and
only scalars transfer. Sampled results are labelled, complex values use magnitude,
and non-finite values are null/N/A.

Playback computes Stage 00 through the selected target for each scene keyframe.
The first pass is **compute-paced**, not real-time trained autonomy. Cached frames
follow recorded timestamp spacing at 0.5x/1x/2x. Missing boundary history/future is
explicitly padded using endpoint frames; padded predictions are illustrative.
The original untrained-head/teaching limitations still apply.

Scene/frame configuration is shared across stages. Frame artifacts are keyed by
source/configuration/dataset identity. Images never pretend to restore tensors:
manual Run To rebuilds context. Pause finishes the current frame; Stop cancels
at safe boundaries. Seek replaces pending work after cancellation. Select any
stage to inspect existing frame outputs, or execute missing dependencies. Static
artifacts remain available. HTTPS images and WSS state are the reliable transport;
WebRTC is optional and no UDP ports are required. Replay requires offline mode.

## Validate

```bash
source .venv/bin/activate
./scripts/check_dashboard.sh
python smoke_test.py
python -m unittest discover -s tests -v
```

For terminal asset updates: `npm ci --ignore-scripts && npm run vendor`. Pinned
xterm assets and licenses are committed; no CDN or Node server is needed on RunPod.

## Troubleshooting

- **Unreachable:** check Pod running, HTTP 8888 exposed, startup command, and
  persistent startup.log. 401 means sign in; proxy 404 often means stopped Pod/port.
- **Port conflict:** disable template Jupyter auto-launch/respawn. The launcher
  refuses to kill arbitrary services; select a free port or stop the owner deliberately.
- **WSS disconnected:** check HTTPS/WSS routing and login. Reconnect reads server
  state; connection failure is independent from pipeline failure.
- **Visual disconnected:** use HTTP fallback; wait for requested stage/frame output
  and inspect Logs. WebRTC failure does not stop inference.
- **GPU/CUDA mismatch:** inspect nvidia-smi, GPU architecture, image versions and
  environment-check output. Use a compatible image or fresh VENV_DIR.
- **Dataset missing:** check dataroot/version/scene.json and saved Settings; Stage 00
  still works without data.
- **Console failed:** check auth, Linux PTY support, Console enabled and WSS routing;
  close unused shells with exit. Session state is lost after server restart.
- **Stage failed/OOM:** Overview and Logs expose the exception/traceback; Tensor Info
  and profile JSON provide context. Prior completed stages remain. Reduce image,
  BEV or history size and Run To again. Stop is cooperative, not a hard CUDA interrupt.
