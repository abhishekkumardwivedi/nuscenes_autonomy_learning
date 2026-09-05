# Start here

1. Mount persistent storage at `/workspace` on a compatible CUDA PyTorch RunPod.
2. Follow [DEPLOY_RUNPOD.md](DEPLOY_RUNPOD.md) once to clone/configure the repository.
3. Set `bash /workspace/autonomy-learning-dashboard/scripts/runpod_boot.sh` as
   the template startup command. Expose HTTP 8888; Jupyter must not respawn there.
4. Open the Pod's 8888 link and sign in with the persistent dashboard token.
5. Select a scene/stage. Run To builds dependencies; Run Stage needs compatible
   context. Stop cancels at a safe boundary; errors expose Retry and full traceback.
6. Inspect left tabs Visual/Inputs/Outputs/Code/Graphs/Artifacts/Console and right
   tabs Overview/Logs/Tensor Info/Parameters/Hardware/Profiler.
7. Play processes each keyframe through the selected stage. Pause finishes the
   current frame; Stop cancels; the timeline seeks. First-pass speed is limited
   by compute. Cached frames follow recorded timestamp spacing.
8. Drag the divider, double-click headers for fullscreen, press Escape to restore.
   Zoom with scroll/+/-; drag to pan and Fit to reset.

Begin with Stage 00 (no dataset needed), then 01 (dataset), 02 (cameras), 05/06
(BEV), 09 (temporal) and downstream prediction/planning. Distinguish trained
inference from untrained demo heads and interpretable baselines.

Basic profiling disables tensor statistics; Learning/Detailed use bounded samples.
Hardware counters are device/container-wide; torch memory is for this server.
Profile JSON lives beside summaries. Keep code, environment, data, caches, logs,
settings and outputs on the volume. Running jobs and tensor caches do not survive
Pod destruction. Deployment docs cover auth, ports, CUDA, data, WSS and Console errors.
