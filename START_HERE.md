# Start here

This project is a **browser-based autonomy learning lab**. The models and nuScenes data run on RunPod; you control and inspect them from one webpage on your local PC.

## Dashboard layout

```text
Top       : Stage 00 ... Stage 20 + Run / Stop / settings
Left      : Visual | Inputs | Outputs | Code | Artifacts
Right     : Overview | Logs | Tensor Info | Parameters
Bottom    : compact full-pipeline status
```

You can click any stage at any time to inspect its saved output or code. Viewing a stage does not rerun it.

- **Run to selected**: execute missing dependencies up to the selected stage.
- **Run selected stage**: rerun that stage using valid cached upstream context.
- Rerunning an upstream stage marks later completed stages **stale** so old downstream results are not silently reused.
- **Stop**: requests cancellation at the next safe stage/substage boundary.

## RunPod

```bash
./scripts/setup_runpod.sh
cp .env.example .env
./download_nuscenes_mini.sh /workspace/data/nuscenes   # only if needed
./scripts/run_dashboard.sh
```

Expose HTTP port **8080**, then open its RunPod endpoint in your local browser.

See `DEPLOY_RUNPOD.md` for deployment details and `CODEX_RUNPOD.md` for the Codex deployment/update contract.

## Learning order

Start at Stage 01 and progress one stage at a time. For each stage:

1. read **Overview**,
2. inspect **Inputs**,
3. run the stage,
4. watch **Logs** and progress,
5. inspect **Tensor Info**,
6. view the image/BEV output,
7. open **Code** and connect implementation to what you just observed.

The command-line `main.py` remains available as a second learning interface.
