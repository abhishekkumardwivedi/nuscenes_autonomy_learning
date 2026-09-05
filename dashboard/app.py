from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import PipelineConfig
from dashboard.events import EventBus
from dashboard.runner import PipelineRunner
from dashboard.stage_metadata import STAGES, BY_NUMBER
from dashboard.visual_hub import VisualHub
from dashboard import webrtc


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"


def initial_config() -> PipelineConfig:
    return PipelineConfig(
        dataroot=os.getenv("NUSCENES_DATAROOT", "/workspace/data/nuscenes"),
        version=os.getenv("NUSCENES_VERSION", "v1.0-mini"),
        output_dir=os.getenv("AUTONOMY_OUTPUT_DIR", "outputs"),
        scene_index=int(os.getenv("SCENE_INDEX", "0")),
        sample_index=int(os.getenv("SAMPLE_INDEX", "-1")),
        history_frames=int(os.getenv("HISTORY_FRAMES", "4")),
        future_frames=int(os.getenv("FUTURE_FRAMES", "6")),
        device=os.getenv("AUTONOMY_DEVICE", "auto"),
        temporal_model=os.getenv("TEMPORAL_MODEL", "ema"),
        planner_mode=os.getenv("PLANNER_MODE", "classical"),
        verbose=int(os.getenv("AUTONOMY_VERBOSE", "2")),
    )


event_bus = EventBus()
visual_hub = VisualHub()
runner = PipelineRunner(initial_config(), event_bus, visual_hub)


@asynccontextmanager
async def lifespan(app: FastAPI):
    event_bus.bind_loop(asyncio.get_running_loop())
    yield
    await webrtc.close_all()


app = FastAPI(title="Autonomy Learning Dashboard", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RunRequest(BaseModel):
    target_stage: int
    mode: str = "run_to"  # run_to | run_stage


class ConfigPatch(BaseModel):
    dataroot: str | None = None
    version: str | None = None
    output_dir: str | None = None
    scene_index: int | None = None
    sample_index: int | None = None
    history_frames: int | None = None
    future_frames: int | None = None
    image_height: int | None = None
    image_width: int | None = None
    bev_resolution: float | None = None
    depth_bins: int | None = None
    pretrained_backbone: bool | None = None
    temporal_model: str | None = None
    planner_mode: str | None = None
    verbose: int | None = None
    save_plots: bool | None = None
    device: str | None = None
    seed: int | None = None
    backend: str | None = None
    carla_host: str | None = None
    carla_port: int | None = None


class WebRTCOffer(BaseModel):
    sdp: str
    type: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    available, error = webrtc.availability()
    return {
        "ok": True,
        "busy": runner.busy,
        "webrtc_available": available,
        "webrtc_import_error": error,
        "dataroot_exists": Path(runner.cfg.dataroot).exists(),
    }


@app.get("/api/stages")
def stages():
    state = runner.snapshot()
    state_by_number = {x["stage"]: x for x in state["stages"]}
    return [
        {**s.to_dict(), "runtime": state_by_number[s.number]}
        for s in STAGES
    ]


@app.get("/api/state")
def state():
    return runner.snapshot()


@app.get("/api/config")
def config():
    return runner.config_dict()


@app.patch("/api/config")
def patch_config(body: ConfigPatch):
    changes = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        return runner.update_config(changes)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/run")
def run(body: RunRequest):
    try:
        if body.mode == "run_stage":
            runner.run_stage(body.target_stage)
        elif body.mode == "run_to":
            runner.run_to(body.target_stage)
        else:
            raise ValueError("mode must be 'run_to' or 'run_stage'")
        return {"accepted": True, "mode": body.mode, "target_stage": body.target_stage}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/stop")
def stop():
    runner.request_stop()
    return {"accepted": True}


@app.post("/api/reset")
def reset():
    try:
        runner.reset()
        return {"ok": True}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/stage/{number}/summary")
def stage_summary(number: int):
    if number not in BY_NUMBER:
        raise HTTPException(status_code=404, detail="Unknown stage")
    data = runner.stage_summary(number)
    if data is None:
        return JSONResponse(status_code=404, content={"detail": "No summary yet. Run the stage first."})
    return data


@app.get("/api/stage/{number}/logs")
def stage_logs(number: int):
    if number not in BY_NUMBER:
        raise HTTPException(status_code=404, detail="Unknown stage")
    return runner.stage_logs(number)


@app.get("/api/stage/{number}/artifacts")
def stage_artifacts(number: int):
    if number not in BY_NUMBER:
        raise HTTPException(status_code=404, detail="Unknown stage")
    return runner.stage_artifacts(number)


@app.get("/api/stage/{number}/code")
def stage_code(number: int, file: str | None = None):
    if number not in BY_NUMBER:
        raise HTTPException(status_code=404, detail="Unknown stage")
    stage = BY_NUMBER[number]
    allowed = {str(Path(x).as_posix()) for x in stage.code_files}
    selected = file or stage.code_files[0]
    selected = str(Path(selected).as_posix())
    if selected not in allowed:
        raise HTTPException(status_code=403, detail="This file is not part of the selected stage's code view.")
    path = (REPO_ROOT / selected).resolve()
    if REPO_ROOT not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Code file not found")
    return {"file": selected, "files": stage.code_files, "content": path.read_text(encoding="utf-8")}


@app.get("/api/artifact")
def artifact(path: str):
    root = runner.cfg.output_path.resolve()
    target = (root / path).resolve()
    if root != target and root not in target.parents:
        raise HTTPException(status_code=403, detail="Invalid artifact path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(target)


@app.get("/api/visual/current")
def current_visual():
    return Response(
        content=visual_hub.jpeg_bytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/system")
def system_info():
    data: dict[str, Any] = {"device": str(runner.cfg.torch_device)}
    try:
        import torch
        data["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            data.update({
                "gpu_name": torch.cuda.get_device_name(idx),
                "allocated_mb": round(torch.cuda.memory_allocated(idx) / 1024**2, 1),
                "reserved_mb": round(torch.cuda.memory_reserved(idx) / 1024**2, 1),
            })
    except Exception as exc:
        data["torch_error"] = str(exc)
    return data


@app.post("/api/webrtc/offer")
async def webrtc_offer(offer: WebRTCOffer):
    available, error = webrtc.availability()
    if not available:
        raise HTTPException(status_code=503, detail=f"WebRTC unavailable: {error}. The dashboard will use HTTP visual fallback.")
    try:
        return await webrtc.handle_offer(offer.sdp, offer.type, visual_hub)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"WebRTC negotiation failed: {exc}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await event_bus.connect(websocket)
    try:
        await websocket.send_text(json.dumps({"type": "hello", "state": runner.snapshot()}))
        while True:
            # Client pings keep reverse proxies from treating the socket as idle.
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        event_bus.disconnect(websocket)
    except Exception:
        event_bus.disconnect(websocket)
