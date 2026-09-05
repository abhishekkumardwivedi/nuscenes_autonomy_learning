from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import threading
import time
from dashboard.profiling import StageProfile
from utils.tensor_utils import PROFILE_LEVEL
import traceback
from typing import Any

from config import PipelineConfig
from pipeline_context import PipelineContext
from utils.logging_utils import LessonLogger
from dashboard.stage_metadata import STAGES, BY_NUMBER, stage_dir


class PipelineStopRequested(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineRunner:
    """Owns one in-memory pipeline context and runs stages in a worker thread.

    The FastAPI event loop never performs model work. This is what keeps the
    browser responsive while PyTorch / nuScenes stages are executing.
    """

    def __init__(self, cfg: PipelineConfig, event_bus, visual_hub, monitor=None) -> None:
        self.cfg = cfg
        self.monitor = monitor
        self._busy = False
        self.event_bus = event_bus
        self.visual_hub = visual_hub
        self.ctx = PipelineContext()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.current_stage: int | None = None
        self.completed_through = -1
        self.stale_from: int | None = None
        self.run_id = 0
        self.states: dict[int, dict[str, Any]] = {}
        self._reset_states()

    def _reset_states(self) -> None:
        self.states = {
            s.number: {
                "stage": s.number,
                "alias": s.alias,
                "title": s.title,
                "status": "not_started",
                "progress": 0,
                "current_step": "",
                "error": None,
                "started_at": None,
                "ended_at": None,
                "logs": [],
                "last_tensor": None,
                "tensors": {},
                "profile": None,
            }
            for s in STAGES
        }

    @property
    def busy(self) -> bool:
        return self._busy

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "busy": self.busy,
                "current_stage": self.current_stage,
                "completed_through": self.completed_through,
                "stale_from": self.stale_from,
                "run_id": self.run_id,
                "config": self.config_dict(),
                "stages": [dict(self.states[s.number], logs=self.states[s.number]["logs"][-80:]) for s in STAGES],
            }

    def config_dict(self) -> dict[str, Any]:
        data = asdict(self.cfg)
        data["bev_height"] = self.cfg.bev_height
        data["bev_width"] = self.cfg.bev_width
        data["torch_device"] = str(self.cfg.torch_device)
        return data

    def update_config(self, changes: dict[str, Any]) -> dict[str, Any]:
        if self.busy:
            raise RuntimeError("Cannot change configuration while a stage is running.")
        allowed = {
            "dataroot", "version", "output_dir", "scene_index", "sample_index",
            "history_frames", "future_frames", "image_height", "image_width",
            "bev_resolution", "depth_bins", "pretrained_backbone", "temporal_model",
            "planner_mode", "verbose", "save_plots", "device", "seed", "backend",
            "carla_host", "carla_port", "profile_level", "playback_mode",
        }
        if changes.get('profile_level', self.cfg.profile_level) not in {'basic', 'learning', 'detailed'}:
            raise ValueError('Profile level must be basic, learning, or detailed')
        for key in ('history_frames', 'future_frames', 'image_height', 'image_width', 'depth_bins'):
            if key in changes and int(changes[key]) < 1:
                raise ValueError(f'{key} must be positive')
        if 'bev_resolution' in changes and float(changes['bev_resolution']) <= 0:
            raise ValueError('BEV resolution must be positive')
        with self.lock:
            for key, value in changes.items():
                if key in allowed and hasattr(self.cfg, key):
                    setattr(self.cfg, key, value)
            self.reset(emit=False)
            self.event_bus.emit({"type": "config_updated", "config": self.config_dict()})
            return self.config_dict()

    def reset(self, emit: bool = True) -> None:
        if self.busy:
            raise RuntimeError("Stop the active run before resetting.")
        with self.lock:
            self.ctx = PipelineContext()
            self.current_stage = None
            self.completed_through = -1
            self.stale_from = None
            self.stop_event.clear()
            self._reset_states()
            self.visual_hub.set_status(0, "Foundation / configuration", "idle")
            if emit:
                self.event_bus.emit({"type": "pipeline_reset", "state": self.snapshot()})

    def request_stop(self) -> None:
        self.stop_event.set()
        self.event_bus.emit({"type": "stop_requested", "message": "Stop requested; execution will stop at the next safe log/substage boundary."})

    def _cancel_check(self) -> None:
        if self.stop_event.is_set():
            raise PipelineStopRequested("Execution stopped by user request.")

    def run_to(self, target: int) -> None:
        target = int(target)
        if self.busy:
            raise RuntimeError("Pipeline is already running.")
        if target not in BY_NUMBER:
            raise ValueError(f"Unknown stage {target}")

        # If anything upstream was rerun, a clean rebuild is safer than silently
        # consuming stale downstream tensors.
        if self.stale_from is not None and target >= self.stale_from:
            self.reset(emit=True)

        if target <= self.completed_through and self.stale_from is None:
            self.event_bus.emit({"type": "already_completed", "stage": target, "message": "This stage is already cached. Use Run Stage to rerun it, or simply inspect it."})
            return

        start = max(0, self.completed_through + 1)
        self._start_worker(mode="run_to", start=start, target=target)

    def run_stage(self, target: int) -> None:
        target = int(target)
        if self.busy:
            raise RuntimeError("Pipeline is already running.")
        if target not in BY_NUMBER:
            raise ValueError(f"Unknown stage {target}")
        if target > self.completed_through + 1:
            raise RuntimeError(
                f"Stage {target:02d} needs upstream context. Run To Stage {target:02d} first."
            )
        if self.stale_from is not None and target >= self.stale_from:
            raise RuntimeError('Upstream results changed. Use Run To to rebuild compatible context.')
        if target <= self.completed_through:
            self._invalidate_downstream(target + 1)
            self.completed_through = target - 1
        self._start_worker(mode="run_stage", start=target, target=target)

    def _start_worker(self, mode: str, start: int, target: int) -> None:
        manifest = self.cfg.output_path / 'frame.json'
        if manifest.exists():
            manifest.unlink()  # This frame is no longer a fully validated cached result.
        self.stop_event.clear()
        self._busy = True
        self.run_id += 1
        run_id = self.run_id
        self.worker = threading.Thread(
            target=self._worker_main,
            args=(mode, start, target, run_id),
            name=f"pipeline-{run_id}",
            daemon=True,
        )
        self.worker.start()

    def _worker_main(self, mode: str, start: int, target: int, run_id: int) -> None:
        self.event_bus.emit({"type": "run_started", "run_id": run_id, "mode": mode, "start": start, "target": target})
        log = LessonLogger(
            self.cfg.verbose,
            event_sink=self._on_log_event,
            cancel_check=self._cancel_check,
        )
        token = PROFILE_LEVEL.set(self.cfg.profile_level if self.cfg.verbose else 'basic')
        try:
            for number in range(start, target + 1):
                self._cancel_check()
                stage = BY_NUMBER[number]
                self._mark_stage_started(stage.number)
                self.ctx.written.clear()
                profile = StageProfile(number, self.monitor, self.cfg.torch_device)
                status = 'failed'
                try:
                    module = importlib.import_module(stage.module)
                    module.run(self.ctx, self.cfg, log)
                    self._cancel_check()
                    status = 'completed'
                except PipelineStopRequested:
                    status = 'cancelled'
                    raise
                finally:
                    try:
                        result = profile.finish({k:self.ctx.get(k) for k in self.ctx.written}, status)
                        result['scene_index'] = self.cfg.scene_index
                        result['sample_index'] = self.cfg.sample_index
                        folder = stage_dir(self.cfg.output_path, stage)
                        folder.mkdir(parents=True, exist_ok=True)
                        (folder / 'profile.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
                        with self.lock:
                            self.states[number]['profile'] = result
                    except Exception as profile_error:
                        self._append_log_file(number, 'PROFILE ERROR', str(profile_error))
                self._mark_stage_completed(stage.number)
                if mode == "run_stage":
                    # Rerunning an upstream stage invalidates every later output.
                    if stage.number < max(self.completed_through, stage.number):
                        self._invalidate_downstream(stage.number + 1)
                    break
            with self.lock:
                self.current_stage = None
            self.event_bus.emit({"type": "run_completed", "run_id": run_id, "target": target, "state": self.snapshot()})
        except PipelineStopRequested as exc:
            self._mark_stage_stopped(self.current_stage, str(exc))
            with self.lock:
                self.current_stage = None
            self.event_bus.emit({"type": "run_stopped", "run_id": run_id, "message": str(exc), "state": self.snapshot()})
        except Exception as exc:
            tb = traceback.format_exc()
            failed_stage = self.current_stage
            self._mark_stage_failed(self.current_stage, exc, tb)
            with self.lock:
                self.current_stage = None
            self.event_bus.emit({
                "type": "run_failed",
                "run_id": run_id,
                "stage": failed_stage,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": tb,
                "state": self.snapshot(),
            })
        finally:
            PROFILE_LEVEL.reset(token)
            self.stop_event.clear()
            self._busy = False
            self.event_bus.emit({'type': 'run_idle', 'state': self.snapshot()})

    def _mark_stage_started(self, number: int) -> None:
        stage = BY_NUMBER[number]
        with self.lock:
            self.current_stage = number
            st = self.states[number]
            st.update(status="running", logs=[], tensors={}, profile=None, progress=2, current_step="Starting stage", error=None, started_at=_now(), ended_at=None)
        self.visual_hub.set_status(number, stage.title, "running")
        self._emit_state(number)

    def _mark_stage_completed(self, number: int) -> None:
        stage = BY_NUMBER[number]
        with self.lock:
            st = self.states[number]
            st.update(status="completed", progress=100, current_step="Completed", ended_at=_now())
            self.completed_through = max(self.completed_through, number)
        self._refresh_visual(number)
        self.visual_hub.set_status(number, stage.title, "completed")
        self._emit_state(number)

    def _mark_stage_failed(self, number: int | None, exc: Exception, tb: str) -> None:
        if number is None:
            return
        stage = BY_NUMBER[number]
        with self.lock:
            st = self.states[number]
            st.update(status="failed", current_step="Failed", error=f"{type(exc).__name__}: {exc}", ended_at=_now())
            st["logs"].append({"ts": _now(), "level": "error", "message": tb})
        self.visual_hub.set_status(number, stage.title, "failed")
        self._append_log_file(number, "ERROR", tb)
        self._emit_state(number)

    def _mark_stage_stopped(self, number: int | None, message: str) -> None:
        if number is None:
            return
        stage = BY_NUMBER[number]
        with self.lock:
            st = self.states[number]
            st.update(status="cancelled", current_step="Stopped by user", error=message, ended_at=_now())
        self.visual_hub.set_status(number, stage.title, "stopped")
        self._emit_state(number)

    def _invalidate_downstream(self, start: int) -> None:
        if start > 20:
            return
        with self.lock:
            self.stale_from = start
            self.completed_through = min(self.completed_through, start - 1)
            for n in range(start, 21):
                if self.states[n]["status"] == "completed":
                    self.states[n]["status"] = "stale"
                    self.states[n]["current_step"] = "Upstream stage changed; rerun required"
        self.event_bus.emit({"type": "downstream_stale", "from_stage": start})

    def _on_log_event(self, event: dict[str, Any]) -> None:
        number = event.get("stage")
        if number is None:
            return
        stage = BY_NUMBER[int(number)]
        kind = event.get("kind", "log")
        message = str(event.get("message", ""))
        level = str(event.get("level", "info"))
        with self.lock:
            st = self.states[int(number)]
            st["logs"].append({"ts": _now(), "level": level, "message": message, "kind": kind})
            if len(st["logs"]) > 1200:
                st["logs"] = st["logs"][-800:]
            if kind == "substage":
                sub = int(event.get("substage", 1))
                total = max(1, stage.substeps)
                st["progress"] = min(95, int(5 + 90 * (sub - 1) / total))
                st["current_step"] = message
            elif kind == "tensor":
                st["last_tensor"] = event.get("tensor_info")
                st["tensors"][event.get("name", "tensor")] = event.get("tensor_info")
                st["current_step"] = event.get("name", message)
        self._append_log_file(int(number), level.upper(), message)
        if kind in {"substage", "outcome"}:
            self._refresh_visual(int(number))
        self.event_bus.emit({"type": "pipeline_event", **event, "state": self._public_stage_state(int(number))})

    def _append_log_file(self, stage: int, level: str, message: str) -> None:
        try:
            out = self.cfg.output_path
            out.mkdir(parents=True, exist_ok=True)
            with (out / "dashboard_run.log").open("a", encoding="utf-8") as f:
                f.write(f"{_now()} [STAGE {stage:02d}] [{level}] {message}\n")
        except Exception:
            pass

    def _public_stage_state(self, number: int) -> dict[str, Any]:
        with self.lock:
            st = dict(self.states[number])
            st["logs"] = st["logs"][-80:]
            return st

    def _emit_state(self, number: int) -> None:
        self.event_bus.emit({"type": "stage_state", "stage": number, "state": self._public_stage_state(number)})

    def _refresh_visual(self, number: int) -> None:
        stage = BY_NUMBER[number]
        folder = stage_dir(self.cfg.output_path, stage)
        if not folder.exists():
            return
        chosen: Path | None = None
        if stage.preferred_visual:
            p = folder / stage.preferred_visual
            if p.exists():
                chosen = p
        if chosen is None:
            images = sorted(folder.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            if images:
                chosen = images[0]
        if chosen is not None:
            self.visual_hub.set_image(chosen, number, stage.title)
            self.event_bus.emit({"type": "visual_updated", "stage": number, "name": chosen.name, "cache_bust": chosen.stat().st_mtime_ns})

    def stage_logs(self, number: int) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.states[number]["logs"])

    def stage_summary(self, number: int) -> dict[str, Any] | None:
        stage = BY_NUMBER[number]
        path = stage_dir(self.cfg.output_path, stage) / "summary.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"_error": str(exc)}

    def stage_artifacts(self, number: int) -> list[dict[str, Any]]:
        stage = BY_NUMBER[number]
        folder = stage_dir(self.cfg.output_path, stage)
        if not folder.exists():
            return []
        items = []
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".json", ".md", ".txt"}:
                items.append({
                    "name": p.name,
                    "relative_path": str(p.relative_to(self.cfg.output_path)),
                    "size": p.stat().st_size,
                    "mtime_ns": p.stat().st_mtime_ns,
                    "kind": "image" if p.suffix.lower() in {".png", ".jpg", ".jpeg"} else "data",
                })
        return items

    def stage_profiles(self):
        profiles = []
        for stage in STAGES:
            path = stage_dir(self.cfg.output_path, stage) / 'profile.json'
            if path.exists():
                try:
                    result = json.loads(path.read_text())
                    result['runtime_status'] = self.states[stage.number]['status']
                    profiles.append(result)
                except (OSError, ValueError):
                    pass
        return profiles
