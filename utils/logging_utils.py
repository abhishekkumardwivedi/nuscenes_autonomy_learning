from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict
from .tensor_utils import tensor_info


EventSink = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], None]


class LessonLogger:
    """Console + structured logger designed for learning.

    The original CLI still prints exactly as before. When the dashboard supplies
    ``event_sink`` the same teaching messages are also emitted as structured
    events, so the browser can update without scraping terminal text.
    """

    def __init__(
        self,
        verbose: int = 2,
        event_sink: EventSink | None = None,
        cancel_check: CancelCheck | None = None,
    ):
        self.verbose = int(verbose)
        self.event_sink = event_sink
        self.cancel_check = cancel_check
        self.current_stage: int | None = None
        self.current_substage: int | None = None

    def _check_cancel(self) -> None:
        if self.cancel_check is not None:
            self.cancel_check()

    def _emit(self, kind: str, message: str, level: str = "info", **extra: Any) -> None:
        if self.event_sink is None:
            return
        event = {
            "kind": kind,
            "level": level,
            "stage": self.current_stage,
            "substage": self.current_substage,
            "message": message,
            **extra,
        }
        self.event_sink(event)

    def stage(self, number: int, title: str) -> None:
        self._check_cancel()
        self.current_stage = int(number)
        self.current_substage = None
        line = "=" * 76
        print(f"\n{line}\n[STAGE {number:02d}] {title}\n{line}")
        self._emit("stage", title, stage=int(number), title=title)

    def substage(self, number: int, sub: int, title: str) -> None:
        self._check_cancel()
        self.current_stage = int(number)
        self.current_substage = int(sub)
        if self.verbose >= 1:
            print(f"\n  [{number:02d}.{sub}] {title}")
            print("  " + "-" * 64)
        self._emit("substage", title, stage=int(number), substage=int(sub))

    def info(self, message: str) -> None:
        if self.verbose >= 1:
            print(f"  {message}")
        self._emit("log", message, level="info")

    def detail(self, message: str) -> None:
        if self.verbose >= 2:
            print(f"    {message}")
        if self.verbose >= 2:
            self._emit("log", message, level="detail")

    def deep(self, message: str) -> None:
        if self.verbose >= 3:
            print(f"      {message}")
            self._emit("log", message, level="deep")

    def tensor(self, name: str, value: Any) -> None:
        if self.verbose >= 1:
            stats = tensor_info(value)
            stats.update(name=name, stage=self.current_stage)
            print(f"    {name}: {stats}")
            self._emit("tensor", f"{name}: {stats}", level="tensor", name=name, tensor_info=stats)

    def outcome(self, message: str) -> None:
        print(f"\n  OUTCOME -> {message}")
        self._emit("outcome", message, level="outcome")


def make_stage_dir(base: Path, number: int, short_name: str) -> Path:
    path = base / f"stage{number:02d}_{short_name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jsonable(value: Any) -> Any:
    """Convert common values to lightweight JSON metadata, not raw tensors."""
    import numpy as np
    import torch

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return tensor_info(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 30:
            return {"length": len(value), "preview": [_jsonable(v) for v in value[:5]]}
        return [_jsonable(v) for v in value]
    # Avoid serializing heavyweight nuScenes objects or model modules.
    return f"<{type(value).__name__}>"


def save_stage_summary(stage_dir: Path, values: Dict[str, Any]) -> Path:
    out = stage_dir / "summary.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(values), f, indent=2)
    return out
