from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from .tensor_utils import tensor_info


class LessonLogger:
    """Console logger designed for learning rather than production telemetry."""

    def __init__(self, verbose: int = 2):
        self.verbose = int(verbose)

    def stage(self, number: int, title: str) -> None:
        line = "=" * 76
        print(f"\n{line}\n[STAGE {number:02d}] {title}\n{line}")

    def substage(self, number: int, sub: int, title: str) -> None:
        if self.verbose >= 1:
            print(f"\n  [{number:02d}.{sub}] {title}")
            print("  " + "-" * 64)

    def info(self, message: str) -> None:
        if self.verbose >= 1:
            print(f"  {message}")

    def detail(self, message: str) -> None:
        if self.verbose >= 2:
            print(f"    {message}")

    def deep(self, message: str) -> None:
        if self.verbose >= 3:
            print(f"      {message}")

    def tensor(self, name: str, value: Any) -> None:
        if self.verbose >= 1:
            stats = tensor_info(value)
            print(f"    {name}: {stats}")

    def outcome(self, message: str) -> None:
        print(f"\n  OUTCOME -> {message}")


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
