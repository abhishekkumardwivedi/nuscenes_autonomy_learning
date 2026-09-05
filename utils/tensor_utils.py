from __future__ import annotations

from typing import Any, Dict
import numpy as np
import torch


def tensor_info(value: Any) -> Dict[str, Any]:
    """Return compact statistics for a torch tensor or numpy array.

    This is deliberately similar to the tensor inspection helper you were already
    using. Every neural stage prints these values so that shape transformations
    remain visible rather than becoming a black box.
    """

    if isinstance(value, torch.Tensor):
        t = value.detach()
        result: Dict[str, Any] = {
            "shape": tuple(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "elements": int(t.numel()),
            "MB": round(t.numel() * t.element_size() / (1024 ** 2), 3),
        }
        if t.numel() > 0 and (t.is_floating_point() or t.is_complex()):
            x = t.float()
            result.update(
                min=float(x.min().item()),
                max=float(x.max().item()),
                mean=float(x.mean().item()),
                std=float(x.std(unbiased=False).item()),
            )
        return result

    if isinstance(value, np.ndarray):
        result = {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "elements": int(value.size),
            "MB": round(value.nbytes / (1024 ** 2), 3),
        }
        if value.size > 0 and np.issubdtype(value.dtype, np.number):
            result.update(
                min=float(np.nanmin(value)),
                max=float(np.nanmax(value)),
                mean=float(np.nanmean(value)),
                std=float(np.nanstd(value)),
            )
        return result

    return {"type": type(value).__name__}
