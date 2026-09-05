"""Bounded on-device inspection. Basic never performs tensor reductions."""
from contextvars import ContextVar
import math
import numpy as np
import torch

PROFILE_LEVEL = ContextVar('profile_level', default='basic')


def tensor_info(value, level=None):
    level = level or PROFILE_LEVEL.get()
    if not isinstance(value, (torch.Tensor, np.ndarray)):
        return {'type': type(value).__name__}
    is_torch = isinstance(value, torch.Tensor)
    count = value.numel() if is_torch else value.size
    size = count * value.element_size() if is_torch else value.nbytes
    result = dict(shape=list(value.shape), dtype=str(value.dtype),
                  device=str(value.device) if is_torch else 'cpu', elements=int(count),
                  memory_bytes=int(size), MB=round(size/1024**2, 3),
                  requires_grad=value.requires_grad if is_torch else False,
                  min=None, max=None, mean=None, std=None, statistics='disabled')
    if level == 'basic' or not count:
        return result
    limit = 65536 if level == 'learning' else 1048576
    x = value.detach() if is_torch else value
    # Slice before flattening: do not allocate a huge non-contiguous copy.
    for axis in range(x.ndim):
        elements = x.numel() if is_torch else x.size
        if elements <= limit:
            break
        selection = [slice(None)] * x.ndim
        selection[axis] = slice(None, None, max(1, math.ceil(elements / limit)))
        x = x[tuple(selection)]
    try:
        complex_input = x.is_complex() if is_torch else np.iscomplexobj(x)
        if complex_input:
            x = x.abs() if is_torch else np.abs(x)
        if is_torch:
            x = x.float()
            stats = [x.min().item(), x.max().item(), x.mean().item(), x.std(unbiased=False).item()]
            sampled = x.numel()
        else:
            x = x.astype(np.float64)
            stats = [np.min(x), np.max(x), np.mean(x), np.std(x)]
            sampled = x.size
        result.update({key: float(v) if math.isfinite(float(v)) else None
                       for key,v in zip(['min','max','mean','std'], stats)})
        result.update(statistics='sampled' if sampled < count else 'exact', sampled_elements=int(sampled),
                      complex_magnitude=bool(complex_input))
    except (TypeError, RuntimeError, ValueError) as exc:
        result['statistics'] = 'unavailable: ' + str(exc)
    return result
