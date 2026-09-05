"""Stage-boundary measurements; sampled NVML peaks are not exact allocation peaks."""
from datetime import datetime, timezone
import time
import psutil
import torch
from dashboard.hardware import torch_memory


def tensor_bytes(values):
    """Unique backing storage of tensors/arrays in stage-written context values."""
    import numpy as np
    seen = set()
    def count(value):
        if isinstance(value, torch.Tensor):
            try:
                storage = value.untyped_storage()
                key = (str(value.device), storage.data_ptr())
                size = storage.nbytes()
            except Exception:
                return 0
        elif isinstance(value, np.ndarray):
            root = value
            while isinstance(root.base, np.ndarray):
                root = root.base
            key, size = ('numpy', id(root)), root.nbytes
        elif isinstance(value, dict):
            return sum(count(v) for v in value.values())
        elif isinstance(value, (list, tuple)):
            return sum(count(v) for v in value)
        else:
            return 0
        if key in seen:
            return 0
        seen.add(key)
        return size
    return count(values)


class StageProfile:
    def __init__(self, stage, monitor, device):
        self.stage, self.monitor, self.device = stage, monitor, device
        self.cuda = device.type == 'cuda' and torch.cuda.is_available()
        if self.cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        self.start = time.time()
        self.tick = time.perf_counter()
        self.cpu = time.process_time()
        self.before = torch_memory()
        self.gpus_before = monitor.gpu_snapshot() if monitor else []

    def finish(self, outputs, status):
        sync_error = None
        if self.cuda:
            try:
                torch.cuda.synchronize(self.device)
            except Exception as exc:
                sync_error = str(exc)
        end = time.time()
        elapsed = time.perf_counter() - self.tick
        after = torch_memory()
        gpus_after = self.monitor.gpu_snapshot() if self.monitor else []
        samples = [x for x in (self.monitor.snapshot()['history'] if self.monitor else [])
                   if self.start <= x['timestamp'] <= end]
        # NVML reports physical devices; retain per-device measurements instead of
        # guessing CUDA_VISIBLE_DEVICES mappings. Torch counters are per process.
        gpu_profiles = []
        for gpu in gpus_after:
            index = gpu['index']
            before = next((g for g in self.gpus_before if g['index'] == index), {})
            sampled = [g for s in samples for g in s.get('gpus', []) if g['index'] == index]
            util = [g['utilization'] for g in sampled if g.get('utilization') is not None]
            used = [g['used'] for g in [before, gpu, *sampled] if g.get('used') is not None]
            gpu_profiles.append(dict(index=index, name=gpu['name'], before=before.get('used'),
                after=gpu.get('used'), sampled_peak=max(used) if used else None,
                utilization_avg=sum(util)/len(util) if util else None, samples=len(sampled)))
        return dict(stage=self.stage, status=status,
            started_at=datetime.fromtimestamp(self.start, timezone.utc).isoformat(),
            ended_at=datetime.fromtimestamp(end, timezone.utc).isoformat(), elapsed_ms=elapsed*1000,
            process_cpu_percent=(time.process_time()-self.cpu)/max(elapsed,.000001)*100,
            cpu_percent_avg=(sum(s['cpu_percent'] for s in samples if 'cpu_percent' in s) /
                             len([s for s in samples if 'cpu_percent' in s])) if any('cpu_percent' in s for s in samples) else None,
            torch_before=self.before, torch_after=after, gpus=gpu_profiles,
            output_memory_bytes=tensor_bytes(outputs), cuda_sync_error=sync_error,
            notes='GPU/VRAM are device-wide; NVML peaks are sampled. Torch peaks reset per stage. '
                  'Process CPU includes dashboard threads; output bytes count unique storage in stage-written values.')
