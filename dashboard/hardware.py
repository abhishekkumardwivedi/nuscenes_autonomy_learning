"""One low-frequency sampler for every browser. Missing counters remain None."""
from collections import deque
from pathlib import Path
import os
import threading
import time

import psutil


def torch_memory():
    import torch
    result = dict(cuda_available=torch.cuda.is_available(), device=None,
                  allocated=None, reserved=None, peak_allocated=None, peak_reserved=None)
    if result['cuda_available']:
        try:
            device = torch.cuda.current_device()
            result.update(device=device, name=torch.cuda.get_device_name(device),
                          allocated=torch.cuda.memory_allocated(device),
                          reserved=torch.cuda.memory_reserved(device),
                          peak_allocated=torch.cuda.max_memory_allocated(device),
                          peak_reserved=torch.cuda.max_memory_reserved(device))
        except Exception as exc:
            result['error'] = str(exc)
    return result


class HardwareMonitor:
    def __init__(self, storage_path=None, interval=None):
        self.storage_path = Path(storage_path or os.getenv('PERSISTENT_ROOT', '/workspace'))
        self.interval = max(.5, float(interval or os.getenv('HARDWARE_INTERVAL', '1')))
        self.history = deque(maxlen=max(2, int(120 / self.interval)))
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.nvml = None
        self.nvml_error = None
        self.previous_network = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
        except Exception as exc:
            self.nvml_error = str(exc)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True, name='hardware-sampler')
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        if self.nvml and not (self.thread and self.thread.is_alive()):
            self.nvml.nvmlShutdown()
            self.nvml = None

    def gpu_snapshot(self):
        gpus = []
        if self.nvml:
            n = self.nvml
            def safe(fn):
                try:
                    value = fn()
                    return value.decode() if isinstance(value, bytes) else value
                except Exception:
                    return None
            count = safe(n.nvmlDeviceGetCount) or 0
            for index in range(count):
                handle = safe(lambda: n.nvmlDeviceGetHandleByIndex(index))
                if handle is None:
                    continue
                memory = safe(lambda: n.nvmlDeviceGetMemoryInfo(handle))
                gpus.append(dict(index=index, name=safe(lambda: n.nvmlDeviceGetName(handle)),
                    utilization=safe(lambda: n.nvmlDeviceGetUtilizationRates(handle).gpu),
                    total=memory.total if memory else None, used=memory.used if memory else None,
                    free=memory.free if memory else None,
                    temperature=safe(lambda: n.nvmlDeviceGetTemperature(handle, n.NVML_TEMPERATURE_GPU)),
                    power_mw=safe(lambda: n.nvmlDeviceGetPowerUsage(handle)),
                    clock_mhz=safe(lambda: n.nvmlDeviceGetClockInfo(handle, n.NVML_CLOCK_SM))))
        return gpus

    def sample(self):
        now = time.time()
        ram = psutil.virtual_memory()
        try:
            disk = psutil.disk_usage(str(self.storage_path))
            storage = dict(path=str(self.storage_path), total=disk.total, used=disk.used, free=disk.free)
        except OSError as exc:
            storage = dict(path=str(self.storage_path), error=str(exc), total=None, used=None, free=None)
        net = psutil.net_io_counters()
        rx = tx = None
        if self.previous_network:
            ts, previous = self.previous_network
            dt = max(.001, now - ts)
            rx = max(0, net.bytes_recv - previous.bytes_recv) / dt
            tx = max(0, net.bytes_sent - previous.bytes_sent) / dt
        self.previous_network = now, net
        sample = dict(timestamp=now, cpu_percent=psutil.cpu_percent(interval=None),
                      cpu_cores=psutil.cpu_count(), ram=dict(total=ram.total, used=ram.used,
                      available=ram.available, percent=ram.percent), storage=storage,
                      network=dict(rx_bytes_sec=rx, tx_bytes_sec=tx),
                      gpus=self.gpu_snapshot(), nvml_error=self.nvml_error, torch=torch_memory())
        with self.lock:
            self.history.append(sample)
        return sample

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self.sample()
            except Exception as exc:
                with self.lock:
                    self.history.append(dict(timestamp=time.time(), error=str(exc)))
            self.stop_event.wait(self.interval)

    def snapshot(self):
        with self.lock:
            return dict(interval=self.interval, latest=self.history[-1] if self.history else None,
                        history=list(self.history))
