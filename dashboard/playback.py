"""Sequential, compute-paced nuScenes replay over the existing HTTP/WSS path.

Each frame reuses the educational Stage 00–20 modules, not a parallel model
implementation. Completed frame artifacts are cached; tensors are not persisted.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading
import time
import traceback

from dashboard.stage_metadata import STAGES, stage_dir


class SceneCatalog:
    def __init__(self):
        self.key = None
        self.scenes = []
        self.samples = {}
        self.lock = threading.Lock()

    def load(self, cfg):
        root = Path(cfg.dataroot) / cfg.version
        key = (str(root.resolve()), (root/'scene.json').stat().st_mtime_ns)
        with self.lock:
            if self.key != key:
                self.scenes = json.loads((root/'scene.json').read_text())
                self.samples = {x['token']:x for x in json.loads((root/'sample.json').read_text())}
                self.key = key
            return self.scenes, self.samples

    def list(self, cfg):
        try:
            scenes, _ = self.load(cfg)
            return [dict(index=i, name=s['name'], description=s['description'], frames=s['nbr_samples']) for i,s in enumerate(scenes)]
        except OSError as exc:
            raise FileNotFoundError(f'nuScenes scene metadata missing at {cfg.dataroot}/{cfg.version}. Download the dataset or update Settings.') from exc

    def frames(self, cfg, scene):
        self.list(cfg)
        scenes, samples = self.load(cfg)
        if not 0 <= scene < len(scenes):
            raise ValueError('Scene index out of range')
        token = scenes[scene]['first_sample_token']
        result, seen = [], set()
        while token:
            if token in seen:
                raise ValueError('Dataset sample chain contains a cycle')
            seen.add(token)
            sample = samples[token]
            result.append(dict(index=len(result), token=token, timestamp=sample['timestamp']))
            token = sample['next']
        return result


class Playback:
    def __init__(self, runner, catalog):
        self.runner, self.catalog = runner, catalog
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.closed = threading.Event()
        self.thread = None
        self.pending = None
        self.playing = False
        self.processing = False
        self.frame = 0
        self.scene = runner.cfg.scene_index
        self.target = 2
        self.speed = 1.
        self.timestamp = None
        self.error = None
        self.cache_hit = False
        self.base_output = runner.cfg.output_dir
        self.base_config = asdict(runner.cfg)

    @property
    def busy(self):
        return self.processing or self.playing or self.pending is not None

    def snapshot(self):
        with self.lock:
            return dict(playing=self.playing, processing=self.processing, frame=self.frame,
                        scene=self.scene, target_stage=self.target, speed=self.speed,
                        timestamp=self.timestamp, error=self.error, cache_hit=self.cache_hit,
                        transport='HTTPS artifacts / WSS status', pacing='compute-paced')

    def command(self, action, scene=None, frame=None, target_stage=None, speed=None):
        with self.runner.lock, self.lock:
            if action == 'pause':
                self.playing = False
                return self.snapshot()
            if action not in {'play', 'seek', 'next', 'previous'}:
                raise ValueError('Unknown playback command')
            if self.runner.busy and not self.processing:
                raise RuntimeError('Stop the manual pipeline run before playback')
            if self.runner.cfg.backend != 'offline':
                raise ValueError('Scene playback requires the offline backend')
            selected_scene = self.scene if scene is None else scene
            frames = self.catalog.frames(self.runner.cfg, selected_scene)
            selected_frame = self.frame if frame is None else frame
            selected_frame += 1 if action == 'next' else -1 if action == 'previous' else 0
            target = self.target if target_stage is None else target_stage
            if not 0 <= target <= 20:
                raise ValueError('Stage must be 00–20')
            if speed is not None and speed not in {.5, 1., 2.}:
                raise ValueError('Playback speed must be 0.5, 1, or 2')
            self.target, self.speed = target, speed or self.speed
            self.pending = (selected_scene, max(0,min(len(frames)-1,selected_frame)), target)
            self.playing = action == 'play'
            self.error = None
            if self.processing:
                self.runner.request_stop()
            self.wake.set()
            if not self.thread or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._loop, daemon=True, name='scene-playback')
                self.thread.start()
            return self.snapshot()

    def stop(self):
        with self.lock:
            self.playing = False
            self.pending = None
        self.runner.request_stop()
        self.wake.set()

    def close(self):
        self.stop()
        self.closed.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=3)

    def _loop(self):
        while not self.closed.is_set():
            with self.lock:
                pending, self.pending = self.pending, None
                if pending:
                    self.processing = True
            if pending is None:
                self.wake.wait(.25)
                self.wake.clear()
                continue
            try:
                scene, frame, target = pending
                frames = self.catalog.frames(self.runner.cfg, scene)
                self._frame(scene, frame, target, frames[frame]['timestamp'])
            except Exception:
                with self.lock:
                    self.error = traceback.format_exc()
                    self.playing = False
            finally:
                with self.lock:
                    self.processing = False
                self.runner.event_bus.emit({'type':'playback', 'playback':self.snapshot(), 'state':self.runner.snapshot()})
            with self.lock:
                if self.pending:
                    continue
                if not self.playing or self.error or self.frame >= len(frames)-1:
                    self.playing = False
                    continue
                delay = max(.01, (frames[self.frame+1]['timestamp'] - frames[self.frame]['timestamp'])/1e6/self.speed)
            self.wake.clear()
            self.wake.wait(delay)
            with self.lock:
                if self.playing and not self.pending:
                    self.pending = (self.scene, self.frame+1, self.target)

    def _frame(self, scene, frame, target, timestamp):
        cfg = self.runner.cfg
        # Include source contents, all model/geometry settings and dataset metadata
        # identity in the key. Pod IDs and public URLs never enter this cache.
        config = asdict(cfg)
        for key in ['output_dir','sample_index','scene_index','playback_mode']:
            config.pop(key, None)
        code = hashlib.sha256()
        root = Path(__file__).resolve().parents[1]
        for path in sorted(list((root/'stages').glob('*.py')) + list((root/'utils').glob('*.py')) + [root/'config.py']):
            code.update(path.read_bytes())
        metadata = Path(cfg.dataroot)/cfg.version/'sample.json'
        key = hashlib.sha256((json.dumps(config,sort_keys=True) + code.hexdigest() + str(metadata.stat().st_mtime_ns)).encode()).hexdigest()[:20]
        folder = Path(self.base_output)/'playback'/key/f'scene{scene:03d}'/f'frame{frame:05d}'
        manifest = folder/'frame.json'
        cached = json.loads(manifest.read_text()) if manifest.exists() else {}
        self.runner.update_config(dict(scene_index=scene, sample_index=frame, playback_mode=True, output_dir=str(folder)))
        self.scene, self.frame, self.timestamp = scene, frame, timestamp
        self.cache_hit = cached.get('completed_through',-1) >= target
        if self.cache_hit:
            # Artifacts can be viewed instantly, but no in-memory context was
            # restored. Manual Run To must still rebuild its dependencies.
            for stage in STAGES[:cached['completed_through']+1]:
                self.runner.states[stage.number].update(status='completed', progress=100, current_step='Cached frame artifacts; Run To rebuilds tensors')
            self.runner._refresh_visual(target)
            return
        self.runner.run_to(target)
        while self.runner.busy:
            if self.closed.wait(.05):
                self.runner.request_stop()
        state = self.runner.states[target]
        if state['status'] == 'failed':
            raise RuntimeError(state['error'])
        if state['status'] != 'completed':
            if self.pending is None:
                self.playing = False
            return
        folder.mkdir(parents=True,exist_ok=True)
        manifest.write_text(json.dumps(dict(scene=scene,frame=frame,timestamp=timestamp,completed_through=target)),encoding='utf-8')
