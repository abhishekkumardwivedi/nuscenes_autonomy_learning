import json
import os
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from config import PipelineConfig
from dashboard.runner import PipelineRunner
from dashboard.visual_hub import VisualHub
from dashboard.hardware import HardwareMonitor
from dashboard.playback import Playback, SceneCatalog
from utils.tensor_utils import tensor_info


class Bus:
    def __init__(self): self.events=[]
    def emit(self, event): self.events.append(event)


def wait(runner):
    deadline=time.monotonic()+30
    while runner.busy and time.monotonic()<deadline: time.sleep(.01)
    assert not runner.busy, 'Worker did not stop'


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.cfg=PipelineConfig(dataroot='/missing-dataset-for-test', output_dir=self.temp.name,device='cpu',verbose=0,save_plots=False)
        self.runner=PipelineRunner(self.cfg,Bus(),VisualHub())
    def tearDown(self): self.temp.cleanup()

    def test_foundation_profile_and_missing_dataset(self):
        self.runner.run_to(0); wait(self.runner)
        self.assertEqual(self.runner.states[0]['status'],'completed')
        path=Path(self.temp.name)/'stage00_foundation/profile.json'
        profile=json.loads(path.read_text())
        self.assertGreater(profile['elapsed_ms'],0)
        self.assertIsNone(profile['torch_after']['allocated']) if not torch.cuda.is_available() else None
        self.runner.run_to(1); wait(self.runner)
        self.assertEqual(self.runner.states[0]['status'],'completed')
        self.assertEqual(self.runner.states[1]['status'],'failed')
        self.assertIn('FileNotFoundError',self.runner.states[1]['logs'][-1]['message'])

    def test_cancel_and_stale_dependencies(self):
        entered=threading.Event()
        def run(ctx,cfg,log):
            log.stage(0,'Test')
            entered.set()
            while True:
                time.sleep(.01)
                log.substage(0,1,'Working')
        with patch('dashboard.runner.importlib.import_module',return_value=types.SimpleNamespace(run=run)):
            self.runner.run_to(0)
            self.assertTrue(entered.wait(5))
            self.runner.request_stop(); wait(self.runner)
        self.assertEqual(self.runner.states[0]['status'],'cancelled')
        def quick(ctx,cfg,log): ctx.set('test',torch.zeros(10))
        with patch('dashboard.runner.importlib.import_module',return_value=types.SimpleNamespace(run=quick)):
            self.runner.run_to(2); wait(self.runner)
            self.runner.run_stage(0); wait(self.runner)
            self.assertEqual(self.runner.states[2]['status'],'stale')
            with self.assertRaisesRegex(RuntimeError,'Upstream'):
                self.runner.run_stage(1)
            self.runner.run_to(2); wait(self.runner)
            self.assertEqual(self.runner.states[2]['status'],'completed')

    def test_tensor_metadata_bounded(self):
        x=torch.ones(1_000_000,requires_grad=True)
        basic=tensor_info(x,'basic')
        self.assertEqual(basic['memory_bytes'],4_000_000)
        self.assertTrue(basic['requires_grad'])
        self.assertIsNone(basic['mean'])
        learning=tensor_info(x,'learning')
        self.assertLessEqual(learning['sampled_elements'],65536)
        self.assertEqual(learning['mean'],1)
        self.assertEqual(tensor_info(np.array([float('nan')]),'learning')['mean'],None)
        if torch.cuda.is_available():
            result=tensor_info(x.cuda(),'learning')
            self.assertTrue(result['device'].startswith('cuda'))
            self.assertEqual(result['mean'],1)

    def test_hardware_without_nvml(self):
        hw=HardwareMonitor(self.temp.name)
        hw.nvml=None
        sample=hw.sample()
        self.assertEqual(sample['gpus'],[])
        self.assertIn('available',sample['ram'])
        self.assertIsNone(sample['network']['rx_bytes_sec'])
        self.assertGreater(sample['storage']['total'],0)

    def test_playback_cache_and_missing_scene(self):
        catalog=SceneCatalog()
        with self.assertRaisesRegex(FileNotFoundError,'metadata missing'):
            catalog.list(self.cfg)
        root=Path(self.temp.name)/'data/v1.0-mini';root.mkdir(parents=True)
        (root/'scene.json').write_text(json.dumps([dict(name='test',description='test',nbr_samples=2,first_sample_token='a')]))
        (root/'sample.json').write_text(json.dumps([dict(token='a',next='b',timestamp=0),dict(token='b',next='',timestamp=500000)]))
        self.cfg.dataroot=str(root.parent)
        playback=Playback(self.runner,catalog)
        self.assertEqual(len(catalog.frames(self.cfg,0)),2)
        playback.command('seek',scene=0,frame=1,target_stage=0)
        deadline=time.time()+15
        while playback.busy and time.time()<deadline: time.sleep(.02)
        self.assertIsNone(playback.error)
        self.assertEqual(playback.frame,1)
        self.assertTrue((self.cfg.output_path/'frame.json').exists())
        playback.command('seek',scene=0,frame=1,target_stage=0)
        while playback.busy and time.time()<deadline: time.sleep(.02)
        self.assertTrue(playback.cache_hit)
        self.assertEqual(self.runner.completed_through,-1)  # no fabricated tensor cache
        playback.close()


class AccessTests(unittest.TestCase):
    def test_shared_http_and_websocket_gate(self):
        from fastapi import FastAPI, WebSocket
        from fastapi.testclient import TestClient
        from dashboard.security import DashboardAccess
        app=FastAPI()
        @app.get('/api/health')
        def health(): return {'ok':True}
        @app.websocket('/ws')
        async def ws(socket:WebSocket):
            await socket.accept(); await socket.send_text('ok'); await socket.close()
        with patch.dict(os.environ,{'DASHBOARD_TOKEN':'test-token','DASHBOARD_TOKEN_FILE':''}):
            app.add_middleware(DashboardAccess)
            with TestClient(app) as client:
                self.assertEqual(client.get('/api/health').status_code,401)
                self.assertEqual(client.post('/login',content='token=wrong').status_code,401)
                self.assertEqual(client.post('/login',content='token=test-token',follow_redirects=False).status_code,303)
                self.assertEqual(client.get('/api/health').status_code,200)
                with client.websocket_connect('/ws') as socket: self.assertEqual(socket.receive_text(),'ok')
                self.assertEqual(client.post('/login',content='token=test-token',headers={'Origin':'https://evil.invalid'}).status_code,403)

    @unittest.skipUnless(os.name=='posix','PTY requires Linux')
    def test_pty_reconnect_resize_interrupt(self):
        from dashboard.terminal import TerminalSession
        session=TerminalSession(tempfile.gettempdir())
        try:
            session.resize(100,30)
            os.write(session.fd,b'printf "PTY_OK\\n"; stty size\n')
            deadline=time.time()+5
            while time.time()<deadline:
                with session.lock: output=''.join(text for _,text in session.chunks)
                if '30 100' in output: break
                time.sleep(.02)
            self.assertIn('PTY_OK',output)
            self.assertIn('30 100',output)
            os.write(session.fd,b'sleep 30\n'); time.sleep(.1)
            os.write(session.fd,b'\x03'); os.write(session.fd,b'echo INTERRUPTED_OK\n')
            time.sleep(.2)
            with session.lock: output=''.join(text for _,text in session.chunks)
            self.assertIn('INTERRUPTED_OK',output)
            self.assertFalse(session.closed)
        finally: session.close()


if __name__=='__main__': unittest.main()
