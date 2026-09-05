"""Opt-in live checks. Executes offline stages/playback; never prints credentials.

Run on the Pod with its token file; LIVE_BASE_URL optionally selects the HTTPS
proxy. The browser still needs a separate visual layout check.
"""
import asyncio
import json
import os
from pathlib import Path
import time
import urllib.request

import websockets

BASE = os.getenv('LIVE_BASE_URL', 'http://127.0.0.1:8888').rstrip('/')
TOKEN = Path(os.getenv('DASHBOARD_TOKEN_FILE', '/workspace/.autonomy/dashboard.token')).read_text().strip()
HEADERS = {'Authorization': 'Bearer '+TOKEN, 'Content-Type':'application/json'}


def request(path, data=None):
    req=urllib.request.Request(BASE+path, data=json.dumps(data).encode() if data is not None else None, headers=HEADERS)
    with urllib.request.urlopen(req,timeout=20) as response:
        body=response.read()
        return json.loads(body) if 'application/json' in response.headers.get('Content-Type','') else body


def idle(timeout=180):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        state=request('/api/state')
        if not state['busy'] and not state['playback']['processing']:
            return state
        assert request('/api/health')['ok']
        time.sleep(.25)
    raise AssertionError('Pipeline did not become idle')


async def socket_checks():
    base=BASE.replace('https:','wss:').replace('http:','ws:')
    async with websockets.connect(base+'/ws', additional_headers=HEADERS) as ws:
        assert json.loads(await ws.recv())['type']=='hello'
        await ws.send('ping')
        assert json.loads(await ws.recv())['type']=='pong'
    async def output_until(ws, text):
        output=''
        for _ in range(150):
            event=json.loads(await asyncio.wait_for(ws.recv(),10))
            output+=event.get('data','')
            if text in output: return
        raise AssertionError('Expected console output missing')
    async with websockets.connect(base+'/ws/console',additional_headers=HEADERS) as ws:
        identity=json.loads(await ws.recv())['id']
        await ws.send(json.dumps(dict(type='resize',cols=110,rows=32)))
        await ws.send(json.dumps(dict(type='input',data='export DASHBOARD_RECONNECT_TEST=ok; stty size\n')))
        await output_until(ws,'32 110')
    async with websockets.connect(base+'/ws/console?session='+identity,additional_headers=HEADERS) as ws:
        assert json.loads(await ws.recv())['id']==identity
        await ws.send(json.dumps(dict(type='input',data='printf "RECONNECTED:%s\\n" "$DASHBOARD_RECONNECT_TEST"\n')))
        await output_until(ws,'RECONNECTED:ok')
        await ws.send(json.dumps(dict(type='input',data='sleep 30\n')))
        await asyncio.sleep(.1)
        await ws.send(json.dumps(dict(type='input',data='\x03echo CTRL_C_OK\n')))
        await output_until(ws,'CTRL_C_OK')
        await ws.send(json.dumps(dict(type='input',data='exit\n')))
    print('Live WSS and console PTY reconnect/resize/Ctrl+C passed',flush=True)


def main():
    assert request('/api/health')['ok']
    assert len(request('/api/stages'))==21
    assert b'console' in request('/')
    assert 'content' in request('/api/stage/9/code')
    first=request('/api/hardware')['latest']
    time.sleep(1.2)
    latest=request('/api/hardware')['latest']
    assert latest['timestamp']>first['timestamp']
    assert latest['torch']['cuda_available']
    print('Live API, hardware updates and CUDA reporting passed',flush=True)
    asyncio.run(socket_checks())
    request('/api/run',dict(target_stage=0,mode='run_to'))
    assert idle()['stages'][0]['status']=='completed'
    assert request('/api/profiles')[0]['elapsed_ms']>0
    scenes=request('/api/scenes'); assert scenes
    frames=request('/api/scenes/0/frames'); assert len(frames)>1
    for frame in [0,1]:
        request('/api/playback',dict(action='seek',scene=0,frame=frame,target_stage=2))
        state=idle()
        assert not state['playback']['error'],state['playback']['error']
        assert state['stages'][2]['status']=='completed',state['stages'][2]
        assert state['config']['sample_index']==frame
        artifacts=request('/api/stage/2/artifacts')
        image=next(x for x in artifacts if x['kind']=='image')
        from urllib.parse import quote
        assert request('/api/artifact?path='+quote(image['relative_path']))
    request('/api/playback',dict(action='seek',scene=0,frame=0,target_stage=2))
    state=idle()
    assert state['playback']['cache_hit']
    print('Live scene seek/next-frame, saved images, profiles and frame cache passed',flush=True)


if __name__=='__main__': main()
