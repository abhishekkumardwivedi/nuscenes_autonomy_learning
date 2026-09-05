"""Linux PTY sessions on the dashboard WebSocket; no additional listening port.

Output is drained even while disconnected. A bounded byte replay buffer lets a
browser reconnect without restarting the shell. Sessions expire after inactivity.
"""
import asyncio
import codecs
import os
import secrets
import signal
import subprocess
import sys
import anyio
import threading
import time
from collections import deque
from pathlib import Path


class TerminalSession:
    def __init__(self, cwd):
        import pty
        self.id = secrets.token_urlsafe(24)
        self.lock = threading.Lock()
        self.chunks = deque()
        self.sequence = 0
        self.size = 0
        self.last_seen = time.monotonic()
        self.closed = False
        # Use a fresh child interpreter to acquire the controlling terminal.
        # No Python code runs between fork and exec in the multithreaded server.
        master, slave = pty.openpty()
        env = dict(os.environ, TERM='xterm-256color', COLORTERM='truecolor')
        env.pop('DASHBOARD_TOKEN', None)
        env['PATH'] = str(Path(sys.prefix)/'bin') + os.pathsep + env.get('PATH','')
        env['HISTFILE'] = str(Path(os.getenv('PERSISTENT_ROOT', cwd))/'.autonomy_console_history')
        helper = "import fcntl,termios,os; fcntl.ioctl(0,termios.TIOCSCTTY,0); os.execvpe('/bin/bash',['/bin/bash','--noprofile','--norc','-i'],os.environ)"
        self.process = subprocess.Popen([sys.executable,'-c',helper], stdin=slave, stdout=slave, stderr=slave,
                                        cwd=cwd, env=env, start_new_session=True, close_fds=True)
        os.close(slave)
        self.pid, self.fd = self.process.pid, master
        self.reader = threading.Thread(target=self._read, daemon=True, name='console-reader')
        self.reader.start()

    def _read(self):
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        try:
            while True:
                data = os.read(self.fd, 8192)
                if not data:
                    break
                text = decoder.decode(data)
                with self.lock:
                    self.sequence += 1
                    self.chunks.append((self.sequence, text))
                    self.size += len(text)
                    while self.size > 512_000:
                        self.size -= len(self.chunks.popleft()[1])
        except OSError:
            pass
        finally:
            self.closed = True
            try:
                self.process.wait()
            except ChildProcessError:
                pass

    def resize(self, cols, rows):
        import fcntl
        import struct
        import termios
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack('HHHH', max(2,min(200,rows)), max(10,min(400,cols)), 0, 0))

    def close(self):
        try:
            if self.process.poll() is None:
                os.killpg(self.pid, signal.SIGHUP)
        except ProcessLookupError:
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass
        self.closed = True


class TerminalManager:
    def __init__(self, cwd):
        self.cwd = str(cwd)
        self.sessions = {}

    def close(self):
        for session in self.sessions.values():
            session.close()
        self.sessions.clear()

    async def connect(self, websocket):
        for key, session in list(self.sessions.items()):
            if session.closed or time.monotonic() - session.last_seen > 3600:
                session.close()
                del self.sessions[key]
        session = self.sessions.get(websocket.query_params.get('session', ''))
        if session is None:
            if len(self.sessions) >= 4:
                await websocket.close(code=1013, reason='At most four consoles; close unused shells with exit.')
                return
            session = TerminalSession(self.cwd)
            self.sessions[session.id] = session
        await websocket.accept()
        await websocket.send_json({'type': 'session', 'id': session.id})
        # Each reconnect resets xterm and replays the bounded terminal history.
        cursor = 0
        async def output():
            nonlocal cursor
            while True:
                with session.lock:
                    pending = [(seq, text) for seq, text in session.chunks if seq > cursor]
                if pending:
                    await websocket.send_json({'type': 'output', 'data': ''.join(text for _,text in pending)})
                    cursor = pending[-1][0]
                if session.closed:
                    await websocket.send_json({'type': 'exit'})
                    return
                session.last_seen = time.monotonic()
                await asyncio.sleep(.04)
        async def input_loop():
            while True:
                msg = await websocket.receive_json()
                session.last_seen = time.monotonic()
                if msg.get('type') == 'input':
                    data = str(msg.get('data', ''))[:16384].encode()
                    await asyncio.to_thread(os.write, session.fd, data)
                elif msg.get('type') == 'resize':
                    session.resize(int(msg.get('cols',80)), int(msg.get('rows',24)))
        tasks = [asyncio.create_task(output()), asyncio.create_task(input_loop())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (Exception, asyncio.CancelledError):
            pass
        finally:
            for task in tasks:
                task.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(*tasks, return_exceptions=True)
                try:
                    await websocket.close()
                except (RuntimeError, OSError):
                    pass
