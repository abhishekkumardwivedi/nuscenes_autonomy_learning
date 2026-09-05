"""Release the dashboard port only from Jupyter; never kill arbitrary services."""
import os
import time
import psutil

port = int(os.getenv('DASHBOARD_PORT', '8888'))
pids = {c.pid for c in psutil.net_connections(kind='inet') if c.status == 'LISTEN' and c.laddr.port == port and c.pid}
for pid in pids:
    process = psutil.Process(pid)
    command = ' '.join(process.cmdline())
    if 'jupyter' not in command.lower():
        raise SystemExit(f'Port {port} is occupied by PID {pid} ({process.name()}). Stop it deliberately or choose another port.')
    print(f'Stopping Jupyter PID {pid} to reserve dashboard port {port}; Jupyter remains installed.')
    process.terminate()
    try:
        process.wait(timeout=10)
    except psutil.TimeoutExpired:
        raise SystemExit('Jupyter did not stop. Disable the image startup supervisor for Jupyter and retry.')
