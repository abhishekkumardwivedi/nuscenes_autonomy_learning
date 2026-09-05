"""Run the autonomy learning dashboard.

Usage:
    python dashboard_server.py
    python dashboard_server.py --host 0.0.0.0 --port 8888
"""
from __future__ import annotations

import argparse
import os
import uvicorn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.getenv("DASHBOARD_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.getenv("DASHBOARD_PORT", "8888")))
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()
    uvicorn.run("dashboard.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
