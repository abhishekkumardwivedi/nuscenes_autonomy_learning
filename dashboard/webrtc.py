from __future__ import annotations

import os
from fractions import Fraction
import time

AVAILABLE = True
_IMPORT_ERROR = None
try:
    import numpy as np
    from av import VideoFrame
    from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
except Exception as exc:  # dashboard remains usable without WebRTC packages
    AVAILABLE = False
    _IMPORT_ERROR = exc


PEERS = set()


def availability() -> tuple[bool, str | None]:
    return AVAILABLE, None if AVAILABLE else str(_IMPORT_ERROR)


def _ice_servers():
    if not AVAILABLE:
        return []
    servers = []
    turn_url = os.getenv("TURN_URL", "").strip()
    stun_url = os.getenv("STUN_URL", "stun:stun.l.google.com:19302").strip()
    if stun_url:
        servers.append(RTCIceServer(urls=[stun_url]))
    if turn_url:
        servers.append(
            RTCIceServer(
                urls=[turn_url],
                username=os.getenv("TURN_USERNAME", ""),
                credential=os.getenv("TURN_PASSWORD", ""),
            )
        )
    return servers


if AVAILABLE:
    class DashboardVideoTrack(VideoStreamTrack):
        """Continuously streams the latest stage visualization as a video track."""

        def __init__(self, visual_hub, fps: int = 5):
            super().__init__()
            self.visual_hub = visual_hub
            self.fps = max(1, int(fps))
            self._pts = 0
            self._time_base = Fraction(1, 90000)

        async def recv(self):
            import asyncio
            await asyncio.sleep(1.0 / self.fps)
            self._pts += int(90000 / self.fps)
            rgb = np.asarray(self.visual_hub.image_rgb())
            frame = VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = self._pts
            frame.time_base = self._time_base
            return frame


async def handle_offer(sdp: str, offer_type: str, visual_hub) -> dict:
    if not AVAILABLE:
        raise RuntimeError(f"WebRTC packages unavailable: {_IMPORT_ERROR}")

    config = RTCConfiguration(iceServers=_ice_servers())
    pc = RTCPeerConnection(configuration=config)
    PEERS.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            if pc.connectionState == "failed":
                await pc.close()
            PEERS.discard(pc)

    pc.addTrack(DashboardVideoTrack(visual_hub))
    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=offer_type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


async def close_all() -> None:
    peers = list(PEERS)
    PEERS.clear()
    for pc in peers:
        try:
            await pc.close()
        except Exception:
            pass
