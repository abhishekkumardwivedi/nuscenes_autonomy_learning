"""Infrastructure routes kept separate from the educational stage API."""
from fastapi import APIRouter, HTTPException, WebSocket
from pydantic import BaseModel, Field
from dashboard.security import terminal_enabled


class PlaybackRequest(BaseModel):
    action: str
    scene: int | None = Field(default=None, ge=0)
    frame: int | None = None
    target_stage: int | None = Field(default=None, ge=0, le=20)
    speed: float | None = None


def infrastructure_routes(runner, monitor, playback, terminal):
    routes = APIRouter()

    @routes.get('/api/hardware')
    def hardware():
        return monitor.snapshot()

    @routes.get('/api/profiles')
    def profiles():
        return runner.stage_profiles()

    @routes.get('/api/stage/{number}/tensors')
    def tensors(number: int):
        if number not in runner.states:
            raise HTTPException(404, 'Unknown stage')
        with runner.lock:
            return list(runner.states[number]['tensors'].values())

    @routes.get('/api/scenes')
    def scenes():
        try:
            return playback.catalog.list(runner.cfg)
        except (OSError, ValueError) as exc:
            raise HTTPException(404, str(exc))

    @routes.get('/api/scenes/{scene}/frames')
    def frames(scene: int):
        try:
            return playback.catalog.frames(runner.cfg, scene)
        except (OSError, ValueError) as exc:
            raise HTTPException(404, str(exc))

    @routes.get('/api/playback')
    def playback_state():
        return playback.snapshot()

    @routes.post('/api/playback')
    def command(body: PlaybackRequest):
        try:
            return playback.command(**body.model_dump(exclude_none=True))
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc))

    @routes.get('/api/console')
    def console_status():
        return dict(enabled=terminal_enabled(),
                    message='Interactive bash PTY; sessions survive reconnects while this server runs.' if terminal_enabled()
                    else 'Console disabled: Linux and dashboard token authentication (or an explicitly trusted authenticated proxy) are required.')

    @routes.websocket('/ws/console')
    async def console(websocket: WebSocket):
        if not terminal_enabled():
            await websocket.close(code=1008, reason='Console requires authenticated access')
            return
        await terminal.connect(websocket)

    return routes
