"""Shared HTTP/WebSocket access gate. A public proxy URL alone is not auth."""
import hashlib
import hmac
import json
import os
from pathlib import Path
from urllib.parse import parse_qs

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse


class DashboardAccess:
    def __init__(self, app):
        self.app = app
        self.token = os.getenv('DASHBOARD_TOKEN', '')
        token_file = os.getenv('DASHBOARD_TOKEN_FILE')
        if token_file:
            self.token = Path(token_file).read_text().strip()
        self.cookie = hmac.new(self.token.encode(), b'dashboard-session-v1', hashlib.sha256).hexdigest()

    async def __call__(self, scope, receive, send):
        if scope['type'] not in {'http', 'websocket'}:
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        host = headers.get(b'host', b'').decode()
        origin = headers.get(b'origin', b'').decode()
        # Same-origin mutations and WebSockets prevent cross-site shell control.
        from urllib.parse import urlsplit
        if origin and urlsplit(origin).netloc != host and (scope['type'] == 'websocket' or scope.get('method') not in {'GET', 'HEAD'}):
            if scope['type'] == 'websocket':
                return await send({'type': 'websocket.close', 'code': 1008})
            return await JSONResponse({'detail': 'Cross-origin request rejected'}, 403)(scope, receive, send)
        if not self.token:
            return await self.app(scope, receive, send)
        if scope['type'] == 'http' and scope['path'] == '/login':
            request = Request(scope, receive)
            if request.method == 'POST':
                body = await request.body()
                if len(body) > 4096:
                    return await JSONResponse({'detail': 'Invalid login'}, 400)(scope, receive, send)
                supplied = parse_qs(body.decode()).get('token', [''])[0]
                if hmac.compare_digest(supplied, self.token):
                    response = RedirectResponse('/', status_code=303)
                    secure = scope.get('scheme') == 'https' or headers.get(b'x-forwarded-proto') == b'https'
                    response.set_cookie('dashboard_session', self.cookie, httponly=True,
                                        secure=secure, samesite='strict', max_age=86400)
                    return await response(scope, receive, send)
                return await HTMLResponse('Invalid dashboard token. <a href="/login">Try again</a>', 401)(scope, receive, send)
            page = '''<!doctype html><meta name="viewport" content="width=device-width"><title>Dashboard sign in</title>
            <body style="background:#0b1020;color:#e7edf7;font:16px system-ui;padding:8vw">
            <h1>Autonomy dashboard</h1><p>Enter the token stored in your persistent dashboard token file.</p>
            <form method="post"><label>Dashboard token <input type="password" name="token" required autocomplete="current-password"></label>
            <button>Sign in</button></form></body>'''
            return await HTMLResponse(page)(scope, receive, send)
        from http.cookies import SimpleCookie
        cookies = SimpleCookie()
        try:
            cookies.load(headers.get(b'cookie', b'').decode())
        except Exception:
            pass
        cookie = cookies.get('dashboard_session')
        auth = headers.get(b'authorization', b'').decode()
        allowed = (cookie and hmac.compare_digest(cookie.value, self.cookie)) or hmac.compare_digest(auth, 'Bearer ' + self.token)
        if not allowed:
            if scope['type'] == 'websocket':
                return await send({'type': 'websocket.close', 'code': 1008})
            response = RedirectResponse('/login', 303) if scope['path'] == '/' else JSONResponse({'detail': 'Sign in at /login'}, 401)
            return await response(scope, receive, send)
        await self.app(scope, receive, send)


def terminal_enabled():
    protected = bool(os.getenv('DASHBOARD_TOKEN') or os.getenv('DASHBOARD_TOKEN_FILE'))
    # This opt-in is appropriate only if an actual authenticating reverse proxy
    # protects EVERY dashboard HTTP and WebSocket route, including direct access.
    protected = protected or os.getenv('DASHBOARD_TRUST_PROXY_AUTH') == '1'
    return os.name == 'posix' and protected and os.getenv('DASHBOARD_CONSOLE', '1') == '1'
