from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from aiohttp import web

from ..config.schema import WebConfig, WebhookConfig
from ..logging import get_logger

if TYPE_CHECKING:
    from ..discord.client import HeimdallBot

log = get_logger("health")

SendMessageCallback = Callable[[str, str], Awaitable[None]]
TriggerCallback = Callable[[str, dict], Awaitable[int]]

# Paths that skip API token authentication
_AUTH_SKIP_PREFIXES = ("/health", "/webhook/", "/ui")

# Exact API paths that skip token auth (login must be accessible unauthenticated)
_AUTH_SKIP_PATHS = frozenset({"/api/auth/login"})

# Rate-limit: max requests per window per IP on /api/ routes
_RATE_LIMIT_MAX = 120
_RATE_LIMIT_WINDOW = 60  # seconds

# Content-Security-Policy for the web UI
_CSP_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self' 'unsafe-eval' https://cdn.tailwindcss.com https://unpkg.com https://cdn.jsdelivr.net",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.tailwindcss.com",
    "font-src 'self' https://fonts.gstatic.com",
    "connect-src 'self' ws: wss:",
    "img-src 'self' data:",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
])


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------

class SessionManager:
    """Server-side session tracking with configurable timeout."""

    def __init__(self, timeout_minutes: int = 0) -> None:
        self._sessions: dict[str, float] = {}  # session_id -> last_activity (monotonic)
        self._timeout = timeout_minutes * 60 if timeout_minutes > 0 else 0

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    @property
    def timeout_seconds(self) -> int:
        return self._timeout

    def create(self) -> tuple[str, int]:
        """Create a new session. Returns (session_id, timeout_seconds)."""
        self.cleanup()
        sid = secrets.token_urlsafe(32)
        self._sessions[sid] = time.monotonic()
        return sid, self._timeout

    def validate(self, sid: str) -> bool:
        """Validate a session ID. Returns False if expired or unknown."""
        ts = self._sessions.get(sid)
        if ts is None:
            return False
        if self._timeout > 0 and (time.monotonic() - ts) > self._timeout:
            del self._sessions[sid]
            return False
        # Refresh activity timestamp
        self._sessions[sid] = time.monotonic()
        return True

    def destroy(self, sid: str) -> bool:
        """Destroy a session. Returns True if it existed."""
        return self._sessions.pop(sid, None) is not None

    def cleanup(self) -> int:
        """Remove expired sessions. Returns count removed."""
        if self._timeout <= 0:
            return 0
        now = time.monotonic()
        expired = [sid for sid, ts in self._sessions.items() if now - ts > self._timeout]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)


# ---------------------------------------------------------------------------
# Middleware factories
# ---------------------------------------------------------------------------

def _make_auth_middleware(
    web_config: WebConfig,
    session_manager: SessionManager,
) -> web.middleware:
    """Create middleware that enforces Bearer token auth on /api/ routes.

    Accepts either the raw api_token or a valid session token.
    """

    @web.middleware
    async def auth_middleware(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        path = request.path
        # Skip auth for non-API routes
        if not path.startswith("/api/"):
            return await handler(request)
        # Skip auth for login endpoint
        if path in _AUTH_SKIP_PATHS:
            return await handler(request)
        # Skip auth if no token configured (dev mode)
        token = web_config.api_token
        if not token:
            return await handler(request)

        # Extract bearer value from Authorization header
        auth_header = request.headers.get("Authorization", "")
        expected_bearer = f"Bearer {token}"
        if hmac.compare_digest(auth_header, expected_bearer):
            return await handler(request)
        # Check session tokens (Bearer <session_id>)
        bearer_prefix = "Bearer "
        if auth_header.startswith(bearer_prefix):
            bearer_value = auth_header[len(bearer_prefix):]
            if session_manager.validate(bearer_value):
                return await handler(request)

        # Check query param token (for downloads, WebSocket)
        query_token = request.query.get("token", "")
        if query_token:
            if hmac.compare_digest(query_token, token):
                return await handler(request)
            if session_manager.validate(query_token):
                return await handler(request)

        return web.json_response({"error": "unauthorized"}, status=401)

    return auth_middleware


def _make_rate_limit_middleware() -> web.middleware:
    """Simple in-memory rate limiter for /api/ routes (per remote IP)."""
    # {ip: [(timestamp, ...),]}
    _buckets: dict[str, list[float]] = defaultdict(list)

    @web.middleware
    async def rate_limit_middleware(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        if not request.path.startswith("/api/"):
            return await handler(request)
        ip = request.remote or "unknown"
        now = time.monotonic()
        window_start = now - _RATE_LIMIT_WINDOW
        # Prune old entries
        bucket = _buckets[ip]
        _buckets[ip] = bucket = [t for t in bucket if t > window_start]
        if len(bucket) >= _RATE_LIMIT_MAX:
            return web.json_response({"error": "rate limit exceeded"}, status=429)
        bucket.append(now)
        return await handler(request)

    return rate_limit_middleware


def _make_security_headers_middleware() -> web.middleware:
    """Add security headers (CSP, X-Frame-Options, etc.) to all responses."""

    @web.middleware
    async def security_headers_middleware(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        try:
            response = await handler(request)
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = _CSP_POLICY
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    return security_headers_middleware


def _make_csrf_middleware() -> web.middleware:
    """Validate Origin/Referer on state-changing requests (defense-in-depth).

    Bearer tokens already prevent CSRF (not auto-sent by browsers), but this
    adds an extra layer by rejecting cross-origin POST/PUT/DELETE requests.
    """

    @web.middleware
    async def csrf_middleware(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        # Only check state-changing methods
        if request.method not in ("POST", "PUT", "DELETE"):
            return await handler(request)
        # Only check API routes
        if not request.path.startswith("/api/"):
            return await handler(request)
        # Skip for login endpoint
        if request.path in _AUTH_SKIP_PATHS:
            return await handler(request)
        # Skip for webhook endpoints (they have their own auth)
        if request.path.startswith("/webhook/"):
            return await handler(request)

        host = request.host  # includes port
        origin = request.headers.get("Origin", "")
        referer = request.headers.get("Referer", "")

        if origin:
            parsed = urlparse(origin)
            if parsed.netloc and parsed.netloc != host:
                log.warning("CSRF blocked: Origin %s != Host %s on %s %s",
                            origin, host, request.method, request.path)
                return web.json_response({"error": "cross-origin request blocked"}, status=403)
        elif referer:
            parsed = urlparse(referer)
            if parsed.netloc and parsed.netloc != host:
                log.warning("CSRF blocked: Referer %s != Host %s on %s %s",
                            referer, host, request.method, request.path)
                return web.json_response({"error": "cross-origin request blocked"}, status=403)
        # If neither Origin nor Referer is present, allow — Bearer token
        # already prevents CSRF since it's not auto-sent by browsers.

        return await handler(request)

    return csrf_middleware


def _make_web_audit_middleware() -> web.middleware:
    """Log state-changing API requests to the audit log."""

    @web.middleware
    async def web_audit_middleware(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        # Only audit state-changing API requests
        if request.method not in ("POST", "PUT", "DELETE") or not request.path.startswith("/api/"):
            return await handler(request)

        start = time.monotonic()
        response = await handler(request)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Fire-and-forget audit log (don't block the response)
        audit = request.app.get("audit_logger")
        if audit:
            try:
                await audit.log_web_action(
                    method=request.method,
                    path=request.path,
                    status=response.status,
                    ip=request.remote or "",
                    execution_time_ms=elapsed_ms,
                )
            except Exception:
                pass  # Never block the response for audit failures

        return response

    return web_audit_middleware


# ---------------------------------------------------------------------------
# Health server
# ---------------------------------------------------------------------------

class HealthServer:
    def __init__(
        self,
        port: int = 3000,
        webhook_config: WebhookConfig | None = None,
        web_config: WebConfig | None = None,
    ) -> None:
        self.port = port
        self._ready = False
        self._webhook_config = webhook_config or WebhookConfig()
        self._web_config = web_config or WebConfig()
        self._send_message: SendMessageCallback | None = None
        self._trigger_callback: TriggerCallback | None = None

        # Session management
        self._session_manager = SessionManager(
            timeout_minutes=self._web_config.session_timeout_minutes,
        )

        middlewares = []
        if self._web_config.enabled:
            middlewares.append(_make_security_headers_middleware())
            middlewares.append(_make_rate_limit_middleware())
            middlewares.append(_make_csrf_middleware())
            middlewares.append(_make_auth_middleware(self._web_config, self._session_manager))
            middlewares.append(_make_web_audit_middleware())
        self._app = web.Application(middlewares=middlewares)
        # Store session_manager on app for access by API routes
        self._app["session_manager"] = self._session_manager
        self._app.router.add_get("/health", self._health)
        if self._webhook_config.enabled:
            self._app.router.add_post("/webhook/gitea", self._webhook_gitea)
            self._app.router.add_post("/webhook/grafana", self._webhook_grafana)
            self._app.router.add_post("/webhook/generic", self._webhook_generic)
            log.info("Webhook endpoints enabled")
        # Serve static UI files if the directory exists
        if self._web_config.enabled:
            ui_dir = Path(__file__).resolve().parent.parent.parent / "ui"
            if ui_dir.is_dir():
                self._app.router.add_get("/", self._redirect_to_ui)
                self._ui_dir = ui_dir
                # Serve static files with a fallback to index.html for SPA routing
                self._app.router.add_get("/ui/{path:.*}", self._serve_ui_file)
                self._app.router.add_get("/ui", self._redirect_to_ui)
                log.info("Serving web UI from %s", ui_dir)
        self._runner: web.AppRunner | None = None

    def set_ready(self, ready: bool = True) -> None:
        self._ready = ready

    def set_send_message(self, callback: SendMessageCallback) -> None:
        self._send_message = callback

    def set_trigger_callback(self, callback: TriggerCallback) -> None:
        """Set callback for webhook-triggered scheduler actions."""
        self._trigger_callback = callback

    def set_bot(self, bot: HeimdallBot) -> None:
        """Wire the bot instance to enable the REST API and WebSocket endpoints."""
        if not self._web_config.enabled:
            return
        from ..web.api import setup_api
        from ..web.websocket import setup_websocket
        setup_api(self._app, bot)
        self._ws_manager = setup_websocket(
            self._app, bot, api_token=self._web_config.api_token,
        )
        # Wire audit events to WebSocket for live dashboard/log updates
        ws_mgr = self._ws_manager
        bot.audit.set_event_callback(ws_mgr.broadcast_event)
        # Store audit logger on app for the web audit middleware
        self._app["audit_logger"] = bot.audit
        log.info("Web management API enabled")

    async def _redirect_to_ui(self, _request: web.Request) -> web.Response:
        """Redirect / to /ui/."""
        raise web.HTTPFound("/ui/")

    async def _serve_ui_file(self, request: web.Request) -> web.Response:
        """Serve static UI files, defaulting to index.html for SPA routing."""
        path = request.match_info.get("path", "")
        if not path or path == "/":
            return web.FileResponse(self._ui_dir / "index.html")
        file = (self._ui_dir / path).resolve()
        # Prevent path traversal
        if not str(file).startswith(str(self._ui_dir.resolve())):
            raise web.HTTPForbidden()
        if file.is_file():
            return web.FileResponse(file)
        # SPA fallback — serve index.html for unmatched routes
        return web.FileResponse(self._ui_dir / "index.html")

    async def start(self) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()
        log.info("Health server listening on port %d", self.port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _health(self, _request: web.Request) -> web.Response:
        if self._ready:
            return web.json_response({"status": "ok"})
        return web.json_response({"status": "starting"}, status=503)

    def _verify_hmac_sha256(self, body: bytes, signature: str) -> bool:
        """Verify HMAC-SHA256 signature against webhook secret.

        Returns False (reject) when no secret is configured — webhooks
        should not be accepted without authentication.
        """
        secret = self._webhook_config.secret
        if not secret:
            log.warning("Webhook rejected: no secret configured for HMAC verification")
            return False
        expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    def _get_channel_id(self, source: str) -> str | None:
        """Get the channel ID for a webhook source."""
        if source == "gitea" and self._webhook_config.gitea_channel_id:
            return self._webhook_config.gitea_channel_id
        if source == "grafana" and self._webhook_config.grafana_channel_id:
            return self._webhook_config.grafana_channel_id
        return self._webhook_config.channel_id or None

    async def _notify_triggers(self, source: str, event_data: dict) -> None:
        """Notify the scheduler about an incoming webhook for trigger matching."""
        if not self._trigger_callback:
            return
        try:
            fired = await self._trigger_callback(source, event_data)
            if fired:
                log.info("Webhook %s fired %d trigger(s)", source, fired)
        except Exception as e:
            log.error("Trigger callback failed for %s: %s", source, e)

    async def _send(self, source: str, text: str) -> web.Response:
        channel_id = self._get_channel_id(source)
        if not channel_id:
            log.warning("Webhook %s: no channel_id configured", source)
            return web.json_response({"error": "no channel configured"}, status=500)
        if not self._send_message:
            log.warning("Webhook %s: no send_message callback", source)
            return web.json_response({"error": "bot not ready"}, status=503)

        try:
            await self._send_message(channel_id, text)
            return web.json_response({"status": "delivered"})
        except Exception as e:
            log.error("Webhook %s delivery failed: %s", source, e)
            return web.json_response({"error": str(e)}, status=500)

    async def _webhook_gitea(self, request: web.Request) -> web.Response:
        body = await request.read()
        signature = request.headers.get("X-Gitea-Signature", "")
        if not self._verify_hmac_sha256(body, signature):
            return web.json_response({"error": "invalid signature"}, status=403)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        event = request.headers.get("X-Gitea-Event", "unknown")
        repo = data.get("repository", {}).get("full_name", "unknown")

        if event == "push":
            pusher = data.get("pusher", {}).get("login", "unknown")
            commits = data.get("commits", [])
            ref = data.get("ref", "").replace("refs/heads/", "")
            commit_lines = []
            for c in commits[:5]:
                msg = c.get("message", "").split("\n")[0][:80]
                commit_lines.append(f"  \u2022 `{c.get('id', '')[:7]}` {msg}")
            commits_text = "\n".join(commit_lines)
            text = f"**Gitea Push** \u2014 `{repo}` (`{ref}`)\nBy: {pusher} | {len(commits)} commit(s)\n{commits_text}"

        elif event in ("pull_request", "pull_request_approved", "pull_request_rejected"):
            pr = data.get("pull_request", {})
            action = data.get("action", "")
            title = pr.get("title", "")
            user = pr.get("user", {}).get("login", "unknown")
            text = f"**Gitea PR** \u2014 `{repo}`\n{action}: **{title}** by {user}"

        elif event == "issues":
            issue = data.get("issue", {})
            action = data.get("action", "")
            title = issue.get("title", "")
            user = data.get("sender", {}).get("login", "unknown")
            text = f"**Gitea Issue** \u2014 `{repo}`\n{action}: **{title}** by {user}"

        else:
            text = f"**Gitea** \u2014 `{repo}` \u2014 event: `{event}`"

        await self._notify_triggers("gitea", {"event": event, "repo": repo})
        return await self._send("gitea", text)

    async def _webhook_grafana(self, request: web.Request) -> web.Response:
        body = await request.read()

        # Authenticate via shared secret header (configure in Grafana contact point)
        if self._webhook_config.secret:
            secret_header = request.headers.get("X-Webhook-Secret", "")
            if not hmac.compare_digest(secret_header, self._webhook_config.secret):
                return web.json_response({"error": "invalid secret"}, status=403)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        # Grafana unified alerting format
        alerts = data.get("alerts", [])
        if not alerts:
            title = data.get("title", data.get("ruleName", "Grafana Alert"))
            message = data.get("message", data.get("state", ""))
            text = f"**Grafana Alert** \u2014 {title}\n{message}"
        else:
            lines = []
            for alert in alerts[:10]:
                status = alert.get("status", "unknown")
                labels = alert.get("labels", {})
                name = labels.get("alertname", alert.get("alertname", "Unknown"))
                instance = labels.get("instance", "")
                emoji = "\U0001f534" if status == "firing" else "\U0001f7e2"
                line = f"{emoji} **{name}** ({status})"
                if instance:
                    line += f" \u2014 `{instance}`"
                annotations = alert.get("annotations", {})
                summary = annotations.get("summary", annotations.get("description", ""))
                if summary:
                    line += f"\n  {summary[:200]}"
                lines.append(line)
            text = f"**Grafana Alerts** ({len(alerts)} alert(s)):\n" + "\n".join(lines)

        # Build event data for trigger matching — use first alert's name if available
        alert_name = ""
        if alerts:
            labels = alerts[0].get("labels", {})
            alert_name = labels.get("alertname", alerts[0].get("alertname", ""))
        else:
            alert_name = data.get("ruleName", data.get("title", ""))
        event_data: dict = {"event": "alert", "alert_name": alert_name}
        await self._notify_triggers("grafana", event_data)
        return await self._send("grafana", text)

    async def _webhook_generic(self, request: web.Request) -> web.Response:
        body = await request.read()
        secret_header = request.headers.get("X-Webhook-Secret", "")
        if self._webhook_config.secret and not hmac.compare_digest(
            secret_header, self._webhook_config.secret
        ):
            return web.json_response({"error": "invalid secret"}, status=403)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        title = data.get("title", "Webhook")
        message = data.get("message", "")
        text = f"**{title}**\n{message}" if message else f"**{title}**"

        event_data_generic: dict = {
            "event": data.get("event", "generic"),
            "title": title,
        }
        await self._notify_triggers("generic", event_data_generic)
        return await self._send("generic", text)
