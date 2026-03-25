from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

from ..config.schema import WebConfig, WebhookConfig
from ..logging import get_logger

if TYPE_CHECKING:
    from ..discord.client import LokiBot

log = get_logger("health")

SendMessageCallback = Callable[[str, str], Awaitable[None]]
TriggerCallback = Callable[[str, dict], Awaitable[int]]

# Paths that skip API token authentication
_AUTH_SKIP_PREFIXES = ("/health", "/webhook/", "/ui")

# Rate-limit: max requests per window per IP on /api/ routes
_RATE_LIMIT_MAX = 120
_RATE_LIMIT_WINDOW = 60  # seconds


def _make_auth_middleware(web_config: WebConfig) -> web.middleware:
    """Create middleware that enforces Bearer token auth on /api/ routes."""

    @web.middleware
    async def auth_middleware(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        path = request.path
        # Skip auth for non-API routes
        if not path.startswith("/api/"):
            return await handler(request)
        # Skip auth if no token configured (dev mode)
        token = web_config.api_token
        if not token:
            return await handler(request)
        # Check Bearer token
        auth_header = request.headers.get("Authorization", "")
        if auth_header == f"Bearer {token}":
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
    """Add security headers to all responses and catch malformed JSON."""

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
        return response

    return security_headers_middleware


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
        middlewares = []
        if self._web_config.enabled:
            middlewares.append(_make_security_headers_middleware())
            middlewares.append(_make_rate_limit_middleware())
            middlewares.append(_make_auth_middleware(self._web_config))
        self._app = web.Application(middlewares=middlewares)
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
                self._app.router.add_static("/ui", ui_dir, show_index=True)
                log.info("Serving web UI from %s", ui_dir)
        self._runner: web.AppRunner | None = None

    def set_ready(self, ready: bool = True) -> None:
        self._ready = ready

    def set_send_message(self, callback: SendMessageCallback) -> None:
        self._send_message = callback

    def set_trigger_callback(self, callback: TriggerCallback) -> None:
        """Set callback for webhook-triggered scheduler actions."""
        self._trigger_callback = callback

    def set_bot(self, bot: LokiBot) -> None:
        """Wire the bot instance to enable the REST API and WebSocket endpoints."""
        if not self._web_config.enabled:
            return
        from ..web.api import setup_api
        from ..web.websocket import setup_websocket
        setup_api(self._app, bot)
        self._ws_manager = setup_websocket(
            self._app, bot, api_token=self._web_config.api_token,
        )
        log.info("Web management API enabled")

    async def _redirect_to_ui(self, _request: web.Request) -> web.Response:
        """Redirect / to /ui/."""
        raise web.HTTPFound("/ui/")

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
