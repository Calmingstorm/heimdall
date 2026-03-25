"""WebSocket handler for live log and event streaming.

Endpoint: /api/ws
- Client sends: {"subscribe": "logs"} or {"subscribe": "events"}
- Server sends: {"type": "log", "line": "..."} or {"type": "event", ...}
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
from aiohttp import web

from ..logging import get_logger

if TYPE_CHECKING:
    from ..discord.client import LokiBot

log = get_logger("web.ws")

# How many lines to send from the end of the log when a client first subscribes
_LOG_TAIL_LINES = 50
# Poll interval for checking new log lines
_LOG_POLL_INTERVAL = 1.0


class WebSocketManager:
    """Manages WebSocket connections and broadcasts events."""

    def __init__(self, bot: LokiBot) -> None:
        self._bot = bot
        self._clients: set[web.WebSocketResponse] = set()
        self._log_subscribers: set[web.WebSocketResponse] = set()
        self._event_subscribers: set[web.WebSocketResponse] = set()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        """Handle a WebSocket connection at /api/ws."""
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)
        self._clients.add(ws)
        log.info("WebSocket client connected (%d total)", len(self._clients))

        log_task: asyncio.Task | None = None

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await ws.send_json({"error": "invalid JSON"})
                        continue

                    sub = data.get("subscribe")
                    unsub = data.get("unsubscribe")

                    if sub == "logs":
                        self._log_subscribers.add(ws)
                        # Start tailing the log file for this client
                        if log_task is None or log_task.done():
                            log_task = asyncio.create_task(
                                self._tail_logs(ws)
                            )
                        await ws.send_json({"type": "subscribed", "channel": "logs"})
                    elif sub == "events":
                        self._event_subscribers.add(ws)
                        await ws.send_json({"type": "subscribed", "channel": "events"})
                    elif unsub == "logs":
                        self._log_subscribers.discard(ws)
                        await ws.send_json({"type": "unsubscribed", "channel": "logs"})
                    elif unsub == "events":
                        self._event_subscribers.discard(ws)
                        await ws.send_json({"type": "unsubscribed", "channel": "events"})
                    else:
                        await ws.send_json({"error": "unknown command"})

                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                    break
        finally:
            if log_task and not log_task.done():
                log_task.cancel()
                try:
                    await log_task
                except asyncio.CancelledError:
                    pass
            self._clients.discard(ws)
            self._log_subscribers.discard(ws)
            self._event_subscribers.discard(ws)
            log.info("WebSocket client disconnected (%d remaining)", len(self._clients))

        return ws

    async def broadcast_event(self, event: dict) -> None:
        """Broadcast an event to all subscribed WebSocket clients."""
        if not self._event_subscribers:
            return
        payload = {"type": "event", **event}
        dead: list[web.WebSocketResponse] = []
        for ws in list(self._event_subscribers):
            try:
                await ws.send_json(payload)
            except (ConnectionError, RuntimeError):
                dead.append(ws)
        for ws in dead:
            self._event_subscribers.discard(ws)
            self._clients.discard(ws)

    async def _tail_logs(self, ws: web.WebSocketResponse) -> None:
        """Tail the audit log file and stream new lines to a client."""
        log_path = Path("./data/audit.jsonl")
        last_pos = 0

        # Send tail of existing log
        if log_path.exists():
            try:
                content = log_path.read_text()
                lines = content.strip().split("\n") if content.strip() else []
                tail = lines[-_LOG_TAIL_LINES:]
                for line in tail:
                    if ws.closed:
                        return
                    await ws.send_json({"type": "log", "line": line})
                last_pos = log_path.stat().st_size
            except OSError:
                pass

        # Poll for new lines
        while not ws.closed and ws in self._log_subscribers:
            try:
                await asyncio.sleep(_LOG_POLL_INTERVAL)
                if not log_path.exists():
                    continue
                current_size = log_path.stat().st_size
                if current_size <= last_pos:
                    if current_size < last_pos:
                        last_pos = 0  # File was truncated/rotated
                    continue
                with open(log_path, "r") as f:
                    f.seek(last_pos)
                    new_data = f.read()
                    last_pos = f.tell()
                for line in new_data.strip().split("\n"):
                    if line and not ws.closed:
                        await ws.send_json({"type": "log", "line": line})
            except asyncio.CancelledError:
                break
            except (OSError, ConnectionError, RuntimeError):
                break


def setup_websocket(app: web.Application, bot: LokiBot) -> WebSocketManager:
    """Register the WebSocket endpoint and return the manager."""
    manager = WebSocketManager(bot)
    app.router.add_get("/api/ws", manager.handle)
    log.info("WebSocket endpoint registered at /api/ws")
    return manager
