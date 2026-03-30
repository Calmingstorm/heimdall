"""REST API for Heimdall web management UI.

All endpoints are prefixed with /api/ and require Bearer token auth
(unless api_token is empty in config, which disables auth for dev mode).
"""
from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from aiohttp import web
from croniter import croniter

from ..config.schema import Config
from ..llm.secret_scrubber import scrub_output_secrets
from ..logging import get_logger
from ..tools.registry import get_tool_definitions
from .chat import MAX_CHAT_CONTENT_LEN, process_web_chat

if TYPE_CHECKING:
    from ..discord.client import HeimdallBot

log = get_logger("web.api")

# Sensitive config fields that should be redacted in API responses
_SENSITIVE_FIELDS = frozenset({
    "token", "api_token", "secret", "ssh_key_path", "credentials_path",
    "api_key", "password",
})


# Input validation limits
_MAX_NAME_LEN = 100
_MAX_CODE_LEN = 50_000
_MAX_CONTENT_LEN = 500_000
_MAX_GOAL_LEN = 2000
_MAX_DESCRIPTION_LEN = 500


def _validate_string(value: str, field: str, max_len: int) -> str | None:
    """Validate a string field. Returns error message or None."""
    if len(value) > max_len:
        return f"{field} exceeds maximum length ({max_len} chars)"
    return None


# Regex: keep only ASCII alphanumeric, hyphen, underscore, period
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_.\-]")


def _safe_filename(name: str, max_len: int = 80) -> str:
    """Sanitize a string for use in Content-Disposition filename."""
    return _SAFE_FILENAME_RE.sub("_", name)[:max_len] or "export"


def _sanitize_error(msg: str) -> str:
    """Scrub secrets from error messages before returning to clients."""
    return scrub_output_secrets(str(msg))


def _contains_blocked_fields(d: dict, blocked: frozenset[str], *, _depth: int = 0) -> bool:
    """Recursively check if any keys in *d* are in *blocked*."""
    if _depth > 10:
        return False
    for key, value in d.items():
        if key in blocked:
            return True
        if isinstance(value, dict) and _contains_blocked_fields(value, blocked, _depth=_depth + 1):
            return True
    return False


def _deep_merge(base: dict, updates: dict, *, _depth: int = 0) -> None:
    """Recursively merge *updates* into *base* in-place."""
    if _depth > 10:
        return
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value, _depth=_depth + 1)
        else:
            base[key] = value


def _redact_config(obj: Any, *, _depth: int = 0) -> Any:
    """Recursively redact sensitive fields from config dicts."""
    if _depth > 10:
        return "..."
    if isinstance(obj, dict):
        return {
            k: "••••••••" if k in _SENSITIVE_FIELDS and isinstance(v, str) and v
            else _redact_config(v, _depth=_depth + 1)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact_config(v, _depth=_depth + 1) for v in obj]
    return obj


def _write_config(path: Path, data: dict) -> None:
    """Write config dict to YAML file."""
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def create_api_routes(bot: HeimdallBot) -> web.RouteTableDef:
    """Create all API route handlers bound to the given bot instance."""
    routes = web.RouteTableDef()

    # ------------------------------------------------------------------
    # Auth (login / logout / session check)
    # ------------------------------------------------------------------

    @routes.post("/api/auth/login")
    async def auth_login(request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        token = (data.get("token") or "").strip()
        if not token:
            return web.json_response({"error": "token is required"}, status=400)

        api_token = bot.config.web.api_token
        if not api_token:
            # No auth configured — dev mode, issue session anyway
            sm = request.app.get("session_manager")
            if sm:
                sid, timeout = sm.create()
                return web.json_response({
                    "session_id": sid,
                    "timeout_seconds": timeout,
                })
            return web.json_response({"error": "no session manager"}, status=500)

        import hmac as _hmac
        if not _hmac.compare_digest(token, api_token):
            return web.json_response({"error": "invalid token"}, status=401)

        sm = request.app.get("session_manager")
        if not sm:
            return web.json_response({"error": "no session manager"}, status=500)

        sid, timeout = sm.create()
        return web.json_response({
            "session_id": sid,
            "timeout_seconds": timeout,
        })

    @routes.post("/api/auth/logout")
    async def auth_logout(request: web.Request) -> web.Response:
        sm = request.app.get("session_manager")
        if not sm:
            return web.json_response({"status": "ok"})

        # Extract session ID from Authorization header
        auth_header = request.headers.get("Authorization", "")
        bearer_prefix = "Bearer "
        if auth_header.startswith(bearer_prefix):
            sid = auth_header[len(bearer_prefix):]
            sm.destroy(sid)

        return web.json_response({"status": "logged_out"})

    @routes.get("/api/auth/session")
    async def auth_session(request: web.Request) -> web.Response:
        sm = request.app.get("session_manager")
        timeout = sm.timeout_seconds if sm else 0
        return web.json_response({
            "authenticated": True,
            "timeout_seconds": timeout,
            "active_sessions": sm.active_count if sm else 0,
        })

    # ------------------------------------------------------------------
    # Status & info
    # ------------------------------------------------------------------

    @routes.get("/api/status")
    async def get_status(_request: web.Request) -> web.Response:
        guilds = [
            {"id": str(g.id), "name": g.name, "member_count": g.member_count or 0}
            for g in bot.guilds
        ]
        user_count = sum(g.member_count or 0 for g in bot.guilds)
        tools = bot._merged_tool_definitions()
        uptime = time.monotonic() - bot._start_time if hasattr(bot, "_start_time") else 0

        # Agent counts
        try:
            agent_agents = bot.agent_manager._agents
            if not isinstance(agent_agents, dict):
                raise AttributeError
            agent_count = len(agent_agents)
            agent_running = sum(
                1 for a in agent_agents.values() if a.status == "running"
            )
        except (AttributeError, TypeError):
            agent_count = 0
            agent_running = 0

        # Process counts
        try:
            proc_procs = bot.tool_executor._process_registry._processes
            if not isinstance(proc_procs, dict):
                raise AttributeError
            process_count = len(proc_procs)
            process_running = sum(
                1 for p in proc_procs.values() if p.status == "running"
            )
        except (AttributeError, TypeError):
            process_count = 0
            process_running = 0

        # Monitoring status
        _default_mon = {
            "enabled": False, "checks": 0, "running": 0, "active_alerts": 0,
        }
        try:
            watcher = bot.infra_watcher
            if watcher is None:
                raise AttributeError
            result = watcher.get_status()
            monitoring = result if isinstance(result, dict) else _default_mon
        except (AttributeError, TypeError):
            monitoring = _default_mon

        return web.json_response({
            "status": "online" if bot.is_ready() else "starting",
            "uptime_seconds": round(uptime, 1),
            "guilds": guilds,
            "guild_count": len(guilds),
            "user_count": user_count,
            "tool_count": len(tools),
            "skill_count": len(bot.skill_manager.list_skills()),
            "session_count": len(bot.sessions._sessions),
            "loop_count": bot.loop_manager.active_count,
            "schedule_count": len(bot.scheduler.list_all()),
            "agent_count": agent_count,
            "agent_running": agent_running,
            "process_count": process_count,
            "process_running": process_running,
            "monitoring": monitoring,
        })

    @routes.get("/api/config")
    async def get_config(_request: web.Request) -> web.Response:
        raw = bot.config.model_dump()
        return web.json_response(_redact_config(raw))

    @routes.put("/api/config")
    async def update_config(request: web.Request) -> web.Response:
        try:
            updates = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        if not isinstance(updates, dict):
            return web.json_response({"error": "expected JSON object"}, status=400)

        # Block sensitive field updates
        if _contains_blocked_fields(updates, _SENSITIVE_FIELDS):
            return web.json_response(
                {"error": "Cannot update sensitive fields via API"}, status=403
            )

        # Deep merge updates into current config
        current = bot.config.model_dump()
        _deep_merge(current, updates)

        # Validate by reconstructing the config model
        try:
            new_config = Config(**current)
        except Exception as e:
            return web.json_response({"error": f"Invalid config: {e}"}, status=400)

        # Apply to bot
        bot.config = new_config

        # Write to disk
        config_path = Path("config.yml")
        if config_path.exists():
            try:
                await asyncio.to_thread(_write_config, config_path, current)
            except Exception:
                log.warning("Config applied in memory but failed to persist to %s", config_path, exc_info=True)

        return web.json_response(_redact_config(new_config.model_dump()))

    # ------------------------------------------------------------------
    # Quick actions
    # ------------------------------------------------------------------

    @routes.post("/api/sessions/clear-all")
    async def clear_all_sessions(_request: web.Request) -> web.Response:
        channel_ids = list(bot.sessions._sessions.keys())
        for cid in channel_ids:
            bot.sessions.reset(cid)
        return web.json_response({"status": "cleared", "count": len(channel_ids)})

    @routes.post("/api/reload")
    async def reload_config(_request: web.Request) -> web.Response:
        bot.context_loader.reload()
        bot._invalidate_prompt_caches()
        bot._system_prompt = bot._build_system_prompt()
        return web.json_response({"status": "reloaded"})

    @routes.post("/api/loops/stop-all")
    async def stop_all_loops(_request: web.Request) -> web.Response:
        result = bot.loop_manager.stop_loop("all")
        return web.json_response({"result": result})

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    @routes.post("/api/chat")
    async def chat(request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        content = (data.get("content") or "").strip()
        if not content:
            return web.json_response({"error": "content is required"}, status=400)
        if len(content) > MAX_CHAT_CONTENT_LEN:
            return web.json_response(
                {"error": f"content exceeds {MAX_CHAT_CONTENT_LEN} chars"}, status=400
            )

        channel_id = data.get("channel_id") or "web-default"
        user_id = data.get("user_id") or "web-user"
        username = data.get("username") or "WebUser"

        result = await process_web_chat(
            bot, content, channel_id,
            user_id=user_id, username=username,
        )
        status = 200 if not result["is_error"] else 502
        return web.json_response({
            "response": result["response"],
            "tools_used": result["tools_used"],
            "is_error": result["is_error"],
        }, status=status)

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    @routes.get("/api/sessions")
    async def list_sessions(_request: web.Request) -> web.Response:
        sessions = []
        for cid, session in bot.sessions._sessions.items():
            # Build preview from last 2 messages
            preview = []
            for m in session.messages[-2:]:
                text = m.content or ""
                if len(text) > 120:
                    text = text[:120] + "..."
                preview.append({"role": m.role, "content": text})
            # Determine source type
            source = "web" if cid.startswith("web-") else "discord"
            sessions.append({
                "channel_id": cid,
                "message_count": len(session.messages),
                "last_active": session.last_active,
                "created_at": session.created_at,
                "has_summary": bool(session.summary),
                "preview": preview,
                "source": source,
                "last_user_id": session.last_user_id,
            })
        sessions.sort(key=lambda s: s["last_active"], reverse=True)
        return web.json_response(sessions)

    @routes.get("/api/sessions/{channel_id}")
    async def get_session(request: web.Request) -> web.Response:
        cid = request.match_info["channel_id"]
        session = bot.sessions._sessions.get(cid)
        if not session:
            return web.json_response({"error": "session not found"}, status=404)
        messages = [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "user_id": m.user_id,
            }
            for m in session.messages
        ]
        return web.json_response({
            "channel_id": cid,
            "messages": messages,
            "summary": session.summary,
            "created_at": session.created_at,
            "last_active": session.last_active,
        })

    @routes.get("/api/sessions/{channel_id}/export")
    async def export_session(request: web.Request) -> web.Response:
        cid = request.match_info["channel_id"]
        session = bot.sessions._sessions.get(cid)
        if not session:
            return web.json_response({"error": "session not found"}, status=404)
        fmt = request.query.get("format", "json")
        messages = [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "user_id": m.user_id,
            }
            for m in session.messages
        ]
        safe_cid = _safe_filename(cid)
        if fmt == "text":
            lines = []
            if session.summary:
                lines.append(f"=== Summary ===\n{session.summary}\n")
            lines.append(f"=== Messages ({len(messages)}) ===")
            for m in messages:
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m["timestamp"])) if m["timestamp"] else "?"
                role = m["role"].upper()
                uid = f" ({m['user_id']})" if m.get("user_id") else ""
                lines.append(f"\n[{ts}] {role}{uid}:\n{m['content']}")
            body = "\n".join(lines)
            return web.Response(
                text=body,
                content_type="text/plain",
                headers={"Content-Disposition": f'attachment; filename="session-{safe_cid}.txt"'},
            )
        # Default: JSON
        export = {
            "channel_id": cid,
            "messages": messages,
            "summary": session.summary,
            "created_at": session.created_at,
            "last_active": session.last_active,
            "exported_at": time.time(),
        }
        return web.json_response(
            export,
            headers={"Content-Disposition": f'attachment; filename="session-{safe_cid}.json"'},
        )

    @routes.delete("/api/sessions/{channel_id}")
    async def delete_session(request: web.Request) -> web.Response:
        cid = request.match_info["channel_id"]
        if cid not in bot.sessions._sessions:
            return web.json_response({"error": "session not found"}, status=404)
        bot.sessions.reset(cid)
        return web.json_response({"status": "cleared"})

    @routes.post("/api/sessions/clear-bulk")
    async def clear_bulk_sessions(request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        channel_ids = data.get("channel_ids", [])
        if not isinstance(channel_ids, list) or not channel_ids:
            return web.json_response(
                {"error": "channel_ids must be a non-empty list"}, status=400
            )
        cleared = 0
        for cid in channel_ids:
            if cid in bot.sessions._sessions:
                bot.sessions.reset(cid)
                cleared += 1
        return web.json_response({"status": "cleared", "count": cleared})

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @routes.get("/api/tools")
    async def list_tools(_request: web.Request) -> web.Response:
        all_tools = get_tool_definitions()
        result = [
            {
                "name": tool["name"],
                "description": tool["description"],
            }
            for tool in all_tools
        ]
        return web.json_response(result)

    @routes.get("/api/tools/stats")
    async def tool_stats(_request: web.Request) -> web.Response:
        counts = await bot.audit.count_by_tool()
        return web.json_response(counts)

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    @routes.get("/api/skills")
    async def list_skills(_request: web.Request) -> web.Response:
        skills = bot.skill_manager.list_skills()
        # Get usage counts from audit log
        counts = await bot.audit.count_by_tool()
        # Add source code and execution stats for each skill
        for skill_info in skills:
            name = skill_info["name"]
            skill_info["code"] = None
            loaded = bot.skill_manager._skills.get(name)
            if loaded and loaded.file_path.exists():
                try:
                    skill_info["code"] = loaded.file_path.read_text()
                except OSError:
                    pass
            skill_info["execution_count"] = counts.get(name, 0)
        return web.json_response(skills)

    @routes.post("/api/skills")
    async def create_skill(request: web.Request) -> web.Response:
        data = await request.json()
        name = data.get("name", "").strip()
        code = data.get("code", "").strip()
        if not name or not code:
            return web.json_response(
                {"error": "name and code are required"}, status=400
            )
        for err in (
            _validate_string(name, "name", _MAX_NAME_LEN),
            _validate_string(code, "code", _MAX_CODE_LEN),
        ):
            if err:
                return web.json_response({"error": err}, status=400)
        result = bot.skill_manager.create_skill(name, code)
        bot._cached_merged_tools = None
        bot._cached_skills_text = None
        is_error = "error" in result.lower() or "failed" in result.lower()
        return web.json_response(
            {"result": result},
            status=400 if is_error else 201,
        )

    @routes.put("/api/skills/{name}")
    async def update_skill(request: web.Request) -> web.Response:
        name = request.match_info["name"]
        data = await request.json()
        code = data.get("code", "").strip()
        if not code:
            return web.json_response({"error": "code is required"}, status=400)
        err = _validate_string(code, "code", _MAX_CODE_LEN)
        if err:
            return web.json_response({"error": err}, status=400)
        result = bot.skill_manager.edit_skill(name, code)
        bot._cached_merged_tools = None
        bot._cached_skills_text = None
        is_error = "error" in result.lower() or "failed" in result.lower()
        return web.json_response(
            {"result": result},
            status=400 if is_error else 200,
        )

    @routes.post("/api/skills/{name}/test")
    async def test_skill(request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if not bot.skill_manager.has_skill(name):
            return web.json_response({"error": "skill not found"}, status=404)
        try:
            result = await bot.skill_manager.execute(name, {})
            is_error = result.startswith("Skill error:") or result.startswith("Skill '")
            return web.json_response({
                "result": result,
                "is_error": is_error,
            })
        except Exception as e:
            return web.json_response({"result": _sanitize_error(e), "is_error": True}, status=500)

    @routes.delete("/api/skills/{name}")
    async def delete_skill(request: web.Request) -> web.Response:
        name = request.match_info["name"]
        result = bot.skill_manager.delete_skill(name)
        bot._cached_merged_tools = None
        bot._cached_skills_text = None
        is_error = "error" in result.lower() or "not found" in result.lower()
        return web.json_response(
            {"result": result},
            status=404 if is_error else 200,
        )

    @routes.get("/api/skills/{name}")
    async def get_skill_detail(request: web.Request) -> web.Response:
        name = request.match_info["name"]
        info = bot.skill_manager.get_skill_info(name)
        if not info:
            return web.json_response({"error": "skill not found"}, status=404)
        return web.json_response(info)

    @routes.post("/api/skills/validate")
    async def validate_skill(request: web.Request) -> web.Response:
        data = await request.json()
        code = data.get("code", "").strip()
        if not code:
            return web.json_response({"error": "code is required"}, status=400)
        err = _validate_string(code, "code", _MAX_CODE_LEN)
        if err:
            return web.json_response({"error": err}, status=400)
        report = bot.skill_manager.validate_skill_code(code)
        return web.json_response(report)

    @routes.post("/api/skills/{name}/enable")
    async def enable_skill(request: web.Request) -> web.Response:
        name = request.match_info["name"]
        result = bot.skill_manager.enable_skill(name)
        if "not found" in result.lower():
            return web.json_response({"result": result}, status=404)
        bot._cached_merged_tools = None
        bot._cached_skills_text = None
        return web.json_response({"result": result})

    @routes.post("/api/skills/{name}/disable")
    async def disable_skill_api(request: web.Request) -> web.Response:
        name = request.match_info["name"]
        result = bot.skill_manager.disable_skill(name)
        if "not found" in result.lower():
            return web.json_response({"result": result}, status=404)
        bot._cached_merged_tools = None
        bot._cached_skills_text = None
        return web.json_response({"result": result})

    @routes.get("/api/skills/{name}/config")
    async def get_skill_config(request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if not bot.skill_manager.has_skill(name):
            return web.json_response({"error": "skill not found"}, status=404)
        info = bot.skill_manager.get_skill_info(name)
        return web.json_response({
            "config": bot.skill_manager.get_skill_config(name),
            "schema": info["metadata"]["config_schema"] if info else {},
        })

    @routes.put("/api/skills/{name}/config")
    async def set_skill_config(request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if not bot.skill_manager.has_skill(name):
            return web.json_response({"error": "skill not found"}, status=404)
        data = await request.json()
        values = data.get("config", {})
        if not isinstance(values, dict):
            return web.json_response({"error": "config must be a dict"}, status=400)
        errors = bot.skill_manager.set_skill_config(name, values)
        if errors:
            return web.json_response({"errors": errors}, status=400)
        return web.json_response({"config": bot.skill_manager.get_skill_config(name)})

    # ------------------------------------------------------------------
    # Knowledge
    # ------------------------------------------------------------------

    @routes.get("/api/knowledge")
    async def list_knowledge(_request: web.Request) -> web.Response:
        store = bot._knowledge_store
        if not store or not store.available:
            return web.json_response({"error": "knowledge store not available"}, status=503)
        return web.json_response(await asyncio.to_thread(store.list_sources))

    @routes.post("/api/knowledge")
    async def ingest_knowledge(request: web.Request) -> web.Response:
        store = bot._knowledge_store
        if not store or not store.available:
            return web.json_response({"error": "knowledge store not available"}, status=503)
        data = await request.json()
        source = data.get("source", "").strip()
        content = data.get("content", "").strip()
        if not source or not content:
            return web.json_response(
                {"error": "source and content are required"}, status=400
            )
        for err in (
            _validate_string(source, "source", _MAX_NAME_LEN),
            _validate_string(content, "content", _MAX_CONTENT_LEN),
        ):
            if err:
                return web.json_response({"error": err}, status=400)
        chunks = await store.ingest(content, source, embedder=bot._embedder, uploader="web-api")
        return web.json_response({"source": source, "chunks": chunks}, status=201)

    @routes.delete("/api/knowledge/{source}")
    async def delete_knowledge(request: web.Request) -> web.Response:
        store = bot._knowledge_store
        if not store or not store.available:
            return web.json_response({"error": "knowledge store not available"}, status=503)
        source = request.match_info["source"]
        deleted = await asyncio.to_thread(store.delete_source, source)
        if deleted == 0:
            return web.json_response({"error": "source not found"}, status=404)
        return web.json_response({"status": "deleted", "chunks_removed": deleted})

    @routes.post("/api/knowledge/{source}/reingest")
    async def reingest_knowledge(request: web.Request) -> web.Response:
        store = bot._knowledge_store
        if not store or not store.available:
            return web.json_response({"error": "knowledge store not available"}, status=503)
        source = request.match_info["source"]
        content = await asyncio.to_thread(store.get_source_content, source)
        if content is None:
            return web.json_response({"error": "source not found"}, status=404)
        chunks = await store.ingest(content, source, embedder=bot._embedder, uploader="web-reingest")
        return web.json_response({"source": source, "chunks": chunks})

    @routes.get("/api/knowledge/search")
    async def search_knowledge(request: web.Request) -> web.Response:
        store = bot._knowledge_store
        if not store or not store.available:
            return web.json_response({"error": "knowledge store not available"}, status=503)
        query = request.query.get("q", "").strip()
        if not query:
            return web.json_response({"error": "q parameter required"}, status=400)
        try:
            limit = min(int(request.query.get("limit", "10")), 50)
        except ValueError:
            return web.json_response({"error": "limit must be an integer"}, status=400)
        results = await store.search_hybrid(query, embedder=bot._embedder, limit=limit)
        return web.json_response(results)

    @routes.get("/api/knowledge/{source}/chunks")
    async def list_knowledge_chunks(request: web.Request) -> web.Response:
        store = bot._knowledge_store
        if not store or not store.available:
            return web.json_response({"error": "knowledge store not available"}, status=503)
        source = request.match_info["source"]
        chunks = await asyncio.to_thread(store.get_source_chunks, source)
        if not chunks:
            return web.json_response({"error": "source not found or empty"}, status=404)
        return web.json_response(chunks)

    # ------------------------------------------------------------------
    # Schedules
    # ------------------------------------------------------------------

    @routes.get("/api/schedules")
    async def list_schedules(_request: web.Request) -> web.Response:
        return web.json_response(bot.scheduler.list_all())

    @routes.post("/api/schedules")
    async def create_schedule(request: web.Request) -> web.Response:
        data = await request.json()
        description = data.get("description", "").strip()
        action = data.get("action", "reminder")
        channel_id = data.get("channel_id", "").strip()
        if not description or not channel_id:
            return web.json_response(
                {"error": "description and channel_id are required"}, status=400
            )
        err = _validate_string(description, "description", _MAX_DESCRIPTION_LEN)
        if err:
            return web.json_response({"error": err}, status=400)
        try:
            schedule = bot.scheduler.add(
                description=description,
                action=action,
                channel_id=channel_id,
                cron=data.get("cron"),
                run_at=data.get("run_at"),
                message=data.get("message"),
                tool_name=data.get("tool_name"),
                tool_input=data.get("tool_input"),
                steps=data.get("steps"),
                trigger=data.get("trigger"),
            )
            return web.json_response(schedule, status=201)
        except (ValueError, TypeError) as e:
            return web.json_response({"error": _sanitize_error(e)}, status=400)

    @routes.delete("/api/schedules/{schedule_id}")
    async def delete_schedule(request: web.Request) -> web.Response:
        sid = request.match_info["schedule_id"]
        if bot.scheduler.delete(sid):
            return web.json_response({"status": "deleted"})
        return web.json_response({"error": "schedule not found"}, status=404)

    @routes.post("/api/schedules/{schedule_id}/run")
    async def run_schedule_now(request: web.Request) -> web.Response:
        sid = request.match_info["schedule_id"]
        schedule = None
        for s in bot.scheduler._schedules:
            if s["id"] == sid:
                schedule = s
                break
        if not schedule:
            return web.json_response({"error": "schedule not found"}, status=404)
        if not bot.scheduler._callback:
            return web.json_response(
                {"error": "scheduler callback not configured"}, status=503
            )
        try:
            schedule["last_run"] = datetime.now().isoformat()
            await bot.scheduler._callback(schedule)
            return web.json_response({"status": "triggered", "schedule_id": sid})
        except Exception as e:
            return web.json_response({"error": _sanitize_error(e)}, status=500)

    @routes.post("/api/schedules/validate-cron")
    async def validate_cron(request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        expr = data.get("expression", "").strip()
        if not expr:
            return web.json_response({"error": "expression is required"}, status=400)
        if not croniter.is_valid(expr):
            return web.json_response({"valid": False, "error": "Invalid cron expression"})
        # Return next 5 run times
        now = datetime.now()
        cr = croniter(expr, now)
        next_runs = [cr.get_next(datetime).isoformat() for _ in range(5)]
        return web.json_response({"valid": True, "next_runs": next_runs})

    # ------------------------------------------------------------------
    # Autonomous loops
    # ------------------------------------------------------------------

    @routes.get("/api/loops")
    async def list_loops(_request: web.Request) -> web.Response:
        loops = []
        for lid, info in bot.loop_manager._loops.items():
            # Include last 5 iteration history entries
            history = list(info._iteration_history[-5:]) if info._iteration_history else []
            loops.append({
                "id": lid,
                "goal": info.goal,
                "mode": info.mode,
                "interval_seconds": info.interval_seconds,
                "stop_condition": info.stop_condition,
                "max_iterations": info.max_iterations,
                "channel_id": info.channel_id,
                "requester_id": info.requester_id,
                "requester_name": info.requester_name,
                "iteration_count": info.iteration_count,
                "last_trigger": info.last_trigger,
                "created_at": info.created_at,
                "status": info.status,
                "iteration_history": history,
            })
        return web.json_response(loops)

    @routes.post("/api/loops")
    async def start_loop(request: web.Request) -> web.Response:
        data = await request.json()
        goal = data.get("goal", "").strip()
        if not goal:
            return web.json_response({"error": "goal is required"}, status=400)
        err = _validate_string(goal, "goal", _MAX_GOAL_LEN)
        if err:
            return web.json_response({"error": err}, status=400)
        channel_id = data.get("channel_id", "").strip()
        if not channel_id:
            return web.json_response(
                {"error": "channel_id is required"}, status=400
            )
        # Find the Discord channel to post to
        try:
            channel = bot.get_channel(int(channel_id))
        except (ValueError, TypeError):
            channel = None
        if not channel:
            return web.json_response({"error": "channel not found"}, status=404)

        requester_id = data.get("requester_id", "web-api")

        # Build iteration callback (same pattern as _handle_start_loop)
        async def _iteration_cb(
            prompt: str, ch: object, prev_context: str | None,
        ) -> str:
            return await bot._run_loop_iteration(
                prompt, ch, prev_context, requester_id,
            )

        result = bot.loop_manager.start_loop(
            goal=goal,
            channel=channel,
            requester_id=requester_id,
            requester_name=data.get("requester_name", "Web API"),
            iteration_callback=_iteration_cb,
            interval_seconds=data.get("interval_seconds", 60),
            mode=data.get("mode", "notify"),
            stop_condition=data.get("stop_condition"),
            max_iterations=data.get("max_iterations", 50),
        )
        if result.startswith("Error"):
            return web.json_response({"error": result}, status=400)
        return web.json_response({"loop_id": result}, status=201)

    @routes.delete("/api/loops/{loop_id}")
    async def stop_loop(request: web.Request) -> web.Response:
        lid = request.match_info["loop_id"]
        result = bot.loop_manager.stop_loop(lid)
        is_error = "not found" in result.lower() or "not running" in result.lower()
        return web.json_response(
            {"result": result}, status=404 if is_error else 200
        )

    @routes.post("/api/loops/{loop_id}/restart")
    async def restart_loop(request: web.Request) -> web.Response:
        lid = request.match_info["loop_id"]
        info = bot.loop_manager._loops.get(lid)
        if not info:
            return web.json_response({"error": "loop not found"}, status=404)

        # Capture config before stopping
        goal = info.goal
        mode = info.mode
        interval_seconds = info.interval_seconds
        stop_condition = info.stop_condition
        max_iterations = info.max_iterations
        channel_id = info.channel_id
        requester_id = info.requester_id
        requester_name = info.requester_name

        # Stop if running
        if info.status == "running":
            bot.loop_manager.stop_loop(lid)

        # Find the channel
        try:
            channel = bot.get_channel(int(channel_id))
        except (ValueError, TypeError):
            channel = None
        if not channel:
            return web.json_response({"error": "channel not found"}, status=404)

        # Build callback
        async def _iteration_cb(
            prompt: str, ch: object, prev_context: str | None,
        ) -> str:
            return await bot._run_loop_iteration(
                prompt, ch, prev_context, requester_id,
            )

        new_id = bot.loop_manager.start_loop(
            goal=goal,
            channel=channel,
            requester_id=requester_id,
            requester_name=requester_name,
            iteration_callback=_iteration_cb,
            interval_seconds=interval_seconds,
            mode=mode,
            stop_condition=stop_condition,
            max_iterations=max_iterations,
        )
        if new_id.startswith("Error"):
            return web.json_response({"error": new_id}, status=400)
        return web.json_response({"old_id": lid, "new_id": new_id}, status=201)

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    @routes.get("/api/agents")
    async def list_agents(_request: web.Request) -> web.Response:
        try:
            agent_agents = bot.agent_manager._agents
            if not isinstance(agent_agents, dict):
                return web.json_response([])
        except (AttributeError, TypeError):
            return web.json_response([])
        agents = []
        now = time.time()
        for aid, info in agent_agents.items():
            runtime = (info.ended_at or now) - info.created_at
            agents.append({
                "id": aid,
                "label": info.label,
                "goal": info.goal[:200],
                "status": info.status,
                "channel_id": info.channel_id,
                "requester_name": info.requester_name,
                "iteration_count": info.iteration_count,
                "tools_used": info.tools_used[-10:],
                "runtime_seconds": round(runtime, 1),
                "created_at": info.created_at,
                "result": (info.result[:200] if info.result else ""),
                "error": (info.error[:200] if info.error else ""),
            })
        return web.json_response(agents)

    @routes.delete("/api/agents/{agent_id}")
    async def kill_agent(request: web.Request) -> web.Response:
        try:
            if not isinstance(bot.agent_manager._agents, dict):
                raise AttributeError
        except (AttributeError, TypeError):
            return web.json_response({"error": "no agent manager"}, status=404)
        agent_id = request.match_info["agent_id"]
        result = bot.agent_manager.kill(agent_id)
        return web.json_response(
            {"result": result}, status=404 if "not found" in result.lower() else 200
        )

    # ------------------------------------------------------------------
    # Processes
    # ------------------------------------------------------------------

    @routes.get("/api/processes")
    async def list_processes(_request: web.Request) -> web.Response:
        registry = getattr(bot.tool_executor, "_process_registry", None)
        if not registry:
            return web.json_response([])
        processes = []
        now = time.time()
        for pid, info in sorted(registry._processes.items()):
            # Last 3 lines of output for inline preview
            output_lines = list(info.output_buffer)
            preview = [line.rstrip("\n") for line in output_lines[-3:]]
            processes.append({
                "pid": pid,
                "command": info.command,
                "host": info.host,
                "status": info.status,
                "exit_code": info.exit_code,
                "uptime_seconds": round(now - info.start_time, 1),
                "start_time": info.start_time,
                "output_preview": preview,
            })
        return web.json_response(processes)

    @routes.delete("/api/processes/{pid}")
    async def kill_process(request: web.Request) -> web.Response:
        registry = getattr(bot.tool_executor, "_process_registry", None)
        if not registry:
            return web.json_response({"error": "no process registry"}, status=404)
        try:
            pid = int(request.match_info["pid"])
        except ValueError:
            return web.json_response({"error": "invalid PID"}, status=400)
        result = await registry.kill(pid)
        is_error = "no process" in result.lower()
        return web.json_response(
            {"result": result}, status=404 if is_error else 200
        )

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    @routes.get("/api/audit")
    async def search_audit(request: web.Request) -> web.Response:
        tool_name = request.query.get("tool") or None
        user = request.query.get("user") or None
        host = request.query.get("host") or None
        keyword = request.query.get("q") or None
        date = request.query.get("date") or None
        error_only = request.query.get("error_only", "").lower() in ("1", "true", "yes")
        try:
            limit = min(int(request.query.get("limit", "50")), 200)
        except ValueError:
            return web.json_response({"error": "limit must be an integer"}, status=400)
        results = await bot.audit.search(
            tool_name=tool_name,
            user=user,
            host=host,
            keyword=keyword,
            date=date,
            limit=limit,
        )
        if error_only:
            results = [r for r in results if r.get("error")]
        return web.json_response(results)

    # ------------------------------------------------------------------
    # Memory (persistent notes — global + per-user scopes)
    # ------------------------------------------------------------------

    @routes.get("/api/memory")
    async def list_memory(_request: web.Request) -> web.Response:
        all_mem = await asyncio.to_thread(
            bot.tool_executor._load_all_memory
        )
        result = {}
        for scope, entries in all_mem.items():
            result[scope] = {
                "keys": list(entries.keys()),
                "count": len(entries),
            }
        return web.json_response(result)

    @routes.get("/api/memory/{scope}/{key}")
    async def get_memory(request: web.Request) -> web.Response:
        scope = request.match_info["scope"]
        key = request.match_info["key"]
        all_mem = await asyncio.to_thread(
            bot.tool_executor._load_all_memory
        )
        section = all_mem.get(scope, {})
        if key not in section:
            return web.json_response({"error": "key not found"}, status=404)
        return web.json_response({"scope": scope, "key": key, "value": section[key]})

    @routes.put("/api/memory/{scope}/{key}")
    async def set_memory(request: web.Request) -> web.Response:
        scope = request.match_info["scope"]
        key = request.match_info["key"]
        data = await request.json()
        value = data.get("value")
        if value is None:
            return web.json_response({"error": "value is required"}, status=400)
        all_mem = await asyncio.to_thread(
            bot.tool_executor._load_all_memory
        )
        if scope not in all_mem:
            all_mem[scope] = {}
        all_mem[scope][key] = str(value)
        await asyncio.to_thread(bot.tool_executor._save_all_memory, all_mem)
        return web.json_response({"status": "saved", "scope": scope, "key": key})

    @routes.delete("/api/memory/{scope}/{key}")
    async def delete_memory(request: web.Request) -> web.Response:
        scope = request.match_info["scope"]
        key = request.match_info["key"]
        all_mem = await asyncio.to_thread(
            bot.tool_executor._load_all_memory
        )
        section = all_mem.get(scope, {})
        if key not in section:
            return web.json_response({"error": "key not found"}, status=404)
        del all_mem[scope][key]
        await asyncio.to_thread(bot.tool_executor._save_all_memory, all_mem)
        return web.json_response({"status": "deleted", "scope": scope, "key": key})

    @routes.post("/api/memory/bulk-delete")
    async def bulk_delete_memory(request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        entries = data.get("entries", [])
        if not isinstance(entries, list) or not entries:
            return web.json_response(
                {"error": "entries must be a non-empty list of {scope, key}"}, status=400
            )
        all_mem = await asyncio.to_thread(
            bot.tool_executor._load_all_memory
        )
        deleted = 0
        for entry in entries:
            scope = entry.get("scope")
            key = entry.get("key")
            if scope and key and scope in all_mem and key in all_mem[scope]:
                del all_mem[scope][key]
                deleted += 1
        if deleted:
            await asyncio.to_thread(bot.tool_executor._save_all_memory, all_mem)
        return web.json_response({"status": "deleted", "count": deleted})

    return routes


def setup_api(app: web.Application, bot: HeimdallBot) -> None:
    """Register all API routes on the given aiohttp application."""
    routes = create_api_routes(bot)
    app.router.add_routes(routes)
    log.info("Web API endpoints registered")
