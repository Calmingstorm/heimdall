"""REST API for Loki web management UI.

All endpoints are prefixed with /api/ and require Bearer token auth
(unless api_token is empty in config, which disables auth for dev mode).
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

from ..logging import get_logger
from ..tools.registry import TOOL_PACKS, get_tool_definitions, get_pack_tool_names

if TYPE_CHECKING:
    from ..discord.client import LokiBot

log = get_logger("web.api")

# Sensitive config fields that should be redacted in API responses
_SENSITIVE_FIELDS = frozenset({
    "token", "api_token", "secret", "ssh_key_path", "credentials_path",
    "api_key", "password",
})


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


def create_api_routes(bot: LokiBot) -> web.RouteTableDef:
    """Create all API route handlers bound to the given bot instance."""
    routes = web.RouteTableDef()

    # ------------------------------------------------------------------
    # Status & info
    # ------------------------------------------------------------------

    @routes.get("/api/status")
    async def get_status(_request: web.Request) -> web.Response:
        guilds = [{"id": str(g.id), "name": g.name} for g in bot.guilds]
        user_count = sum(g.member_count or 0 for g in bot.guilds)
        tools = bot._merged_tool_definitions()
        uptime = time.monotonic() - bot._start_time if hasattr(bot, "_start_time") else 0
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
        })

    @routes.get("/api/config")
    async def get_config(_request: web.Request) -> web.Response:
        raw = bot.config.model_dump()
        return web.json_response(_redact_config(raw))

    @routes.put("/api/config")
    async def update_config(request: web.Request) -> web.Response:
        # Read-only for now — Round 22+ may add selective updates
        return web.json_response({"error": "config updates not yet supported"}, status=501)

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    @routes.get("/api/sessions")
    async def list_sessions(_request: web.Request) -> web.Response:
        sessions = []
        for cid, session in bot.sessions._sessions.items():
            sessions.append({
                "channel_id": cid,
                "message_count": len(session.messages),
                "last_active": session.last_active,
                "created_at": session.created_at,
                "has_summary": bool(session.summary),
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

    @routes.delete("/api/sessions/{channel_id}")
    async def delete_session(request: web.Request) -> web.Response:
        cid = request.match_info["channel_id"]
        if cid not in bot.sessions._sessions:
            return web.json_response({"error": "session not found"}, status=404)
        bot.sessions.reset(cid)
        return web.json_response({"status": "cleared"})

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @routes.get("/api/tools")
    async def list_tools(_request: web.Request) -> web.Response:
        all_tools = get_tool_definitions()  # All 81
        pack_names = get_pack_tool_names(list(TOOL_PACKS.keys()))
        result = []
        for tool in all_tools:
            name = tool["name"]
            pack = None
            for pack_name, pack_tools in TOOL_PACKS.items():
                if name in pack_tools:
                    pack = pack_name
                    break
            result.append({
                "name": name,
                "description": tool["description"],
                "pack": pack,
                "is_core": name not in pack_names,
            })
        return web.json_response(result)

    @routes.get("/api/tools/packs")
    async def list_packs(_request: web.Request) -> web.Response:
        enabled = bot.config.tools.tool_packs
        packs = {}
        for pack_name, tool_names in TOOL_PACKS.items():
            packs[pack_name] = {
                "enabled": not enabled or pack_name in enabled,
                "tools": tool_names,
                "tool_count": len(tool_names),
            }
        return web.json_response({
            "packs": packs,
            "enabled_packs": enabled,
            "all_packs_loaded": not enabled,
        })

    @routes.put("/api/tools/packs")
    async def update_packs(request: web.Request) -> web.Response:
        data = await request.json()
        packs = data.get("packs", [])
        valid = set(TOOL_PACKS.keys())
        invalid = [p for p in packs if p not in valid]
        if invalid:
            return web.json_response(
                {"error": f"unknown packs: {invalid}", "valid": sorted(valid)},
                status=400,
            )
        bot.config.tools.tool_packs = packs
        bot._cached_merged_tools = None
        return web.json_response({"status": "updated", "packs": packs})

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    @routes.get("/api/skills")
    async def list_skills(_request: web.Request) -> web.Response:
        skills = bot.skill_manager.list_skills()
        # Add source code for each skill
        for skill_info in skills:
            name = skill_info["name"]
            loaded = bot.skill_manager._skills.get(name)
            if loaded and loaded.file_path.exists():
                try:
                    skill_info["code"] = loaded.file_path.read_text()
                except OSError:
                    skill_info["code"] = None
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
        result = bot.skill_manager.edit_skill(name, code)
        bot._cached_merged_tools = None
        bot._cached_skills_text = None
        is_error = "error" in result.lower() or "failed" in result.lower()
        return web.json_response(
            {"result": result},
            status=400 if is_error else 200,
        )

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

    # ------------------------------------------------------------------
    # Knowledge
    # ------------------------------------------------------------------

    @routes.get("/api/knowledge")
    async def list_knowledge(_request: web.Request) -> web.Response:
        store = bot._knowledge_store
        if not store or not store.available:
            return web.json_response({"error": "knowledge store not available"}, status=503)
        return web.json_response(store.list_sources())

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
        chunks = await store.ingest(content, source, embedder=bot._embedder, uploader="web-api")
        return web.json_response({"source": source, "chunks": chunks}, status=201)

    @routes.delete("/api/knowledge/{source}")
    async def delete_knowledge(request: web.Request) -> web.Response:
        store = bot._knowledge_store
        if not store or not store.available:
            return web.json_response({"error": "knowledge store not available"}, status=503)
        source = request.match_info["source"]
        deleted = store.delete_source(source)
        if deleted == 0:
            return web.json_response({"error": "source not found"}, status=404)
        return web.json_response({"status": "deleted", "chunks_removed": deleted})

    @routes.get("/api/knowledge/search")
    async def search_knowledge(request: web.Request) -> web.Response:
        store = bot._knowledge_store
        if not store or not store.available:
            return web.json_response({"error": "knowledge store not available"}, status=503)
        query = request.query.get("q", "").strip()
        if not query:
            return web.json_response({"error": "q parameter required"}, status=400)
        limit = min(int(request.query.get("limit", "10")), 50)
        results = await store.search_hybrid(query, embedder=bot._embedder, limit=limit)
        return web.json_response(results)

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
            return web.json_response({"error": str(e)}, status=400)

    @routes.delete("/api/schedules/{schedule_id}")
    async def delete_schedule(request: web.Request) -> web.Response:
        sid = request.match_info["schedule_id"]
        if bot.scheduler.delete(sid):
            return web.json_response({"status": "deleted"})
        return web.json_response({"error": "schedule not found"}, status=404)

    # ------------------------------------------------------------------
    # Autonomous loops
    # ------------------------------------------------------------------

    @routes.get("/api/loops")
    async def list_loops(_request: web.Request) -> web.Response:
        loops = []
        for lid, info in bot.loop_manager._loops.items():
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
            })
        return web.json_response(loops)

    @routes.post("/api/loops")
    async def start_loop(request: web.Request) -> web.Response:
        data = await request.json()
        goal = data.get("goal", "").strip()
        if not goal:
            return web.json_response({"error": "goal is required"}, status=400)
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
            processes.append({
                "pid": pid,
                "command": info.command,
                "host": info.host,
                "status": info.status,
                "exit_code": info.exit_code,
                "uptime_seconds": round(now - info.start_time, 1),
                "start_time": info.start_time,
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
        limit = min(int(request.query.get("limit", "50")), 200)
        results = await bot.audit.search(
            tool_name=tool_name,
            user=user,
            host=host,
            keyword=keyword,
            date=date,
            limit=limit,
        )
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

    return routes


def setup_api(app: web.Application, bot: LokiBot) -> None:
    """Register all API routes on the given aiohttp application."""
    routes = create_api_routes(bot)
    app.router.add_routes(routes)
    log.info("Web API endpoints registered")
