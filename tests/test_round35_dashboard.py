"""Round 35 — Dashboard redesign tests.

Tests the expanded /api/status endpoint (agent/process/monitoring data),
new /api/agents endpoint, dashboard JS component structure, and CSS classes.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from src.config.schema import WebConfig
from src.health.server import (
    _make_auth_middleware,
    _make_rate_limit_middleware,
    _make_security_headers_middleware,
)
from src.web.api import create_api_routes, setup_api


# ---------------------------------------------------------------------------
# Helper: mock bot with all dashboard-relevant attributes
# ---------------------------------------------------------------------------

def _make_bot():
    bot = MagicMock()

    # Discord
    guild = MagicMock()
    guild.id = 111
    guild.name = "TestGuild"
    guild.member_count = 42
    bot.guilds = [guild]
    bot.is_ready = MagicMock(return_value=True)
    bot._start_time = time.monotonic() - 3600

    # Tools
    bot._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command"}, {"name": "read_file"},
    ])

    # Sessions
    bot.sessions = MagicMock()
    bot.sessions._sessions = {"chan1": MagicMock()}

    # Skills
    bot.skill_manager = MagicMock()
    bot.skill_manager.list_skills = MagicMock(return_value=[
        {"name": "joke"}, {"name": "hello"},
    ])

    # Scheduler
    bot.scheduler = MagicMock()
    bot.scheduler.list_all = MagicMock(return_value=[{"id": "s1"}])

    # Loops
    bot.loop_manager = MagicMock()
    bot.loop_manager.active_count = 2
    bot.loop_manager._loops = {}

    # Agents
    agent1 = MagicMock()
    agent1.id = "agt001"
    agent1.label = "disk-checker"
    agent1.goal = "Check disk usage on all hosts"
    agent1.status = "running"
    agent1.channel_id = "ch1"
    agent1.requester_id = "u1"
    agent1.requester_name = "Alice"
    agent1.iteration_count = 5
    agent1.tools_used = ["run_command", "read_file", "check_disk"]
    agent1.created_at = time.time() - 120
    agent1.ended_at = None
    agent1.result = ""
    agent1.error = ""

    agent2 = MagicMock()
    agent2.id = "agt002"
    agent2.label = "log-parser"
    agent2.goal = "Parse recent nginx logs for errors"
    agent2.status = "completed"
    agent2.channel_id = "ch1"
    agent2.requester_id = "u2"
    agent2.requester_name = "Bob"
    agent2.iteration_count = 8
    agent2.tools_used = ["run_command", "read_file"]
    agent2.created_at = time.time() - 300
    agent2.ended_at = time.time() - 60
    agent2.result = "Found 12 nginx errors in access.log"
    agent2.error = ""

    bot.agent_manager = MagicMock()
    bot.agent_manager._agents = {"agt001": agent1, "agt002": agent2}
    bot.agent_manager.kill = MagicMock(return_value="Kill signal sent to agent 'disk-checker'.")

    # Processes
    proc1 = MagicMock()
    proc1.command = "htop"
    proc1.host = "localhost"
    proc1.status = "running"
    proc1.exit_code = None
    proc1.start_time = time.time() - 60
    proc1.output_buffer = deque(["line1\n"], maxlen=500)

    proc2 = MagicMock()
    proc2.command = "tail -f /var/log/syslog"
    proc2.host = "server1"
    proc2.status = "completed"
    proc2.exit_code = 0
    proc2.start_time = time.time() - 300
    proc2.output_buffer = deque(["done\n"], maxlen=500)

    registry = MagicMock()
    registry._processes = {100: proc1, 200: proc2}
    registry.kill = AsyncMock(return_value="Process killed")
    bot.tool_executor = MagicMock()
    bot.tool_executor._process_registry = registry

    # Monitoring
    bot.infra_watcher = MagicMock()
    bot.infra_watcher.get_status = MagicMock(return_value={
        "enabled": True, "checks": 5, "running": 3, "active_alerts": 1,
    })

    # Config (for reload support)
    bot.config = MagicMock()
    bot.config.model_dump = MagicMock(return_value={"tools": {}})
    bot.context_loader = MagicMock()
    bot._invalidate_prompt_caches = MagicMock()

    # Audit
    bot.audit = MagicMock()
    bot.audit.search = AsyncMock(return_value=[])
    bot.audit.count_by_tool = AsyncMock(return_value={})

    # Memory
    bot.tool_executor._load_all_memory = MagicMock(return_value={})
    bot.tool_executor._save_all_memory = MagicMock()

    return bot


def _make_app(bot=None, *, api_token=""):
    if bot is None:
        bot = _make_bot()
    web_config = WebConfig(api_token=api_token)
    app = web.Application(middlewares=[
        _make_security_headers_middleware(),
        _make_rate_limit_middleware(),
        _make_auth_middleware(web_config),
    ])
    setup_api(app, bot)
    return app, bot


# ===================================================================
# Status endpoint — expanded fields
# ===================================================================


class TestExpandedStatus:
    """Verify /api/status includes agent, process, and monitoring data."""

    @pytest.mark.asyncio
    async def test_status_includes_agent_counts(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            body = await resp.json()
            assert body["agent_count"] == 2  # agt001 + agt002
            assert body["agent_running"] == 1  # only agt001 is running

    @pytest.mark.asyncio
    async def test_status_includes_process_counts(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            body = await resp.json()
            assert body["process_count"] == 2  # proc1 + proc2
            assert body["process_running"] == 1  # only proc1 is running

    @pytest.mark.asyncio
    async def test_status_includes_monitoring(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            body = await resp.json()
            assert "monitoring" in body
            assert body["monitoring"]["enabled"] is True
            assert body["monitoring"]["checks"] == 5
            assert body["monitoring"]["active_alerts"] == 1

    @pytest.mark.asyncio
    async def test_status_no_agent_manager(self):
        bot = _make_bot()
        del bot.agent_manager
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            body = await resp.json()
            assert body["agent_count"] == 0
            assert body["agent_running"] == 0

    @pytest.mark.asyncio
    async def test_status_no_watcher(self):
        bot = _make_bot()
        bot.infra_watcher = None
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            body = await resp.json()
            mon = body["monitoring"]
            assert mon["enabled"] is False
            assert mon["active_alerts"] == 0

    @pytest.mark.asyncio
    async def test_status_no_process_registry(self):
        bot = _make_bot()
        bot.tool_executor._process_registry = None
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            body = await resp.json()
            assert body["process_count"] == 0
            assert body["process_running"] == 0

    @pytest.mark.asyncio
    async def test_status_preserves_original_fields(self):
        """Expanded status still has all original fields."""
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            body = await resp.json()
            for field in [
                "status", "uptime_seconds", "guilds", "guild_count",
                "user_count", "tool_count", "skill_count", "session_count",
                "loop_count", "schedule_count",
            ]:
                assert field in body, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_status_all_new_fields_present(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            body = await resp.json()
            for field in [
                "agent_count", "agent_running",
                "process_count", "process_running",
                "monitoring",
            ]:
                assert field in body, f"Missing new field: {field}"


# ===================================================================
# Agents endpoint
# ===================================================================


class TestAgentsEndpoint:
    """Verify /api/agents returns agent data and /api/agents/:id DELETE works."""

    @pytest.mark.asyncio
    async def test_list_agents(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents")
            body = await resp.json()
            assert len(body) == 2
            ids = {a["id"] for a in body}
            assert "agt001" in ids
            assert "agt002" in ids

    @pytest.mark.asyncio
    async def test_agent_fields(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents")
            body = await resp.json()
            running = [a for a in body if a["id"] == "agt001"][0]
            assert running["label"] == "disk-checker"
            assert running["status"] == "running"
            assert running["iteration_count"] == 5
            assert "run_command" in running["tools_used"]
            assert running["runtime_seconds"] > 0
            assert running["goal"].startswith("Check disk")

    @pytest.mark.asyncio
    async def test_completed_agent_has_result(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents")
            body = await resp.json()
            completed = [a for a in body if a["id"] == "agt002"][0]
            assert completed["status"] == "completed"
            assert "12 nginx errors" in completed["result"]

    @pytest.mark.asyncio
    async def test_kill_agent(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/agents/agt001")
            body = await resp.json()
            assert "Kill signal" in body["result"]
            bot.agent_manager.kill.assert_called_once_with("agt001")

    @pytest.mark.asyncio
    async def test_kill_nonexistent_agent(self):
        bot = _make_bot()
        bot.agent_manager.kill = MagicMock(
            return_value="Error: Agent 'xyz' not found."
        )
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/agents/xyz")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_agents_no_manager(self):
        bot = _make_bot()
        del bot.agent_manager
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents")
            body = await resp.json()
            assert body == []

    @pytest.mark.asyncio
    async def test_agents_empty(self):
        bot = _make_bot()
        bot.agent_manager._agents = {}
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents")
            body = await resp.json()
            assert body == []

    @pytest.mark.asyncio
    async def test_agent_goal_truncated(self):
        """Agent goal should be truncated to 200 chars."""
        bot = _make_bot()
        agent = list(bot.agent_manager._agents.values())[0]
        agent.goal = "x" * 500
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents")
            body = await resp.json()
            goals = [a["goal"] for a in body if a["id"] == agent.id]
            assert len(goals[0]) == 200

    @pytest.mark.asyncio
    async def test_agent_tools_truncated(self):
        """Only last 10 tools shown."""
        bot = _make_bot()
        agent = list(bot.agent_manager._agents.values())[0]
        agent.tools_used = [f"tool_{i}" for i in range(20)]
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents")
            body = await resp.json()
            agent_data = [a for a in body if a["id"] == agent.id][0]
            assert len(agent_data["tools_used"]) == 10
            # Should be last 10
            assert agent_data["tools_used"][0] == "tool_10"

    @pytest.mark.asyncio
    async def test_kill_agent_no_manager(self):
        bot = _make_bot()
        del bot.agent_manager
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/agents/xyz")
            assert resp.status == 404


# ===================================================================
# Dashboard JS — template and component structure
# ===================================================================


DASHBOARD_JS = Path("ui/js/pages/dashboard.js").read_text()


class TestDashboardTemplate:
    """Verify the dashboard JS has key UI sections and data bindings."""

    def test_has_hero_status_banner(self):
        assert "dash-hero" in DASHBOARD_JS

    def test_has_uptime_ring(self):
        assert "dash-hero-ring" in DASHBOARD_JS
        assert "uptimeRingOffset" in DASHBOARD_JS

    def test_has_stat_cards(self):
        assert "dash-stat" in DASHBOARD_JS
        assert "dash-stat-value" in DASHBOARD_JS

    def test_has_health_indicators(self):
        assert "dash-health-bar" in DASHBOARD_JS
        assert "healthIndicators" in DASHBOARD_JS

    def test_has_agent_panel(self):
        assert "dash-agent-list" in DASHBOARD_JS
        assert "Active Agents" in DASHBOARD_JS

    def test_has_activity_feed(self):
        assert "dash-activity-list" in DASHBOARD_JS
        assert "Recent Activity" in DASHBOARD_JS

    def test_has_guild_panel(self):
        assert "dash-guild-item" in DASHBOARD_JS
        assert "Guilds" in DASHBOARD_JS

    def test_has_error_panel(self):
        assert "dash-error-list" in DASHBOARD_JS
        assert "Recent Errors" in DASHBOARD_JS

    def test_has_quick_actions(self):
        assert "reloadConfig" in DASHBOARD_JS
        assert "clearSessions" in DASHBOARD_JS
        assert "stopAllLoops" in DASHBOARD_JS

    def test_fetches_agents_api(self):
        assert "/api/agents" in DASHBOARD_JS

    def test_fetches_status_api(self):
        assert "/api/status" in DASHBOARD_JS

    def test_auto_refresh_status(self):
        assert "setInterval(fetchStatus" in DASHBOARD_JS

    def test_auto_refresh_agents(self):
        assert "setInterval(fetchAgents" in DASHBOARD_JS

    def test_websocket_subscription(self):
        assert "ws.subscribe" in DASHBOARD_JS
        assert "ws.unsubscribe" in DASHBOARD_JS

    def test_flash_animation_on_new_events(self):
        assert "flash-new" in DASHBOARD_JS
        assert "_isNew" in DASHBOARD_JS

    def test_loading_skeleton(self):
        assert "skeleton" in DASHBOARD_JS

    def test_error_state(self):
        assert "error-state" in DASHBOARD_JS

    def test_has_format_duration(self):
        assert "formatDuration" in DASHBOARD_JS

    def test_uptime_ring_computed(self):
        """Uptime ring offset should be computed from uptime_seconds."""
        assert "125.66" in DASHBOARD_JS
        assert "86400" in DASHBOARD_JS


class TestDashboardStats:
    """Verify stat card definitions include new fields."""

    def test_agents_stat(self):
        assert "'Agents'" in DASHBOARD_JS
        assert "agent_running" in DASHBOARD_JS

    def test_processes_stat(self):
        assert "'Processes'" in DASHBOARD_JS
        assert "process_running" in DASHBOARD_JS

    def test_monitoring_stat(self):
        assert "'Alerts'" in DASHBOARD_JS
        assert "active_alerts" in DASHBOARD_JS

    def test_all_original_stats(self):
        for stat in ["Guilds", "Sessions", "Tools", "Loops", "Schedules", "Users"]:
            assert f"'{stat}'" in DASHBOARD_JS, f"Missing stat: {stat}"

    def test_skill_count_shown(self):
        assert "skill_count" in DASHBOARD_JS

    def test_stat_highlight_for_active(self):
        assert "dash-stat-highlight" in DASHBOARD_JS


class TestDashboardHealthIndicators:
    """Verify health indicator computation references all sources."""

    def test_bot_status_indicator(self):
        assert "'Bot'" in DASHBOARD_JS

    def test_monitoring_indicator(self):
        assert "'Monitoring'" in DASHBOARD_JS
        assert "mon.enabled" in DASHBOARD_JS or "monitoring" in DASHBOARD_JS

    def test_loops_indicator(self):
        assert "loop_count" in DASHBOARD_JS

    def test_agents_indicator(self):
        assert "agent_running" in DASHBOARD_JS


# ===================================================================
# Dashboard CSS — new design classes
# ===================================================================


CSS_CONTENT = Path("ui/css/style.css").read_text()


class TestDashboardCSS:
    """Verify all new CSS classes for the dashboard redesign exist."""

    def test_hero_classes(self):
        for cls in ["dash-hero", "dash-hero-left", "dash-hero-ring",
                     "dash-hero-name", "dash-hero-sub", "dash-hero-actions"]:
            assert cls in CSS_CONTENT, f"Missing CSS class: {cls}"

    def test_ring_variants(self):
        assert "ring-online" in CSS_CONTENT
        assert "ring-starting" in CSS_CONTENT

    def test_stat_classes(self):
        for cls in ["dash-stat", "dash-stat-header", "dash-stat-icon",
                     "dash-stat-label", "dash-stat-value", "dash-stat-sub",
                     "dash-stat-highlight"]:
            assert cls in CSS_CONTENT, f"Missing CSS class: {cls}"

    def test_health_bar_classes(self):
        for cls in ["dash-health-bar", "dash-health-items", "dash-health-item",
                     "dash-health-dot", "dash-health-ok", "dash-health-warn",
                     "dash-health-error", "dash-health-label"]:
            assert cls in CSS_CONTENT, f"Missing CSS class: {cls}"

    def test_panel_classes(self):
        for cls in ["dash-panel", "dash-panel-header", "dash-panel-title"]:
            assert cls in CSS_CONTENT, f"Missing CSS class: {cls}"

    def test_agent_classes(self):
        for cls in ["dash-agent-list", "dash-agent-item", "dash-agent-top",
                     "dash-agent-dot", "dash-agent-label", "dash-agent-running",
                     "dash-agent-completed", "dash-agent-failed"]:
            assert cls in CSS_CONTENT, f"Missing CSS class: {cls}"

    def test_activity_classes(self):
        for cls in ["dash-activity-list", "dash-activity-item",
                     "dash-activity-dot", "dash-activity-tool", "dash-activity-time"]:
            assert cls in CSS_CONTENT, f"Missing CSS class: {cls}"

    def test_error_classes(self):
        for cls in ["dash-error-list", "dash-error-item", "dash-error-tool"]:
            assert cls in CSS_CONTENT, f"Missing CSS class: {cls}"

    def test_guild_classes(self):
        assert "dash-guild-item" in CSS_CONTENT

    def test_empty_state(self):
        assert "dash-empty" in CSS_CONTENT

    def test_uses_design_tokens(self):
        """New CSS should use design tokens, not hard-coded colors."""
        # Check a few representative rules use tokens
        assert "--hm-success" in CSS_CONTENT
        assert "--hm-accent" in CSS_CONTENT
        assert "--hm-surface" in CSS_CONTENT


# ===================================================================
# Agent manager integration (verify data shapes)
# ===================================================================


class TestAgentDataShapes:
    """Verify the agent endpoint data matches what the dashboard expects."""

    @pytest.mark.asyncio
    async def test_agent_has_all_dashboard_fields(self):
        """Each agent in the list should have all fields the dashboard binds to."""
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents")
            body = await resp.json()
            required_fields = {
                "id", "label", "goal", "status", "channel_id",
                "requester_name", "iteration_count", "tools_used",
                "runtime_seconds", "created_at", "result", "error",
            }
            for agent in body:
                assert required_fields.issubset(set(agent.keys())), \
                    f"Agent missing fields: {required_fields - set(agent.keys())}"

    @pytest.mark.asyncio
    async def test_status_monitoring_is_dict(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            body = await resp.json()
            mon = body["monitoring"]
            assert isinstance(mon, dict)
            assert "enabled" in mon
            assert "checks" in mon
            assert "active_alerts" in mon
