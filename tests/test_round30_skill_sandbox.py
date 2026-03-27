"""Round 30 — Skill sandboxing tests.

Tests output limits, resource tracking, restricted file access, URL blocking,
and per-execution rate limits on tool calls, HTTP requests, and messages.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.schema import ToolsConfig, ToolHost
from src.tools.executor import ToolExecutor
from src.tools.skill_context import (
    MAX_SKILL_HTTP_REQUESTS,
    MAX_SKILL_MESSAGES,
    MAX_SKILL_TOOL_CALLS,
    ResourceTracker,
    SkillContext,
    is_path_denied,
    is_url_blocked,
)
from src.tools.skill_manager import (
    MAX_SKILL_OUTPUT_CHARS,
    SkillExecutionStats,
    SkillManager,
    SkillStatus,
)


VALID_SKILL_CODE = '''
SKILL_DEFINITION = {
    "name": "test_skill",
    "description": "A test skill",
    "input_schema": {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
    },
}

async def execute(inp, context):
    return f"Got: {inp.get('msg', 'nothing')}"
'''

LARGE_OUTPUT_SKILL = '''
SKILL_DEFINITION = {
    "name": "big_output",
    "description": "Produces large output",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    return "x" * 100000
'''

TRACKING_SKILL = '''
SKILL_DEFINITION = {
    "name": "tracker_skill",
    "description": "Uses various resources",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    await context.post_message("hello")
    return "tracked"
'''


@pytest.fixture
def skill_mgr(tmp_dir: Path, tools_config: ToolsConfig) -> SkillManager:
    executor = ToolExecutor(tools_config)
    skills_dir = tmp_dir / "skills"
    skills_dir.mkdir()
    return SkillManager(str(skills_dir), executor)


def _create_skill(mgr: SkillManager, name: str = "test_skill", code: str | None = None) -> str:
    if code is None:
        code = VALID_SKILL_CODE.replace("test_skill", name)
    return mgr.create_skill(name, code)


# ---------------------------------------------------------------------------
# Path denial
# ---------------------------------------------------------------------------

class TestPathDenied:
    def test_env_file(self):
        assert is_path_denied(".env") is True
        assert is_path_denied("/app/.env") is True
        assert is_path_denied("/app/.env.local") is True

    def test_config_yml(self):
        assert is_path_denied("config.yml") is True
        assert is_path_denied("/etc/app/config.yaml") is True

    def test_etc_shadow(self):
        assert is_path_denied("/etc/shadow") is True

    def test_ssh_keys(self):
        assert is_path_denied("/home/user/.ssh/id_rsa") is True
        assert is_path_denied("/home/user/.ssh/id_ed25519") is True
        assert is_path_denied("id_rsa") is True
        assert is_path_denied("id_ed25519") is True

    def test_ssh_directory(self):
        assert is_path_denied("/home/user/.ssh/config") is True
        assert is_path_denied("/home/user/.ssh/authorized_keys") is True

    def test_credentials_json(self):
        assert is_path_denied("/app/credentials.json") is True
        assert is_path_denied("credentials.json") is True

    def test_kube_config(self):
        assert is_path_denied("/home/user/.kube/config") is True

    def test_allowed_paths(self):
        assert is_path_denied("/var/log/syslog") is False
        assert is_path_denied("/home/user/app.py") is False
        assert is_path_denied("/tmp/output.txt") is False
        assert is_path_denied("/etc/hostname") is False

    def test_env_not_substring(self):
        # "environment.py" should NOT be blocked
        assert is_path_denied("/app/environment.py") is False


# ---------------------------------------------------------------------------
# URL blocking
# ---------------------------------------------------------------------------

class TestURLBlocked:
    def test_localhost(self):
        assert is_url_blocked("http://localhost:8080/api") is True
        assert is_url_blocked("http://127.0.0.1:3000") is True
        assert is_url_blocked("http://0.0.0.0:9090") is True

    def test_ipv6_loopback(self):
        assert is_url_blocked("http://[::1]:8080") is True

    def test_private_ips(self):
        assert is_url_blocked("http://10.0.0.1:3000") is True
        assert is_url_blocked("http://192.168.1.1") is True
        assert is_url_blocked("http://172.16.0.1") is True

    def test_cloud_metadata(self):
        assert is_url_blocked("http://169.254.169.254/latest/meta-data/") is True
        assert is_url_blocked("http://metadata.google.internal/computeMetadata/v1/") is True

    def test_public_urls_allowed(self):
        assert is_url_blocked("https://api.example.com/data") is False
        assert is_url_blocked("https://httpbin.org/get") is False

    def test_malformed_url_blocked(self):
        assert is_url_blocked("not a url at all ://") is True

    def test_link_local(self):
        assert is_url_blocked("http://169.254.1.1/something") is True


# ---------------------------------------------------------------------------
# ResourceTracker
# ---------------------------------------------------------------------------

class TestResourceTracker:
    def test_initial_values(self):
        t = ResourceTracker()
        assert t.tool_calls == 0
        assert t.http_requests == 0
        assert t.messages_sent == 0
        assert t.files_sent == 0
        assert t.bytes_downloaded == 0

    def test_to_dict(self):
        t = ResourceTracker(tool_calls=3, http_requests=1, messages_sent=2)
        d = t.to_dict()
        assert d["tool_calls"] == 3
        assert d["http_requests"] == 1
        assert d["messages_sent"] == 2

    def test_increment(self):
        t = ResourceTracker()
        t.tool_calls += 1
        t.http_requests += 2
        assert t.tool_calls == 1
        assert t.http_requests == 2


# ---------------------------------------------------------------------------
# SkillExecutionStats
# ---------------------------------------------------------------------------

class TestSkillExecutionStats:
    def test_to_dict(self):
        stats = SkillExecutionStats(
            wall_time_ms=123.456,
            output_chars=500,
            truncated=False,
            tool_calls=2,
            http_requests=1,
            messages_sent=1,
            files_sent=0,
            timestamp="2024-01-01T00:00:00",
        )
        d = stats.to_dict()
        assert d["wall_time_ms"] == 123.5
        assert d["output_chars"] == 500
        assert d["truncated"] is False
        assert d["tool_calls"] == 2
        assert d["timestamp"] == "2024-01-01T00:00:00"

    def test_defaults(self):
        stats = SkillExecutionStats()
        assert stats.wall_time_ms == 0.0
        assert stats.truncated is False


# ---------------------------------------------------------------------------
# Output limits
# ---------------------------------------------------------------------------

class TestOutputLimits:
    async def test_large_output_truncated(self, skill_mgr: SkillManager):
        _create_skill(skill_mgr, "big_output", LARGE_OUTPUT_SKILL)
        result = await skill_mgr.execute("big_output", {})
        assert len(result) < 100000
        assert "truncated" in result.lower()

    async def test_normal_output_not_truncated(self, skill_mgr: SkillManager):
        _create_skill(skill_mgr)
        result = await skill_mgr.execute("test_skill", {"msg": "hi"})
        assert result == "Got: hi"
        assert "truncated" not in result

    async def test_truncation_stats_recorded(self, skill_mgr: SkillManager):
        _create_skill(skill_mgr, "big_output", LARGE_OUTPUT_SKILL)
        await skill_mgr.execute("big_output", {})
        skill = skill_mgr._skills["big_output"]
        assert skill.last_execution is not None
        assert skill.last_execution.truncated is True

    async def test_max_output_constant(self):
        assert MAX_SKILL_OUTPUT_CHARS == 50000


# ---------------------------------------------------------------------------
# Execution stats tracking
# ---------------------------------------------------------------------------

class TestExecutionStats:
    async def test_stats_recorded_after_execute(self, skill_mgr: SkillManager):
        _create_skill(skill_mgr)
        await skill_mgr.execute("test_skill", {"msg": "hello"})
        skill = skill_mgr._skills["test_skill"]
        assert skill.last_execution is not None
        assert skill.last_execution.wall_time_ms > 0
        assert skill.last_execution.output_chars > 0
        assert skill.last_execution.timestamp != ""

    async def test_total_executions_incremented(self, skill_mgr: SkillManager):
        _create_skill(skill_mgr)
        assert skill_mgr._skills["test_skill"].total_executions == 0
        await skill_mgr.execute("test_skill", {"msg": "a"})
        assert skill_mgr._skills["test_skill"].total_executions == 1
        await skill_mgr.execute("test_skill", {"msg": "b"})
        assert skill_mgr._skills["test_skill"].total_executions == 2

    async def test_stats_in_list_skills(self, skill_mgr: SkillManager):
        _create_skill(skill_mgr)
        await skill_mgr.execute("test_skill", {"msg": "hi"})
        listing = skill_mgr.list_skills()
        assert len(listing) == 1
        assert listing[0]["total_executions"] == 1
        assert listing[0]["last_execution"] is not None
        assert listing[0]["last_execution"]["output_chars"] > 0

    async def test_stats_in_get_skill_info(self, skill_mgr: SkillManager):
        _create_skill(skill_mgr)
        await skill_mgr.execute("test_skill", {"msg": "hi"})
        info = skill_mgr.get_skill_info("test_skill")
        assert info["total_executions"] == 1
        assert info["last_execution"]["wall_time_ms"] >= 0

    async def test_no_stats_before_execution(self, skill_mgr: SkillManager):
        _create_skill(skill_mgr)
        listing = skill_mgr.list_skills()
        assert listing[0]["last_execution"] is None
        assert listing[0]["total_executions"] == 0

    async def test_message_tracking(self, skill_mgr: SkillManager):
        callback = AsyncMock()
        _create_skill(skill_mgr, "tracker_skill", TRACKING_SKILL)
        await skill_mgr.execute("tracker_skill", {}, message_callback=callback)
        skill = skill_mgr._skills["tracker_skill"]
        assert skill.last_execution is not None
        assert skill.last_execution.messages_sent == 1


# ---------------------------------------------------------------------------
# SkillContext file access restriction
# ---------------------------------------------------------------------------

class TestSkillContextFileRestriction:
    @pytest.fixture
    def context(self, tools_config: ToolsConfig) -> SkillContext:
        executor = ToolExecutor(tools_config)
        return SkillContext(executor, "test_skill")

    async def test_read_file_denied_path(self, context: SkillContext):
        result = await context.read_file("server", "/etc/shadow")
        assert "access denied" in result.lower()

    async def test_read_file_denied_env(self, context: SkillContext):
        result = await context.read_file("server", "/app/.env")
        assert "access denied" in result.lower()

    async def test_read_file_denied_ssh_key(self, context: SkillContext):
        result = await context.read_file("server", "/root/.ssh/id_ed25519")
        assert "access denied" in result.lower()

    @patch("src.tools.executor.ToolExecutor.execute", new_callable=AsyncMock)
    async def test_read_file_allowed_path(self, mock_exec, context: SkillContext):
        mock_exec.return_value = "file content"
        result = await context.read_file("server", "/var/log/syslog")
        assert result == "file content"

    async def test_execute_tool_read_file_denied(self, context: SkillContext):
        """execute_tool('read_file', ...) also checks path restrictions."""
        result = await context.execute_tool("read_file", {"host": "server", "path": "/etc/shadow"})
        assert "access denied" in result.lower()

    @patch("src.tools.executor.ToolExecutor.execute", new_callable=AsyncMock)
    async def test_execute_tool_read_file_allowed(self, mock_exec, context: SkillContext):
        mock_exec.return_value = "ok"
        result = await context.execute_tool("read_file", {"host": "server", "path": "/tmp/out.log"})
        assert result == "ok"


# ---------------------------------------------------------------------------
# SkillContext URL restriction
# ---------------------------------------------------------------------------

class TestSkillContextURLRestriction:
    @pytest.fixture
    def context(self, tools_config: ToolsConfig) -> SkillContext:
        executor = ToolExecutor(tools_config)
        return SkillContext(executor, "test_skill")

    async def test_http_get_localhost_blocked(self, context: SkillContext):
        result = await context.http_get("http://localhost:8080/api")
        assert "access denied" in result.lower()

    async def test_http_get_private_ip_blocked(self, context: SkillContext):
        result = await context.http_get("http://10.0.0.1:3000/data")
        assert "access denied" in result.lower()

    async def test_http_post_localhost_blocked(self, context: SkillContext):
        result = await context.http_post("http://127.0.0.1/webhook", json={"a": 1})
        assert "access denied" in result.lower()

    async def test_http_get_metadata_blocked(self, context: SkillContext):
        result = await context.http_get("http://169.254.169.254/latest/meta-data/")
        assert "access denied" in result.lower()

    async def test_http_post_metadata_blocked(self, context: SkillContext):
        result = await context.http_post("http://metadata.google.internal/computeMetadata/v1/")
        assert "access denied" in result.lower()


# ---------------------------------------------------------------------------
# Rate limits — tool calls
# ---------------------------------------------------------------------------

class TestToolCallLimit:
    @pytest.fixture
    def context(self, tools_config: ToolsConfig) -> SkillContext:
        executor = ToolExecutor(tools_config)
        tracker = ResourceTracker(tool_calls=MAX_SKILL_TOOL_CALLS)
        return SkillContext(executor, "test_skill", resource_tracker=tracker)

    async def test_tool_calls_exceeded(self, context: SkillContext):
        result = await context.execute_tool("check_disk", {"host": "server"})
        assert "limit" in result.lower()
        assert str(MAX_SKILL_TOOL_CALLS) in result

    @patch("src.tools.executor.ToolExecutor.execute", new_callable=AsyncMock)
    async def test_tool_calls_within_limit(self, mock_exec, tools_config: ToolsConfig):
        executor = ToolExecutor(tools_config)
        tracker = ResourceTracker(tool_calls=0)
        ctx = SkillContext(executor, "test_skill", resource_tracker=tracker)
        mock_exec.return_value = "ok"
        result = await ctx.execute_tool("check_disk", {"host": "server"})
        assert result == "ok"
        assert tracker.tool_calls == 1


# ---------------------------------------------------------------------------
# Rate limits — HTTP requests
# ---------------------------------------------------------------------------

class TestHTTPRequestLimit:
    @pytest.fixture
    def context(self, tools_config: ToolsConfig) -> SkillContext:
        executor = ToolExecutor(tools_config)
        tracker = ResourceTracker(http_requests=MAX_SKILL_HTTP_REQUESTS)
        return SkillContext(executor, "test_skill", resource_tracker=tracker)

    async def test_http_get_exceeded(self, context: SkillContext):
        result = await context.http_get("https://example.com/api")
        assert "limit" in result.lower()

    async def test_http_post_exceeded(self, context: SkillContext):
        result = await context.http_post("https://example.com/api")
        assert "limit" in result.lower()


# ---------------------------------------------------------------------------
# Rate limits — messages
# ---------------------------------------------------------------------------

class TestMessageLimit:
    async def test_message_limit_enforced(self, tools_config: ToolsConfig):
        executor = ToolExecutor(tools_config)
        tracker = ResourceTracker(messages_sent=MAX_SKILL_MESSAGES)
        callback = AsyncMock()
        ctx = SkillContext(
            executor, "test_skill",
            message_callback=callback,
            resource_tracker=tracker,
        )
        await ctx.post_message("should be blocked")
        callback.assert_not_called()

    async def test_file_limit_enforced(self, tools_config: ToolsConfig):
        executor = ToolExecutor(tools_config)
        tracker = ResourceTracker(files_sent=MAX_SKILL_MESSAGES)
        callback = AsyncMock()
        ctx = SkillContext(
            executor, "test_skill",
            file_callback=callback,
            resource_tracker=tracker,
        )
        await ctx.post_file(b"data", "file.txt")
        callback.assert_not_called()

    async def test_messages_within_limit(self, tools_config: ToolsConfig):
        executor = ToolExecutor(tools_config)
        tracker = ResourceTracker()
        callback = AsyncMock()
        ctx = SkillContext(
            executor, "test_skill",
            message_callback=callback,
            resource_tracker=tracker,
        )
        await ctx.post_message("hello")
        callback.assert_called_once_with("hello")
        assert tracker.messages_sent == 1


# ---------------------------------------------------------------------------
# Resource tracker in context
# ---------------------------------------------------------------------------

class TestResourceTrackerInContext:
    async def test_tracker_default_created(self, tools_config: ToolsConfig):
        executor = ToolExecutor(tools_config)
        ctx = SkillContext(executor, "test_skill")
        assert ctx._tracker is not None
        assert ctx._tracker.tool_calls == 0

    async def test_tracker_injected(self, tools_config: ToolsConfig):
        executor = ToolExecutor(tools_config)
        tracker = ResourceTracker(tool_calls=5)
        ctx = SkillContext(executor, "test_skill", resource_tracker=tracker)
        assert ctx._tracker.tool_calls == 5

    @patch("src.tools.executor.ToolExecutor.execute", new_callable=AsyncMock)
    async def test_tracker_increments_on_tool_call(self, mock_exec, tools_config: ToolsConfig):
        executor = ToolExecutor(tools_config)
        tracker = ResourceTracker()
        ctx = SkillContext(executor, "test_skill", resource_tracker=tracker)
        mock_exec.return_value = "ok"
        await ctx.execute_tool("check_disk", {"host": "server"})
        await ctx.execute_tool("check_memory", {"host": "server"})
        assert tracker.tool_calls == 2


# ---------------------------------------------------------------------------
# Integration: sandbox limits with full skill execution
# ---------------------------------------------------------------------------

class TestSandboxIntegration:
    async def test_stats_recorded_on_timeout(self, skill_mgr: SkillManager):
        """Even on timeout, stats are recorded."""
        timeout_skill = '''
SKILL_DEFINITION = {
    "name": "slow_skill",
    "description": "Takes forever",
    "input_schema": {"type": "object", "properties": {}},
}

import asyncio

async def execute(inp, context):
    await asyncio.sleep(999)
    return "done"
'''
        _create_skill(skill_mgr, "slow_skill", timeout_skill)
        # Patch timeout to be very short
        with patch("src.tools.skill_manager.SKILL_EXECUTE_TIMEOUT", 0.01):
            result = await skill_mgr.execute("slow_skill", {})
        assert "timed out" in result.lower()
        skill = skill_mgr._skills["slow_skill"]
        assert skill.last_execution is not None
        assert skill.total_executions == 1

    async def test_stats_recorded_on_error(self, skill_mgr: SkillManager):
        """Even on error, stats are recorded."""
        error_skill = '''
SKILL_DEFINITION = {
    "name": "err_skill",
    "description": "Always errors",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    raise ValueError("boom")
'''
        _create_skill(skill_mgr, "err_skill", error_skill)
        result = await skill_mgr.execute("err_skill", {})
        assert "error" in result.lower()
        skill = skill_mgr._skills["err_skill"]
        assert skill.last_execution is not None
        assert skill.total_executions == 1

    async def test_disabled_skill_no_stats(self, skill_mgr: SkillManager):
        """Disabled skills don't get stats recorded."""
        _create_skill(skill_mgr)
        skill_mgr.disable_skill("test_skill")
        result = await skill_mgr.execute("test_skill", {})
        assert "disabled" in result.lower()
        skill = skill_mgr._skills["test_skill"]
        assert skill.last_execution is None
        assert skill.total_executions == 0


# ---------------------------------------------------------------------------
# Sandbox constants
# ---------------------------------------------------------------------------

class TestSandboxConstants:
    def test_tool_call_limit(self):
        assert MAX_SKILL_TOOL_CALLS == 50

    def test_http_request_limit(self):
        assert MAX_SKILL_HTTP_REQUESTS == 20

    def test_message_limit(self):
        assert MAX_SKILL_MESSAGES == 10

    def test_output_limit(self):
        assert MAX_SKILL_OUTPUT_CHARS == 50000
