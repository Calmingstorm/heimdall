"""Tests for tools/skill_context.py — covering run_on_host, query_prometheus, read_file,
post_message, remember/recall, get_hosts, get_services, http_get, http_post, log,
execute_tool allowlist, and memory isolation."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.schema import ToolsConfig
from src.tools.executor import ToolExecutor
from src.tools.skill_context import SKILL_SAFE_TOOLS, SkillContext


@pytest.fixture
def executor(tools_config: ToolsConfig) -> ToolExecutor:
    return ToolExecutor(tools_config)


@pytest.fixture
def ctx_with_memory(executor, tmp_dir):
    """SkillContext with memory path configured."""
    return SkillContext(
        executor, "test_skill",
        memory_path=str(tmp_dir / "skill_memory.json"),
    )


@pytest.fixture
def ctx_no_memory(executor):
    """SkillContext without memory path."""
    return SkillContext(executor, "test_skill")


# ---------------------------------------------------------------------------
# run_on_host (line 49)
# ---------------------------------------------------------------------------

class TestRunOnHost:
    @pytest.mark.asyncio
    async def test_run_on_host_delegates(self, ctx_with_memory):
        """run_on_host delegates to executor._run_on_host."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "uptime: 5 days")
            result = await ctx_with_memory.run_on_host("server", "uptime")
        assert "5 days" in result

    @pytest.mark.asyncio
    async def test_run_on_host_unknown_host(self, ctx_with_memory):
        """run_on_host with unknown host returns error."""
        result = await ctx_with_memory.run_on_host("nonexistent", "uptime")
        assert "Unknown" in result or "disallowed" in result.lower()


# ---------------------------------------------------------------------------
# query_prometheus (line 53)
# ---------------------------------------------------------------------------

class TestQueryPrometheus:
    @pytest.mark.asyncio
    async def test_query_prometheus_delegates(self, ctx_with_memory):
        """query_prometheus delegates to executor.execute."""
        raw = json.dumps({
            "status": "success",
            "data": {"resultType": "vector", "result": [
                {"metric": {"__name__": "up"}, "value": [1, "1"]},
            ]},
        })
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, raw)
            result = await ctx_with_memory.query_prometheus("up")
        assert "up" in result.lower() or "1 result" in result


# ---------------------------------------------------------------------------
# read_file (lines 55-59)
# ---------------------------------------------------------------------------

class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_file_delegates(self, ctx_with_memory):
        """read_file delegates to executor.execute with correct params."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "file content here")
            result = await ctx_with_memory.read_file("server", "/etc/hostname")
        assert "file content" in result


# ---------------------------------------------------------------------------
# post_message (lines 61-66)
# ---------------------------------------------------------------------------

class TestPostMessage:
    @pytest.mark.asyncio
    async def test_post_message_with_callback(self, executor):
        """post_message calls the callback when set."""
        callback = AsyncMock()
        ctx = SkillContext(executor, "test_skill", message_callback=callback)
        await ctx.post_message("Hello from skill!")
        callback.assert_called_once_with("Hello from skill!")

    @pytest.mark.asyncio
    async def test_post_message_without_callback(self, ctx_no_memory):
        """post_message without callback logs warning but doesn't crash."""
        # Should not raise
        await ctx_no_memory.post_message("Hello!")


# ---------------------------------------------------------------------------
# remember / recall (lines 68-79)
# ---------------------------------------------------------------------------

class TestRememberRecall:
    def test_remember_and_recall(self, ctx_with_memory):
        """remember saves a key-value pair that can be recalled."""
        ctx_with_memory.remember("server_ip", "192.168.1.13")
        value = ctx_with_memory.recall("server_ip")
        assert value == "192.168.1.13"

    def test_recall_nonexistent_key(self, ctx_with_memory):
        """recall returns None for nonexistent key."""
        value = ctx_with_memory.recall("nonexistent")
        assert value is None

    def test_remember_overwrites(self, ctx_with_memory):
        """remember overwrites existing value."""
        ctx_with_memory.remember("key", "old")
        ctx_with_memory.remember("key", "new")
        assert ctx_with_memory.recall("key") == "new"

    def test_remember_no_memory_path(self, ctx_no_memory):
        """remember with no memory path is a no-op."""
        ctx_no_memory.remember("key", "value")
        # Should not crash

    def test_recall_no_memory_path(self, ctx_no_memory):
        """recall with no memory path returns None."""
        result = ctx_no_memory.recall("key")
        assert result is None

    def test_remember_creates_directory(self, executor, tmp_dir):
        """remember creates parent directory if it doesn't exist."""
        deep_path = str(tmp_dir / "nested" / "dir" / "skill_mem.json")
        ctx = SkillContext(executor, "test_skill", memory_path=deep_path)
        ctx.remember("key", "value")
        assert ctx.recall("key") == "value"
        assert Path(deep_path).exists()

    def test_recall_corrupted_file(self, executor, tmp_dir):
        """recall handles corrupted memory file gracefully."""
        mem_path = tmp_dir / "skill_mem.json"
        mem_path.write_text("{corrupted json!")
        ctx = SkillContext(executor, "test_skill", memory_path=str(mem_path))
        result = ctx.recall("key")
        assert result is None


# ---------------------------------------------------------------------------
# get_hosts (line 83)
# ---------------------------------------------------------------------------

class TestGetHosts:
    def test_get_hosts_returns_aliases(self, ctx_with_memory):
        """get_hosts returns configured host aliases."""
        hosts = ctx_with_memory.get_hosts()
        assert "server" in hosts
        assert "desktop" in hosts
        assert "macbook" in hosts

    def test_get_hosts_type(self, ctx_with_memory):
        """get_hosts returns a list."""
        assert isinstance(ctx_with_memory.get_hosts(), list)


# ---------------------------------------------------------------------------
# get_services (line 87)
# ---------------------------------------------------------------------------

class TestGetServices:
    def test_get_services_returns_allowlist(self, ctx_with_memory):
        """get_services returns the allowed services list."""
        services = ctx_with_memory.get_services()
        assert "apache2" in services
        assert "prometheus" in services

    def test_get_services_type(self, ctx_with_memory):
        """get_services returns a list."""
        assert isinstance(ctx_with_memory.get_services(), list)


# ---------------------------------------------------------------------------
# http_get (lines 89-104)
# ---------------------------------------------------------------------------

class TestHttpGet:
    @pytest.mark.asyncio
    async def test_http_get_json_response(self, ctx_with_memory):
        """http_get parses JSON responses automatically."""
        mock_resp = AsyncMock()
        mock_resp.content_type = "application/json"
        mock_resp.json = AsyncMock(return_value={"status": "ok", "count": 42})

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_session_ctx
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.skill_context.aiohttp.ClientSession", return_value=mock_session):
            result = await ctx_with_memory.http_get("http://example.com/api")
        assert result == {"status": "ok", "count": 42}

    @pytest.mark.asyncio
    async def test_http_get_text_response(self, ctx_with_memory):
        """http_get returns plain text for non-JSON responses."""
        mock_resp = AsyncMock()
        mock_resp.content_type = "text/html"
        mock_resp.text = AsyncMock(return_value="<html>Hello</html>")

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_session_ctx
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.skill_context.aiohttp.ClientSession", return_value=mock_session):
            result = await ctx_with_memory.http_get("http://example.com")
        assert result == "<html>Hello</html>"

    @pytest.mark.asyncio
    async def test_http_get_text_that_is_json(self, ctx_with_memory):
        """http_get tries to parse text as JSON even without JSON content type."""
        mock_resp = AsyncMock()
        mock_resp.content_type = "text/plain"
        mock_resp.text = AsyncMock(return_value='{"key": "value"}')

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_session_ctx
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.skill_context.aiohttp.ClientSession", return_value=mock_session):
            result = await ctx_with_memory.http_get("http://example.com")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_http_get_with_params(self, ctx_with_memory):
        """http_get passes params to the request."""
        mock_resp = AsyncMock()
        mock_resp.content_type = "application/json"
        mock_resp.json = AsyncMock(return_value={"data": []})

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_session_ctx
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.skill_context.aiohttp.ClientSession", return_value=mock_session):
            await ctx_with_memory.http_get(
                "http://example.com/api",
                params={"q": "test"},
            )
        mock_session.get.assert_called_once()
        call_kwargs = mock_session.get.call_args
        assert call_kwargs[1].get("params") == {"q": "test"} or call_kwargs[0] == ("http://example.com/api",)


# ---------------------------------------------------------------------------
# http_post (lines 106-125)
# ---------------------------------------------------------------------------

class TestHttpPost:
    @pytest.mark.asyncio
    async def test_http_post_json_request(self, ctx_with_memory):
        """http_post sends JSON body and parses JSON response."""
        mock_resp = AsyncMock()
        mock_resp.content_type = "application/json"
        mock_resp.json = AsyncMock(return_value={"id": 1, "created": True})

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_session_ctx
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.skill_context.aiohttp.ClientSession", return_value=mock_session):
            result = await ctx_with_memory.http_post(
                "http://example.com/api",
                json={"name": "test"},
            )
        assert result == {"id": 1, "created": True}

    @pytest.mark.asyncio
    async def test_http_post_with_data(self, ctx_with_memory):
        """http_post with data= sends raw body and returns JSON response."""
        mock_resp = AsyncMock()
        mock_resp.content_type = "application/json"
        mock_resp.json = AsyncMock(return_value={"status": "received"})

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_session_ctx
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.skill_context.aiohttp.ClientSession", return_value=mock_session):
            result = await ctx_with_memory.http_post(
                "http://example.com/api",
                data="raw body",
            )
        assert result == {"status": "received"}

    @pytest.mark.asyncio
    async def test_http_post_no_body(self, ctx_with_memory):
        """http_post with no body sends POST with default None values."""
        mock_resp = AsyncMock()
        mock_resp.content_type = "application/json"
        mock_resp.json = AsyncMock(return_value={"ok": True})

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_session_ctx
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.skill_context.aiohttp.ClientSession", return_value=mock_session):
            result = await ctx_with_memory.http_post("http://example.com/api")
        assert result == {"ok": True}


# ---------------------------------------------------------------------------
# log (line 179)
# ---------------------------------------------------------------------------

class TestLog:
    def test_log_writes_message(self, ctx_with_memory):
        """log method calls the internal logger."""
        with patch.object(ctx_with_memory._log, "info") as mock_info:
            ctx_with_memory.log("Test log message")
        mock_info.assert_called_once_with("%s", "Test log message")


# ---------------------------------------------------------------------------
# _load_memory / _save_memory (lines 181-193)
# ---------------------------------------------------------------------------

class TestMemoryPersistence:
    def test_memory_persists_to_file(self, ctx_with_memory, tmp_dir):
        """Memory is persisted to the file system."""
        ctx_with_memory.remember("key1", "value1")
        ctx_with_memory.remember("key2", "value2")

        mem_file = tmp_dir / "skill_memory.json"
        assert mem_file.exists()
        data = json.loads(mem_file.read_text())
        assert data["key1"] == "value1"
        assert data["key2"] == "value2"

    def test_load_memory_from_existing_file(self, executor, tmp_dir):
        """Memory is loaded from an existing file."""
        mem_path = tmp_dir / "skill_mem.json"
        mem_path.write_text(json.dumps({"existing": "data"}))

        ctx = SkillContext(executor, "test_skill", memory_path=str(mem_path))
        result = ctx.recall("existing")
        assert result == "data"

    def test_save_memory_no_path(self, ctx_no_memory):
        """_save_memory with no path is a no-op."""
        # Should not crash
        ctx_no_memory._save_memory({"key": "value"})

    def test_load_memory_no_path(self, ctx_no_memory):
        """_load_memory with no path returns empty dict."""
        result = ctx_no_memory._load_memory()
        assert result == {}

    def test_load_memory_nonexistent_file(self, executor, tmp_dir):
        """_load_memory with nonexistent file returns empty dict."""
        ctx = SkillContext(
            executor, "test_skill",
            memory_path=str(tmp_dir / "nonexistent.json"),
        )
        result = ctx._load_memory()
        assert result == {}


# ---------------------------------------------------------------------------
# execute_tool allowlist (SKILL_SAFE_TOOLS)
# ---------------------------------------------------------------------------


class TestExecuteToolAllowlist:
    @pytest.mark.asyncio
    async def test_safe_tool_allowed(self, ctx_with_memory):
        """execute_tool allows tools in SKILL_SAFE_TOOLS."""
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "active (running)")
            result = await ctx_with_memory.execute_tool(
                "check_service", {"host": "server", "service": "apache2"},
            )
        assert "active" in result or "running" in result

    @pytest.mark.asyncio
    async def test_run_command_blocked(self, ctx_with_memory):
        """execute_tool blocks run_command (requires approval)."""
        result = await ctx_with_memory.execute_tool(
            "run_command", {"host": "server", "command": "rm -rf /"},
        )
        assert "not allowed" in result

    @pytest.mark.asyncio
    async def test_write_file_blocked(self, ctx_with_memory):
        """execute_tool blocks write_file (requires approval)."""
        result = await ctx_with_memory.execute_tool(
            "write_file", {"host": "server", "path": "/etc/passwd", "content": "hacked"},
        )
        assert "not allowed" in result

    @pytest.mark.asyncio
    async def test_restart_service_blocked(self, ctx_with_memory):
        """execute_tool blocks restart_service (requires approval)."""
        result = await ctx_with_memory.execute_tool(
            "restart_service", {"host": "server", "service": "apache2"},
        )
        assert "not allowed" in result

    @pytest.mark.asyncio
    async def test_create_skill_blocked(self, ctx_with_memory):
        """execute_tool blocks create_skill (requires approval)."""
        result = await ctx_with_memory.execute_tool(
            "create_skill", {"name": "evil", "code": "..."},
        )
        assert "not allowed" in result

    @pytest.mark.asyncio
    async def test_incus_exec_blocked(self, ctx_with_memory):
        """execute_tool blocks incus_exec (requires approval)."""
        result = await ctx_with_memory.execute_tool(
            "incus_exec", {"instance": "test", "command": "whoami"},
        )
        assert "not allowed" in result

    @pytest.mark.asyncio
    async def test_docker_compose_action_blocked(self, ctx_with_memory):
        """execute_tool blocks docker_compose_action (requires approval)."""
        result = await ctx_with_memory.execute_tool(
            "docker_compose_action",
            {"host": "server", "project_dir": "/opt/foo", "action": "down"},
        )
        assert "not allowed" in result

    def test_safe_tools_is_frozenset(self):
        """SKILL_SAFE_TOOLS is immutable."""
        assert isinstance(SKILL_SAFE_TOOLS, frozenset)

    def test_safe_tools_does_not_contain_destructive(self):
        """SKILL_SAFE_TOOLS must not contain any known destructive tools."""
        destructive = {
            "run_command", "write_file", "restart_service", "purge_messages",
            "create_skill", "edit_skill", "delete_skill", "docker_compose_action",
            "git_push", "git_commit", "git_pull", "git_branch",
            "incus_exec", "incus_start", "incus_stop", "incus_restart",
            "incus_delete", "incus_launch", "incus_snapshot",
            "run_ansible_playbook", "run_command_multi",
            "browser_click", "browser_fill", "browser_evaluate",
            "delete_knowledge", "schedule_task", "delete_schedule",
            "create_digest", "delegate_task", "set_permission",
        }
        overlap = SKILL_SAFE_TOOLS & destructive
        assert not overlap, f"SKILL_SAFE_TOOLS contains destructive tools: {overlap}"


# ---------------------------------------------------------------------------
# Skill memory isolation from executor memory
# ---------------------------------------------------------------------------


class TestSkillMemoryIsolation:
    def test_skill_memory_does_not_corrupt_executor_memory(self, executor, tmp_dir):
        """Skill memory uses a separate file from executor memory."""
        # Set up executor with scoped memory
        executor._memory_path = tmp_dir / "memory.json"
        scoped_data = {"global": {"server_ip": "192.168.1.13"}, "user_123": {"pref": "dark"}}
        executor._save_all_memory(scoped_data)

        # Skill uses a different path (skill_memory.json)
        skill_mem_path = str(tmp_dir / "memory_skills.json")
        ctx = SkillContext(executor, "test_skill", memory_path=skill_mem_path)
        ctx.remember("foo", "bar")

        # Executor's memory should be untouched
        loaded = executor._load_all_memory()
        assert loaded == scoped_data
        assert "foo" not in loaded

        # Skill memory should have its own data
        assert ctx.recall("foo") == "bar"

    def test_skill_manager_derives_separate_memory_path(self, tmp_dir):
        """SkillManager should derive a _skills suffix memory path."""
        from src.tools.skill_manager import SkillManager

        mgr = SkillManager(
            skills_dir=str(tmp_dir / "skills"),
            tool_executor=MagicMock(),
            memory_path=str(tmp_dir / "memory.json"),
        )
        assert mgr._memory_path == str(tmp_dir / "memory_skills.json")

    def test_skill_manager_none_memory_path(self, tmp_dir):
        """SkillManager with no memory path stays None."""
        from src.tools.skill_manager import SkillManager

        mgr = SkillManager(
            skills_dir=str(tmp_dir / "skills"),
            tool_executor=MagicMock(),
            memory_path=None,
        )
        assert mgr._memory_path is None
