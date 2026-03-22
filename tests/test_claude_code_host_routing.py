"""Tests for context-aware claude_code host/directory routing.

Round 2, Session 8: The claude_code routing path hardcoded host='desktop'
and working_directory='/root/project'. This meant all code analysis went
to the desktop even when the user asked about server files, production
config, or server-hosted services.

resolve_claude_code_target() analyzes the message content to pick the
right host (server vs desktop) and working directory. Only these two
hosts have claude CLI installed.

Tests cover:
- Default routing: generic code messages → desktop
- Explicit server mentions: "on server", "on the server"
- Server service config patterns: grafana/prometheus/loki/gitea/nginx + config
- Server paths: /opt/ and /opt/project
- Production context: "production config", "deployed version"
- False negatives: messages that look server-related but should stay on desktop
- Case insensitivity
- Edge cases (empty, ambiguous)
- Integration: client.py passes resolved host to _handle_claude_code
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import LokiBot  # noqa: E402
from src.discord.routing import (  # noqa: E402
    CLAUDE_CODE_DEFAULTS,
    resolve_claude_code_target,
)


class TestResolveClaudeCodeTargetDefaults:
    """Generic code analysis messages should route to desktop."""

    @pytest.mark.parametrize("msg", [
        "review my code for bugs",
        "explain the check_service function",
        "what does this class do?",
        "write a python script to parse logs",
        "refactor the routing module",
        "summarize the recent commits",
        "how does the session manager work?",
        "can you analyze this code?",
        "",
    ])
    def test_generic_messages_route_to_desktop(self, msg):
        host, directory = resolve_claude_code_target(msg)
        assert host == "desktop"
        assert directory == CLAUDE_CODE_DEFAULTS["desktop"]

    def test_default_directory_is_root_project(self):
        _, directory = resolve_claude_code_target("review the code")
        assert directory == "/root/project"


class TestResolveClaudeCodeTargetExplicitServer:
    """Explicit server mentions should route to server."""

    @pytest.mark.parametrize("msg", [
        "check the code on server",
        "look at the config on the server",
        "review the changes on server",
        "what's in the config on server?",
        "read the logs on the server please",
    ])
    def test_on_server_routes_to_server(self, msg):
        host, directory = resolve_claude_code_target(msg)
        assert host == "server"
        assert directory == CLAUDE_CODE_DEFAULTS["server"]

    @pytest.mark.parametrize("msg", [
        "server config looks wrong",
        "check the server logs",
        "read the server files",
        "server setup needs review",
        "analyze the server version",
    ])
    def test_server_noun_phrases_route_to_server(self, msg):
        host, directory = resolve_claude_code_target(msg)
        assert host == "server"
        assert directory == "/opt/project"


class TestResolveClaudeCodeTargetServerPaths:
    """Messages mentioning server-specific paths should route to server."""

    @pytest.mark.parametrize("msg", [
        "read the file at /opt/project/config.yml",
        "what's in /opt/project/docker-compose.yml?",
        "look at /opt/some-other-app/main.py",
        "check /opt/grafana/defaults.ini",
    ])
    def test_opt_paths_route_to_server(self, msg):
        host, directory = resolve_claude_code_target(msg)
        assert host == "server"
        assert directory == "/opt/project"


class TestResolveClaudeCodeTargetServiceConfigs:
    """Server-hosted service + config context should route to server."""

    @pytest.mark.parametrize("msg", [
        "show me the grafana config",
        "review the prometheus rules",
        "check the loki config file",
        "analyze the gitea config",
        "read the nginx conf",
        "grafana dashboard setup",
        "prometheus config looks wrong",
        "nginx setup needs work",
    ])
    def test_service_config_routes_to_server(self, msg):
        host, directory = resolve_claude_code_target(msg)
        assert host == "server"
        assert directory == "/opt/project"

    @pytest.mark.parametrize("msg", [
        "config for grafana seems broken",
        "setup for prometheus monitoring",
        "check the config for nginx",
        "review the conf for loki",
    ])
    def test_config_for_service_routes_to_server(self, msg):
        host, directory = resolve_claude_code_target(msg)
        assert host == "server"
        assert directory == "/opt/project"


class TestResolveClaudeCodeTargetProductionContext:
    """Production/deployment context should route to server."""

    @pytest.mark.parametrize("msg", [
        "check the production config",
        "compare the deployed version",
        "what's in the deployed code?",
        "review the production files",
        "check the running version",
        "running config looks outdated",
    ])
    def test_production_context_routes_to_server(self, msg):
        host, directory = resolve_claude_code_target(msg)
        assert host == "server"
        assert directory == "/opt/project"


class TestResolveClaudeCodeTargetFalseNegatives:
    """Messages that mention server-adjacent terms but should stay on desktop."""

    @pytest.mark.parametrize("msg", [
        # Mentioning services without config/file context — could be querying them
        "what is grafana?",
        "how does prometheus work?",
        "explain nginx reverse proxying",
        "what does loki do?",
        # Desktop-specific references
        "look at the code on desktop",
        "check /root/project/src/main.py",
        # Generic code analysis (no host/service indicator)
        "review the routing module",
        "explain the session manager class",
        "what does _handle_claude_code do?",
        # Mentions "server" as part of code, not the machine
        # (these are tricky — "server" word boundary matches, but
        # as a variable or module name it's still caught. Accepted trade-off.)
    ])
    def test_stays_on_desktop(self, msg):
        host, _ = resolve_claude_code_target(msg)
        assert host == "desktop"


class TestResolveClaudeCodeTargetCaseInsensitivity:
    """Patterns should match regardless of case."""

    @pytest.mark.parametrize("msg", [
        "Check the code ON SERVER",
        "SERVER CONFIG needs review",
        "GRAFANA CONFIG is wrong",
        "Read /OPT/project/config.yml",
        "PRODUCTION CONFIG check",
    ])
    def test_case_insensitive_matching(self, msg):
        host, _ = resolve_claude_code_target(msg)
        assert host == "server"


class TestResolveClaudeCodeTargetEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_message(self):
        host, directory = resolve_claude_code_target("")
        assert host == "desktop"
        assert directory == "/root/project"

    def test_whitespace_only(self):
        host, _ = resolve_claude_code_target("   ")
        assert host == "desktop"

    def test_server_in_non_matching_context(self):
        """'server' without 'on' prefix or config/log suffix stays desktop."""
        host, _ = resolve_claude_code_target("the health server module")
        assert host == "desktop"

    def test_multiple_indicators_still_routes_server(self):
        host, _ = resolve_claude_code_target(
            "check the grafana config on the server at /opt/project"
        )
        assert host == "server"

    def test_return_type(self):
        result = resolve_claude_code_target("anything")
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestClaudeCodeDefaults:
    """CLAUDE_CODE_DEFAULTS dict is correct."""

    def test_desktop_default(self):
        assert CLAUDE_CODE_DEFAULTS["desktop"] == "/root/project"

    def test_server_default(self):
        assert CLAUDE_CODE_DEFAULTS["server"] == "/opt/project"

    def test_only_two_hosts(self):
        assert set(CLAUDE_CODE_DEFAULTS.keys()) == {"desktop", "server"}


# --- Integration tests: client.py uses resolved host ---


def _make_bot_stub():
    """Minimal LokiBot stub for routing tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._last_tool_use = {}
    stub._system_prompt = "initial system prompt"
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = []
    stub.config.tools.approval_timeout_seconds = 30
    stub.sessions = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock(return_value=True)
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.claude = MagicMock()
    stub.classifier.classify = AsyncMock(return_value="claude_code")
    stub.codex_client = None
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor._handle_claude_code = AsyncMock(
        return_value="Analysis result."
    )
    stub._build_system_prompt = MagicMock(return_value="full system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat system prompt")
    stub._send_with_retry = AsyncMock()
    stub._send_chunked = AsyncMock()
    stub._process_with_tools = AsyncMock(
        return_value=("Codex fallback response", False, False, [], False)
    )
    stub._merged_tool_definitions = MagicMock(return_value=[])
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.voice_manager = None
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._pending_files = {}
    return stub


def _make_message(channel_id="chan-1"):
    msg = AsyncMock()
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.author = MagicMock()
    msg.author.id = "user-1"
    msg.reply = AsyncMock()
    return msg


class TestIntegrationServerRouting:
    """client.py should pass resolved host/dir to _handle_claude_code."""

    async def test_server_message_routes_to_server_host(self):
        """Message mentioning 'on server' should pass host='server'."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(
                msg, "review the config on server", "chan-1"
            )

        stub.tool_executor._handle_claude_code.assert_called_once()
        call_args = stub.tool_executor._handle_claude_code.call_args[0][0]
        assert call_args["host"] == "server"
        assert call_args["working_directory"] == "/opt/project"

    async def test_generic_message_routes_to_desktop_host(self):
        """Generic code analysis message should pass host='desktop'."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(
                msg, "explain the session manager", "chan-1"
            )

        call_args = stub.tool_executor._handle_claude_code.call_args[0][0]
        assert call_args["host"] == "desktop"
        assert call_args["working_directory"] == "/root/project"

    async def test_keyword_bypass_with_server_mention(self):
        """Keyword bypass path should also use resolved host."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        # "review code on server" — matches claude_code keyword AND server indicator
        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(
                msg, "review the code on server", "chan-1"
            )

        call_args = stub.tool_executor._handle_claude_code.call_args[0][0]
        assert call_args["host"] == "server"
        assert call_args["working_directory"] == "/opt/project"

    async def test_grafana_config_routes_to_server(self):
        """Message about grafana config should route to server."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(
                msg, "analyze the grafana config", "chan-1"
            )

        call_args = stub.tool_executor._handle_claude_code.call_args[0][0]
        assert call_args["host"] == "server"

    async def test_opt_path_routes_to_server(self):
        """Message mentioning /opt/ path should route to server."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(
                msg, "read /opt/project/docker-compose.yml", "chan-1"
            )

        call_args = stub.tool_executor._handle_claude_code.call_args[0][0]
        assert call_args["host"] == "server"
        assert call_args["working_directory"] == "/opt/project"

    async def test_allow_edits_always_false_on_routing_path(self):
        """Routing path should always set allow_edits=False regardless of host."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(
                msg, "check the config on server", "chan-1"
            )

        call_args = stub.tool_executor._handle_claude_code.call_args[0][0]
        assert call_args["allow_edits"] is False

    async def test_max_output_chars_preserved(self):
        """Routing path should still pass max_output_chars=8000 for both hosts."""
        stub = _make_bot_stub()
        msg = _make_message()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(
                msg, "review the server config", "chan-1"
            )

        call_args = stub.tool_executor._handle_claude_code.call_args[0][0]
        assert call_args["max_output_chars"] == 8000
