"""Round 27 — Integration tests verifying everything works together.

Covers:
- System prompt size invariant (<5000 chars)
- Detection functions present and operational
- tool_choice is "auto"
- Config save error handling (write failure resilience)
- Web UI page routing completeness
- Chat interface end-to-end flow
- Tool registry integrity
- Session defense layers present
- Critical invariants that must hold across all rounds
"""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from src.config.schema import Config, WebConfig
from src.discord.client import (
    detect_code_hedging,
    detect_fabrication,
    detect_hedging,
    detect_premature_failure,
    _FABRICATION_RETRY_MSG,
    _HEDGING_RETRY_MSG,
    _CODE_HEDGING_RETRY_MSG,
)
from src.health.server import (
    SessionManager,
    _make_auth_middleware,
    _make_rate_limit_middleware,
    _make_security_headers_middleware,
)
from src.llm.secret_scrubber import scrub_output_secrets
from src.llm.system_prompt import SYSTEM_PROMPT_TEMPLATE, build_system_prompt
from src.tools.registry import TOOLS, get_tool_definitions
from src.web.api import (
    _deep_merge,
    _redact_config,
    _write_config,
    setup_api,
)
from src.web.chat import MAX_CHAT_CONTENT_LEN, process_web_chat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bot(*, config_dump=None):
    """Build a mock bot for API testing."""
    bot = MagicMock()
    guild = MagicMock()
    guild.id = 111
    guild.name = "TestGuild"
    guild.member_count = 10
    bot.guilds = [guild]
    bot.is_ready = MagicMock(return_value=True)
    bot._start_time = time.monotonic() - 60

    dump = config_dump or {
        "discord": {"token": "secret"},
        "tools": {"tool_packs": []},
        "web": {"api_token": "", "enabled": True},
    }
    bot.config = MagicMock()
    bot.config.model_dump = MagicMock(return_value=dump)
    bot.config.tools.

    bot._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command", "description": "Run", "input_schema": {}},
    ])
    bot._cached_merged_tools = None
    bot._cached_skills_text = None
    bot.skill_manager = MagicMock()
    bot.skill_manager.list_skills = MagicMock(return_value=[])
    bot.skill_manager._skills = {}

    bot.sessions = MagicMock()
    bot.sessions._sessions = {}
    bot.sessions.add_message = MagicMock()
    bot.sessions.remove_last_message = MagicMock()
    bot.sessions.prune = MagicMock()
    bot.sessions.save = MagicMock()
    bot.sessions.reset = MagicMock()
    bot.sessions.get_task_history = AsyncMock(return_value=[])

    bot.codex_client = MagicMock()
    bot._build_system_prompt = MagicMock(return_value="You are Heimdall.")
    bot._inject_tool_hints = AsyncMock(return_value="You are Heimdall.")
    bot._process_with_tools = AsyncMock(
        return_value=("Hello from Heimdall!", False, False, [], False)
    )

    bot.permissions = MagicMock()
    bot.audit = MagicMock()
    bot.audit.log_execution = AsyncMock()
    bot.audit.search = AsyncMock(return_value=[])
    bot.audit.count_by_tool = AsyncMock(return_value={})

    bot._knowledge_store = None
    bot._embedder = None
    bot.scheduler = MagicMock()
    bot.scheduler.list_all = MagicMock(return_value=[])
    bot.scheduler._schedules = []
    bot.scheduler._callback = None
    bot.loop_manager = MagicMock()
    bot.loop_manager.active_count = 0
    bot.loop_manager._loops = {}
    bot.tool_executor = MagicMock()
    bot.tool_executor._process_registry = None
    bot.tool_executor._load_all_memory = MagicMock(return_value={})
    bot.tool_executor._save_all_memory = MagicMock()
    bot.context_loader = MagicMock()
    bot._invalidate_prompt_caches = MagicMock()
    bot._system_prompt = "system prompt"

    return bot


def _make_app(bot=None, *, api_token=""):
    if bot is None:
        bot = _make_bot()
    web_config = WebConfig(api_token=api_token)
    app = web.Application(middlewares=[
        _make_security_headers_middleware(),
        _make_rate_limit_middleware(),
        _make_auth_middleware(web_config, SessionManager()),
    ])
    setup_api(app, bot)
    return app, bot


# ===================================================================
# 1. System Prompt Invariants
# ===================================================================


class TestSystemPromptInvariants:
    """System prompt must stay under 5000 chars in all configurations."""

    def test_template_under_5000_chars(self):
        assert len(SYSTEM_PROMPT_TEMPLATE) < 5000

    def test_built_prompt_under_5000_chars_minimal(self):
        """Minimal config still produces a valid prompt under limit."""
        prompt = build_system_prompt(
            context="", hosts={}
        )
        assert len(prompt) < 5000

    def test_built_prompt_under_5000_chars_moderate(self):
        """Moderate config (several hosts, services) stays under limit."""
        prompt = build_system_prompt(
            context="Production cluster with 3 nodes.",
            hosts={"web1": "10.0.0.1", "web2": "10.0.0.2", "db1": "10.0.0.3"},
        )
        assert len(prompt) < 5000

    def test_template_contains_executor_directive(self):
        """Prompt must contain the EXECUTOR behavior directive."""
        assert "EXECUTOR" in SYSTEM_PROMPT_TEMPLATE

    def test_template_contains_no_emojis_rule(self):
        """Prompt must contain the no-emojis rule."""
        assert "Never use emojis" in SYSTEM_PROMPT_TEMPLATE

    def test_template_contains_secret_protection(self):
        """Prompt must contain the secret protection rule."""
        assert "NEVER reveal API keys" in SYSTEM_PROMPT_TEMPLATE


# ===================================================================
# 2. Detection Functions Present and Working
# ===================================================================


class TestDetectionFunctionsPresent:
    """All detection functions must exist and be operational."""

    def test_detect_fabrication_exists(self):
        assert callable(detect_fabrication)

    def test_detect_hedging_exists(self):
        assert callable(detect_hedging)

    def test_detect_premature_failure_exists(self):
        assert callable(detect_premature_failure)

    def test_detect_code_hedging_exists(self):
        assert callable(detect_code_hedging)


class TestDetectionFunctionsWork:
    """Detection functions must correctly identify their respective patterns."""

    def test_fabrication_catches_fake_output(self):
        assert detect_fabrication(
            "I ran the command and got: disk usage is 50%", []
        )

    def test_fabrication_ignores_with_tools(self):
        assert not detect_fabrication(
            "I ran the command and got: disk usage is 50%", ["run_command"]
        )

    def test_fabrication_ignores_short_text(self):
        assert not detect_fabrication("ok", [])

    def test_hedging_catches_permission_asking(self):
        assert detect_hedging("Shall I check the disk usage for you?", [])

    def test_hedging_catches_would_you_like(self):
        assert detect_hedging("Would you like me to restart the service?", [])

    def test_hedging_ignores_with_tools(self):
        assert not detect_hedging("Shall I check?", ["run_command"])

    def test_premature_failure_catches_give_up(self):
        assert detect_premature_failure(
            "Failed to get the disk information, here is a workaround",
            ["run_command"],
        )

    def test_premature_failure_requires_tools(self):
        """Premature failure only fires when tools WERE used (partial exec)."""
        assert not detect_premature_failure(
            "Failed to get the disk information", []
        )

    def test_code_hedging_catches_bash_block(self):
        assert detect_code_hedging(
            "Try this:\n```bash\ndf -h\n```", []
        )

    def test_code_hedging_ignores_with_tools(self):
        assert not detect_code_hedging(
            "```bash\ndf -h\n```", ["run_command"]
        )


class TestDetectionRetryMessages:
    """Retry messages must have correct structure."""

    def test_fabrication_retry_is_developer_role(self):
        assert _FABRICATION_RETRY_MSG["role"] == "developer"
        assert len(_FABRICATION_RETRY_MSG["content"]) > 10

    def test_hedging_retry_is_developer_role(self):
        assert _HEDGING_RETRY_MSG["role"] == "developer"
        assert len(_HEDGING_RETRY_MSG["content"]) > 10

    def test_code_hedging_retry_is_developer_role(self):
        assert _CODE_HEDGING_RETRY_MSG["role"] == "developer"
        assert len(_CODE_HEDGING_RETRY_MSG["content"]) > 10


# ===================================================================
# 3. Tool Choice is "auto"
# ===================================================================


class TestToolChoiceAuto:
    """tool_choice must remain "auto" in the Codex client."""

    def test_tool_choice_in_source(self):
        src = Path("src/llm/openai_codex.py").read_text()
        assert '"tool_choice": "auto"' in src


# ===================================================================
# 4. Config Save Error Handling
# ===================================================================


class TestConfigSaveResilience:
    """Config write failures must not crash the API."""

    @pytest.mark.asyncio
    async def test_config_write_failure_still_returns_200(self):
        """If _write_config raises, the config is still applied in memory."""
        bot = _make_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            with patch("src.web.api._write_config", side_effect=OSError("disk full")):
                with patch("src.web.api.Path") as mock_path:
                    mock_path.return_value.exists.return_value = True
                    resp = await client.put(
                        "/api/config", json={"timezone": "UTC"}
                    )
            assert resp.status == 200
            data = await resp.json()
            # Config was applied in memory despite write failure
            assert data.get("timezone") == "UTC"

    @pytest.mark.asyncio
    async def test_config_write_success(self):
        """Normal config save works end-to-end."""
        bot = _make_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            with patch("src.web.api._write_config"):
                resp = await client.put(
                    "/api/config", json={"timezone": "UTC"}
                )
            assert resp.status == 200
            data = await resp.json()
            # Response should be redacted (uses •••••••• as placeholder)
            assert data.get("discord", {}).get("token") == "••••••••"

    @pytest.mark.asyncio
    async def test_config_invalid_json_returns_400(self):
        bot = _make_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/config",
                data=b"not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_config_non_dict_returns_400(self):
        bot = _make_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.put("/api/config", json=[1, 2, 3])
            assert resp.status == 400


# ===================================================================
# 5. Web UI Page Routing Completeness
# ===================================================================


class TestWebUIPageRouting:
    """All 13 page components must exist and have valid Vue exports."""

    EXPECTED_PAGES = [
        "dashboard", "chat", "sessions", "tools", "skills",
        "knowledge", "schedules", "loops", "processes",
        "audit", "config", "logs", "memory",
    ]

    def test_all_page_files_exist(self):
        for page in self.EXPECTED_PAGES:
            path = Path(f"ui/js/pages/{page}.js")
            assert path.exists(), f"Missing page file: {path}"

    def test_all_pages_export_component(self):
        """Each page file must contain 'export default' (Vue component)."""
        for page in self.EXPECTED_PAGES:
            path = Path(f"ui/js/pages/{page}.js")
            content = path.read_text()
            assert "export default" in content, f"{page}.js missing 'export default'"

    def test_all_pages_have_template(self):
        """Each page must have a template string."""
        for page in self.EXPECTED_PAGES:
            path = Path(f"ui/js/pages/{page}.js")
            content = path.read_text()
            assert "template:" in content or "template :" in content, \
                f"{page}.js missing template"

    def test_app_js_routes_all_pages(self):
        """app.js must have routes for all pages."""
        app_js = Path("ui/js/app.js").read_text()
        for page in self.EXPECTED_PAGES:
            assert f"/{page}" in app_js, f"Missing route for /{page} in app.js"

    def test_index_html_exists(self):
        assert Path("ui/index.html").exists()

    def test_style_css_exists(self):
        assert Path("ui/css/style.css").exists()

    def test_api_js_exists(self):
        assert Path("ui/js/api.js").exists()

    def test_app_js_exists(self):
        assert Path("ui/js/app.js").exists()


# ===================================================================
# 6. Chat Interface End-to-End
# ===================================================================


class TestChatEndToEnd:
    """Chat flow must work from REST to LLM and back."""

    @pytest.mark.asyncio
    async def test_chat_rest_success(self):
        """POST /api/chat returns LLM response."""
        bot = _make_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat",
                json={"content": "Hello Heimdall"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert "response" in data
            assert data["response"] == "Hello from Heimdall!"
            assert data["is_error"] is False

    @pytest.mark.asyncio
    async def test_chat_empty_content_rejected(self):
        bot = _make_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat", json={"content": ""})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_chat_whitespace_only_rejected(self):
        bot = _make_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat", json={"content": "   "})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_chat_too_long_rejected(self):
        bot = _make_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat",
                json={"content": "a" * (MAX_CHAT_CONTENT_LEN + 1)},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_chat_scrubs_secrets(self):
        """Secrets in LLM response must be scrubbed before returning."""
        bot = _make_bot()
        bot._process_with_tools = AsyncMock(
            return_value=("password=s3cr3t123", False, False, [], False)
        )
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat", json={"content": "test"}
            )
            data = await resp.json()
            assert "s3cr3t123" not in data["response"]

    @pytest.mark.asyncio
    async def test_chat_with_tools_returns_tool_names(self):
        bot = _make_bot()
        bot._process_with_tools = AsyncMock(
            return_value=("Done!", False, False, ["run_command"], False)
        )
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat", json={"content": "check disk"}
            )
            data = await resp.json()
            assert "run_command" in data["tools_used"]


# ===================================================================
# 7. Tool Registry Integrity
# ===================================================================


class TestToolRegistryIntegrity:
    """Tool registry must be consistent and complete."""

    def test_all_tools_have_required_fields(self):
        for tool in TOOLS:
            assert "name" in tool, f"Tool missing name: {tool}"
            assert "description" in tool, f"Tool {tool.get('name')} missing description"
            assert "input_schema" in tool, f"Tool {tool.get('name')} missing input_schema"

    def test_no_duplicate_tool_names(self):
        names = [t["name"] for t in TOOLS]
        assert len(names) == len(set(names)), "Duplicate tool names found"

    def test_get_tool_definitions_returns_list(self):
        defs = get_tool_definitions()
        assert isinstance(defs, list)
        assert len(defs) > 0


# ===================================================================
# 8. Session Defense Layers
# ===================================================================


class TestSessionDefenseLayers:
    """The 5-layer session defense must be intact."""

    def test_context_separator_exists(self):
        """Layer 1: context separator must exist in client.py."""
        src = Path("src/discord/client.py").read_text()
        assert "CURRENT REQUEST" in src

    def test_selective_saving_logic(self):
        """Layer 2: tool-less responses not saved."""
        src = Path("src/discord/client.py").read_text()
        # The selective saving check — only save when tools were used
        assert "tools_used_in_loop" in src

    def test_task_history_windowed(self):
        """Layer 3: get_task_history returns windowed subset."""
        src = Path("src/sessions/manager.py").read_text()
        assert "get_task_history" in src

    def test_compaction_omits_errors(self):
        """Layer 4: compaction prompt omits errors."""
        src = Path("src/sessions/manager.py").read_text()
        assert "OMIT" in src or "omit" in src

    def test_fabrication_detection_in_tool_loop(self):
        """Layer 5: fabrication detection fires in tool loop."""
        src = Path("src/discord/client.py").read_text()
        assert "detect_fabrication" in src
        assert "detect_hedging" in src


# ===================================================================
# 9. Secret Scrubbing Integration
# ===================================================================


class TestSecretScrubbing:
    """Secret scrubber must catch common secret patterns."""

    def test_scrubs_password(self):
        result = scrub_output_secrets("Found password=s3cr3t in config")
        assert "s3cr3t" not in result

    def test_scrubs_api_key(self):
        result = scrub_output_secrets("api_key=abc123def456")
        assert "abc123def456" not in result

    def test_scrubs_openai_key(self):
        result = scrub_output_secrets("sk-abc123def456ghi789jklmnop")
        assert "sk-abc123" not in result

    def test_scrubs_aws_key(self):
        result = scrub_output_secrets("AKIAIOSFODNN7EXAMPLE")
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    def test_clean_text_unchanged(self):
        text = "System is running normally, disk usage at 50%"
        assert scrub_output_secrets(text) == text


# ===================================================================
# 10. Deep Merge Helper
# ===================================================================


class TestDeepMerge:
    """_deep_merge must correctly handle nested config updates."""

    def test_flat_merge(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 3, "c": 4})
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        _deep_merge(base, {"a": {"y": 9}})
        assert base["a"]["x"] == 1
        assert base["a"]["y"] == 9
        assert base["b"] == 3

    def test_depth_limit(self):
        """Merges beyond depth 10 are silently skipped."""
        # Build 12 levels of dict nesting (depth 0..11, leaf at depth 11)
        base = {"k": "original"}
        for _ in range(11):
            base = {"k": base}
        update = {"k": "replaced"}
        for _ in range(11):
            update = {"k": update}

        _deep_merge(base, update)

        # Navigate to the leaf at depth 11
        inner = base
        for _ in range(11):
            inner = inner["k"]
        # Depth limit (>10) prevents merge at depth 11
        assert inner["k"] == "original"


# ===================================================================
# 11. MAX_CHAT_CONTENT_LEN
# ===================================================================


class TestChatConstants:
    """Chat content length limits must be consistent."""

    def test_max_content_len_is_4000(self):
        assert MAX_CHAT_CONTENT_LEN == 4000

    def test_api_uses_max_content_len(self):
        src = Path("src/web/api.py").read_text()
        assert "MAX_CHAT_CONTENT_LEN" in src

    def test_websocket_uses_max_content_len(self):
        src = Path("src/web/websocket.py").read_text()
        assert "MAX_CHAT_CONTENT_LEN" in src


# ===================================================================
# 12. Config Redaction
# ===================================================================


class TestConfigRedaction:
    """Config redaction must hide all sensitive fields."""

    def test_redacts_token(self):
        raw = {"discord": {"token": "my-secret-token"}}
        redacted = _redact_config(raw)
        assert redacted["discord"]["token"] == "••••••••"

    def test_redacts_nested(self):
        raw = {"web": {"api_token": "secret123"}}
        redacted = _redact_config(raw)
        assert redacted["web"]["api_token"] == "••••••••"

    def test_preserves_non_sensitive(self):
        raw = {"timezone": "UTC", "discord": {"token": "x"}}
        redacted = _redact_config(raw)
        assert redacted["timezone"] == "UTC"
