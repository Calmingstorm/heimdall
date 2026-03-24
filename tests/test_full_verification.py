"""Full functionality verification tests — Round 13.

Comprehensive end-to-end verification of all major subsystems after
the Loki transformation (Rounds 1-12).
"""
from __future__ import annotations

import importlib
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


# ---------------------------------------------------------------------------
# 1. Identity verification — all references are "Loki" not "Ansiblex"
# ---------------------------------------------------------------------------

class TestIdentityComplete:
    """Verify the Loki identity rename is 100% complete."""

    def _source_files(self):
        for f in SRC.rglob("*.py"):
            yield f

    def test_no_ansiblex_class_name(self):
        """No AnsiblexBot class reference in any source file."""
        for f in self._source_files():
            content = f.read_text()
            assert "AnsiblexBot" not in content, f"{f.name} contains AnsiblexBot"

    def test_loki_bot_class_exists(self):
        from src.discord.client import LokiBot
        assert LokiBot.__name__ == "LokiBot"

    def test_loki_bot_exported(self):
        from src.discord import LokiBot
        assert LokiBot is not None

    def test_logger_namespace_is_loki(self):
        from src.logging import get_logger
        log = get_logger("test")
        assert log.name.startswith("loki")

    def test_main_entry_says_loki(self):
        content = (SRC / "__main__.py").read_text()
        assert "Starting Loki" in content
        assert "Loki stopped" in content

    def test_system_prompt_says_loki(self):
        from src.llm.system_prompt import build_system_prompt, build_chat_system_prompt
        sp = build_system_prompt("", {}, [], [])
        assert "Loki" in sp
        assert "Ansiblex" not in sp
        cp = build_chat_system_prompt()
        assert "Loki" in cp
        assert "Ansiblex" not in cp

    def test_pyproject_name_is_loki(self):
        content = (ROOT / "pyproject.toml").read_text()
        assert 'name = "loki"' in content

    def test_dockerfile_user_is_loki(self):
        content = (ROOT / "Dockerfile").read_text()
        assert "loki" in content.lower()

    def test_docker_compose_services_are_loki(self):
        content = (ROOT / "docker-compose.yml").read_text()
        assert "loki-bot" in content
        assert "loki-browser" in content
        assert "loki-voice" in content


# ---------------------------------------------------------------------------
# 2. Personal data absence — covered by test_security_audit.py (42 tests)
# ---------------------------------------------------------------------------
# Removed: TestNoPersonalData class was redundant with the comprehensive
# test_security_audit.py which scans all tracked files for personal IPs,
# Discord IDs, names, paths, and URLs.


# ---------------------------------------------------------------------------
# 3. Configuration system verification
# ---------------------------------------------------------------------------

class TestConfigSystem:
    """Verify config loading, env var substitution, and defaults."""

    def test_env_var_required_raises(self):
        from src.config.schema import _substitute_env_vars
        with pytest.raises(ValueError, match="not set"):
            _substitute_env_vars("${DEFINITELY_MISSING_VAR_12345}")

    def test_env_var_with_default(self):
        from src.config.schema import _substitute_env_vars
        result = _substitute_env_vars("${MISSING_VAR_XYZ:-fallback}")
        assert result == "fallback"

    def test_env_var_empty_default(self):
        from src.config.schema import _substitute_env_vars
        result = _substitute_env_vars("${MISSING_VAR_XYZ:-}")
        assert result == ""

    def test_env_var_set_overrides_default(self):
        from src.config.schema import _substitute_env_vars
        os.environ["_TEST_VERIFY_VAR"] = "real_value"
        try:
            result = _substitute_env_vars("${_TEST_VERIFY_VAR:-fallback}")
            assert result == "real_value"
        finally:
            del os.environ["_TEST_VERIFY_VAR"]

    def test_config_defaults_timezone(self):
        from src.config.schema import Config
        cfg = Config(discord={"token": "t"})
        assert cfg.timezone == "UTC"

    def test_config_defaults_bot_interaction(self):
        from src.config.schema import Config
        cfg = Config(discord={"token": "t"})
        assert cfg.discord.respond_to_bots is False
        assert cfg.discord.require_mention is False

    def test_config_defaults_tools(self):
        from src.config.schema import Config
        cfg = Config(discord={"token": "t"})
        assert cfg.tools.prometheus_host == ""
        assert cfg.tools.ansible_host == ""
        assert cfg.tools.claude_code_host == ""
        assert cfg.tools.claude_code_user == ""
        assert cfg.tools.claude_code_dir == "/opt/project"
        assert cfg.tools.incus_host == ""

    def test_config_loads_actual_yml(self):
        """config.yml loads without error when env vars are set."""
        from src.config.schema import load_config
        os.environ.setdefault("DISCORD_TOKEN", "test-token")
        os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
        cfg = load_config(ROOT / "config.yml")
        assert cfg.timezone == "UTC"
        assert cfg.tools.hosts == {}

    def test_empty_config_raises_system_exit(self):
        from src.config.schema import load_config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("")
            f.flush()
            with pytest.raises(SystemExit, match="empty or invalid"):
                load_config(f.name)
            os.unlink(f.name)


# ---------------------------------------------------------------------------
# 4. Tool system verification
# ---------------------------------------------------------------------------

class TestToolSystem:
    """Verify all tools are registered and properly structured."""

    def test_tool_count(self):
        from src.tools.registry import TOOLS
        assert len(TOOLS) == 78

    def test_all_tools_have_required_keys(self):
        from src.tools.registry import TOOLS
        for tool in TOOLS:
            assert "name" in tool, f"Tool missing name: {tool}"
            assert "description" in tool, f"Tool {tool.get('name')} missing description"
            assert "input_schema" in tool, f"Tool {tool['name']} missing input_schema"
            assert "requires_approval" not in tool, f"Tool {tool['name']} should not have requires_approval"

    def test_tool_names_are_unique(self):
        from src.tools.registry import TOOLS
        names = [t["name"] for t in TOOLS]
        assert len(names) == len(set(names)), "Duplicate tool names found"

    def test_no_personal_host_examples_in_tools(self):
        from src.tools.registry import TOOLS
        import json
        tools_text = json.dumps(TOOLS)
        assert "192.168.1." not in tools_text
        assert "desktop" not in tools_text.lower() or "myserver" in tools_text.lower()

    def test_key_tool_categories_present(self):
        from src.tools.registry import TOOLS
        names = {t["name"] for t in TOOLS}
        # SSH/infrastructure
        assert "check_service" in names
        assert "check_docker" in names
        assert "check_disk" in names
        assert "check_memory" in names
        assert "run_command" in names
        assert "run_command_multi" in names
        # Docker
        assert "docker_logs" in names
        assert "docker_compose_action" in names
        # Git
        assert "git_status" in names
        assert "git_pull" in names
        assert "git_push" in names
        # Incus
        assert "incus_list" in names
        assert "incus_exec" in names
        assert "incus_launch" in names
        # Knowledge/search
        assert "search_knowledge" in names
        assert "ingest_document" in names
        # Browser
        assert "browser_screenshot" in names
        assert "browser_click" in names
        # Scheduling
        assert "schedule_task" in names
        assert "parse_time" in names
        # Claude Code
        assert "claude_code" in names
        # Skills
        assert "create_skill" in names
        assert "list_skills" in names
        # Web
        assert "web_search" in names
        assert "fetch_url" in names

    def test_incus_tools_count(self):
        from src.tools.registry import TOOLS
        incus_tools = [t for t in TOOLS if t["name"].startswith("incus_")]
        assert len(incus_tools) >= 11


# ---------------------------------------------------------------------------
# 5. System prompt verification
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    """Verify system prompt generation works correctly."""

    def test_prompt_with_no_hosts(self):
        from src.llm.system_prompt import build_system_prompt
        p = build_system_prompt("", {}, [], [])
        assert "None configured" in p

    def test_prompt_with_hosts(self):
        from src.llm.system_prompt import build_system_prompt
        p = build_system_prompt("", {"web": "root@10.0.0.1"}, [], [])
        assert "web" in p
        assert "10.0.0.1" in p

    def test_prompt_with_context(self):
        from src.llm.system_prompt import build_system_prompt
        p = build_system_prompt("Custom infrastructure notes", {}, [], [])
        assert "Custom infrastructure notes" in p

    def test_prompt_with_services_and_playbooks(self):
        from src.llm.system_prompt import build_system_prompt
        p = build_system_prompt("", {}, ["nginx", "docker"], ["deploy.yml"])
        assert "nginx" in p
        assert "docker" in p
        assert "deploy.yml" in p

    def test_prompt_timezone_utc(self):
        from src.llm.system_prompt import build_system_prompt
        p = build_system_prompt("", {}, [], [], tz="UTC")
        assert "UTC" in p

    def test_prompt_timezone_custom(self):
        from src.llm.system_prompt import build_system_prompt
        p = build_system_prompt("", {}, [], [], tz="Asia/Tokyo")
        assert "JST" in p or "Asia/Tokyo" in p or "+09" in p

    def test_chat_prompt_is_lightweight(self):
        from src.llm.system_prompt import build_system_prompt, build_chat_system_prompt
        full = build_system_prompt("ctx", {"h": "a"}, ["s"], ["p"])
        chat = build_chat_system_prompt()
        assert len(chat) < len(full)

    def test_chat_prompt_no_tools_section(self):
        from src.llm.system_prompt import build_chat_system_prompt
        p = build_chat_system_prompt()
        assert "Claude Code Delegation" not in p
        assert "Available Hosts" not in p


# ---------------------------------------------------------------------------
# 7. Search subsystem verification
# ---------------------------------------------------------------------------

class TestSearchSubsystem:
    """Verify FTS5 and RRF search subsystems."""

    def test_fts5_index_and_search(self):
        from src.search.fts import FullTextIndex
        with tempfile.TemporaryDirectory() as tmp:
            idx = FullTextIndex(os.path.join(tmp, "test.db"))
            idx.index_knowledge_chunk("doc1", "hello world search test", "source", 0)
            results = idx.search_knowledge("hello")
            assert len(results) >= 1

    def test_fts5_no_results(self):
        from src.search.fts import FullTextIndex
        with tempfile.TemporaryDirectory() as tmp:
            idx = FullTextIndex(os.path.join(tmp, "test.db"))
            idx.index_knowledge_chunk("doc1", "hello world", "src", 0)
            results = idx.search_knowledge("zzzzzznonexistent")
            assert len(results) == 0

    def test_rrf_merges_results(self):
        from src.search.hybrid import reciprocal_rank_fusion
        list1 = [{"doc_id": "a", "text": "first"}, {"doc_id": "b", "text": "second"}]
        list2 = [{"doc_id": "b", "text": "second"}, {"doc_id": "c", "text": "third"}]
        results = reciprocal_rank_fusion(list1, list2)
        assert len(results) == 3
        # b appears in both lists, should rank highest
        assert results[0]["doc_id"] == "b"
        # All have rrf_score
        for r in results:
            assert "rrf_score" in r

    def test_rrf_empty_lists(self):
        from src.search.hybrid import reciprocal_rank_fusion
        results = reciprocal_rank_fusion([], [])
        assert results == []


# ---------------------------------------------------------------------------
# 8. Secret scrubber verification
# ---------------------------------------------------------------------------

class TestSecretScrubber:
    """Verify the secret scrubber catches sensitive patterns."""

    def test_scrubs_api_keys(self):
        from src.llm.secret_scrubber import scrub_output_secrets
        text = "Found key: sk-1234567890abcdefghijklmnop"
        result = scrub_output_secrets(text)
        assert "sk-1234567890" not in result

    def test_preserves_safe_text(self):
        from src.llm.secret_scrubber import scrub_output_secrets
        text = "Hello, the disk usage is 42%"
        result = scrub_output_secrets(text)
        assert result == text


# ---------------------------------------------------------------------------
# 9. Deployment verification
# ---------------------------------------------------------------------------

class TestDeploymentFiles:
    """Verify deployment files are generic and properly configured."""

    def test_docker_compose_no_hardcoded_timezone(self):
        content = (ROOT / "docker-compose.yml").read_text()
        assert "America/New_York" not in content
        assert "${TZ:-UTC}" in content

    def test_dockerfile_creates_ssh_dir(self):
        content = (ROOT / "Dockerfile").read_text()
        assert ".ssh" in content

    def test_dockerignore_excludes_secrets(self):
        content = (ROOT / ".dockerignore").read_text()
        assert "ssh/" in content
        assert ".env" in content

    def test_env_example_has_required_vars(self):
        content = (ROOT / ".env.example").read_text()
        assert "DISCORD_TOKEN" in content
        # Anthropic removed — no classifier

    def test_incus_deploy_script_exists(self):
        script = ROOT / "scripts" / "incus-deploy.sh"
        assert script.exists()
        assert os.access(str(script), os.X_OK)

    def test_monitor_script_multi_deploy(self):
        content = (ROOT / "scripts" / "monitor.sh").read_text()
        assert "docker" in content
        assert "incus" in content
        assert "local" in content


# ---------------------------------------------------------------------------
# 10. Skill system verification
# ---------------------------------------------------------------------------

class TestSkillSystem:
    """Verify the skill system is generic and properly documented."""

    def test_skill_templates_exist(self):
        templates = list((ROOT / "data" / "skills").glob("*.template"))
        assert len(templates) >= 3

    def test_skill_templates_valid_python(self):
        for tmpl in (ROOT / "data" / "skills").glob("*.template"):
            content = tmpl.read_text()
            # Should compile without syntax errors
            compile(content, str(tmpl), "exec")

    def test_skill_module_prefix_is_loki(self):
        content = (SRC / "tools" / "skill_manager.py").read_text()
        assert "loki_skill_" in content
        assert "ansiblex_skill_" not in content

    def test_create_skill_tool_documents_api(self):
        from src.tools.registry import TOOLS
        create_tool = next(t for t in TOOLS if t["name"] == "create_skill")
        desc = create_tool["description"]
        # Key API methods should be documented
        assert "run_on_host" in desc
        assert "remember" in desc or "recall" in desc
        assert "template" in desc.lower() or "data/skills" in desc


# ---------------------------------------------------------------------------
# 11. Circuit breaker verification
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    """Verify circuit breaker state machine."""

    def test_starts_closed(self):
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test")
        assert cb._state == "closed"

    def test_opens_after_failures(self):
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb._state == "open"

    def test_success_resets_count(self):
        from src.llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._failure_count == 0
        assert cb._state == "closed"


# ---------------------------------------------------------------------------
# 12. Audit logger verification
# ---------------------------------------------------------------------------

class TestAuditLogger:
    """Verify audit logger initializes correctly."""

    def test_init_creates_logger(self):
        from src.audit.logger import AuditLogger
        with tempfile.TemporaryDirectory() as tmp:
            al = AuditLogger(os.path.join(tmp, "audit.jsonl"))
            assert al is not None


# ---------------------------------------------------------------------------
# 13. Module import verification
# ---------------------------------------------------------------------------

class TestModuleImports:
    """Verify all critical modules import without errors."""

    MODULES = [
        "src.discord.client",
        "src.discord.background_task",
        "src.discord.voice",
        "src.llm.system_prompt",
        "src.llm.openai_codex",
        "src.llm.circuit_breaker",
        "src.llm.secret_scrubber",
        "src.llm.types",
        "src.config.schema",
        "src.tools.registry",
        "src.tools.executor",
        "src.tools.skill_manager",
        "src.tools.skill_context",
        "src.tools.browser",
        "src.tools.web",
        "src.tools.ssh",
        "src.tools.time_parser",
        "src.health.server",
        "src.monitoring.watcher",
        "src.sessions.manager",
        "src.learning.reflector",
        "src.search.fts",
        "src.search.hybrid",
        "src.audit.logger",
        "src.scheduler.scheduler",
        "src.context.loader",
    ]

    @pytest.mark.parametrize("module", MODULES)
    def test_module_imports(self, module):
        """Every module in the project should import without error."""
        importlib.import_module(module)
