"""Tests for Round 31: Skill Discovery + Sharing.

Tests install_from_url, export_skill, and skill_status features.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.schema import ToolsConfig, ToolHost
from src.tools.executor import ToolExecutor
from src.tools.skill_manager import (
    MAX_SKILL_DOWNLOAD_BYTES,
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
    "version": "1.2.0",
    "author": "Tester",
    "homepage": "https://example.com",
    "tags": ["test", "demo"],
}

async def execute(inp, context):
    return f"Got: {inp.get('msg', 'nothing')}"
'''

VALID_SKILL_CODE_NAMED = '''
SKILL_DEFINITION = {{
    "name": "{name}",
    "description": "A test skill named {name}",
    "input_schema": {{
        "type": "object",
        "properties": {{"msg": {{"type": "string"}}}},
    }},
}}

async def execute(inp, context):
    return f"Got: {{inp.get('msg', 'nothing')}}"
'''

INVALID_SKILL_CODE = '''
# Missing SKILL_DEFINITION
async def execute(inp, context):
    return "broken"
'''

SKILL_WITH_DEPS = '''
SKILL_DEFINITION = {
    "name": "dep_skill",
    "description": "Skill with deps",
    "input_schema": {"type": "object", "properties": {}},
    "dependencies": ["requests"],
    "config_schema": {
        "type": "object",
        "properties": {
            "api_key": {"type": "string", "default": "test123"},
        },
    },
}

async def execute(inp, context):
    return "ok"
'''


@pytest.fixture
def skill_mgr(tmp_dir: Path, tools_config: ToolsConfig) -> SkillManager:
    executor = ToolExecutor(tools_config)
    skills_dir = tmp_dir / "skills"
    skills_dir.mkdir()
    return SkillManager(str(skills_dir), executor)


def _mock_response(status=200, content=b"", content_length=None):
    """Create a mock aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.content_length = content_length
    resp.content = AsyncMock()
    resp.content.read = AsyncMock(return_value=content)
    return resp


def _mock_session(resp):
    """Create a mock aiohttp session context manager."""
    session = AsyncMock()
    session.get = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=resp),
        __aexit__=AsyncMock(return_value=False),
    ))
    session_cls = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=session),
        __aexit__=AsyncMock(return_value=False),
    ))
    return session_cls


# ========================
# install_from_url tests
# ========================

class TestInstallFromUrl:
    """Tests for SkillManager.install_from_url()."""

    async def test_invalid_scheme_ftp(self, skill_mgr: SkillManager):
        result = await skill_mgr.install_from_url("ftp://example.com/skill.py")
        assert "Invalid URL scheme" in result

    async def test_invalid_scheme_file(self, skill_mgr: SkillManager):
        result = await skill_mgr.install_from_url("file:///etc/passwd")
        assert "Invalid URL scheme" in result

    async def test_empty_host(self, skill_mgr: SkillManager):
        result = await skill_mgr.install_from_url("https://")
        assert "no host" in result.lower() or "Invalid URL" in result

    async def test_http_error_404(self, skill_mgr: SkillManager):
        resp = _mock_response(status=404)
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "HTTP 404" in result

    async def test_http_error_500(self, skill_mgr: SkillManager):
        resp = _mock_response(status=500)
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "HTTP 500" in result

    async def test_file_too_large_via_header(self, skill_mgr: SkillManager):
        resp = _mock_response(
            status=200,
            content=b"x",
            content_length=MAX_SKILL_DOWNLOAD_BYTES + 1,
        )
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "too large" in result.lower()

    async def test_file_too_large_via_body(self, skill_mgr: SkillManager):
        big = b"x" * (MAX_SKILL_DOWNLOAD_BYTES + 1)
        resp = _mock_response(status=200, content=big)
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "too large" in result.lower()

    async def test_non_utf8(self, skill_mgr: SkillManager):
        resp = _mock_response(status=200, content=b"\xff\xfe\x00\x01")
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "UTF-8" in result

    async def test_invalid_code(self, skill_mgr: SkillManager):
        resp = _mock_response(status=200, content=INVALID_SKILL_CODE.encode())
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "Invalid skill code" in result

    async def test_syntax_error(self, skill_mgr: SkillManager):
        bad = b"def foo(\n"
        resp = _mock_response(status=200, content=bad)
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "Invalid skill code" in result or "Syntax" in result

    async def test_success(self, skill_mgr: SkillManager):
        code = VALID_SKILL_CODE_NAMED.format(name="url_skill")
        resp = _mock_response(status=200, content=code.encode())
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/url_skill.py")
        assert "successfully" in result
        assert skill_mgr.has_skill("url_skill")

    async def test_duplicate(self, skill_mgr: SkillManager):
        # First install
        code = VALID_SKILL_CODE_NAMED.format(name="dup_skill")
        resp = _mock_response(status=200, content=code.encode())
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            await skill_mgr.install_from_url("https://example.com/skill.py")
        # Second install of same name
        resp2 = _mock_response(status=200, content=code.encode())
        with patch("aiohttp.ClientSession", _mock_session(resp2)):
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "already exists" in result

    async def test_download_timeout(self, skill_mgr: SkillManager):
        async def raise_timeout(*args, **kwargs):
            raise asyncio.TimeoutError()

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__ = raise_timeout
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "timed out" in result.lower() or "error" in result.lower()

    async def test_missing_name_in_definition(self, skill_mgr: SkillManager):
        code = b'''
SKILL_DEFINITION = {
    "description": "Missing name",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    return "ok"
'''
        resp = _mock_response(status=200, content=code)
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "Invalid skill code" in result or "missing" in result.lower()

    async def test_http_url_allowed(self, skill_mgr: SkillManager):
        """Plain http:// should be allowed (not just https)."""
        code = VALID_SKILL_CODE_NAMED.format(name="http_skill")
        resp = _mock_response(status=200, content=code.encode())
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("http://example.com/http_skill.py")
        assert "successfully" in result

    async def test_download_exception(self, skill_mgr: SkillManager):
        """Generic download exceptions are caught."""

        async def raise_error(*args, **kwargs):
            raise OSError("Connection refused")

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__ = raise_error
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "error" in result.lower()


# ========================
# export_skill tests
# ========================

class TestExportSkill:
    """Tests for SkillManager.export_skill()."""

    def test_not_found(self, skill_mgr: SkillManager):
        result = skill_mgr.export_skill("nonexistent")
        assert isinstance(result, str)
        assert "not found" in result

    def test_success(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        result = skill_mgr.export_skill("test_skill")
        assert isinstance(result, tuple)
        file_bytes, filename = result
        assert filename == "test_skill.py"
        assert isinstance(file_bytes, bytes)
        assert b"SKILL_DEFINITION" in file_bytes

    def test_export_content_matches_source(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        result = skill_mgr.export_skill("test_skill")
        file_bytes, _ = result
        assert file_bytes.decode("utf-8") == VALID_SKILL_CODE

    def test_export_disabled_skill(self, skill_mgr: SkillManager):
        """Can export a disabled skill (for sharing, even if not active)."""
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        skill_mgr.disable_skill("test_skill")
        result = skill_mgr.export_skill("test_skill")
        assert isinstance(result, tuple)
        file_bytes, filename = result
        assert filename == "test_skill.py"

    def test_export_file_read_error(self, skill_mgr: SkillManager):
        """If file can't be read, returns error string."""
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        # Delete the file behind the manager's back
        skill = skill_mgr._skills["test_skill"]
        skill.file_path.unlink()
        result = skill_mgr.export_skill("test_skill")
        assert isinstance(result, str)
        assert "Failed" in result or "error" in result.lower()


# ========================
# skill_status tests
# ========================

class TestSkillStatus:
    """Tests for SkillManager.skill_status()."""

    def test_not_found(self, skill_mgr: SkillManager):
        result = skill_mgr.skill_status("nonexistent")
        assert "not found" in result

    def test_basic_status(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        result = skill_mgr.skill_status("test_skill")
        assert "**Skill: test_skill**" in result
        assert "A test skill" in result
        assert "loaded" in result
        assert "1.2.0" in result
        assert "Tester" in result
        assert "https://example.com" in result
        assert "test, demo" in result
        assert "Total executions: 0" in result

    def test_status_disabled(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        skill_mgr.disable_skill("test_skill")
        result = skill_mgr.skill_status("test_skill")
        assert "disabled" in result

    async def test_status_after_execution(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        await skill_mgr.execute("test_skill", {"msg": "hi"})
        result = skill_mgr.skill_status("test_skill")
        assert "Total executions: 1" in result
        assert "Last execution:" in result

    def test_status_with_deps(self, skill_mgr: SkillManager):
        with patch("src.tools.skill_manager._is_package_installed", return_value=True):
            skill_mgr.create_skill("dep_skill", SKILL_WITH_DEPS)
        result = skill_mgr.skill_status("dep_skill")
        assert "Dependencies:" in result
        assert "requests" in result

    def test_status_with_config(self, skill_mgr: SkillManager):
        with patch("src.tools.skill_manager._is_package_installed", return_value=True):
            skill_mgr.create_skill("dep_skill", SKILL_WITH_DEPS)
        result = skill_mgr.skill_status("dep_skill")
        assert "Config:" in result
        assert "test123" in result

    def test_status_with_diagnostics(self, skill_mgr: SkillManager):
        """Skills with diagnostics show them in status."""
        code = '''
SKILL_DEFINITION = {
    "name": "diag_skill",
    "description": "Skill with bad version",
    "input_schema": {"type": "object", "properties": {}},
    "version": "not-a-version",
}

async def execute(inp, context):
    return "ok"
'''
        skill_mgr.create_skill("diag_skill", code)
        result = skill_mgr.skill_status("diag_skill")
        assert "Diagnostics:" in result
        assert "warn" in result.lower()


# ========================
# Tool definition tests
# ========================

class TestToolDefinitions:
    """Verify the new tool definitions exist in registry."""

    def test_install_skill_tool(self):
        from src.tools.registry import TOOLS
        names = {t["name"] for t in TOOLS}
        assert "install_skill" in names

    def test_export_skill_tool(self):
        from src.tools.registry import TOOLS
        names = {t["name"] for t in TOOLS}
        assert "export_skill" in names

    def test_skill_status_tool(self):
        from src.tools.registry import TOOLS
        names = {t["name"] for t in TOOLS}
        assert "skill_status" in names

    def test_install_skill_requires_url(self):
        from src.tools.registry import TOOLS
        tool = next(t for t in TOOLS if t["name"] == "install_skill")
        assert "url" in tool["input_schema"]["required"]

    def test_export_skill_requires_name(self):
        from src.tools.registry import TOOLS
        tool = next(t for t in TOOLS if t["name"] == "export_skill")
        assert "name" in tool["input_schema"]["required"]

    def test_skill_status_requires_name(self):
        from src.tools.registry import TOOLS
        tool = next(t for t in TOOLS if t["name"] == "skill_status")
        assert "name" in tool["input_schema"]["required"]


# ========================
# Client handler tests
# ========================

class TestClientHandlers:
    """Test that client.py handlers are wired correctly for new skill tools."""

    def test_install_skill_handler_exists(self):
        """Verify install_skill handler exists in client.py source."""
        import src.discord.client as client_mod
        source = Path(client_mod.__file__).read_text()
        assert 'tool_name == "install_skill"' in source

    def test_export_skill_handler_exists(self):
        """Verify export_skill handler exists in client.py source."""
        import src.discord.client as client_mod
        source = Path(client_mod.__file__).read_text()
        assert 'tool_name == "export_skill"' in source

    def test_skill_status_handler_exists(self):
        """Verify skill_status handler exists in client.py source."""
        import src.discord.client as client_mod
        source = Path(client_mod.__file__).read_text()
        assert 'tool_name == "skill_status"' in source


# ========================
# Edge cases
# ========================

class TestInstallExportRoundtrip:
    """Test installing, exporting, and re-installing a skill."""

    async def test_export_and_reimport(self, skill_mgr: SkillManager):
        """Export a skill and verify the exported code is valid."""
        code = VALID_SKILL_CODE_NAMED.format(name="roundtrip_skill")
        skill_mgr.create_skill("roundtrip_skill", code)

        export_result = skill_mgr.export_skill("roundtrip_skill")
        assert isinstance(export_result, tuple)
        file_bytes, filename = export_result

        # Validate the exported code
        report = skill_mgr.validate_skill_code(file_bytes.decode("utf-8"))
        assert report["valid"]

    async def test_install_then_export(self, skill_mgr: SkillManager):
        """Install from URL, then export — should produce valid code."""
        code = VALID_SKILL_CODE_NAMED.format(name="install_export")
        resp = _mock_response(status=200, content=code.encode())
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/install_export.py")
        assert "successfully" in result

        export_result = skill_mgr.export_skill("install_export")
        assert isinstance(export_result, tuple)

    def test_export_then_create_new_manager(self, tmp_dir: Path, tools_config: ToolsConfig):
        """Export and load in a new SkillManager."""
        skills1 = tmp_dir / "skills1"
        skills1.mkdir()
        mgr1 = SkillManager(str(skills1), ToolExecutor(tools_config))
        code = VALID_SKILL_CODE_NAMED.format(name="portable_skill")
        mgr1.create_skill("portable_skill", code)
        export_result = mgr1.export_skill("portable_skill")
        file_bytes, filename = export_result

        # Create new manager and install the exported code
        skills2 = tmp_dir / "skills2"
        skills2.mkdir()
        mgr2 = SkillManager(str(skills2), ToolExecutor(tools_config))
        result = mgr2.create_skill("portable_skill", file_bytes.decode("utf-8"))
        assert "successfully" in result
        assert mgr2.has_skill("portable_skill")


class TestInstallUrlEdgeCases:
    """Additional edge cases for URL install."""

    async def test_empty_response_body(self, skill_mgr: SkillManager):
        resp = _mock_response(status=200, content=b"")
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "Invalid" in result or "error" in result.lower()

    async def test_name_conflicts_builtin(self, skill_mgr: SkillManager):
        """Skill with name matching a built-in tool should fail."""
        code = '''
SKILL_DEFINITION = {
    "name": "check_disk",
    "description": "Fake disk check",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    return "ok"
'''
        resp = _mock_response(status=200, content=code.encode())
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "conflicts" in result or "Invalid" in result


class TestSkillStatusEdgeCases:
    """Edge cases for skill_status."""

    def test_minimal_skill(self, skill_mgr: SkillManager):
        """Skill with no optional metadata still works."""
        code = '''
SKILL_DEFINITION = {
    "name": "minimal_skill",
    "description": "Bare bones",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    return "ok"
'''
        skill_mgr.create_skill("minimal_skill", code)
        result = skill_mgr.skill_status("minimal_skill")
        assert "**Skill: minimal_skill**" in result
        assert "0.0.0" in result
        # Should not crash on missing optional fields
        assert "Author" not in result  # No author set
        assert "Homepage" not in result  # No homepage set
        assert "Tags" not in result  # No tags set
