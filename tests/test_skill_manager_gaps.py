"""Tests for skill_manager.py coverage gaps.

Targets uncovered lines: 67-69, 71, 78-79, 94-96, 101-103, 119-122,
153-154, 179-180, 262, 264-268.
Also covers: edit_skill name validation, _unload_skill module name,
LoadedSkill.module_name tracking.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from src.config.schema import ToolsConfig, ToolHost
from src.tools.executor import ToolExecutor
from src.tools.skill_manager import SkillManager


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

# Skill that returns a non-string result
NON_STRING_RESULT_CODE = '''
SKILL_DEFINITION = {
    "name": "non_string_skill",
    "description": "Returns non-string",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    return 42
'''

# Skill that raises an exception
ERROR_SKILL_CODE = '''
SKILL_DEFINITION = {
    "name": "error_skill",
    "description": "Raises error",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    raise RuntimeError("skill boom")
'''

# Skill that takes forever (for timeout)
TIMEOUT_SKILL_CODE = '''
import asyncio

SKILL_DEFINITION = {
    "name": "slow_skill",
    "description": "Takes forever",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    await asyncio.sleep(9999)
    return "never"
'''

# Missing execute function
NO_EXECUTE_CODE = '''
SKILL_DEFINITION = {
    "name": "no_exec",
    "description": "No execute",
    "input_schema": {"type": "object", "properties": {}},
}
'''

# Missing required key in SKILL_DEFINITION
MISSING_KEY_CODE = '''
SKILL_DEFINITION = {
    "name": "missing_key",
    "description": "Missing input_schema",
}

async def execute(inp, context):
    return "ok"
'''

# Syntax error in skill code
SYNTAX_ERROR_CODE = '''
SKILL_DEFINITION = {
    "name": "broken",
    this is not valid python
}
'''


@pytest.fixture
def skill_mgr(tmp_dir: Path, tools_config: ToolsConfig) -> SkillManager:
    executor = ToolExecutor(tools_config)
    skills_dir = tmp_dir / "skills"
    skills_dir.mkdir()
    return SkillManager(str(skills_dir), executor)


class TestLoadAll:
    def test_load_all_loads_existing_skills(self, tmp_dir: Path, tools_config: ToolsConfig):
        """_load_all loads .py files from the skills dir on init (lines 67-69, 71)."""
        skills_dir = tmp_dir / "skills2"
        skills_dir.mkdir()
        # Pre-create a valid skill file
        (skills_dir / "test_skill.py").write_text(VALID_SKILL_CODE)

        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(skills_dir), executor)
        assert mgr.has_skill("test_skill")
        assert len(mgr.list_skills()) == 1

    def test_load_all_skips_invalid_files(self, tmp_dir: Path, tools_config: ToolsConfig):
        """_load_all skips files that fail validation (lines 67-69)."""
        skills_dir = tmp_dir / "skills3"
        skills_dir.mkdir()
        (skills_dir / "bad_skill.py").write_text("# nothing here")

        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(skills_dir), executor)
        assert len(mgr.list_skills()) == 0


class TestLoadSkillEdgeCases:
    def test_no_spec_returns_none(self, skill_mgr: SkillManager, tmp_dir: Path):
        """When spec_from_file_location returns None (lines 78-79)."""
        path = tmp_dir / "skills" / "bad.py"
        path.write_text("x = 1")
        with patch("src.tools.skill_manager.importlib.util.spec_from_file_location", return_value=None):
            result = skill_mgr._load_skill(path)
        assert result is None

    def test_missing_required_key(self, skill_mgr: SkillManager):
        """Missing 'input_schema' key in SKILL_DEFINITION (lines 94-96)."""
        result = skill_mgr.create_skill("missing_key", MISSING_KEY_CODE)
        assert "failed to load" in result.lower()
        assert not skill_mgr.has_skill("missing_key")

    def test_no_execute_function(self, skill_mgr: SkillManager):
        """Missing execute() function (lines 101-103)."""
        result = skill_mgr.create_skill("no_exec", NO_EXECUTE_CODE)
        assert "failed to load" in result.lower()
        assert not skill_mgr.has_skill("no_exec")

    def test_syntax_error_in_skill(self, skill_mgr: SkillManager):
        """Syntax error triggers exception path (lines 119-122)."""
        result = skill_mgr.create_skill("broken", SYNTAX_ERROR_CODE)
        assert "failed to load" in result.lower()
        assert not skill_mgr.has_skill("broken")
        # Module should be cleaned up from sys.modules
        assert "heimdall_skill_broken" not in sys.modules


class TestCreateSkillEdgeCases:
    def test_write_error(self, skill_mgr: SkillManager):
        """Write failure returns error (lines 153-154)."""
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            result = skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        assert "Failed to write" in result


class TestEditSkillEdgeCases:
    def test_edit_write_error(self, skill_mgr: SkillManager):
        """Write failure during edit returns error (lines 179-180)."""
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            result = skill_mgr.edit_skill("test_skill", VALID_SKILL_CODE.replace("A test", "Updated"))
        assert "Failed to write" in result


class TestExecuteEdgeCases:
    @pytest.mark.asyncio
    async def test_non_string_result_converted(self, skill_mgr: SkillManager):
        """Non-string result is str()-converted (line 262)."""
        skill_mgr.create_skill("non_string_skill", NON_STRING_RESULT_CODE)
        result = await skill_mgr.execute("non_string_skill", {})
        assert result == "42"

    @pytest.mark.asyncio
    async def test_timeout_handling(self, skill_mgr: SkillManager):
        """Skill that exceeds timeout returns timeout message (lines 264-265)."""
        skill_mgr.create_skill("slow_skill", TIMEOUT_SKILL_CODE)
        # Patch the timeout to be very short
        with patch("src.tools.skill_manager.SKILL_EXECUTE_TIMEOUT", 0.01):
            result = await skill_mgr.execute("slow_skill", {})
        assert "timed out" in result

    @pytest.mark.asyncio
    async def test_execution_exception(self, skill_mgr: SkillManager):
        """Skill that raises returns error message (lines 267-268)."""
        skill_mgr.create_skill("error_skill", ERROR_SKILL_CODE)
        result = await skill_mgr.execute("error_skill", {})
        assert "Skill error" in result
        assert "skill boom" in result


# Skill whose SKILL_DEFINITION.name differs from the intended filename
RENAMED_SKILL_CODE = '''
SKILL_DEFINITION = {
    "name": "different_name",
    "description": "Name doesn't match filename",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    return "ok"
'''


class TestEditSkillNameValidation:
    """edit_skill must reject code where SKILL_DEFINITION.name != filename."""

    def test_edit_rejects_name_mismatch(self, skill_mgr: SkillManager):
        """Editing a skill with a mismatched SKILL_DEFINITION.name should revert."""
        # Create a valid skill first
        result = skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        assert "successfully" in result
        assert skill_mgr.has_skill("test_skill")

        # Edit with code that has a different SKILL_DEFINITION.name
        result = skill_mgr.edit_skill("test_skill", RENAMED_SKILL_CODE)
        assert "doesn't match" in result
        assert "Reverted" in result

        # Original skill should still work
        assert skill_mgr.has_skill("test_skill")
        defs = skill_mgr.get_tool_definitions()
        names = [d["name"] for d in defs]
        assert "test_skill" in names
        assert "different_name" not in names

    def test_edit_accepts_matching_name(self, skill_mgr: SkillManager):
        """Editing with matching SKILL_DEFINITION.name should succeed."""
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)

        updated_code = VALID_SKILL_CODE.replace("A test skill", "Updated description")
        result = skill_mgr.edit_skill("test_skill", updated_code)
        assert "successfully" in result

        defs = skill_mgr.get_tool_definitions()
        desc = [d["description"] for d in defs if d["name"] == "test_skill"]
        assert desc == ["Updated description"]


class TestUnloadSkillModuleName:
    """_unload_skill should use the stored module_name from LoadedSkill."""

    def test_module_name_stored_on_load(self, skill_mgr: SkillManager):
        """LoadedSkill should track the actual module_name from load."""
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        skill = skill_mgr._skills["test_skill"]
        assert skill.module_name == "heimdall_skill_test_skill"

    def test_unload_cleans_correct_module(self, skill_mgr: SkillManager):
        """_unload_skill removes the correct module from sys.modules."""
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        assert "heimdall_skill_test_skill" in sys.modules

        skill_mgr._unload_skill("test_skill")
        assert "heimdall_skill_test_skill" not in sys.modules
        assert "test_skill" not in skill_mgr._skills

    def test_unload_uses_stored_module_name(self, tmp_dir: Path, tools_config: ToolsConfig):
        """When file stem != skill name, _unload_skill uses stored module_name."""
        skills_dir = tmp_dir / "skills_mismatch"
        skills_dir.mkdir()
        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(skills_dir), executor)

        # Manually place a skill file where stem != SKILL_DEFINITION.name
        code = '''
SKILL_DEFINITION = {
    "name": "my_tool",
    "description": "Test",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    return "ok"
'''
        path = skills_dir / "my_tool_file.py"
        path.write_text(code)

        # Load manually
        skill = mgr._load_skill(path)
        assert skill is not None
        assert skill.module_name == "heimdall_skill_my_tool_file"
        mgr._skills[skill.name] = skill

        # Verify module is in sys.modules under the file-stem-based name
        assert "heimdall_skill_my_tool_file" in sys.modules

        # Unload using the skill name ("my_tool")
        mgr._unload_skill("my_tool")

        # Should have cleaned up the correct module
        assert "heimdall_skill_my_tool_file" not in sys.modules
