"""Tests for tools/skill_manager.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config.schema import ToolsConfig, ToolHost
from src.tools.executor import ToolExecutor
from src.tools.skill_manager import SkillManager, SKILL_NAME_PATTERN


VALID_SKILL_CODE = '''
SKILL_DEFINITION = {
    "name": "test_skill",
    "description": "A test skill",
    "input_schema": {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
    },
    "requires_approval": False,
}

async def execute(inp, context):
    return f"Got: {inp.get('msg', 'nothing')}"
'''

INVALID_SKILL_CODE = '''
# Missing SKILL_DEFINITION
async def execute(inp, context):
    return "broken"
'''


@pytest.fixture
def skill_mgr(tmp_dir: Path, tools_config: ToolsConfig) -> SkillManager:
    executor = ToolExecutor(tools_config)
    skills_dir = tmp_dir / "skills"
    skills_dir.mkdir()
    return SkillManager(str(skills_dir), executor)


class TestNameValidation:
    def test_valid_names(self):
        assert SKILL_NAME_PATTERN.match("check_ssl")
        assert SKILL_NAME_PATTERN.match("a")
        assert SKILL_NAME_PATTERN.match("my_skill_123")

    def test_invalid_names(self):
        assert not SKILL_NAME_PATTERN.match("")
        assert not SKILL_NAME_PATTERN.match("123abc")
        assert not SKILL_NAME_PATTERN.match("CamelCase")
        assert not SKILL_NAME_PATTERN.match("has-dash")
        assert not SKILL_NAME_PATTERN.match("_leading")

    def test_builtin_collision(self, skill_mgr: SkillManager):
        error = skill_mgr._validate_name("check_disk")
        assert error is not None
        assert "conflicts" in error


class TestCreateSkill:
    def test_create_valid(self, skill_mgr: SkillManager):
        result = skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        assert "successfully" in result
        assert skill_mgr.has_skill("test_skill")

    def test_create_duplicate(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        result = skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        assert "already exists" in result

    def test_create_invalid_code(self, skill_mgr: SkillManager):
        result = skill_mgr.create_skill("bad_skill", INVALID_SKILL_CODE)
        assert "failed to load" in result.lower()
        assert not skill_mgr.has_skill("bad_skill")

    def test_create_invalid_name(self, skill_mgr: SkillManager):
        result = skill_mgr.create_skill("BadName", VALID_SKILL_CODE)
        assert "Invalid" in result

    def test_name_mismatch(self, skill_mgr: SkillManager):
        code = VALID_SKILL_CODE.replace('"test_skill"', '"wrong_name"')
        result = skill_mgr.create_skill("test_skill", code)
        assert "doesn't match" in result


class TestEditSkill:
    def test_edit_valid(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        new_code = VALID_SKILL_CODE.replace("A test skill", "Updated skill")
        result = skill_mgr.edit_skill("test_skill", new_code)
        assert "updated" in result.lower()

    def test_edit_bad_code_reverts(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        result = skill_mgr.edit_skill("test_skill", INVALID_SKILL_CODE)
        assert "Reverted" in result
        # Original should still be loaded
        assert skill_mgr.has_skill("test_skill")

    def test_edit_nonexistent(self, skill_mgr: SkillManager):
        result = skill_mgr.edit_skill("nonexistent", VALID_SKILL_CODE)
        assert "not found" in result


class TestDeleteSkill:
    def test_delete(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        result = skill_mgr.delete_skill("test_skill")
        assert "deleted" in result.lower()
        assert not skill_mgr.has_skill("test_skill")

    def test_delete_nonexistent(self, skill_mgr: SkillManager):
        result = skill_mgr.delete_skill("nonexistent")
        assert "not found" in result


class TestListSkills:
    def test_list_empty(self, skill_mgr: SkillManager):
        assert skill_mgr.list_skills() == []

    def test_list_with_skills(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        skills = skill_mgr.list_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "test_skill"
        assert skills[0]["requires_approval"] is False


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_skill(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        result = await skill_mgr.execute("test_skill", {"msg": "hello"})
        assert result == "Got: hello"

    @pytest.mark.asyncio
    async def test_execute_nonexistent(self, skill_mgr: SkillManager):
        result = await skill_mgr.execute("nonexistent", {})
        assert "not found" in result


class TestToolDefinitions:
    def test_returns_api_format(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        defs = skill_mgr.get_tool_definitions()
        assert len(defs) == 1
        assert "name" in defs[0]
        assert "description" in defs[0]
        assert "input_schema" in defs[0]
        assert "requires_approval" not in defs[0]


class TestRequiresApproval:
    def test_skill_approval(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        assert skill_mgr.requires_approval("test_skill") is False

    def test_non_skill_returns_none(self, skill_mgr: SkillManager):
        assert skill_mgr.requires_approval("check_disk") is None
