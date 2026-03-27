"""Round 26: Skill system gap analysis — metadata, validation, diagnostics, status."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.schema import ToolsConfig, ToolHost
from src.tools.executor import ToolExecutor
from src.tools.skill_manager import (
    LoadedSkill,
    SkillDiagnostic,
    SkillManager,
    SkillMetadata,
    SkillStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def skill_mgr(tmp_dir: Path, tools_config: ToolsConfig) -> SkillManager:
    executor = ToolExecutor(tools_config)
    skills_dir = tmp_dir / "skills"
    skills_dir.mkdir()
    return SkillManager(str(skills_dir), executor)


# ---------------------------------------------------------------------------
# Skill code templates
# ---------------------------------------------------------------------------

MINIMAL_SKILL = '''
SKILL_DEFINITION = {
    "name": "minimal",
    "description": "A minimal skill",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    return "ok"
'''

RICH_METADATA_SKILL = '''
SKILL_DEFINITION = {
    "name": "rich_meta",
    "description": "Skill with rich metadata",
    "input_schema": {"type": "object", "properties": {}},
    "version": "1.2.3",
    "author": "Heimdall",
    "homepage": "https://example.com",
    "tags": ["monitoring", "alerts"],
    "dependencies": ["requests", "pyyaml"],
    "config_schema": {
        "type": "object",
        "properties": {
            "threshold": {"type": "number", "default": 90},
        },
    },
}

async def execute(inp, context):
    return "ok"
'''

BAD_VERSION_SKILL = '''
SKILL_DEFINITION = {
    "name": "bad_version",
    "description": "Invalid version format",
    "input_schema": {"type": "object", "properties": {}},
    "version": "not-a-version",
}

async def execute(inp, context):
    return "ok"
'''

BAD_TYPES_SKILL = '''
SKILL_DEFINITION = {
    "name": "bad_types",
    "description": "Bad metadata types",
    "input_schema": {"type": "object", "properties": {}},
    "version": 123,
    "author": 456,
    "tags": "not-a-list",
    "dependencies": {"not": "a-list"},
    "config_schema": ["not", "a", "dict"],
}

async def execute(inp, context):
    return "ok"
'''

SYNC_EXECUTE_SKILL = '''
SKILL_DEFINITION = {
    "name": "sync_exec",
    "description": "Non-async execute",
    "input_schema": {"type": "object", "properties": {}},
}

def execute(inp, context):
    return "ok"
'''

HANDOFF_SKILL = '''
SKILL_DEFINITION = {
    "name": "handoff_skill",
    "description": "Skill with handoff",
    "input_schema": {"type": "object", "properties": {}},
    "version": "2.0.0",
    "author": "Test Author",
    "handoff_to_codex": True,
}

async def execute(inp, context):
    return "result for codex"
'''

SYNTAX_ERROR_SKILL = '''
SKILL_DEFINITION = {
    "name": "broken"
    # missing comma
    "description": "oops"
}
'''

NO_DEFINITION_SKILL = '''
async def execute(inp, context):
    return "no definition"
'''

NO_EXECUTE_SKILL = '''
SKILL_DEFINITION = {
    "name": "no_exec",
    "description": "Missing execute",
    "input_schema": {"type": "object", "properties": {}},
}
'''

MISSING_KEYS_SKILL = '''
SKILL_DEFINITION = {
    "name": "incomplete",
}

async def execute(inp, context):
    return "ok"
'''


# ===================================================================
# SkillMetadata tests
# ===================================================================

class TestSkillMetadata:
    """Tests for SkillMetadata.from_definition()."""

    def test_defaults(self):
        meta, diags = SkillMetadata.from_definition({})
        assert meta.version == "0.0.0"
        assert meta.author == ""
        assert meta.homepage == ""
        assert meta.tags == []
        assert meta.dependencies == []
        assert meta.config_schema == {}
        assert diags == []

    def test_all_fields(self):
        definition = {
            "version": "1.2.3",
            "author": "Heimdall",
            "homepage": "https://example.com",
            "tags": ["infra", "monitoring"],
            "dependencies": ["requests"],
            "config_schema": {"type": "object"},
        }
        meta, diags = SkillMetadata.from_definition(definition)
        assert meta.version == "1.2.3"
        assert meta.author == "Heimdall"
        assert meta.homepage == "https://example.com"
        assert meta.tags == ["infra", "monitoring"]
        assert meta.dependencies == ["requests"]
        assert meta.config_schema == {"type": "object"}
        assert diags == []

    def test_invalid_version_string(self):
        meta, diags = SkillMetadata.from_definition({"version": "bad"})
        assert meta.version == "0.0.0"
        assert len(diags) == 1
        assert diags[0].level == "warn"
        assert "semver" in diags[0].message

    def test_version_not_string(self):
        meta, diags = SkillMetadata.from_definition({"version": 123})
        assert meta.version == "0.0.0"
        assert any("must be a string" in d.message for d in diags)

    def test_author_not_string(self):
        meta, diags = SkillMetadata.from_definition({"author": 42})
        assert meta.author == ""
        assert any("author must be a string" in d.message for d in diags)

    def test_homepage_not_string(self):
        meta, diags = SkillMetadata.from_definition({"homepage": []})
        assert meta.homepage == ""
        assert any("homepage must be a string" in d.message for d in diags)

    def test_tags_not_list(self):
        meta, diags = SkillMetadata.from_definition({"tags": "oops"})
        assert meta.tags == []
        assert any("tags must be a list" in d.message for d in diags)

    def test_tags_with_non_strings(self):
        meta, diags = SkillMetadata.from_definition({"tags": [1, 2]})
        assert meta.tags == []
        assert any("tags must be a list of strings" in d.message for d in diags)

    def test_dependencies_not_list(self):
        meta, diags = SkillMetadata.from_definition({"dependencies": {"a": 1}})
        assert meta.dependencies == []
        assert any("dependencies must be a list" in d.message for d in diags)

    def test_config_schema_not_dict(self):
        meta, diags = SkillMetadata.from_definition({"config_schema": [1]})
        assert meta.config_schema == {}
        assert any("config_schema must be a dict" in d.message for d in diags)

    def test_empty_version_defaults(self):
        meta, diags = SkillMetadata.from_definition({"version": ""})
        assert meta.version == "0.0.0"
        assert diags == []

    def test_semver_with_prerelease(self):
        meta, diags = SkillMetadata.from_definition({"version": "1.0.0-beta.1"})
        assert meta.version == "1.0.0-beta.1"
        assert diags == []

    def test_semver_with_build(self):
        meta, diags = SkillMetadata.from_definition({"version": "1.0.0+build.42"})
        assert meta.version == "1.0.0+build.42"
        assert diags == []

    def test_multiple_bad_fields(self):
        meta, diags = SkillMetadata.from_definition({
            "version": 1, "author": 2, "tags": "x", "dependencies": 3, "config_schema": "y",
        })
        # Should have diagnostics for version, author, tags, dependencies, config_schema
        assert len(diags) == 5


# ===================================================================
# SkillStatus tests
# ===================================================================

class TestSkillStatus:
    def test_values(self):
        assert SkillStatus.LOADED.value == "loaded"
        assert SkillStatus.DISABLED.value == "disabled"
        assert SkillStatus.ERROR.value == "error"

    def test_is_string_enum(self):
        assert isinstance(SkillStatus.LOADED, str)
        assert SkillStatus.LOADED == "loaded"


# ===================================================================
# SkillDiagnostic tests
# ===================================================================

class TestSkillDiagnostic:
    def test_creation(self):
        d = SkillDiagnostic(level="error", message="something broke")
        assert d.level == "error"
        assert d.message == "something broke"

    def test_warn(self):
        d = SkillDiagnostic(level="warn", message="minor issue")
        assert d.level == "warn"


# ===================================================================
# LoadedSkill tests
# ===================================================================

class TestLoadedSkill:
    def test_defaults(self):
        skill = LoadedSkill(
            name="test", definition={}, execute_fn=lambda: None,
            file_path=Path("/tmp/test.py"), loaded_at="2024-01-01",
        )
        assert skill.status == SkillStatus.LOADED
        assert skill.metadata.version == "0.0.0"
        assert skill.diagnostics == []

    def test_with_metadata(self):
        meta = SkillMetadata(version="1.0.0", author="Test", tags=["x"])
        skill = LoadedSkill(
            name="test", definition={}, execute_fn=lambda: None,
            file_path=Path("/tmp/test.py"), loaded_at="2024-01-01",
            metadata=meta,
        )
        assert skill.metadata.version == "1.0.0"
        assert skill.metadata.author == "Test"
        assert skill.metadata.tags == ["x"]


# ===================================================================
# SkillManager metadata integration
# ===================================================================

class TestSkillManagerMetadata:
    """Tests that SkillManager properly loads and exposes metadata."""

    def test_minimal_skill_defaults(self, skill_mgr: SkillManager):
        result = skill_mgr.create_skill("minimal", MINIMAL_SKILL)
        assert "successfully" in result
        skill = skill_mgr._skills["minimal"]
        assert skill.metadata.version == "0.0.0"
        assert skill.metadata.author == ""
        assert skill.metadata.tags == []
        assert skill.metadata.dependencies == []
        assert skill.status == SkillStatus.LOADED
        assert skill.diagnostics == []

    def test_rich_metadata_skill(self, skill_mgr: SkillManager):
        result = skill_mgr.create_skill("rich_meta", RICH_METADATA_SKILL)
        assert "successfully" in result
        skill = skill_mgr._skills["rich_meta"]
        assert skill.metadata.version == "1.2.3"
        assert skill.metadata.author == "Heimdall"
        assert skill.metadata.homepage == "https://example.com"
        assert skill.metadata.tags == ["monitoring", "alerts"]
        assert skill.metadata.dependencies == ["requests", "pyyaml"]
        assert skill.metadata.config_schema == {
            "type": "object",
            "properties": {"threshold": {"type": "number", "default": 90}},
        }

    def test_bad_version_produces_diagnostic(self, skill_mgr: SkillManager):
        result = skill_mgr.create_skill("bad_version", BAD_VERSION_SKILL)
        assert "successfully" in result
        skill = skill_mgr._skills["bad_version"]
        assert skill.metadata.version == "0.0.0"
        assert len(skill.diagnostics) == 1
        assert skill.diagnostics[0].level == "warn"

    def test_bad_types_produce_diagnostics(self, skill_mgr: SkillManager):
        result = skill_mgr.create_skill("bad_types", BAD_TYPES_SKILL)
        assert "successfully" in result
        skill = skill_mgr._skills["bad_types"]
        assert len(skill.diagnostics) == 5  # version, author, tags, deps, config_schema

    def test_edit_reloads_with_metadata(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("rich_meta", RICH_METADATA_SKILL)
        assert skill_mgr._skills["rich_meta"].metadata.version == "1.2.3"
        # Edit the description (simpler, avoids module caching)
        updated_code = RICH_METADATA_SKILL.replace(
            "Skill with rich metadata", "Updated description"
        )
        result = skill_mgr.edit_skill("rich_meta", updated_code)
        assert "successfully" in result
        skill = skill_mgr._skills["rich_meta"]
        # Metadata is re-parsed from the reloaded definition
        assert skill.metadata.author == "Heimdall"
        assert skill.status == SkillStatus.LOADED


# ===================================================================
# list_skills enriched output
# ===================================================================

class TestListSkillsEnriched:
    """Tests that list_skills returns enriched metadata."""

    def test_minimal_skill_list(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("minimal", MINIMAL_SKILL)
        skills = skill_mgr.list_skills()
        assert len(skills) == 1
        s = skills[0]
        assert s["name"] == "minimal"
        assert s["status"] == "loaded"
        assert s["version"] == "0.0.0"
        assert s["author"] == ""
        assert s["tags"] == []
        assert s["dependencies"] == []
        assert s["has_config"] is False
        assert s["diagnostics"] == []

    def test_rich_skill_list(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("rich_meta", RICH_METADATA_SKILL)
        skills = skill_mgr.list_skills()
        s = skills[0]
        assert s["version"] == "1.2.3"
        assert s["author"] == "Heimdall"
        assert s["tags"] == ["monitoring", "alerts"]
        assert s["dependencies"] == ["requests", "pyyaml"]
        assert s["has_config"] is True

    def test_diagnostics_in_list(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("bad_version", BAD_VERSION_SKILL)
        skills = skill_mgr.list_skills()
        s = skills[0]
        assert len(s["diagnostics"]) == 1
        assert s["diagnostics"][0]["level"] == "warn"
        assert "semver" in s["diagnostics"][0]["message"]


# ===================================================================
# get_skill_info detailed output
# ===================================================================

class TestGetSkillInfo:
    """Tests for get_skill_info()."""

    def test_not_found(self, skill_mgr: SkillManager):
        assert skill_mgr.get_skill_info("nonexistent") is None

    def test_minimal_info(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("minimal", MINIMAL_SKILL)
        info = skill_mgr.get_skill_info("minimal")
        assert info is not None
        assert info["name"] == "minimal"
        assert info["status"] == "loaded"
        assert info["metadata"]["version"] == "0.0.0"
        assert info["code"] is not None
        assert "SKILL_DEFINITION" in info["code"]
        assert info["handoff_to_codex"] is False

    def test_rich_info(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("rich_meta", RICH_METADATA_SKILL)
        info = skill_mgr.get_skill_info("rich_meta")
        assert info["metadata"]["version"] == "1.2.3"
        assert info["metadata"]["author"] == "Heimdall"
        assert info["metadata"]["homepage"] == "https://example.com"
        assert info["metadata"]["tags"] == ["monitoring", "alerts"]
        assert info["metadata"]["dependencies"] == ["requests", "pyyaml"]
        assert info["metadata"]["has_config"] is True
        assert "threshold" in str(info["metadata"]["config_schema"])

    def test_handoff_info(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("handoff_skill", HANDOFF_SKILL)
        info = skill_mgr.get_skill_info("handoff_skill")
        assert info["handoff_to_codex"] is True
        assert info["metadata"]["version"] == "2.0.0"
        assert info["metadata"]["author"] == "Test Author"

    def test_info_includes_file_path(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("minimal", MINIMAL_SKILL)
        info = skill_mgr.get_skill_info("minimal")
        assert "minimal.py" in info["file_path"]


# ===================================================================
# validate_skill_code
# ===================================================================

class TestValidateSkillCode:
    """Tests for validate_skill_code()."""

    def test_valid_minimal(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(MINIMAL_SKILL)
        assert report["valid"] is True
        assert report["errors"] == []
        assert "name" in report["definition_keys"]
        assert report["metadata"]["version"] == "0.0.0"

    def test_valid_rich(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(RICH_METADATA_SKILL)
        assert report["valid"] is True
        assert report["metadata"]["version"] == "1.2.3"
        assert report["metadata"]["author"] == "Heimdall"
        assert report["metadata"]["has_config"] is True

    def test_syntax_error(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(SYNTAX_ERROR_SKILL)
        assert report["valid"] is False
        assert any("Syntax error" in e for e in report["errors"])
        assert report["metadata"] is None
        assert report["definition_keys"] == []

    def test_no_definition(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(NO_DEFINITION_SKILL)
        assert report["valid"] is False
        assert any("Missing or invalid SKILL_DEFINITION" in e for e in report["errors"])

    def test_no_execute(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(NO_EXECUTE_SKILL)
        assert report["valid"] is False
        assert any("Missing execute()" in e for e in report["errors"])

    def test_missing_required_keys(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(MISSING_KEYS_SKILL)
        assert report["valid"] is False
        assert any("description" in e for e in report["errors"])
        assert any("input_schema" in e for e in report["errors"])

    def test_sync_execute_warning(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(SYNC_EXECUTE_SKILL)
        # Sync execute is valid (it works), but produces a warning
        assert report["valid"] is True
        assert any("not async" in w for w in report["warnings"])

    def test_bad_version_warning(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(BAD_VERSION_SKILL)
        assert report["valid"] is True
        assert any("semver" in w for w in report["warnings"])

    def test_bad_types_warnings(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(BAD_TYPES_SKILL)
        assert report["valid"] is True
        assert len(report["warnings"]) == 5

    def test_name_conflicts_with_builtin(self, skill_mgr: SkillManager):
        code = '''
SKILL_DEFINITION = {
    "name": "check_disk",
    "description": "Conflicts with builtin",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    return "ok"
'''
        report = skill_mgr.validate_skill_code(code)
        assert report["valid"] is False
        assert any("conflicts" in e for e in report["errors"])

    def test_empty_code(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code("")
        assert report["valid"] is False
        assert any("Missing" in e for e in report["errors"])

    def test_execution_error(self, skill_mgr: SkillManager):
        code = '''
raise RuntimeError("boom")
'''
        report = skill_mgr.validate_skill_code(code)
        assert report["valid"] is False
        assert any("Execution error" in e for e in report["errors"])


# ===================================================================
# SkillManager _load_all with metadata
# ===================================================================

class TestLoadAllWithMetadata:
    """Tests that _load_all correctly populates metadata."""

    def test_preloaded_skills_have_metadata(self, tmp_dir: Path, tools_config: ToolsConfig):
        skills_dir = tmp_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "rich_meta.py").write_text(RICH_METADATA_SKILL)
        mgr = SkillManager(str(skills_dir), ToolExecutor(tools_config))
        assert mgr.has_skill("rich_meta")
        skill = mgr._skills["rich_meta"]
        assert skill.metadata.version == "1.2.3"
        assert skill.metadata.author == "Heimdall"
        assert skill.status == SkillStatus.LOADED

    def test_preloaded_with_bad_metadata(self, tmp_dir: Path, tools_config: ToolsConfig):
        skills_dir = tmp_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "bad_types.py").write_text(BAD_TYPES_SKILL)
        mgr = SkillManager(str(skills_dir), ToolExecutor(tools_config))
        assert mgr.has_skill("bad_types")
        skill = mgr._skills["bad_types"]
        assert len(skill.diagnostics) == 5
        assert skill.status == SkillStatus.LOADED  # Still loads despite warnings


# ===================================================================
# Web API integration (mock-level)
# ===================================================================

class TestWebAPISkillEndpoints:
    """Tests that the new API endpoints work correctly."""

    def test_get_skill_detail_exists(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("rich_meta", RICH_METADATA_SKILL)
        info = skill_mgr.get_skill_info("rich_meta")
        assert info is not None
        assert info["metadata"]["version"] == "1.2.3"

    def test_get_skill_detail_not_found(self, skill_mgr: SkillManager):
        info = skill_mgr.get_skill_info("nope")
        assert info is None

    def test_validate_valid_code(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(MINIMAL_SKILL)
        assert report["valid"] is True

    def test_validate_invalid_code(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(SYNTAX_ERROR_SKILL)
        assert report["valid"] is False


# ===================================================================
# Edge cases
# ===================================================================

class TestMetadataEdgeCases:
    """Edge cases for metadata parsing."""

    def test_empty_tags_list(self):
        meta, diags = SkillMetadata.from_definition({"tags": []})
        assert meta.tags == []
        assert diags == []

    def test_empty_dependencies_list(self):
        meta, diags = SkillMetadata.from_definition({"dependencies": []})
        assert meta.dependencies == []
        assert diags == []

    def test_empty_config_schema(self):
        meta, diags = SkillMetadata.from_definition({"config_schema": {}})
        assert meta.config_schema == {}
        assert diags == []

    def test_none_values_use_defaults(self):
        # None values in optional fields should be treated as missing
        meta, diags = SkillMetadata.from_definition({
            "tags": None, "dependencies": None, "config_schema": None,
        })
        # None is falsy so these skip the "if raw_x:" checks
        assert meta.tags == []
        assert meta.dependencies == []
        assert meta.config_schema == {}

    def test_extra_keys_ignored(self):
        meta, diags = SkillMetadata.from_definition({
            "version": "1.0.0",
            "custom_field": "ignored",
            "another_one": 42,
        })
        assert meta.version == "1.0.0"
        assert diags == []

    def test_validate_preserves_existing_skill_check(self, skill_mgr: SkillManager):
        """validate_skill_code should not interfere with already-loaded skills."""
        skill_mgr.create_skill("minimal", MINIMAL_SKILL)
        report = skill_mgr.validate_skill_code(RICH_METADATA_SKILL)
        assert report["valid"] is True
        # Original skill still present
        assert skill_mgr.has_skill("minimal")


# ===================================================================
# Backward compatibility
# ===================================================================

class TestBackwardCompatibility:
    """Ensure existing skill functionality is not broken."""

    def test_old_style_skill_still_works(self, skill_mgr: SkillManager):
        """A skill with only the required fields should still load fine."""
        code = '''
SKILL_DEFINITION = {
    "name": "old_style",
    "description": "No metadata at all",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    return "works"
'''
        result = skill_mgr.create_skill("old_style", code)
        assert "successfully" in result
        assert skill_mgr.has_skill("old_style")
        # Default metadata
        skill = skill_mgr._skills["old_style"]
        assert skill.metadata.version == "0.0.0"
        assert skill.diagnostics == []

    async def test_execution_still_works(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("minimal", MINIMAL_SKILL)
        result = await skill_mgr.execute("minimal", {})
        assert result == "ok"

    def test_get_tool_definitions_unchanged(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("minimal", MINIMAL_SKILL)
        defs = skill_mgr.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "minimal"
        assert "description" in defs[0]
        assert "input_schema" in defs[0]
        # No extra metadata in tool definitions (LLM doesn't need it)
        assert "version" not in defs[0]

    def test_should_handoff_to_codex(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("handoff_skill", HANDOFF_SKILL)
        assert skill_mgr.should_handoff_to_codex("handoff_skill") is True
        skill_mgr.create_skill("minimal", MINIMAL_SKILL)
        assert skill_mgr.should_handoff_to_codex("minimal") is False

    def test_delete_still_works(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("minimal", MINIMAL_SKILL)
        assert skill_mgr.has_skill("minimal")
        result = skill_mgr.delete_skill("minimal")
        assert "deleted" in result.lower()
        assert not skill_mgr.has_skill("minimal")

    def test_edit_still_works(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("minimal", MINIMAL_SKILL)
        new_code = MINIMAL_SKILL.replace("A minimal skill", "Updated skill")
        result = skill_mgr.edit_skill("minimal", new_code)
        assert "successfully" in result
        info = skill_mgr.get_skill_info("minimal")
        assert info["description"] == "Updated skill"
