"""Round 33: Comprehensive skill system tests.

Tests edge cases, error paths, and integration scenarios across skill_manager.py
and skill_context.py that existing test files do not cover.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.schema import ToolsConfig, ToolHost
from src.tools.executor import ToolExecutor
from src.tools.skill_manager import (
    BUILTIN_TOOL_NAMES,
    MAX_SKILL_DEPENDENCIES,
    MAX_SKILL_DOWNLOAD_BYTES,
    MAX_SKILL_OUTPUT_CHARS,
    SKILL_EXECUTE_TIMEOUT,
    SKILL_NAME_PATTERN,
    LoadedSkill,
    SkillDiagnostic,
    SkillExecutionStats,
    SkillManager,
    SkillMetadata,
    SkillStatus,
    _extract_dependencies_from_source,
    _install_packages,
    _is_package_installed,
    _parse_package_name,
    apply_defaults,
    resolve_dependencies,
    validate_config,
    validate_config_value,
)
from src.tools.skill_context import (
    MAX_SKILL_FILES,
    MAX_SKILL_HTTP_REQUESTS,
    MAX_SKILL_MESSAGES,
    MAX_SKILL_TOOL_CALLS,
    SKILL_SAFE_TOOLS,
    ResourceTracker,
    SkillContext,
    is_path_denied,
    is_url_blocked,
)


# ---------------------------------------------------------------------------
# Skill code templates
# ---------------------------------------------------------------------------

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

SKILL_WITH_METADATA = '''
SKILL_DEFINITION = {
    "name": "meta_skill",
    "description": "Skill with full metadata",
    "input_schema": {"type": "object", "properties": {}},
    "version": "2.1.0",
    "author": "Test Author",
    "homepage": "https://example.com",
    "tags": ["test", "demo"],
    "dependencies": [],
    "config_schema": {
        "type": "object",
        "properties": {
            "greeting": {"type": "string", "default": "hello"},
            "count": {"type": "integer", "default": 5, "minimum": 1, "maximum": 100},
        },
        "required": ["greeting"],
    },
    "handoff_to_codex": True,
}

async def execute(inp, context):
    cfg = context.get_all_config()
    return f"greeting={cfg.get('greeting')}, count={cfg.get('count')}"
'''

SKILL_WITH_DEPS = '''
SKILL_DEFINITION = {
    "name": "dep_skill",
    "description": "Has dependencies",
    "input_schema": {"type": "object", "properties": {}},
    "dependencies": ["requests>=2.0"],
}

async def execute(inp, context):
    return "ok"
'''

SYNC_EXECUTE_SKILL = '''
SKILL_DEFINITION = {
    "name": "sync_skill",
    "description": "Sync execute",
    "input_schema": {"type": "object", "properties": {}},
}

def execute(inp, context):
    return "sync result"
'''


@pytest.fixture
def skill_mgr(tmp_dir: Path, tools_config: ToolsConfig) -> SkillManager:
    executor = ToolExecutor(tools_config)
    skills_dir = tmp_dir / "skills"
    skills_dir.mkdir()
    return SkillManager(str(skills_dir), executor)


@pytest.fixture
def skill_mgr_with_memory(tmp_dir: Path, tools_config: ToolsConfig) -> SkillManager:
    executor = ToolExecutor(tools_config)
    skills_dir = tmp_dir / "skills"
    skills_dir.mkdir()
    memory_path = str(tmp_dir / "tool_memory.json")
    return SkillManager(str(skills_dir), executor, memory_path=memory_path)


# ===========================================================================
# Part 1: Helper function tests
# ===========================================================================


class TestParsePackageName:
    """Tests for _parse_package_name()."""

    def test_simple_name(self):
        assert _parse_package_name("requests") == "requests"

    def test_version_constraint(self):
        assert _parse_package_name("requests>=2.0") == "requests"

    def test_exact_version(self):
        assert _parse_package_name("flask==2.3.1") == "flask"

    def test_compatible_release(self):
        assert _parse_package_name("aiohttp~=3.9") == "aiohttp"

    def test_not_equal(self):
        assert _parse_package_name("pkg!=1.0") == "pkg"

    def test_extras(self):
        assert _parse_package_name("Pillow[jpeg]") == "Pillow"

    def test_extras_with_version(self):
        assert _parse_package_name("requests[security]>=2.20") == "requests"

    def test_whitespace(self):
        assert _parse_package_name("  requests  ") == "requests"

    def test_empty_string(self):
        assert _parse_package_name("") == ""

    def test_whitespace_only(self):
        assert _parse_package_name("   ") == ""

    def test_hyphenated_name(self):
        assert _parse_package_name("my-package") == "my-package"

    def test_dotted_name(self):
        assert _parse_package_name("zope.interface") == "zope.interface"

    def test_underscored_name(self):
        assert _parse_package_name("my_package") == "my_package"

    def test_single_char(self):
        assert _parse_package_name("a") == "a"

    def test_numeric_start(self):
        assert _parse_package_name("3to2") == "3to2"


class TestIsPackageInstalled:
    """Tests for _is_package_installed()."""

    def test_installed_package(self):
        assert _is_package_installed("pytest") is True

    def test_missing_package(self):
        assert _is_package_installed("nonexistent_pkg_xyz_999") is False

    def test_empty_name(self):
        # Empty string raises ValueError from importlib.metadata
        with pytest.raises(ValueError):
            _is_package_installed("")


class TestInstallPackages:
    """Tests for _install_packages()."""

    def test_timeout_returns_failure(self):
        with patch("subprocess.run", side_effect=subprocess_timeout()):
            ok, output = _install_packages(["fakepkg"], timeout=1)
            assert ok is False
            assert "timed out" in output

    def test_generic_exception(self):
        with patch("subprocess.run", side_effect=OSError("no pip")):
            ok, output = _install_packages(["fakepkg"])
            assert ok is False
            assert "no pip" in output

    def test_success(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            ok, output = _install_packages(["fakepkg"])
            assert ok is True

    def test_failure_returncode(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="ERROR: no matching dist"
            )
            ok, output = _install_packages(["fakepkg"])
            assert ok is False
            assert "no matching dist" in output


class TestExtractDependenciesFromSource:
    """Tests for _extract_dependencies_from_source()."""

    def test_simple_deps(self):
        code = '''
SKILL_DEFINITION = {
    "name": "x",
    "description": "y",
    "input_schema": {},
    "dependencies": ["requests", "flask>=2.0"],
}
'''
        deps = _extract_dependencies_from_source(code)
        assert deps == ["requests", "flask>=2.0"]

    def test_no_deps_key(self):
        code = '''
SKILL_DEFINITION = {
    "name": "x",
    "description": "y",
    "input_schema": {},
}
'''
        assert _extract_dependencies_from_source(code) == []

    def test_no_definition(self):
        code = 'x = 1\n'
        assert _extract_dependencies_from_source(code) == []

    def test_syntax_error(self):
        assert _extract_dependencies_from_source("def (broken:") == []

    def test_dynamic_definition_not_extractable(self):
        code = '''
import os
SKILL_DEFINITION = dict(name=os.getenv("X"), description="y", input_schema={}, dependencies=["a"])
'''
        # dict() call is not a literal — cannot extract
        assert _extract_dependencies_from_source(code) == []

    def test_deps_not_list(self):
        code = '''
SKILL_DEFINITION = {
    "name": "x",
    "description": "y",
    "input_schema": {},
    "dependencies": "requests",
}
'''
        assert _extract_dependencies_from_source(code) == []

    def test_deps_list_with_non_strings(self):
        code = '''
SKILL_DEFINITION = {
    "name": "x",
    "description": "y",
    "input_schema": {},
    "dependencies": ["requests", 42],
}
'''
        assert _extract_dependencies_from_source(code) == []

    def test_multiple_assignments_takes_first(self):
        code = '''
SKILL_DEFINITION = {"name": "a", "description": "b", "input_schema": {}, "dependencies": ["first"]}
SKILL_DEFINITION = {"name": "c", "description": "d", "input_schema": {}, "dependencies": ["second"]}
'''
        deps = _extract_dependencies_from_source(code)
        assert deps == ["first"]

    def test_definition_inside_class(self):
        code = '''
class Foo:
    SKILL_DEFINITION = {"name": "x", "description": "y", "input_schema": {}, "dependencies": ["inner"]}
'''
        # ast.walk should still find the assignment inside a class
        deps = _extract_dependencies_from_source(code)
        assert deps == ["inner"]


class TestResolveDependencies:
    """Tests for resolve_dependencies()."""

    def test_empty_deps(self):
        installed, newly, diags = resolve_dependencies([])
        assert installed == []
        assert newly == []
        assert diags == []

    def test_too_many_deps(self):
        deps = [f"pkg{i}" for i in range(MAX_SKILL_DEPENDENCIES + 1)]
        installed, newly, diags = resolve_dependencies(deps)
        assert installed == []
        assert newly == []
        assert any("Too many" in d.message for d in diags)

    def test_invalid_spec(self):
        _, _, diags = resolve_dependencies([""])
        assert any("Invalid dependency spec" in d.message for d in diags)

    def test_already_installed(self):
        with patch("src.tools.skill_manager._is_package_installed", return_value=True):
            installed, newly, diags = resolve_dependencies(["pytest"])
            assert "pytest" in installed
            assert newly == []

    def test_install_success(self):
        with patch("src.tools.skill_manager._is_package_installed", return_value=False):
            with patch("src.tools.skill_manager._install_packages", return_value=(True, "")):
                installed, newly, diags = resolve_dependencies(["newpkg"])
                assert installed == []
                assert "newpkg" in newly

    def test_install_failure(self):
        with patch("src.tools.skill_manager._is_package_installed", return_value=False):
            with patch("src.tools.skill_manager._install_packages", return_value=(False, "fail")):
                installed, newly, diags = resolve_dependencies(["badpkg"])
                assert installed == []
                assert newly == []
                assert any("Failed to install" in d.message for d in diags)


# ===========================================================================
# Part 2: Config validation tests
# ===========================================================================


class TestValidateConfigValue:
    """Tests for validate_config_value()."""

    def test_string_valid(self):
        assert validate_config_value("x", {"type": "string"}, "hello") is None

    def test_string_invalid(self):
        err = validate_config_value("x", {"type": "string"}, 42)
        assert err is not None
        assert "expected string" in err

    def test_integer_valid(self):
        assert validate_config_value("x", {"type": "integer"}, 5) is None

    def test_integer_bool_rejected(self):
        err = validate_config_value("x", {"type": "integer"}, True)
        assert err is not None
        assert "expected integer" in err

    def test_number_valid_float(self):
        assert validate_config_value("x", {"type": "number"}, 3.14) is None

    def test_number_valid_int(self):
        assert validate_config_value("x", {"type": "number"}, 7) is None

    def test_number_bool_rejected(self):
        err = validate_config_value("x", {"type": "number"}, False)
        assert err is not None

    def test_boolean_valid(self):
        assert validate_config_value("x", {"type": "boolean"}, True) is None

    def test_boolean_invalid(self):
        err = validate_config_value("x", {"type": "boolean"}, 1)
        assert err is not None
        assert "expected boolean" in err

    def test_enum_valid(self):
        schema = {"type": "string", "enum": ["a", "b", "c"]}
        assert validate_config_value("x", schema, "b") is None

    def test_enum_invalid(self):
        schema = {"type": "string", "enum": ["a", "b"]}
        err = validate_config_value("x", schema, "z")
        assert err is not None
        assert "not in allowed values" in err

    def test_minimum(self):
        schema = {"type": "integer", "minimum": 0}
        assert validate_config_value("x", schema, 0) is None
        err = validate_config_value("x", schema, -1)
        assert err is not None
        assert "below minimum" in err

    def test_maximum(self):
        schema = {"type": "integer", "maximum": 10}
        assert validate_config_value("x", schema, 10) is None
        err = validate_config_value("x", schema, 11)
        assert err is not None
        assert "exceeds maximum" in err

    def test_min_length(self):
        schema = {"type": "string", "minLength": 3}
        assert validate_config_value("x", schema, "abc") is None
        err = validate_config_value("x", schema, "ab")
        assert err is not None
        assert "minLength" in err

    def test_max_length(self):
        schema = {"type": "string", "maxLength": 5}
        assert validate_config_value("x", schema, "abcde") is None
        err = validate_config_value("x", schema, "abcdef")
        assert err is not None
        assert "maxLength" in err

    def test_no_type_defaults_to_string(self):
        # No "type" key → defaults to string check
        assert validate_config_value("x", {}, "hello") is None

    def test_null_enum_ignored(self):
        schema = {"type": "string", "enum": None}
        assert validate_config_value("x", schema, "anything") is None


class TestValidateConfig:
    """Tests for validate_config()."""

    def test_valid_config(self):
        schema = {
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assert validate_config(schema, {"name": "test"}) == []

    def test_missing_required_no_default(self):
        schema = {
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        errors = validate_config(schema, {})
        assert any("Missing required" in e for e in errors)

    def test_missing_required_with_default(self):
        schema = {
            "properties": {"name": {"type": "string", "default": "anon"}},
            "required": ["name"],
        }
        # Required but has default → no error
        assert validate_config(schema, {}) == []

    def test_unknown_field(self):
        schema = {"properties": {"name": {"type": "string"}}}
        errors = validate_config(schema, {"name": "ok", "extra": "bad"})
        assert any("Unknown field" in e for e in errors)

    def test_type_mismatch(self):
        schema = {"properties": {"count": {"type": "integer"}}}
        errors = validate_config(schema, {"count": "five"})
        assert len(errors) == 1
        assert "expected integer" in errors[0]

    def test_empty_schema_empty_values(self):
        assert validate_config({}, {}) == []

    def test_multiple_errors(self):
        schema = {
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "string"},
            },
            "required": ["a", "b"],
        }
        errors = validate_config(schema, {"a": "not_int", "b": 42})
        assert len(errors) == 2


class TestApplyDefaults:
    """Tests for apply_defaults()."""

    def test_fills_missing_defaults(self):
        schema = {
            "properties": {
                "x": {"type": "string", "default": "hello"},
                "y": {"type": "integer", "default": 10},
            },
        }
        result = apply_defaults(schema, {})
        assert result == {"x": "hello", "y": 10}

    def test_preserves_provided_values(self):
        schema = {"properties": {"x": {"type": "string", "default": "hello"}}}
        result = apply_defaults(schema, {"x": "custom"})
        assert result == {"x": "custom"}

    def test_no_default_no_fill(self):
        schema = {"properties": {"x": {"type": "string"}}}
        result = apply_defaults(schema, {})
        assert result == {}

    def test_empty_schema(self):
        result = apply_defaults({}, {"anything": "works"})
        assert result == {"anything": "works"}

    def test_complex_default_values(self):
        schema = {
            "properties": {
                "tags": {"type": "string", "default": "a,b,c"},
                "flag": {"type": "boolean", "default": False},
            },
        }
        result = apply_defaults(schema, {"flag": True})
        assert result == {"flag": True, "tags": "a,b,c"}


# ===========================================================================
# Part 3: Data class tests
# ===========================================================================


class TestSkillDiagnostic:
    """Tests for SkillDiagnostic dataclass."""

    def test_creation(self):
        d = SkillDiagnostic("warn", "test message")
        assert d.level == "warn"
        assert d.message == "test message"

    def test_error_level(self):
        d = SkillDiagnostic("error", "bad thing")
        assert d.level == "error"


class TestSkillMetadata:
    """Tests for SkillMetadata dataclass and from_definition()."""

    def test_defaults(self):
        m = SkillMetadata()
        assert m.version == "0.0.0"
        assert m.author == ""
        assert m.homepage == ""
        assert m.tags == []
        assert m.dependencies == []
        assert m.config_schema == {}

    def test_from_definition_all_fields(self):
        defn = {
            "version": "1.2.3",
            "author": "Alice",
            "homepage": "https://example.com",
            "tags": ["infra", "monitoring"],
            "dependencies": ["requests"],
            "config_schema": {"type": "object", "properties": {}},
        }
        meta, diags = SkillMetadata.from_definition(defn)
        assert diags == []
        assert meta.version == "1.2.3"
        assert meta.author == "Alice"
        assert meta.homepage == "https://example.com"
        assert meta.tags == ["infra", "monitoring"]
        assert meta.dependencies == ["requests"]
        assert meta.config_schema == {"type": "object", "properties": {}}

    def test_invalid_version_format(self):
        meta, diags = SkillMetadata.from_definition({"version": "not-semver"})
        assert meta.version == "0.0.0"
        assert any("Invalid version" in d.message for d in diags)

    def test_non_string_version(self):
        meta, diags = SkillMetadata.from_definition({"version": 123})
        assert meta.version == "0.0.0"
        assert any("version must be a string" in d.message for d in diags)

    def test_empty_version(self):
        meta, diags = SkillMetadata.from_definition({"version": ""})
        assert meta.version == "0.0.0"
        assert diags == []

    def test_prerelease_version(self):
        meta, diags = SkillMetadata.from_definition({"version": "1.0.0-beta.1"})
        assert meta.version == "1.0.0-beta.1"
        assert diags == []

    def test_build_metadata_version(self):
        meta, diags = SkillMetadata.from_definition({"version": "1.0.0+build.42"})
        assert meta.version == "1.0.0+build.42"
        assert diags == []

    def test_non_string_author(self):
        meta, diags = SkillMetadata.from_definition({"author": 42})
        assert meta.author == ""
        assert any("author must be a string" in d.message for d in diags)

    def test_non_string_homepage(self):
        meta, diags = SkillMetadata.from_definition({"homepage": True})
        assert meta.homepage == ""
        assert any("homepage must be a string" in d.message for d in diags)

    def test_non_list_tags(self):
        meta, diags = SkillMetadata.from_definition({"tags": "single"})
        assert meta.tags == []
        assert any("tags must be a list" in d.message for d in diags)

    def test_tags_with_non_strings(self):
        meta, diags = SkillMetadata.from_definition({"tags": ["ok", 42]})
        assert meta.tags == []
        assert any("tags must be a list" in d.message for d in diags)

    def test_non_list_dependencies(self):
        meta, diags = SkillMetadata.from_definition({"dependencies": "single"})
        assert meta.dependencies == []
        assert any("dependencies must be a list" in d.message for d in diags)

    def test_non_dict_config_schema(self):
        meta, diags = SkillMetadata.from_definition({"config_schema": "bad"})
        assert meta.config_schema == {}
        assert any("config_schema must be a dict" in d.message for d in diags)

    def test_falsy_tags_no_warning(self):
        """Empty list/None for tags should not produce a warning."""
        meta, diags = SkillMetadata.from_definition({"tags": []})
        assert meta.tags == []
        assert not any("tags" in d.message for d in diags)

    def test_falsy_deps_no_warning(self):
        meta, diags = SkillMetadata.from_definition({"dependencies": []})
        assert meta.dependencies == []
        assert not any("dependencies" in d.message for d in diags)

    def test_falsy_config_schema_no_warning(self):
        meta, diags = SkillMetadata.from_definition({"config_schema": {}})
        assert meta.config_schema == {}
        assert not any("config_schema" in d.message for d in diags)


class TestSkillExecutionStats:
    """Tests for SkillExecutionStats dataclass."""

    def test_defaults(self):
        stats = SkillExecutionStats()
        assert stats.wall_time_ms == 0.0
        assert stats.output_chars == 0
        assert stats.truncated is False
        assert stats.tool_calls == 0
        assert stats.http_requests == 0
        assert stats.messages_sent == 0
        assert stats.files_sent == 0
        assert stats.timestamp == ""

    def test_to_dict(self):
        stats = SkillExecutionStats(
            wall_time_ms=123.456,
            output_chars=500,
            truncated=True,
            tool_calls=3,
            http_requests=2,
            messages_sent=1,
            files_sent=1,
            timestamp="2024-01-01T00:00:00",
        )
        d = stats.to_dict()
        assert d["wall_time_ms"] == 123.5  # rounded to 1 decimal
        assert d["output_chars"] == 500
        assert d["truncated"] is True
        assert d["tool_calls"] == 3
        assert d["http_requests"] == 2
        assert d["messages_sent"] == 1
        assert d["files_sent"] == 1
        assert d["timestamp"] == "2024-01-01T00:00:00"

    def test_zero_wall_time(self):
        stats = SkillExecutionStats(wall_time_ms=0.0)
        assert stats.to_dict()["wall_time_ms"] == 0.0


class TestResourceTracker:
    """Tests for ResourceTracker dataclass."""

    def test_defaults(self):
        t = ResourceTracker()
        assert t.tool_calls == 0
        assert t.http_requests == 0
        assert t.messages_sent == 0
        assert t.files_sent == 0
        assert t.bytes_downloaded == 0

    def test_to_dict(self):
        t = ResourceTracker(tool_calls=5, http_requests=3, messages_sent=2, files_sent=1, bytes_downloaded=1024)
        d = t.to_dict()
        assert d == {
            "tool_calls": 5,
            "http_requests": 3,
            "messages_sent": 2,
            "files_sent": 1,
            "bytes_downloaded": 1024,
        }

    def test_mutation(self):
        t = ResourceTracker()
        t.tool_calls += 1
        t.http_requests += 2
        assert t.tool_calls == 1
        assert t.http_requests == 2


class TestSkillStatus:
    """Tests for SkillStatus enum."""

    def test_values(self):
        assert SkillStatus.LOADED.value == "loaded"
        assert SkillStatus.DISABLED.value == "disabled"
        assert SkillStatus.ERROR.value == "error"

    def test_string_comparison(self):
        assert SkillStatus.LOADED == "loaded"
        assert SkillStatus.DISABLED == "disabled"


# ===========================================================================
# Part 4: SkillManager tests
# ===========================================================================


class TestSkillManagerInit:
    """Tests for SkillManager initialization and loading."""

    def test_creates_skills_dir(self, tmp_dir: Path, tools_config: ToolsConfig):
        skills_dir = tmp_dir / "new_skills_dir"
        assert not skills_dir.exists()
        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(skills_dir), executor)
        assert skills_dir.exists()

    def test_creates_config_dir(self, tmp_dir: Path, tools_config: ToolsConfig):
        skills_dir = tmp_dir / "skills2"
        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(skills_dir), executor)
        assert (skills_dir / "config").is_dir()

    def test_memory_path_derivation(self, tmp_dir: Path, tools_config: ToolsConfig):
        skills_dir = tmp_dir / "skills3"
        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(skills_dir), executor, memory_path=str(tmp_dir / "memory.json"))
        assert mgr._memory_path == str(tmp_dir / "memory_skills.json")

    def test_no_memory_path(self, skill_mgr: SkillManager):
        assert skill_mgr._memory_path is None

    def test_loads_existing_skills(self, tmp_dir: Path, tools_config: ToolsConfig):
        skills_dir = tmp_dir / "skills4"
        skills_dir.mkdir()
        (skills_dir / "test_skill.py").write_text(VALID_SKILL_CODE)
        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(skills_dir), executor)
        assert mgr.has_skill("test_skill")

    def test_bad_skill_file_skipped(self, tmp_dir: Path, tools_config: ToolsConfig):
        skills_dir = tmp_dir / "skills5"
        skills_dir.mkdir()
        (skills_dir / "broken.py").write_text("this is not valid python {{{")
        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(skills_dir), executor)
        assert not mgr.has_skill("broken")

    @pytest.mark.skipif(os.getuid() == 0, reason="root can read any file regardless of permissions")
    def test_unreadable_skill_file(self, tmp_dir: Path, tools_config: ToolsConfig):
        skills_dir = tmp_dir / "skills6"
        skills_dir.mkdir()
        p = skills_dir / "unreadable.py"
        p.write_text(VALID_SKILL_CODE.replace("test_skill", "unreadable"))
        # Make unreadable
        p.chmod(0o000)
        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(skills_dir), executor)
        assert not mgr.has_skill("unreadable")
        # Restore permissions for cleanup
        p.chmod(0o644)


class TestSkillManagerDisabledState:
    """Tests for disabled skill persistence."""

    def test_load_disabled_set_empty(self, skill_mgr: SkillManager):
        assert skill_mgr._disabled == set()

    def test_load_disabled_set_malformed_json(self, tmp_dir: Path, tools_config: ToolsConfig):
        skills_dir = tmp_dir / "skills_dis"
        skills_dir.mkdir()
        (skills_dir / ".disabled.json").write_text("not json{{{")
        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(skills_dir), executor)
        assert mgr._disabled == set()

    def test_load_disabled_set_non_list(self, tmp_dir: Path, tools_config: ToolsConfig):
        skills_dir = tmp_dir / "skills_dis2"
        skills_dir.mkdir()
        (skills_dir / ".disabled.json").write_text('{"not": "a list"}')
        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(skills_dir), executor)
        assert mgr._disabled == set()

    def test_load_disabled_set_filters_non_strings(self, tmp_dir: Path, tools_config: ToolsConfig):
        skills_dir = tmp_dir / "skills_dis3"
        skills_dir.mkdir()
        (skills_dir / ".disabled.json").write_text('["valid_name", 42, null]')
        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(skills_dir), executor)
        assert mgr._disabled == {"valid_name"}

    def test_disabled_state_applied_on_load(self, tmp_dir: Path, tools_config: ToolsConfig):
        skills_dir = tmp_dir / "skills_dis4"
        skills_dir.mkdir()
        (skills_dir / "test_skill.py").write_text(VALID_SKILL_CODE)
        (skills_dir / ".disabled.json").write_text('["test_skill"]')
        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(skills_dir), executor)
        assert mgr.has_skill("test_skill")
        assert not mgr.is_enabled("test_skill")
        assert mgr._skills["test_skill"].status == SkillStatus.DISABLED

    def test_save_disabled_set(self, skill_mgr: SkillManager):
        skill_mgr._disabled = {"foo", "bar"}
        skill_mgr._save_disabled_set()
        data = json.loads(skill_mgr._disabled_path.read_text())
        assert sorted(data) == ["bar", "foo"]


class TestSkillManagerCRUD:
    """Tests for create, edit, delete skill operations."""

    def test_create_invalid_name_uppercase(self, skill_mgr: SkillManager):
        result = skill_mgr.create_skill("BadName", VALID_SKILL_CODE)
        assert "Invalid skill name" in result

    def test_create_invalid_name_dash(self, skill_mgr: SkillManager):
        result = skill_mgr.create_skill("my-skill", VALID_SKILL_CODE)
        assert "Invalid skill name" in result

    def test_create_invalid_name_leading_underscore(self, skill_mgr: SkillManager):
        result = skill_mgr.create_skill("_hidden", VALID_SKILL_CODE)
        assert "Invalid skill name" in result

    def test_create_builtin_collision(self, skill_mgr: SkillManager):
        code = VALID_SKILL_CODE.replace('"test_skill"', '"run_command"')
        result = skill_mgr.create_skill("run_command", code)
        assert "conflicts" in result

    def test_create_duplicate(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        result = skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        assert "already exists" in result

    def test_create_name_mismatch(self, skill_mgr: SkillManager):
        code = VALID_SKILL_CODE.replace('"test_skill"', '"other_name"')
        result = skill_mgr.create_skill("test_skill", code)
        assert "doesn't match" in result
        assert not skill_mgr.has_skill("test_skill")
        assert not skill_mgr.has_skill("other_name")

    def test_create_broken_code(self, skill_mgr: SkillManager):
        result = skill_mgr.create_skill("broken", "not valid python {{{")
        assert "failed to load" in result.lower()

    def test_create_missing_execute(self, skill_mgr: SkillManager):
        code = '''
SKILL_DEFINITION = {
    "name": "no_exec",
    "description": "Missing execute",
    "input_schema": {"type": "object", "properties": {}},
}
'''
        result = skill_mgr.create_skill("no_exec", code)
        assert "failed to load" in result.lower()

    def test_edit_nonexistent(self, skill_mgr: SkillManager):
        result = skill_mgr.edit_skill("nope", "code")
        assert "not found" in result

    def test_edit_success(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        new_code = VALID_SKILL_CODE.replace("A test skill", "Updated description")
        result = skill_mgr.edit_skill("test_skill", new_code)
        assert "updated" in result.lower()
        info = skill_mgr.get_skill_info("test_skill")
        assert info["description"] == "Updated description"

    def test_edit_broken_reverts(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        result = skill_mgr.edit_skill("test_skill", "broken code {{{")
        assert "Reverted" in result
        assert skill_mgr.has_skill("test_skill")

    def test_edit_name_mismatch_reverts(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        bad_code = VALID_SKILL_CODE.replace('"test_skill"', '"different"')
        result = skill_mgr.edit_skill("test_skill", bad_code)
        assert "doesn't match" in result
        assert "Reverted" in result
        assert skill_mgr.has_skill("test_skill")

    def test_delete_nonexistent(self, skill_mgr: SkillManager):
        result = skill_mgr.delete_skill("nope")
        assert "not found" in result

    def test_delete_cleans_config(self, skill_mgr: SkillManager):
        code = SKILL_WITH_METADATA.replace("meta_skill", "cfg_skill")
        skill_mgr.create_skill("cfg_skill", code)
        skill_mgr.set_skill_config("cfg_skill", {"greeting": "hi"})
        config_path = skill_mgr._config_dir / "cfg_skill.json"
        assert config_path.exists()
        skill_mgr.delete_skill("cfg_skill")
        assert not config_path.exists()

    def test_delete_cleans_disabled_state(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        skill_mgr.disable_skill("test_skill")
        assert "test_skill" in skill_mgr._disabled
        skill_mgr.delete_skill("test_skill")
        assert "test_skill" not in skill_mgr._disabled


class TestSkillManagerEnableDisable:
    """Tests for enable/disable skill functionality."""

    def test_enable_not_found(self, skill_mgr: SkillManager):
        assert "not found" in skill_mgr.enable_skill("nope")

    def test_disable_not_found(self, skill_mgr: SkillManager):
        assert "not found" in skill_mgr.disable_skill("nope")

    def test_enable_already_enabled(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        assert "already enabled" in skill_mgr.enable_skill("test_skill")

    def test_disable_already_disabled(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        skill_mgr.disable_skill("test_skill")
        assert "already disabled" in skill_mgr.disable_skill("test_skill")

    def test_is_enabled_nonexistent(self, skill_mgr: SkillManager):
        assert skill_mgr.is_enabled("nope") is False

    def test_disable_excludes_from_tool_definitions(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        assert any(t["name"] == "test_skill" for t in skill_mgr.get_tool_definitions())
        skill_mgr.disable_skill("test_skill")
        assert not any(t["name"] == "test_skill" for t in skill_mgr.get_tool_definitions())


class TestSkillManagerExecution:
    """Tests for skill execution."""

    async def test_execute_not_found(self, skill_mgr: SkillManager):
        result = await skill_mgr.execute("nope", {})
        assert "not found" in result

    async def test_execute_disabled(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        skill_mgr.disable_skill("test_skill")
        result = await skill_mgr.execute("test_skill", {"msg": "hi"})
        assert "disabled" in result

    async def test_execute_success(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        result = await skill_mgr.execute("test_skill", {"msg": "hello"})
        assert result == "Got: hello"

    async def test_execute_tracks_stats(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        await skill_mgr.execute("test_skill", {"msg": "hi"})
        skill = skill_mgr._skills["test_skill"]
        assert skill.total_executions == 1
        assert skill.last_execution is not None
        assert skill.last_execution.wall_time_ms > 0
        assert skill.last_execution.output_chars > 0
        assert skill.last_execution.timestamp != ""

    async def test_execute_increments_total(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        await skill_mgr.execute("test_skill", {"msg": "1"})
        await skill_mgr.execute("test_skill", {"msg": "2"})
        assert skill_mgr._skills["test_skill"].total_executions == 2

    async def test_execute_non_string_result_cast(self, skill_mgr: SkillManager):
        code = '''
SKILL_DEFINITION = {
    "name": "int_skill",
    "description": "Returns int",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    return 42
'''
        skill_mgr.create_skill("int_skill", code)
        result = await skill_mgr.execute("int_skill", {})
        assert result == "42"

    async def test_execute_output_truncation(self, skill_mgr: SkillManager):
        code = f'''
SKILL_DEFINITION = {{
    "name": "big_output",
    "description": "Big output",
    "input_schema": {{"type": "object", "properties": {{}}}},
}}

async def execute(inp, context):
    return "x" * {MAX_SKILL_OUTPUT_CHARS + 1000}
'''
        skill_mgr.create_skill("big_output", code)
        result = await skill_mgr.execute("big_output", {})
        assert len(result) <= MAX_SKILL_OUTPUT_CHARS + 100  # truncation message
        assert "truncated" in result
        stats = skill_mgr._skills["big_output"].last_execution
        assert stats.truncated is True

    async def test_execute_exception(self, skill_mgr: SkillManager):
        code = '''
SKILL_DEFINITION = {
    "name": "err_skill",
    "description": "Raises",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    raise ValueError("boom")
'''
        skill_mgr.create_skill("err_skill", code)
        result = await skill_mgr.execute("err_skill", {})
        assert "Skill error" in result
        assert "boom" in result

    async def test_execute_timeout(self, skill_mgr: SkillManager):
        code = '''
import asyncio
SKILL_DEFINITION = {
    "name": "slow_skill",
    "description": "Slow",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    await asyncio.sleep(9999)
    return "done"
'''
        skill_mgr.create_skill("slow_skill", code)
        with patch("src.tools.skill_manager.SKILL_EXECUTE_TIMEOUT", 0.01):
            result = await skill_mgr.execute("slow_skill", {})
            assert "timed out" in result

    async def test_execute_with_config(self, skill_mgr: SkillManager):
        code = SKILL_WITH_METADATA.replace("meta_skill", "cfg_skill2")
        skill_mgr.create_skill("cfg_skill2", code)
        skill_mgr.set_skill_config("cfg_skill2", {"greeting": "yo"})
        result = await skill_mgr.execute("cfg_skill2", {})
        assert "greeting=yo" in result
        assert "count=5" in result  # default from schema


class TestSkillManagerConfig:
    """Tests for skill configuration management."""

    def test_get_config_no_skill(self, skill_mgr: SkillManager):
        assert skill_mgr.get_skill_config("nope") == {}

    def test_set_config_no_skill(self, skill_mgr: SkillManager):
        errors = skill_mgr.set_skill_config("nope", {"x": 1})
        assert any("not found" in e for e in errors)

    def test_set_config_validation(self, skill_mgr: SkillManager):
        code = SKILL_WITH_METADATA.replace("meta_skill", "cfg_test")
        skill_mgr.create_skill("cfg_test", code)
        errors = skill_mgr.set_skill_config("cfg_test", {"greeting": 42})  # wrong type
        assert len(errors) > 0

    def test_set_config_no_schema(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        errors = skill_mgr.set_skill_config("test_skill", {"any": "value"})
        assert errors == []  # No schema → no validation

    def test_config_persistence(self, skill_mgr: SkillManager):
        code = SKILL_WITH_METADATA.replace("meta_skill", "persist_test")
        skill_mgr.create_skill("persist_test", code)
        skill_mgr.set_skill_config("persist_test", {"greeting": "saved"})
        config = skill_mgr.get_skill_config("persist_test")
        assert config["greeting"] == "saved"
        assert config["count"] == 5  # default applied

    def test_load_config_file_corrupt(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        config_path = skill_mgr._config_dir / "test_skill.json"
        config_path.write_text("{{not json")
        assert skill_mgr._load_config_file("test_skill") == {}


class TestSkillManagerValidation:
    """Tests for validate_skill_code()."""

    def test_valid_code(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(VALID_SKILL_CODE)
        assert report["valid"] is True
        assert report["errors"] == []

    def test_syntax_error(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code("def broken(:")
        assert report["valid"] is False
        assert any("Syntax error" in e for e in report["errors"])

    def test_execution_error(self, skill_mgr: SkillManager):
        code = 'raise RuntimeError("crash")\nSKILL_DEFINITION = {}'
        report = skill_mgr.validate_skill_code(code)
        assert report["valid"] is False
        assert any("Execution error" in e for e in report["errors"])

    def test_missing_definition(self, skill_mgr: SkillManager):
        code = 'async def execute(inp, ctx): return "ok"'
        report = skill_mgr.validate_skill_code(code)
        assert report["valid"] is False
        assert any("SKILL_DEFINITION" in e for e in report["errors"])

    def test_missing_keys(self, skill_mgr: SkillManager):
        code = '''
SKILL_DEFINITION = {"name": "x"}
async def execute(inp, ctx): return "ok"
'''
        report = skill_mgr.validate_skill_code(code)
        assert report["valid"] is False
        assert any("description" in e for e in report["errors"])

    def test_sync_execute_warning(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(SYNC_EXECUTE_SKILL)
        assert report["valid"] is True
        assert any("not async" in w for w in report["warnings"])

    def test_missing_execute(self, skill_mgr: SkillManager):
        code = '''
SKILL_DEFINITION = {
    "name": "no_exec",
    "description": "No execute",
    "input_schema": {"type": "object", "properties": {}},
}
'''
        report = skill_mgr.validate_skill_code(code)
        assert report["valid"] is False
        assert any("Missing execute" in e for e in report["errors"])

    def test_name_validation_in_validate(self, skill_mgr: SkillManager):
        code = '''
SKILL_DEFINITION = {
    "name": "BadName",
    "description": "Bad name",
    "input_schema": {"type": "object", "properties": {}},
}
async def execute(inp, ctx): return "ok"
'''
        report = skill_mgr.validate_skill_code(code)
        assert report["valid"] is False
        assert any("Invalid skill name" in e for e in report["errors"])

    def test_non_string_name(self, skill_mgr: SkillManager):
        code = '''
SKILL_DEFINITION = {
    "name": 42,
    "description": "num name",
    "input_schema": {"type": "object", "properties": {}},
}
async def execute(inp, ctx): return "ok"
'''
        report = skill_mgr.validate_skill_code(code)
        assert report["valid"] is False
        assert any("must be a string" in e for e in report["errors"])

    def test_metadata_in_report(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(SKILL_WITH_METADATA.replace("meta_skill", "val_meta"))
        assert report["valid"] is True
        assert report["metadata"] is not None
        assert report["metadata"]["version"] == "2.1.0"
        assert report["metadata"]["author"] == "Test Author"
        assert "definition_keys" in report
        assert "name" in report["definition_keys"]


class TestSkillManagerInfo:
    """Tests for get_skill_info() and list_skills()."""

    def test_info_nonexistent(self, skill_mgr: SkillManager):
        assert skill_mgr.get_skill_info("nope") is None

    def test_info_complete(self, skill_mgr: SkillManager):
        code = SKILL_WITH_METADATA.replace("meta_skill", "info_skill")
        skill_mgr.create_skill("info_skill", code)
        info = skill_mgr.get_skill_info("info_skill")
        assert info is not None
        assert info["name"] == "info_skill"
        assert info["description"] == "Skill with full metadata"
        assert info["status"] == "loaded"
        assert info["metadata"]["version"] == "2.1.0"
        assert info["metadata"]["author"] == "Test Author"
        assert info["metadata"]["homepage"] == "https://example.com"
        assert info["metadata"]["tags"] == ["test", "demo"]
        assert info["metadata"]["has_config"] is True
        assert "code" in info
        assert info["handoff_to_codex"] is True
        assert info["total_executions"] == 0
        assert info["last_execution"] is None

    def test_list_skills_empty(self, skill_mgr: SkillManager):
        assert skill_mgr.list_skills() == []

    def test_list_skills_with_entries(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        skills = skill_mgr.list_skills()
        assert len(skills) == 1
        s = skills[0]
        assert s["name"] == "test_skill"
        assert s["status"] == "loaded"
        assert s["version"] == "0.0.0"
        assert s["total_executions"] == 0
        assert s["last_execution"] is None

    async def test_list_skills_with_execution_stats(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        await skill_mgr.execute("test_skill", {"msg": "hi"})
        skills = skill_mgr.list_skills()
        assert skills[0]["total_executions"] == 1
        assert skills[0]["last_execution"] is not None
        assert skills[0]["last_execution"]["wall_time_ms"] >= 0


class TestSkillManagerMisc:
    """Tests for miscellaneous SkillManager methods."""

    def test_has_skill(self, skill_mgr: SkillManager):
        assert skill_mgr.has_skill("nope") is False
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        assert skill_mgr.has_skill("test_skill") is True

    def test_should_handoff_to_codex(self, skill_mgr: SkillManager):
        assert skill_mgr.should_handoff_to_codex("nope") is False
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        assert skill_mgr.should_handoff_to_codex("test_skill") is False

    def test_should_handoff_to_codex_true(self, skill_mgr: SkillManager):
        code = SKILL_WITH_METADATA.replace("meta_skill", "ho_skill")
        skill_mgr.create_skill("ho_skill", code)
        assert skill_mgr.should_handoff_to_codex("ho_skill") is True

    def test_check_dependencies_not_found(self, skill_mgr: SkillManager):
        result = skill_mgr.check_dependencies("nope")
        assert "error" in result

    def test_check_dependencies_none(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        result = skill_mgr.check_dependencies("test_skill")
        assert result["all_satisfied"] is True
        assert result["dependencies"] == []

    def test_check_dependencies_with_deps(self, skill_mgr: SkillManager):
        code = SKILL_WITH_DEPS.replace("dep_skill", "dep_check")
        with patch("src.tools.skill_manager._is_package_installed", return_value=True):
            with patch("src.tools.skill_manager._install_packages", return_value=(True, "")):
                skill_mgr.create_skill("dep_check", code)
        result = skill_mgr.check_dependencies("dep_check")
        assert len(result["dependencies"]) == 1
        assert result["dependencies"][0]["spec"] == "requests>=2.0"

    def test_skill_status_not_found(self, skill_mgr: SkillManager):
        assert "not found" in skill_mgr.skill_status("nope")

    def test_skill_status_basic(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        status = skill_mgr.skill_status("test_skill")
        assert "**Skill: test_skill**" in status
        assert "Status: loaded" in status
        assert "Total executions: 0" in status

    def test_skill_status_with_metadata(self, skill_mgr: SkillManager):
        code = SKILL_WITH_METADATA.replace("meta_skill", "st_skill")
        skill_mgr.create_skill("st_skill", code)
        status = skill_mgr.skill_status("st_skill")
        assert "Author: Test Author" in status
        assert "Homepage: https://example.com" in status
        assert "Tags: test, demo" in status
        assert "Config:" in status

    async def test_skill_status_with_execution(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        await skill_mgr.execute("test_skill", {"msg": "hi"})
        status = skill_mgr.skill_status("test_skill")
        assert "Last execution:" in status
        assert "Total executions: 1" in status

    def test_get_tool_definitions_empty(self, skill_mgr: SkillManager):
        assert skill_mgr.get_tool_definitions() == []

    def test_get_tool_definitions(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        defs = skill_mgr.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "test_skill"
        assert "description" in defs[0]
        assert "input_schema" in defs[0]


class TestSkillManagerExport:
    """Tests for export_skill()."""

    def test_export_not_found(self, skill_mgr: SkillManager):
        result = skill_mgr.export_skill("nope")
        assert isinstance(result, str)
        assert "not found" in result

    def test_export_success(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        result = skill_mgr.export_skill("test_skill")
        assert isinstance(result, tuple)
        data, filename = result
        assert filename == "test_skill.py"
        assert b"SKILL_DEFINITION" in data


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


class TestSkillManagerInstallFromUrl:
    """Tests for install_from_url()."""

    async def test_bad_scheme(self, skill_mgr: SkillManager):
        result = await skill_mgr.install_from_url("ftp://example.com/skill.py")
        assert "Invalid URL scheme" in result

    async def test_no_host(self, skill_mgr: SkillManager):
        result = await skill_mgr.install_from_url("https://")
        assert "no host" in result

    async def test_download_timeout(self, skill_mgr: SkillManager):
        async def raise_timeout(*args, **kwargs):
            raise asyncio.TimeoutError()

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__ = raise_timeout
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "timed out" in result.lower() or "error" in result.lower()

    async def test_download_unicode_error(self, skill_mgr: SkillManager):
        resp = _mock_response(status=200, content=b"\xff\xfe\x00\x01")
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "UTF-8" in result

    async def test_download_http_error(self, skill_mgr: SkillManager):
        resp = _mock_response(status=404)
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "HTTP 404" in result

    async def test_download_generic_error(self, skill_mgr: SkillManager):
        async def raise_error(*args, **kwargs):
            raise OSError("Connection refused")

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__ = raise_error
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "error" in result.lower()

    async def test_file_too_large_via_header(self, skill_mgr: SkillManager):
        resp = _mock_response(
            status=200, content=b"x",
            content_length=MAX_SKILL_DOWNLOAD_BYTES + 1,
        )
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "too large" in result.lower()

    async def test_invalid_code_rejected(self, skill_mgr: SkillManager):
        bad_code = b"async def execute(inp, ctx): return 'ok'"
        resp = _mock_response(status=200, content=bad_code)
        with patch("aiohttp.ClientSession", _mock_session(resp)):
            result = await skill_mgr.install_from_url("https://example.com/skill.py")
        assert "Invalid skill code" in result


class TestSkillManagerUnload:
    """Tests for _unload_skill()."""

    def test_unload_removes_from_skills(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        assert skill_mgr.has_skill("test_skill")
        skill_mgr._unload_skill("test_skill")
        assert not skill_mgr.has_skill("test_skill")

    def test_unload_removes_from_sys_modules(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("test_skill", VALID_SKILL_CODE)
        mod_name = skill_mgr._skills["test_skill"].module_name
        assert mod_name in sys.modules
        skill_mgr._unload_skill("test_skill")
        assert mod_name not in sys.modules

    def test_unload_nonexistent_no_error(self, skill_mgr: SkillManager):
        # Should not raise
        skill_mgr._unload_skill("doesnt_exist")


class TestSkillManagerSetServices:
    """Tests for set_services()."""

    def test_set_services(self, skill_mgr: SkillManager):
        ks = MagicMock()
        emb = MagicMock()
        sm = MagicMock()
        sched = MagicMock()
        skill_mgr.set_services(
            knowledge_store=ks,
            embedder=emb,
            session_manager=sm,
            scheduler=sched,
        )
        assert skill_mgr._knowledge_store is ks
        assert skill_mgr._embedder is emb
        assert skill_mgr._session_manager is sm
        assert skill_mgr._scheduler is sched

    def test_set_services_partial(self, skill_mgr: SkillManager):
        skill_mgr.set_services(knowledge_store=MagicMock())
        assert skill_mgr._knowledge_store is not None
        assert skill_mgr._embedder is None


# ===========================================================================
# Part 5: SkillContext tests
# ===========================================================================


@pytest.fixture
def skill_context(tools_config: ToolsConfig, tmp_dir: Path) -> SkillContext:
    executor = ToolExecutor(tools_config)
    tracker = ResourceTracker()
    return SkillContext(
        executor, "test_ctx",
        memory_path=str(tmp_dir / "ctx_memory.json"),
        message_callback=AsyncMock(),
        file_callback=AsyncMock(),
        resource_tracker=tracker,
    )


class TestSkillContextExecuteTool:
    """Tests for SkillContext.execute_tool()."""

    async def test_blocked_tool(self, skill_context: SkillContext):
        result = await skill_context.execute_tool("run_command")
        assert "not allowed" in result

    async def test_empty_tool_name(self, skill_context: SkillContext):
        result = await skill_context.execute_tool("")
        assert "not allowed" in result

    async def test_tool_call_limit(self, skill_context: SkillContext):
        skill_context._tracker.tool_calls = MAX_SKILL_TOOL_CALLS
        result = await skill_context.execute_tool("search_knowledge")
        assert "limit" in result.lower()

    async def test_safe_tool_increments_tracker(self, skill_context: SkillContext):
        with patch.object(skill_context._executor, "execute", new_callable=AsyncMock, return_value="ok"):
            result = await skill_context.execute_tool("search_knowledge", {"query": "test"})
            assert result == "ok"
            assert skill_context._tracker.tool_calls == 1

    async def test_read_file_path_denied(self, skill_context: SkillContext):
        result = await skill_context.execute_tool("read_file", {"path": "/etc/shadow"})
        assert "Access denied" in result

    async def test_read_file_path_allowed(self, skill_context: SkillContext):
        with patch.object(skill_context._executor, "execute", new_callable=AsyncMock, return_value="content"):
            result = await skill_context.execute_tool("read_file", {"path": "/var/log/syslog"})
            assert result == "content"


class TestSkillContextPostMessage:
    """Tests for SkillContext.post_message()."""

    async def test_post_message_success(self, skill_context: SkillContext):
        await skill_context.post_message("hello")
        skill_context._message_callback.assert_called_once_with("hello")
        assert skill_context._tracker.messages_sent == 1

    async def test_post_message_limit(self, skill_context: SkillContext):
        skill_context._tracker.messages_sent = MAX_SKILL_MESSAGES
        await skill_context.post_message("blocked")
        skill_context._message_callback.assert_not_called()

    async def test_post_message_no_callback(self, tools_config: ToolsConfig):
        ctx = SkillContext(ToolExecutor(tools_config), "test")
        # Should not raise
        await ctx.post_message("test")


class TestSkillContextPostFile:
    """Tests for SkillContext.post_file()."""

    async def test_post_file_success(self, skill_context: SkillContext):
        await skill_context.post_file(b"data", "test.txt", "caption")
        skill_context._file_callback.assert_called_once_with(b"data", "test.txt", "caption")
        assert skill_context._tracker.files_sent == 1

    async def test_post_file_limit(self, skill_context: SkillContext):
        skill_context._tracker.files_sent = MAX_SKILL_FILES
        await skill_context.post_file(b"data", "test.txt")
        skill_context._file_callback.assert_not_called()

    async def test_post_file_no_callback(self, tools_config: ToolsConfig):
        ctx = SkillContext(ToolExecutor(tools_config), "test")
        await ctx.post_file(b"data", "test.txt")


class TestSkillContextMemory:
    """Tests for SkillContext memory operations."""

    def test_remember_and_recall(self, skill_context: SkillContext):
        skill_context.remember("key1", "value1")
        assert skill_context.recall("key1") == "value1"

    def test_recall_missing_key(self, skill_context: SkillContext):
        assert skill_context.recall("nonexistent") is None

    def test_remember_overwrites(self, skill_context: SkillContext):
        skill_context.remember("key", "old")
        skill_context.remember("key", "new")
        assert skill_context.recall("key") == "new"

    def test_no_memory_path(self, tools_config: ToolsConfig):
        ctx = SkillContext(ToolExecutor(tools_config), "test")
        ctx.remember("key", "val")  # should not raise
        assert ctx.recall("key") is None

    def test_memory_special_chars(self, skill_context: SkillContext):
        skill_context.remember("key", 'value with "quotes" and\nnewlines')
        result = skill_context.recall("key")
        assert '"quotes"' in result
        assert "\n" in result

    def test_memory_persistence(self, skill_context: SkillContext):
        skill_context.remember("persist", "data")
        # Load memory from file directly
        data = json.loads(skill_context._memory_path.read_text())
        assert data["persist"] == "data"

    def test_corrupt_memory_file(self, skill_context: SkillContext):
        skill_context._memory_path.write_text("{{broken json")
        assert skill_context.recall("key") is None


class TestSkillContextHosts:
    """Tests for SkillContext host/service info."""

    def test_get_hosts(self, skill_context: SkillContext):
        hosts = skill_context.get_hosts()
        assert "server" in hosts
        assert "desktop" in hosts

    def test_get_services(self, skill_context: SkillContext):
        services = skill_context.get_services()
        # Systemd tools removed — get_services() returns empty list
        assert services == []


class TestSkillContextConfig:
    """Tests for SkillContext config access."""

    def test_get_config_default(self, skill_context: SkillContext):
        assert skill_context.get_config("missing") is None
        assert skill_context.get_config("missing", "fallback") == "fallback"

    def test_get_config_set(self):
        executor = MagicMock()
        ctx = SkillContext(executor, "test", skill_config={"key": "value"})
        assert ctx.get_config("key") == "value"

    def test_get_all_config(self):
        executor = MagicMock()
        config = {"a": 1, "b": "two"}
        ctx = SkillContext(executor, "test", skill_config=config)
        result = ctx.get_all_config()
        assert result == {"a": 1, "b": "two"}
        # Should be a copy
        result["c"] = 3
        assert "c" not in ctx.get_all_config()


class TestSkillContextKnowledge:
    """Tests for knowledge base integration."""

    async def test_search_knowledge_no_store(self, skill_context: SkillContext):
        result = await skill_context.search_knowledge("test")
        assert result == []

    async def test_search_knowledge_with_store(self):
        ks = AsyncMock()
        ks.search_hybrid = AsyncMock(return_value=[{"content": "found", "score": 0.9}])
        emb = MagicMock()
        executor = MagicMock()
        ctx = SkillContext(executor, "test", knowledge_store=ks, embedder=emb)
        result = await ctx.search_knowledge("query")
        assert len(result) == 1
        ks.search_hybrid.assert_called_once_with("query", emb, limit=5)

    async def test_ingest_document_no_store(self, skill_context: SkillContext):
        result = await skill_context.ingest_document("content", "source")
        assert result == 0

    async def test_ingest_document_with_store(self):
        ks = AsyncMock()
        ks.ingest = AsyncMock(return_value=3)
        emb = MagicMock()
        executor = MagicMock()
        ctx = SkillContext(executor, "test", knowledge_store=ks, embedder=emb)
        result = await ctx.ingest_document("content", "source")
        assert result == 3


class TestSkillContextHistory:
    """Tests for session history search."""

    async def test_search_history_no_manager(self, skill_context: SkillContext):
        result = await skill_context.search_history("test")
        assert result == []

    async def test_search_history_with_manager(self):
        sm = AsyncMock()
        sm.search_history = AsyncMock(return_value=[{"content": "msg", "timestamp": "now"}])
        executor = MagicMock()
        ctx = SkillContext(executor, "test", session_manager=sm)
        result = await ctx.search_history("test", limit=3)
        assert len(result) == 1
        sm.search_history.assert_called_once_with("test", limit=3)


class TestSkillContextScheduler:
    """Tests for scheduler integration."""

    def test_schedule_task_no_scheduler(self, skill_context: SkillContext):
        result = skill_context.schedule_task("desc", "action", "ch1")
        assert result is None

    def test_schedule_task_with_scheduler(self):
        sched = MagicMock()
        sched.add.return_value = {"id": "s1", "description": "test"}
        executor = MagicMock()
        ctx = SkillContext(executor, "test", scheduler=sched)
        result = ctx.schedule_task("desc", "action", "ch1", cron="* * * * *")
        assert result["id"] == "s1"
        sched.add.assert_called_once_with("desc", "action", "ch1", cron="* * * * *")

    def test_list_schedules_no_scheduler(self, skill_context: SkillContext):
        assert skill_context.list_schedules() == []

    def test_delete_schedule_no_scheduler(self, skill_context: SkillContext):
        assert skill_context.delete_schedule("id1") is False


class TestSkillContextLog:
    """Tests for SkillContext.log()."""

    def test_log(self, skill_context: SkillContext):
        # Should not raise
        skill_context.log("test message")


class TestSkillContextRunOnHost:
    """Tests for SkillContext.run_on_host()."""

    async def test_run_on_host(self, skill_context: SkillContext):
        with patch.object(
            skill_context._executor, "_run_on_host",
            new_callable=AsyncMock, return_value="output"
        ):
            result = await skill_context.run_on_host("server", "ls")
            assert result == "output"


class TestSkillContextQueryPrometheus:
    """Tests for SkillContext.query_prometheus()."""

    async def test_query_prometheus(self, skill_context: SkillContext):
        with patch.object(
            skill_context._executor, "execute",
            new_callable=AsyncMock, return_value='{"status":"success"}'
        ):
            result = await skill_context.query_prometheus("up")
            assert "success" in result


class TestSkillContextReadFile:
    """Tests for SkillContext.read_file()."""

    async def test_read_file_denied(self, skill_context: SkillContext):
        result = await skill_context.read_file("server", ".env")
        assert "Access denied" in result

    async def test_read_file_allowed(self, skill_context: SkillContext):
        with patch.object(
            skill_context._executor, "execute",
            new_callable=AsyncMock, return_value="file content"
        ):
            result = await skill_context.read_file("server", "/var/log/app.log")
            assert result == "file content"


# ===========================================================================
# Part 6: Path denial and URL blocking edge cases
# ===========================================================================


class TestIsPathDenied:
    """Edge cases for is_path_denied()."""

    def test_env_file(self):
        assert is_path_denied(".env") is True

    def test_env_local(self):
        assert is_path_denied(".env.local") is True

    def test_nested_env(self):
        assert is_path_denied("/app/.env") is True

    def test_config_yml(self):
        assert is_path_denied("config.yml") is True

    def test_config_yaml(self):
        assert is_path_denied("config.yaml") is True

    def test_nested_config(self):
        assert is_path_denied("/opt/app/config.yml") is True

    def test_etc_shadow(self):
        assert is_path_denied("/etc/shadow") is True

    def test_ssh_keys(self):
        assert is_path_denied("id_rsa") is True
        assert is_path_denied("id_ed25519") is True
        assert is_path_denied("id_ecdsa") is True
        assert is_path_denied("id_dsa") is True

    def test_ssh_dir(self):
        assert is_path_denied("/home/user/.ssh/authorized_keys") is True
        assert is_path_denied(".ssh/config") is True

    def test_credentials_json(self):
        assert is_path_denied("credentials.json") is True
        assert is_path_denied("/path/to/credentials.json") is True

    def test_kube_config(self):
        assert is_path_denied("/home/user/.kube/config") is True

    def test_allowed_paths(self):
        assert is_path_denied("/var/log/syslog") is False
        assert is_path_denied("/etc/hostname") is False
        assert is_path_denied("/app/main.py") is False
        assert is_path_denied("README.md") is False

    def test_partial_match_not_denied(self):
        # "config.yml.bak" should still match (contains config.yml at end)
        # Actually, the regex is config\.ya?ml$ so .bak would not match
        assert is_path_denied("config.yml.bak") is False

    def test_env_in_name_not_denied(self):
        # "environment.py" should not be denied
        assert is_path_denied("environment.py") is False


class TestIsUrlBlocked:
    """Edge cases for is_url_blocked()."""

    def test_localhost(self):
        assert is_url_blocked("http://localhost/api") is True

    def test_ipv4_loopback(self):
        assert is_url_blocked("http://127.0.0.1/") is True

    def test_ipv6_loopback(self):
        assert is_url_blocked("http://[::1]/") is True

    def test_zero_address(self):
        assert is_url_blocked("http://0.0.0.0/") is True

    def test_cloud_metadata(self):
        assert is_url_blocked("http://169.254.169.254/latest/meta-data/") is True
        assert is_url_blocked("http://metadata.google.internal/") is True

    def test_private_ip_10(self):
        assert is_url_blocked("http://10.0.0.1/") is True

    def test_private_ip_172(self):
        assert is_url_blocked("http://172.16.0.1/") is True

    def test_link_local(self):
        assert is_url_blocked("http://169.254.0.1/") is True

    def test_public_url_allowed(self):
        assert is_url_blocked("https://api.example.com/data") is False

    def test_malformed_url(self):
        assert is_url_blocked("not a url at all") is True

    def test_empty_host(self):
        assert is_url_blocked("http:///path") is True

    def test_hostname_not_ip(self):
        # Regular hostnames should be allowed
        assert is_url_blocked("https://myserver.internal.corp/api") is False

    def test_url_with_port(self):
        assert is_url_blocked("http://localhost:8080/") is True

    def test_url_with_auth(self):
        assert is_url_blocked("http://user:pass@localhost/") is True


# ===========================================================================
# Part 7: SKILL_SAFE_TOOLS validation
# ===========================================================================


class TestSafeToolsAllowlist:
    """Verify SKILL_SAFE_TOOLS contains expected tools and excludes dangerous ones."""

    def test_safe_tools_is_frozenset(self):
        assert isinstance(SKILL_SAFE_TOOLS, frozenset)

    def test_read_only_tools_present(self):
        expected = {
            "read_file", "search_knowledge", "list_skills",
            "web_search", "fetch_url", "browser_screenshot",
        }
        for tool in expected:
            assert tool in SKILL_SAFE_TOOLS, f"{tool} should be in SKILL_SAFE_TOOLS"

    def test_dangerous_tools_excluded(self):
        dangerous = {
            "run_command", "run_script", "write_file",
            "delete_file", "manage_skill", "manage_service",
        }
        for tool in dangerous:
            assert tool not in SKILL_SAFE_TOOLS, f"{tool} should NOT be in SKILL_SAFE_TOOLS"


# ===========================================================================
# Part 8: Integration tests
# ===========================================================================


class TestSkillExecutionWithContext:
    """Integration tests for skill execution with full context wiring."""

    async def test_skill_uses_memory(self, skill_mgr_with_memory: SkillManager):
        code = '''
SKILL_DEFINITION = {
    "name": "mem_skill",
    "description": "Uses memory",
    "input_schema": {"type": "object", "properties": {"action": {"type": "string"}}},
}

async def execute(inp, context):
    action = inp.get("action", "read")
    if action == "write":
        context.remember("test_key", "test_value")
        return "written"
    else:
        val = context.recall("test_key")
        return f"recalled: {val}"
'''
        mgr = skill_mgr_with_memory
        mgr.create_skill("mem_skill", code)
        # Write
        result = await mgr.execute("mem_skill", {"action": "write"})
        assert result == "written"
        # Read back
        result = await mgr.execute("mem_skill", {"action": "read"})
        assert result == "recalled: test_value"

    async def test_skill_posts_message(self, skill_mgr: SkillManager):
        code = '''
SKILL_DEFINITION = {
    "name": "msg_skill",
    "description": "Posts messages",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    await context.post_message("hello from skill")
    return "sent"
'''
        skill_mgr.create_skill("msg_skill", code)
        callback = AsyncMock()
        result = await skill_mgr.execute("msg_skill", {}, message_callback=callback)
        assert result == "sent"
        callback.assert_called_once_with("hello from skill")

    async def test_skill_posts_file(self, skill_mgr: SkillManager):
        code = '''
SKILL_DEFINITION = {
    "name": "file_skill",
    "description": "Posts files",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    await context.post_file(b"file data", "output.txt", "here you go")
    return "file sent"
'''
        skill_mgr.create_skill("file_skill", code)
        file_cb = AsyncMock()
        result = await skill_mgr.execute("file_skill", {}, file_callback=file_cb)
        assert result == "file sent"
        file_cb.assert_called_once_with(b"file data", "output.txt", "here you go")

    async def test_skill_resource_tracking(self, skill_mgr: SkillManager):
        code = '''
SKILL_DEFINITION = {
    "name": "tracked_skill",
    "description": "Resource tracked",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    await context.post_message("msg1")
    await context.post_message("msg2")
    return "done"
'''
        skill_mgr.create_skill("tracked_skill", code)
        msg_cb = AsyncMock()
        result = await skill_mgr.execute("tracked_skill", {}, message_callback=msg_cb)
        assert result == "done"
        stats = skill_mgr._skills["tracked_skill"].last_execution
        assert stats.messages_sent == 2

    async def test_skill_config_during_execution(self, skill_mgr: SkillManager):
        code = '''
SKILL_DEFINITION = {
    "name": "cfg_exec",
    "description": "Config exec",
    "input_schema": {"type": "object", "properties": {}},
    "config_schema": {
        "type": "object",
        "properties": {
            "prefix": {"type": "string", "default": ">>"},
        },
    },
}

async def execute(inp, context):
    prefix = context.get_config("prefix", ">")
    return f"{prefix} hello"
'''
        skill_mgr.create_skill("cfg_exec", code)
        # Default config
        result = await skill_mgr.execute("cfg_exec", {})
        assert result == ">> hello"
        # Set custom config
        skill_mgr.set_skill_config("cfg_exec", {"prefix": "##"})
        result = await skill_mgr.execute("cfg_exec", {})
        assert result == "## hello"


class TestSkillNamePattern:
    """Tests for SKILL_NAME_PATTERN regex."""

    def test_max_length(self):
        name = "a" * 50
        assert SKILL_NAME_PATTERN.match(name)

    def test_over_max_length(self):
        name = "a" * 51
        assert not SKILL_NAME_PATTERN.match(name)

    def test_min_length(self):
        assert SKILL_NAME_PATTERN.match("a")

    def test_numbers_after_start(self):
        assert SKILL_NAME_PATTERN.match("a123")

    def test_underscores(self):
        assert SKILL_NAME_PATTERN.match("my_cool_skill")

    def test_all_numbers_after_first(self):
        assert SKILL_NAME_PATTERN.match("x0123456789")


class TestBuiltinToolNames:
    """Tests for BUILTIN_TOOL_NAMES set."""

    def test_is_set(self):
        assert isinstance(BUILTIN_TOOL_NAMES, set)

    def test_contains_core_tools(self):
        assert "run_command" in BUILTIN_TOOL_NAMES
        assert "read_file" in BUILTIN_TOOL_NAMES

    def test_skill_names_not_in_builtins(self):
        assert "my_custom_skill" not in BUILTIN_TOOL_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import subprocess

def subprocess_timeout():
    """Return a side_effect function that raises subprocess.TimeoutExpired."""
    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="pip", timeout=1)
    return _raise
