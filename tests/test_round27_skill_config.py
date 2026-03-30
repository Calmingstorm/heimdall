"""Round 27: Skill metadata + configuration — config validation, storage, access, API."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.schema import ToolsConfig, ToolHost
from src.tools.executor import ToolExecutor
from src.tools.skill_manager import (
    SkillManager,
    SkillMetadata,
    validate_config,
    validate_config_value,
    apply_defaults,
    _CONFIG_FIELD_TYPES,
)
from src.tools.skill_context import SkillContext


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

CONFIG_SKILL = '''
SKILL_DEFINITION = {
    "name": "configurable",
    "description": "A skill with config",
    "input_schema": {"type": "object", "properties": {}},
    "version": "1.0.0",
    "config_schema": {
        "type": "object",
        "properties": {
            "threshold": {"type": "number", "default": 90, "description": "Alert threshold"},
            "label": {"type": "string", "default": "default", "description": "Display label"},
            "enabled": {"type": "boolean", "default": True, "description": "Enable feature"},
            "retries": {"type": "integer", "default": 3, "minimum": 0, "maximum": 10},
        },
        "required": ["threshold"],
    },
}

async def execute(inp, context):
    threshold = context.get_config("threshold")
    label = context.get_config("label", "fallback")
    return f"threshold={threshold}, label={label}"
'''

NO_CONFIG_SKILL = '''
SKILL_DEFINITION = {
    "name": "no_config",
    "description": "Skill without config",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    return "ok"
'''

ENUM_CONFIG_SKILL = '''
SKILL_DEFINITION = {
    "name": "enum_skill",
    "description": "Skill with enum config",
    "input_schema": {"type": "object", "properties": {}},
    "config_schema": {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["fast", "slow", "balanced"], "default": "balanced"},
        },
    },
}

async def execute(inp, context):
    return context.get_config("mode", "balanced")
'''


# ===========================================================================
# Test validate_config_value — individual field validation
# ===========================================================================

class TestValidateConfigValue:

    def test_string_valid(self):
        assert validate_config_value("f", {"type": "string"}, "hello") is None

    def test_string_invalid(self):
        err = validate_config_value("f", {"type": "string"}, 42)
        assert err and "expected string" in err

    def test_integer_valid(self):
        assert validate_config_value("f", {"type": "integer"}, 5) is None

    def test_integer_rejects_float(self):
        err = validate_config_value("f", {"type": "integer"}, 5.5)
        assert err and "expected integer" in err

    def test_integer_rejects_bool(self):
        err = validate_config_value("f", {"type": "integer"}, True)
        assert err and "expected integer" in err

    def test_number_valid_int(self):
        assert validate_config_value("f", {"type": "number"}, 5) is None

    def test_number_valid_float(self):
        assert validate_config_value("f", {"type": "number"}, 3.14) is None

    def test_number_rejects_string(self):
        err = validate_config_value("f", {"type": "number"}, "3")
        assert err and "expected number" in err

    def test_number_rejects_bool(self):
        err = validate_config_value("f", {"type": "number"}, False)
        assert err and "expected number" in err

    def test_boolean_valid(self):
        assert validate_config_value("f", {"type": "boolean"}, True) is None
        assert validate_config_value("f", {"type": "boolean"}, False) is None

    def test_boolean_rejects_int(self):
        err = validate_config_value("f", {"type": "boolean"}, 1)
        assert err and "expected boolean" in err

    def test_enum_valid(self):
        schema = {"type": "string", "enum": ["a", "b"]}
        assert validate_config_value("f", schema, "a") is None

    def test_enum_invalid(self):
        schema = {"type": "string", "enum": ["a", "b"]}
        err = validate_config_value("f", schema, "c")
        assert err and "not in allowed values" in err

    def test_minimum(self):
        schema = {"type": "integer", "minimum": 0}
        assert validate_config_value("f", schema, 0) is None
        err = validate_config_value("f", schema, -1)
        assert err and "below minimum" in err

    def test_maximum(self):
        schema = {"type": "integer", "maximum": 10}
        assert validate_config_value("f", schema, 10) is None
        err = validate_config_value("f", schema, 11)
        assert err and "exceeds maximum" in err

    def test_min_max_combined(self):
        schema = {"type": "number", "minimum": 0, "maximum": 100}
        assert validate_config_value("f", schema, 50.0) is None
        assert validate_config_value("f", schema, -1.0) is not None
        assert validate_config_value("f", schema, 101.0) is not None

    def test_minLength(self):
        schema = {"type": "string", "minLength": 3}
        assert validate_config_value("f", schema, "abc") is None
        err = validate_config_value("f", schema, "ab")
        assert err and "below minLength" in err

    def test_maxLength(self):
        schema = {"type": "string", "maxLength": 5}
        assert validate_config_value("f", schema, "hello") is None
        err = validate_config_value("f", schema, "toolong")
        assert err and "exceeds maxLength" in err

    def test_default_type_is_string(self):
        # No 'type' key — defaults to string
        assert validate_config_value("f", {}, "hello") is None

    def test_field_name_in_error(self):
        err = validate_config_value("my_field", {"type": "integer"}, "oops")
        assert "'my_field'" in err


# ===========================================================================
# Test validate_config — full config validation
# ===========================================================================

class TestValidateConfig:

    def test_valid_config(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
        }
        errors = validate_config(schema, {"name": "test", "count": 5})
        assert errors == []

    def test_missing_required(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        errors = validate_config(schema, {})
        assert any("Missing required field 'name'" in e for e in errors)

    def test_required_with_default_ok(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string", "default": "anon"}},
            "required": ["name"],
        }
        # Required but has default — not an error when missing
        errors = validate_config(schema, {})
        assert errors == []

    def test_unknown_field(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        errors = validate_config(schema, {"name": "ok", "extra": "bad"})
        assert any("Unknown field 'extra'" in e for e in errors)

    def test_type_error(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
        errors = validate_config(schema, {"count": "not-int"})
        assert len(errors) == 1
        assert "expected integer" in errors[0]

    def test_empty_schema(self):
        errors = validate_config({}, {"anything": "goes"})
        # No properties defined — any field is unknown
        assert any("Unknown field" in e for e in errors)

    def test_empty_values(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        errors = validate_config(schema, {})
        assert errors == []

    def test_multiple_errors(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "string"},
            },
            "required": ["a", "b"],
        }
        errors = validate_config(schema, {"a": "wrong"})
        assert len(errors) >= 2  # type error + missing required


# ===========================================================================
# Test apply_defaults
# ===========================================================================

class TestApplyDefaults:

    def test_fills_missing_defaults(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string", "default": "hello"},
                "b": {"type": "integer", "default": 42},
            },
        }
        result = apply_defaults(schema, {})
        assert result == {"a": "hello", "b": 42}

    def test_does_not_override_provided(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string", "default": "hello"},
            },
        }
        result = apply_defaults(schema, {"a": "custom"})
        assert result == {"a": "custom"}

    def test_mixed_provided_and_defaults(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string", "default": "def_a"},
                "b": {"type": "integer"},
                "c": {"type": "boolean", "default": False},
            },
        }
        result = apply_defaults(schema, {"b": 10})
        assert result == {"a": "def_a", "b": 10, "c": False}

    def test_empty_schema(self):
        result = apply_defaults({}, {"x": 1})
        assert result == {"x": 1}

    def test_no_defaults_in_schema(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
        }
        result = apply_defaults(schema, {})
        assert result == {}


# ===========================================================================
# Test _CONFIG_FIELD_TYPES constant
# ===========================================================================

class TestConfigFieldTypes:

    def test_supported_types(self):
        assert "string" in _CONFIG_FIELD_TYPES
        assert "integer" in _CONFIG_FIELD_TYPES
        assert "number" in _CONFIG_FIELD_TYPES
        assert "boolean" in _CONFIG_FIELD_TYPES

    def test_is_frozenset(self):
        assert isinstance(_CONFIG_FIELD_TYPES, frozenset)

    def test_four_types(self):
        assert len(_CONFIG_FIELD_TYPES) == 4


# ===========================================================================
# Test SkillManager config storage
# ===========================================================================

class TestSkillManagerConfigStorage:

    def test_config_dir_created(self, skill_mgr: SkillManager):
        assert (skill_mgr.skills_dir / "config").is_dir()

    def test_get_config_empty_for_unknown(self, skill_mgr: SkillManager):
        assert skill_mgr.get_skill_config("nonexistent") == {}

    def test_set_config_unknown_skill(self, skill_mgr: SkillManager):
        errors = skill_mgr.set_skill_config("nonexistent", {"x": 1})
        assert errors
        assert "not found" in errors[0]

    def test_set_and_get_config(self, skill_mgr: SkillManager):
        result = skill_mgr.create_skill("configurable", CONFIG_SKILL)
        assert "created" in result.lower()
        errors = skill_mgr.set_skill_config("configurable", {"threshold": 75})
        assert errors == []
        config = skill_mgr.get_skill_config("configurable")
        assert config["threshold"] == 75
        # Defaults filled
        assert config["label"] == "default"
        assert config["enabled"] is True
        assert config["retries"] == 3

    def test_set_config_validates(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("configurable", CONFIG_SKILL)
        errors = skill_mgr.set_skill_config("configurable", {"threshold": "not-a-number"})
        assert errors
        assert "expected number" in errors[0]

    def test_set_config_rejects_unknown_field(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("configurable", CONFIG_SKILL)
        errors = skill_mgr.set_skill_config("configurable", {"threshold": 50, "unknown": True})
        assert any("Unknown field" in e for e in errors)

    def test_config_persisted_to_file(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("configurable", CONFIG_SKILL)
        skill_mgr.set_skill_config("configurable", {"threshold": 80})
        config_path = skill_mgr._config_dir / "configurable.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data["threshold"] == 80

    def test_no_config_skill_accepts_empty(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("no_config", NO_CONFIG_SKILL)
        errors = skill_mgr.set_skill_config("no_config", {})
        assert errors == []

    def test_no_config_skill_stores_arbitrary_values(self, skill_mgr: SkillManager):
        # Skills without config_schema accept any values (no validation)
        skill_mgr.create_skill("no_config", NO_CONFIG_SKILL)
        errors = skill_mgr.set_skill_config("no_config", {"any": "value"})
        assert errors == []
        config = skill_mgr.get_skill_config("no_config")
        assert config["any"] == "value"

    def test_delete_removes_config(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("configurable", CONFIG_SKILL)
        skill_mgr.set_skill_config("configurable", {"threshold": 50})
        config_path = skill_mgr._config_dir / "configurable.json"
        assert config_path.exists()
        skill_mgr.delete_skill("configurable")
        assert not config_path.exists()

    def test_enum_validation(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("enum_skill", ENUM_CONFIG_SKILL)
        errors = skill_mgr.set_skill_config("enum_skill", {"mode": "fast"})
        assert errors == []
        errors = skill_mgr.set_skill_config("enum_skill", {"mode": "invalid"})
        assert any("not in allowed values" in e for e in errors)

    def test_range_validation(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("configurable", CONFIG_SKILL)
        errors = skill_mgr.set_skill_config("configurable", {"threshold": 50, "retries": -1})
        assert any("below minimum" in e for e in errors)
        errors = skill_mgr.set_skill_config("configurable", {"threshold": 50, "retries": 20})
        assert any("exceeds maximum" in e for e in errors)


# ===========================================================================
# Test SkillContext config access
# ===========================================================================

class TestSkillContextConfig:

    def _make_context(self, config: dict | None = None) -> SkillContext:
        executor = MagicMock()
        executor.config = MagicMock()
        executor.config.hosts = {}
        return SkillContext(
            executor, "test_skill",
            skill_config=config,
        )

    def test_get_config_returns_value(self):
        ctx = self._make_context({"threshold": 90})
        assert ctx.get_config("threshold") == 90

    def test_get_config_default(self):
        ctx = self._make_context({})
        assert ctx.get_config("missing", 42) == 42

    def test_get_config_none_default(self):
        ctx = self._make_context({})
        assert ctx.get_config("missing") is None

    def test_get_all_config(self):
        ctx = self._make_context({"a": 1, "b": "two"})
        all_config = ctx.get_all_config()
        assert all_config == {"a": 1, "b": "two"}

    def test_get_all_config_is_copy(self):
        config = {"a": 1}
        ctx = self._make_context(config)
        result = ctx.get_all_config()
        result["a"] = 999
        assert ctx.get_config("a") == 1  # Original unchanged

    def test_no_config_gives_empty(self):
        ctx = self._make_context()
        assert ctx.get_all_config() == {}

    def test_config_none_gives_empty(self):
        ctx = self._make_context(None)
        assert ctx.get_all_config() == {}


# ===========================================================================
# Test config integration with skill execution
# ===========================================================================

class TestConfigExecution:

    @pytest.fixture
    def exec_mgr(self, tmp_dir: Path, tools_config: ToolsConfig) -> SkillManager:
        executor = ToolExecutor(tools_config)
        skills_dir = tmp_dir / "skills"
        skills_dir.mkdir()
        mgr = SkillManager(str(skills_dir), executor)
        return mgr

    async def test_config_available_during_execute(self, exec_mgr: SkillManager):
        exec_mgr.create_skill("configurable", CONFIG_SKILL)
        exec_mgr.set_skill_config("configurable", {"threshold": 99, "label": "custom"})
        result = await exec_mgr.execute("configurable", {})
        assert "threshold=99" in result
        assert "label=custom" in result

    async def test_defaults_used_when_no_config_set(self, exec_mgr: SkillManager):
        exec_mgr.create_skill("configurable", CONFIG_SKILL)
        result = await exec_mgr.execute("configurable", {})
        assert "threshold=90" in result
        assert "label=default" in result

    async def test_partial_config_with_defaults(self, exec_mgr: SkillManager):
        exec_mgr.create_skill("configurable", CONFIG_SKILL)
        exec_mgr.set_skill_config("configurable", {"threshold": 50})
        result = await exec_mgr.execute("configurable", {})
        assert "threshold=50" in result
        assert "label=default" in result


# ===========================================================================
# Test get_skill_info includes config
# ===========================================================================

class TestSkillInfoConfig:

    def test_info_includes_config_key(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("configurable", CONFIG_SKILL)
        info = skill_mgr.get_skill_info("configurable")
        assert "config" in info

    def test_info_config_has_defaults(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("configurable", CONFIG_SKILL)
        info = skill_mgr.get_skill_info("configurable")
        assert info["config"]["threshold"] == 90
        assert info["config"]["label"] == "default"

    def test_info_config_reflects_set_values(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("configurable", CONFIG_SKILL)
        skill_mgr.set_skill_config("configurable", {"threshold": 42})
        info = skill_mgr.get_skill_info("configurable")
        assert info["config"]["threshold"] == 42

    def test_info_still_has_schema(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("configurable", CONFIG_SKILL)
        info = skill_mgr.get_skill_info("configurable")
        assert "config_schema" in info["metadata"]
        assert "threshold" in info["metadata"]["config_schema"]["properties"]


# ===========================================================================
# Test web API endpoints (mocked)
# ===========================================================================

class TestWebAPISkillConfig:

    def _make_mock_bot(self, skill_mgr: SkillManager) -> MagicMock:
        bot = MagicMock()
        bot.skill_manager = skill_mgr
        return bot

    async def test_get_config_endpoint_exists(self):
        """Verify the route is registered in the API setup code."""
        import src.web.api as api_mod
        source = Path(api_mod.__file__).read_text()
        assert "/api/skills/{name}/config" in source
        assert "get_skill_config" in source

    async def test_put_config_endpoint_exists(self):
        """Verify the PUT route is registered."""
        import src.web.api as api_mod
        source = Path(api_mod.__file__).read_text()
        assert "set_skill_config" in source

    async def test_get_config_route_method(self):
        import src.web.api as api_mod
        source = Path(api_mod.__file__).read_text()
        assert '@routes.get("/api/skills/{name}/config")' in source

    async def test_put_config_route_method(self):
        import src.web.api as api_mod
        source = Path(api_mod.__file__).read_text()
        assert '@routes.put("/api/skills/{name}/config")' in source


# ===========================================================================
# Test config schema in validate_skill_code
# ===========================================================================

class TestValidateSkillCodeConfig:

    def test_reports_config_schema(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(CONFIG_SKILL)
        assert report["valid"] is True
        assert report["metadata"]["has_config"] is True

    def test_no_config_reports_false(self, skill_mgr: SkillManager):
        report = skill_mgr.validate_skill_code(NO_CONFIG_SKILL)
        assert report["valid"] is True
        assert report["metadata"]["has_config"] is False


# ===========================================================================
# Test config file edge cases
# ===========================================================================

class TestConfigEdgeCases:

    def test_corrupt_config_file(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("configurable", CONFIG_SKILL)
        config_path = skill_mgr._config_dir / "configurable.json"
        config_path.write_text("not json!!!")
        config = skill_mgr.get_skill_config("configurable")
        # Should gracefully return defaults
        assert config["threshold"] == 90

    def test_config_overwrite(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("configurable", CONFIG_SKILL)
        skill_mgr.set_skill_config("configurable", {"threshold": 10})
        skill_mgr.set_skill_config("configurable", {"threshold": 20})
        config = skill_mgr.get_skill_config("configurable")
        assert config["threshold"] == 20

    def test_multiple_skills_independent_config(self, skill_mgr: SkillManager):
        skill_mgr.create_skill("configurable", CONFIG_SKILL)
        skill_mgr.create_skill("enum_skill", ENUM_CONFIG_SKILL)
        skill_mgr.set_skill_config("configurable", {"threshold": 50})
        skill_mgr.set_skill_config("enum_skill", {"mode": "fast"})
        assert skill_mgr.get_skill_config("configurable")["threshold"] == 50
        assert skill_mgr.get_skill_config("enum_skill")["mode"] == "fast"

    def test_config_path_helper(self, skill_mgr: SkillManager):
        path = skill_mgr._config_path("myskill")
        assert path.name == "myskill.json"
        assert path.parent == skill_mgr._config_dir


# ===========================================================================
# Source verification
# ===========================================================================

class TestSourceVerification:

    def test_skill_manager_has_validate_config(self):
        from src.tools.skill_manager import validate_config
        assert callable(validate_config)

    def test_skill_manager_has_validate_config_value(self):
        from src.tools.skill_manager import validate_config_value
        assert callable(validate_config_value)

    def test_skill_manager_has_apply_defaults(self):
        from src.tools.skill_manager import apply_defaults
        assert callable(apply_defaults)

    def test_skill_context_has_get_config(self):
        assert hasattr(SkillContext, "get_config")

    def test_skill_context_has_get_all_config(self):
        assert hasattr(SkillContext, "get_all_config")

    def test_skill_manager_has_get_skill_config(self):
        assert hasattr(SkillManager, "get_skill_config")

    def test_skill_manager_has_set_skill_config(self):
        assert hasattr(SkillManager, "set_skill_config")

    def test_config_field_types_constant(self):
        assert _CONFIG_FIELD_TYPES == {"string", "integer", "number", "boolean"}

    def test_json_import_in_manager(self):
        import src.tools.skill_manager as mod
        source = Path(mod.__file__).read_text()
        assert "import json" in source

    def test_skill_context_accepts_skill_config(self):
        import inspect
        sig = inspect.signature(SkillContext.__init__)
        assert "skill_config" in sig.parameters

    def test_execute_passes_config(self):
        import src.tools.skill_manager as mod
        source = Path(mod.__file__).read_text()
        assert "skill_config=skill_config" in source

    def test_get_skill_info_includes_config(self):
        import src.tools.skill_manager as mod
        source = Path(mod.__file__).read_text()
        assert '"config": self.get_skill_config(name)' in source
