from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from collections.abc import Callable
from typing import Any

from ..logging import get_logger
from .executor import ToolExecutor
from .registry import TOOLS
from .skill_context import SkillContext

log = get_logger("skills")

SKILL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,49}$")
BUILTIN_TOOL_NAMES = {t["name"] for t in TOOLS}
SKILL_EXECUTE_TIMEOUT = 120  # seconds

# Supported metadata keys in SKILL_DEFINITION (beyond the required ones).
_METADATA_KEYS = ("version", "author", "homepage", "tags", "dependencies", "config_schema")

# Valid semantic version pattern (loose — major.minor.patch with optional pre-release).
_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[-+].+)?$")

# Supported types for config_schema fields.
_CONFIG_FIELD_TYPES = frozenset({"string", "integer", "number", "boolean"})


def validate_config_value(field_name: str, field_schema: dict, value: Any) -> str | None:
    """Validate a single config value against its field schema.

    Returns an error string or None if valid.
    """
    ftype = field_schema.get("type", "string")

    # Type check
    if ftype == "string":
        if not isinstance(value, str):
            return f"'{field_name}': expected string, got {type(value).__name__}"
    elif ftype == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"'{field_name}': expected integer, got {type(value).__name__}"
    elif ftype == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"'{field_name}': expected number, got {type(value).__name__}"
    elif ftype == "boolean":
        if not isinstance(value, bool):
            return f"'{field_name}': expected boolean, got {type(value).__name__}"

    # Enum constraint
    enum = field_schema.get("enum")
    if enum is not None and isinstance(enum, list) and value not in enum:
        return f"'{field_name}': value {value!r} not in allowed values {enum}"

    # Numeric constraints
    if ftype in ("integer", "number") and isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = field_schema.get("minimum")
        if minimum is not None and value < minimum:
            return f"'{field_name}': value {value} is below minimum {minimum}"
        maximum = field_schema.get("maximum")
        if maximum is not None and value > maximum:
            return f"'{field_name}': value {value} exceeds maximum {maximum}"

    # String constraints
    if ftype == "string" and isinstance(value, str):
        min_len = field_schema.get("minLength")
        if min_len is not None and len(value) < min_len:
            return f"'{field_name}': length {len(value)} is below minLength {min_len}"
        max_len = field_schema.get("maxLength")
        if max_len is not None and len(value) > max_len:
            return f"'{field_name}': length {len(value)} exceeds maxLength {max_len}"

    return None


def validate_config(schema: dict, values: dict) -> list[str]:
    """Validate a full config dict against a config_schema.

    The schema follows JSON Schema 'object' format:
    {
      "type": "object",
      "properties": {
        "field_name": {"type": "string", "default": "...", "description": "..."},
        ...
      },
      "required": ["field_name"]
    }

    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    # Check required fields
    for req in required:
        if req not in values:
            # Only error if there's no default in the schema
            prop = properties.get(req, {})
            if "default" not in prop:
                errors.append(f"Missing required field '{req}'")

    # Validate each provided value
    for key, value in values.items():
        if key not in properties:
            errors.append(f"Unknown field '{key}'")
            continue
        err = validate_config_value(key, properties[key], value)
        if err:
            errors.append(err)

    return errors


def apply_defaults(schema: dict, values: dict) -> dict:
    """Return config values with defaults filled in from schema."""
    result = dict(values)
    properties = schema.get("properties", {})
    for key, prop in properties.items():
        if key not in result and "default" in prop:
            result[key] = prop["default"]
    return result


class SkillStatus(str, Enum):
    """Lifecycle state of a skill."""
    LOADED = "loaded"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class SkillDiagnostic:
    """A warning or error collected during skill load/validation."""
    level: str  # "warn" or "error"
    message: str


@dataclass
class SkillMetadata:
    """Rich metadata parsed from SKILL_DEFINITION."""
    version: str = "0.0.0"
    author: str = ""
    homepage: str = ""
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_definition(cls, definition: dict) -> tuple[SkillMetadata, list[SkillDiagnostic]]:
        """Parse metadata from a SKILL_DEFINITION dict, collecting diagnostics."""
        diagnostics: list[SkillDiagnostic] = []
        kwargs: dict[str, Any] = {}

        # version
        raw_version = definition.get("version", "0.0.0")
        if isinstance(raw_version, str):
            if raw_version and not _SEMVER_PATTERN.match(raw_version):
                diagnostics.append(SkillDiagnostic(
                    "warn", f"Invalid version '{raw_version}', expected semver (e.g. 1.0.0). Using 0.0.0.",
                ))
                kwargs["version"] = "0.0.0"
            else:
                kwargs["version"] = raw_version or "0.0.0"
        else:
            diagnostics.append(SkillDiagnostic("warn", "version must be a string. Using 0.0.0."))
            kwargs["version"] = "0.0.0"

        # author
        raw_author = definition.get("author", "")
        if isinstance(raw_author, str):
            kwargs["author"] = raw_author
        else:
            diagnostics.append(SkillDiagnostic("warn", "author must be a string. Ignored."))

        # homepage
        raw_homepage = definition.get("homepage", "")
        if isinstance(raw_homepage, str):
            kwargs["homepage"] = raw_homepage
        else:
            diagnostics.append(SkillDiagnostic("warn", "homepage must be a string. Ignored."))

        # tags
        raw_tags = definition.get("tags", [])
        if isinstance(raw_tags, list) and all(isinstance(t, str) for t in raw_tags):
            kwargs["tags"] = raw_tags
        elif raw_tags:
            diagnostics.append(SkillDiagnostic("warn", "tags must be a list of strings. Ignored."))

        # dependencies (pip packages)
        raw_deps = definition.get("dependencies", [])
        if isinstance(raw_deps, list) and all(isinstance(d, str) for d in raw_deps):
            kwargs["dependencies"] = raw_deps
        elif raw_deps:
            diagnostics.append(SkillDiagnostic("warn", "dependencies must be a list of strings. Ignored."))

        # config_schema (JSON Schema dict)
        raw_cs = definition.get("config_schema", {})
        if isinstance(raw_cs, dict):
            kwargs["config_schema"] = raw_cs
        elif raw_cs:
            diagnostics.append(SkillDiagnostic("warn", "config_schema must be a dict. Ignored."))

        return cls(**kwargs), diagnostics


@dataclass
class LoadedSkill:
    name: str
    definition: dict
    execute_fn: Callable
    file_path: Path
    loaded_at: str
    module_name: str = ""
    status: SkillStatus = SkillStatus.LOADED
    metadata: SkillMetadata = field(default_factory=SkillMetadata)
    diagnostics: list[SkillDiagnostic] = field(default_factory=list)


class SkillManager:
    """Manages user-created Python skill files in data/skills/."""

    def __init__(
        self, skills_dir: str, tool_executor: ToolExecutor,
        memory_path: str | None = None,
    ) -> None:
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._config_dir = self.skills_dir / "config"
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._executor = tool_executor
        # Derive a separate skill memory file to avoid corrupting the
        # executor's scoped memory structure (global/user_* namespaces).
        if memory_path:
            p = Path(memory_path)
            self._memory_path = str(p.parent / f"{p.stem}_skills{p.suffix}")
        else:
            self._memory_path = None
        self._skills: dict[str, LoadedSkill] = {}
        # Optional service references — set after construction via set_services()
        self._knowledge_store = None
        self._embedder = None
        self._session_manager = None
        self._scheduler = None
        self._load_all()

    def set_services(
        self,
        knowledge_store=None,
        embedder=None,
        session_manager=None,
        scheduler=None,
    ) -> None:
        """Inject optional service references for skill context expansion."""
        self._knowledge_store = knowledge_store
        self._embedder = embedder
        self._session_manager = session_manager
        self._scheduler = scheduler

    def _load_all(self) -> None:
        for path in sorted(self.skills_dir.glob("*.py")):
            skill = self._load_skill(path)
            if skill:
                self._skills[skill.name] = skill
        if self._skills:
            log.info("Loaded %d skill(s): %s", len(self._skills), ", ".join(self._skills))

    def _load_skill(self, path: Path) -> LoadedSkill | None:
        module_name = f"heimdall_skill_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if not spec or not spec.loader:
                log.warning("Cannot create module spec for %s", path)
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Validate SKILL_DEFINITION
            definition = getattr(module, "SKILL_DEFINITION", None)
            if not isinstance(definition, dict):
                log.warning("Skill %s: missing or invalid SKILL_DEFINITION dict", path.name)
                del sys.modules[module_name]
                return None

            for key in ("name", "description", "input_schema"):
                if key not in definition:
                    log.warning("Skill %s: SKILL_DEFINITION missing '%s'", path.name, key)
                    del sys.modules[module_name]
                    return None

            # Validate execute function
            execute_fn = getattr(module, "execute", None)
            if not callable(execute_fn):
                log.warning("Skill %s: missing execute() function", path.name)
                del sys.modules[module_name]
                return None

            # Parse rich metadata from definition
            metadata, diagnostics = SkillMetadata.from_definition(definition)
            if diagnostics:
                for d in diagnostics:
                    log.warning("Skill %s metadata: %s", path.name, d.message)

            name = definition["name"]
            log.info("Loaded skill: %s from %s", name, path.name)
            return LoadedSkill(
                name=name,
                definition=definition,
                execute_fn=execute_fn,
                file_path=path,
                loaded_at=datetime.now().isoformat(),
                module_name=module_name,
                status=SkillStatus.LOADED,
                metadata=metadata,
                diagnostics=diagnostics,
            )

        except Exception as e:
            log.error("Failed to load skill %s: %s", path.name, e)
            sys.modules.pop(module_name, None)
            return None

    def _unload_skill(self, name: str) -> None:
        if name in self._skills:
            skill = self._skills[name]
            # Use the actual module name stored during load (based on file stem)
            # rather than deriving from skill name, which could differ for
            # manually placed skill files.
            mod_name = skill.module_name or f"heimdall_skill_{name}"
            sys.modules.pop(mod_name, None)
            del self._skills[name]

    def _validate_name(self, name: str) -> str | None:
        """Validate skill name. Returns error string or None if valid."""
        if not SKILL_NAME_PATTERN.match(name):
            return (
                f"Invalid skill name '{name}'. Must be lowercase alphanumeric with underscores, "
                "1-50 chars, starting with a letter."
            )
        if name in BUILTIN_TOOL_NAMES:
            return f"Name '{name}' conflicts with a built-in tool."
        return None

    def create_skill(self, name: str, code: str) -> str:
        """Write a new skill file and hot-load it."""
        error = self._validate_name(name)
        if error:
            return error

        if name in self._skills:
            return f"Skill '{name}' already exists. Use edit_skill to modify it."

        path = self.skills_dir / f"{name}.py"
        try:
            path.write_text(code)
        except Exception as e:
            return f"Failed to write skill file: {e}"

        skill = self._load_skill(path)
        if not skill:
            path.unlink(missing_ok=True)
            return f"Skill file written but failed to load. Check syntax and required exports (SKILL_DEFINITION dict + async execute function)."

        if skill.name != name:
            self._unload_skill(skill.name)
            path.unlink(missing_ok=True)
            return f"SKILL_DEFINITION.name ('{skill.name}') doesn't match filename ('{name}'). They must be identical."

        self._skills[name] = skill
        return f"Skill '{name}' created and loaded successfully. It's now available as a tool."

    def edit_skill(self, name: str, code: str) -> str:
        """Replace a skill's code and reload it."""
        if name not in self._skills:
            return f"Skill '{name}' not found."

        path = self.skills_dir / f"{name}.py"
        old_code = path.read_text() if path.exists() else ""

        try:
            path.write_text(code)
        except Exception as e:
            return f"Failed to write skill file: {e}"

        self._unload_skill(name)
        skill = self._load_skill(path)
        if not skill:
            # Restore old code
            path.write_text(old_code)
            old_skill = self._load_skill(path)
            if old_skill:
                self._skills[name] = old_skill
            return f"New code failed to load. Reverted to previous version."

        if skill.name != name:
            # Name mismatch — revert to old code
            self._unload_skill(skill.name)
            path.write_text(old_code)
            old_skill = self._load_skill(path)
            if old_skill:
                self._skills[name] = old_skill
            return (
                f"SKILL_DEFINITION.name ('{skill.name}') doesn't match filename "
                f"('{name}'). They must be identical. Reverted to previous version."
            )

        self._skills[name] = skill
        return f"Skill '{name}' updated and reloaded successfully."

    def delete_skill(self, name: str) -> str:
        """Delete a skill file and unload it."""
        if name not in self._skills:
            return f"Skill '{name}' not found."

        path = self.skills_dir / f"{name}.py"
        self._unload_skill(name)
        path.unlink(missing_ok=True)
        # Clean up config file
        config_path = self._config_dir / f"{name}.json"
        config_path.unlink(missing_ok=True)
        return f"Skill '{name}' deleted."

    # -- Config management --

    def _config_path(self, name: str) -> Path:
        return self._config_dir / f"{name}.json"

    def get_skill_config(self, name: str) -> dict:
        """Get runtime config for a skill, with defaults applied from schema."""
        skill = self._skills.get(name)
        if not skill:
            return {}
        raw = self._load_config_file(name)
        schema = skill.metadata.config_schema
        if schema:
            return apply_defaults(schema, raw)
        return raw

    def set_skill_config(self, name: str, values: dict) -> list[str]:
        """Set runtime config for a skill. Returns list of validation errors (empty = success)."""
        skill = self._skills.get(name)
        if not skill:
            return [f"Skill '{name}' not found."]
        schema = skill.metadata.config_schema
        if schema:
            errors = validate_config(schema, values)
            if errors:
                return errors
        self._save_config_file(name, values)
        return []

    def _load_config_file(self, name: str) -> dict:
        path = self._config_path(name)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    def _save_config_file(self, name: str, values: dict) -> None:
        path = self._config_path(name)
        path.write_text(json.dumps(values, indent=2))

    def list_skills(self) -> list[dict]:
        """Return metadata for all loaded skills."""
        return [
            {
                "name": s.name,
                "description": s.definition.get("description", ""),
                "loaded_at": s.loaded_at,
                "status": s.status.value,
                "version": s.metadata.version,
                "author": s.metadata.author,
                "tags": s.metadata.tags,
                "dependencies": s.metadata.dependencies,
                "has_config": bool(s.metadata.config_schema),
                "diagnostics": [
                    {"level": d.level, "message": d.message}
                    for d in s.diagnostics
                ],
            }
            for s in self._skills.values()
        ]

    def has_skill(self, name: str) -> bool:
        return name in self._skills

    def should_handoff_to_codex(self, name: str) -> bool:
        """Check if a skill wants its result handed to Codex for the response."""
        skill = self._skills.get(name)
        if not skill:
            return False
        return bool(skill.definition.get("handoff_to_codex", False))

    def get_tool_definitions(self) -> list[dict]:
        """Return tool definitions for all loaded skills."""
        return [
            {
                "name": s.definition["name"],
                "description": s.definition["description"],
                "input_schema": s.definition["input_schema"],
            }
            for s in self._skills.values()
        ]

    def validate_skill_code(self, code: str, filename: str = "<string>") -> dict:
        """Validate skill code without loading it. Returns a report dict.

        The report contains:
          valid (bool), errors (list[str]), warnings (list[str]),
          metadata (dict|None), definition_keys (list[str])
        """
        errors: list[str] = []
        warnings: list[str] = []
        metadata_out: dict | None = None
        definition_keys: list[str] = []

        # 1. Syntax check
        try:
            compile(code, filename, "exec")
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return {
                "valid": False, "errors": errors, "warnings": warnings,
                "metadata": None, "definition_keys": [],
            }

        # 2. Execute in a temporary namespace to inspect exports
        ns: dict[str, Any] = {}
        try:
            exec(code, ns)  # noqa: S102 — skill validation needs exec
        except Exception as e:
            errors.append(f"Execution error: {e}")
            return {
                "valid": False, "errors": errors, "warnings": warnings,
                "metadata": None, "definition_keys": [],
            }

        # 3. Check SKILL_DEFINITION
        definition = ns.get("SKILL_DEFINITION")
        if not isinstance(definition, dict):
            errors.append("Missing or invalid SKILL_DEFINITION dict.")
        else:
            definition_keys = list(definition.keys())
            for key in ("name", "description", "input_schema"):
                if key not in definition:
                    errors.append(f"SKILL_DEFINITION missing required key '{key}'.")

            # Validate name if present
            name = definition.get("name")
            if isinstance(name, str):
                name_err = self._validate_name(name)
                if name_err:
                    errors.append(name_err)
            elif name is not None:
                errors.append("SKILL_DEFINITION 'name' must be a string.")

            # Parse metadata
            meta, diags = SkillMetadata.from_definition(definition)
            metadata_out = {
                "version": meta.version, "author": meta.author,
                "homepage": meta.homepage, "tags": meta.tags,
                "dependencies": meta.dependencies,
                "has_config": bool(meta.config_schema),
            }
            for d in diags:
                warnings.append(d.message)

        # 4. Check execute function
        execute_fn = ns.get("execute")
        if not callable(execute_fn):
            errors.append("Missing execute() function.")
        elif not asyncio.iscoroutinefunction(execute_fn):
            warnings.append("execute() is not async. It should be 'async def execute(inp, context)'.")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "metadata": metadata_out,
            "definition_keys": definition_keys,
        }

    def get_skill_info(self, name: str) -> dict | None:
        """Return detailed info for a single skill, or None if not found."""
        skill = self._skills.get(name)
        if not skill:
            return None
        code = None
        try:
            code = skill.file_path.read_text()
        except Exception:
            pass
        return {
            "name": skill.name,
            "description": skill.definition.get("description", ""),
            "input_schema": skill.definition.get("input_schema", {}),
            "loaded_at": skill.loaded_at,
            "status": skill.status.value,
            "file_path": str(skill.file_path),
            "metadata": {
                "version": skill.metadata.version,
                "author": skill.metadata.author,
                "homepage": skill.metadata.homepage,
                "tags": skill.metadata.tags,
                "dependencies": skill.metadata.dependencies,
                "has_config": bool(skill.metadata.config_schema),
                "config_schema": skill.metadata.config_schema,
            },
            "config": self.get_skill_config(name),
            "diagnostics": [
                {"level": d.level, "message": d.message}
                for d in skill.diagnostics
            ],
            "handoff_to_codex": skill.definition.get("handoff_to_codex", False),
            "code": code,
        }

    async def execute(
        self, tool_name: str, tool_input: dict,
        message_callback: Callable | None = None,
        file_callback: Callable | None = None,
    ) -> str:
        """Execute a user-created skill with timeout."""
        skill = self._skills.get(tool_name)
        if not skill:
            return f"Skill '{tool_name}' not found."

        # Load config with defaults applied
        skill_config = self.get_skill_config(tool_name)

        context = SkillContext(
            self._executor, tool_name,
            memory_path=self._memory_path,
            message_callback=message_callback,
            file_callback=file_callback,
            knowledge_store=self._knowledge_store,
            embedder=self._embedder,
            session_manager=self._session_manager,
            scheduler=self._scheduler,
            skill_config=skill_config,
        )
        try:
            result = await asyncio.wait_for(
                skill.execute_fn(tool_input, context),
                timeout=SKILL_EXECUTE_TIMEOUT,
            )
            if not isinstance(result, str):
                result = str(result)
            return result
        except asyncio.TimeoutError:
            return f"Skill '{tool_name}' timed out after {SKILL_EXECUTE_TIMEOUT}s."
        except Exception as e:
            log.error("Skill %s execution error: %s", tool_name, e, exc_info=True)
            return f"Skill error: {e}"
