from __future__ import annotations

import asyncio
import importlib.util
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Callable

from ..logging import get_logger
from .executor import ToolExecutor
from .registry import TOOLS
from .skill_context import SkillContext

log = get_logger("skills")

SKILL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,49}$")
BUILTIN_TOOL_NAMES = {t["name"] for t in TOOLS}
SKILL_EXECUTE_TIMEOUT = 120  # seconds


@dataclass
class LoadedSkill:
    name: str
    definition: dict
    execute_fn: Callable
    file_path: Path
    loaded_at: str
    module_name: str = ""


class SkillManager:
    """Manages user-created Python skill files in data/skills/."""

    def __init__(
        self, skills_dir: str, tool_executor: ToolExecutor,
        memory_path: str | None = None,
    ) -> None:
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
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

            name = definition["name"]
            log.info("Loaded skill: %s from %s", name, path.name)
            return LoadedSkill(
                name=name,
                definition=definition,
                execute_fn=execute_fn,
                file_path=path,
                loaded_at=datetime.now().isoformat(),
                module_name=module_name,
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
        return f"Skill '{name}' deleted."

    def list_skills(self) -> list[dict]:
        """Return metadata for all loaded skills."""
        return [
            {
                "name": s.name,
                "description": s.definition.get("description", ""),
                "loaded_at": s.loaded_at,
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

    async def execute(
        self, tool_name: str, tool_input: dict,
        message_callback: Callable | None = None,
        file_callback: Callable | None = None,
    ) -> str:
        """Execute a user-created skill with timeout."""
        skill = self._skills.get(tool_name)
        if not skill:
            return f"Skill '{tool_name}' not found."

        context = SkillContext(
            self._executor, tool_name,
            memory_path=self._memory_path,
            message_callback=message_callback,
            file_callback=file_callback,
            knowledge_store=self._knowledge_store,
            embedder=self._embedder,
            session_manager=self._session_manager,
            scheduler=self._scheduler,
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
