"""Tests verifying dead code cleanup — unused imports, dead stubs, and code hygiene."""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _get_source_files() -> list[Path]:
    """Return all .py files under src/."""
    return sorted(SRC_DIR.rglob("*.py"))


def _get_imports(tree: ast.Module) -> dict[str, int]:
    """Extract top-level import names and line numbers (exclude __future__)."""
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                imports[name] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            if getattr(node, "module", "") == "__future__":
                continue
            for alias in node.names:
                name = alias.asname or alias.name
                imports[name] = node.lineno
    return imports


def _get_used_names(tree: ast.Module) -> set[str]:
    """Extract all Name and first-level Attribute references."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                names.add(node.value.id)
    return names


class TestNoUnusedImportsInCleanedFiles:
    """Verify the specific files cleaned in Round 16 stay clean."""

    CLEANED_FILES = [
        "src/audit/logger.py",
        "src/discord/background_task.py",
        "src/tools/browser.py",
    ]

    @pytest.mark.parametrize("rel_path", CLEANED_FILES)
    def test_no_unused_imports(self, rel_path: str):
        path = SRC_DIR.parent / rel_path
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        imports = _get_imports(tree)
        used = _get_used_names(tree)

        unused = [
            f"{name} (line {lineno})"
            for name, lineno in imports.items()
            if name not in used
            # __init__.py re-exports are intentional
            and not path.name == "__init__.py"
        ]
        assert not unused, f"Unused imports in {rel_path}: {unused}"


class TestSpecificCleanupsApplied:
    """Verify each specific cleanup from Round 16."""

    def test_audit_logger_no_time_import(self):
        """audit/logger.py should not import time (uses datetime instead)."""
        source = (SRC_DIR / "audit" / "logger.py").read_text()
        tree = ast.parse(source)
        imports = _get_imports(tree)
        assert "time" not in imports, "audit/logger.py still imports unused 'time'"

    def test_background_task_no_re_import(self):
        """background_task.py should not import re (never uses regex)."""
        source = (SRC_DIR / "discord" / "background_task.py").read_text()
        tree = ast.parse(source)
        imports = _get_imports(tree)
        assert "re" not in imports, "background_task.py still imports unused 're'"

    def test_voice_no_time_import(self):
        """voice.py should not import time at module level."""
        source = (SRC_DIR / "discord" / "voice.py").read_text()
        # Check module-level imports only (not inside functions)
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "time", "voice.py still imports 'time' at module level"

    def test_voice_no_opus_decoder_import(self):
        """voice.py _patch_voice_recv_dave should not import OpusDecoder."""
        source = (SRC_DIR / "discord" / "voice.py").read_text()
        assert "OpusDecoder" not in source, "voice.py still references unused OpusDecoder"

    def test_voice_no_add_reaction_stub(self):
        """VoiceMessageProxy should not have add_reaction method."""
        source = (SRC_DIR / "discord" / "voice.py").read_text()
        assert "add_reaction" not in source, "voice.py still has dead add_reaction stub"

    def test_knowledge_store_no_path_import(self):
        """knowledge/store.py should not import Path (never used)."""
        source = (SRC_DIR / "knowledge" / "store.py").read_text()
        tree = ast.parse(source)
        imports = _get_imports(tree)
        assert "Path" not in imports, "knowledge/store.py still imports unused 'Path'"

    def test_browser_no_re_import(self):
        """browser.py should not import re (never uses regex)."""
        source = (SRC_DIR / "tools" / "browser.py").read_text()
        tree = ast.parse(source)
        imports = _get_imports(tree)
        assert "re" not in imports, "browser.py still imports unused 're'"

    def test_skill_manager_no_awaitable(self):
        """skill_manager.py should not import Awaitable (unused)."""
        source = (SRC_DIR / "tools" / "skill_manager.py").read_text()
        tree = ast.parse(source)
        imports = _get_imports(tree)
        # Any is used by SkillMetadata (config_schema: dict[str, Any])
        assert "Awaitable" not in imports, "skill_manager.py still imports unused 'Awaitable'"

    def test_skill_manager_still_imports_callable(self):
        """skill_manager.py should still import Callable (it's used)."""
        source = (SRC_DIR / "tools" / "skill_manager.py").read_text()
        tree = ast.parse(source)
        imports = _get_imports(tree)
        assert "Callable" in imports, "skill_manager.py lost required 'Callable' import"


class TestVoiceMessageProxyStillWorks:
    """Verify VoiceMessageProxy retains its essential interface."""

    def test_has_reply_method(self):
        """VoiceMessageProxy must still have reply() for message routing."""
        source = (SRC_DIR / "discord" / "voice.py").read_text()
        assert "async def reply(" in source

    def test_has_required_fields(self):
        """VoiceMessageProxy must still have author, channel, id, guild."""
        source = (SRC_DIR / "discord" / "voice.py").read_text()
        for field in ["author", "channel", "id", "guild"]:
            assert f"{field}:" in source or f"{field} :" in source


class TestLegacyCodeIntentionallyKept:
    """Verify legacy/migration code paths are still present (intentional)."""

    def test_memory_migration_still_present(self):
        """Memory format migration in executor.py is kept for existing users."""
        source = (SRC_DIR / "tools" / "executor.py").read_text()
        assert "Migrate old flat format" in source

    def test_grocery_migration_still_present(self):
        """Grocery list migration in executor.py is kept for existing users."""
        source = (SRC_DIR / "tools" / "executor.py").read_text()
        assert "grocery_list.json" in source

    def test_set_user_context_still_present(self):
        """set_user_context() is kept for backward compatibility with tests."""
        source = (SRC_DIR / "tools" / "executor.py").read_text()
        assert "def set_user_context(" in source

    def test_reflector_legacy_user_id_fallback(self):
        """Reflector user_id fallback is kept for backward compat with callers."""
        source = (SRC_DIR / "learning" / "reflector.py").read_text()
        assert "legacy single user_id" in source
