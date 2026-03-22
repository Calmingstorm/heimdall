"""Tests verifying Round 19 polish changes — naming consistency, type fixes, imports."""
from __future__ import annotations

import ast
import collections.abc
import importlib
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


# ---------------------------------------------------------------------------
# 1. Stale OllamaClassifier references removed
# ---------------------------------------------------------------------------
class TestNoOllamaClassifierReferences:
    def test_haiku_classifier_no_ollama_ref(self):
        source = (SRC / "llm" / "haiku_classifier.py").read_text()
        assert "OllamaClassifier" not in source

    def test_no_ollama_ref_in_any_source(self):
        for py in SRC.rglob("*.py"):
            source = py.read_text()
            assert "OllamaClassifier" not in source, f"Found OllamaClassifier in {py}"


# ---------------------------------------------------------------------------
# 2. "whitelisted" → "allowlisted" terminology
# ---------------------------------------------------------------------------
class TestNoWhitelistTerminology:
    def test_permissions_uses_allowlisted(self):
        source = (SRC / "permissions" / "manager.py").read_text()
        assert "allowlisted" in source
        assert "whitelisted" not in source

    def test_no_whitelist_in_any_source(self):
        for py in SRC.rglob("*.py"):
            source = py.read_text().lower()
            assert "whitelist" not in source, f"Found 'whitelist' in {py}"


# ---------------------------------------------------------------------------
# 3. CompactionFn/TextFn type aliases use Awaitable[str]
# ---------------------------------------------------------------------------
class TestAsyncTypeAliases:
    def test_compaction_fn_awaitable(self):
        source = (SRC / "sessions" / "manager.py").read_text()
        assert "Awaitable[str]" in source
        # Should NOT have the old plain str return type
        assert "Callable[[list[dict], str], str]" not in source

    def test_text_fn_awaitable(self):
        source = (SRC / "learning" / "reflector.py").read_text()
        assert "Awaitable[str]" in source
        assert "Callable[[list[dict], str], str]" not in source

    def test_compaction_fn_runtime_value(self):
        from src.sessions.manager import CompactionFn
        # Should reference Awaitable in its args
        assert CompactionFn is not None

    def test_text_fn_runtime_value(self):
        from src.learning.reflector import TextFn
        assert TextFn is not None


# ---------------------------------------------------------------------------
# 4. Consistent Callable/Awaitable imports from collections.abc
# ---------------------------------------------------------------------------
_CALLABLE_FILES = [
    "discord/client.py",
    "discord/voice.py",
    "monitoring/watcher.py",
    "scheduler/scheduler.py",
    "health/server.py",
    "tools/skill_context.py",
    "tools/skill_manager.py",
    "sessions/manager.py",
    "learning/reflector.py",
]


class TestConsistentCallableImports:
    def test_no_callable_from_typing(self):
        """No source file should import Callable or Awaitable from typing."""
        for py in SRC.rglob("*.py"):
            tree = ast.parse(py.read_text(), filename=str(py))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "typing":
                    names = [alias.name for alias in node.names]
                    assert "Callable" not in names, (
                        f"{py} imports Callable from typing"
                    )
                    assert "Awaitable" not in names, (
                        f"{py} imports Awaitable from typing"
                    )

    def test_callable_files_use_collections_abc(self):
        """All files that use Callable import it from collections.abc."""
        for relpath in _CALLABLE_FILES:
            path = SRC / relpath
            tree = ast.parse(path.read_text(), filename=relpath)
            found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "collections.abc":
                    names = [alias.name for alias in node.names]
                    if "Callable" in names:
                        found = True
            assert found, f"{relpath} should import Callable from collections.abc"


# ---------------------------------------------------------------------------
# 5. ssl=False comment accuracy
# ---------------------------------------------------------------------------
class TestSSLComment:
    def test_ssl_comment_mentions_disable(self):
        source = (SRC / "tools" / "web.py").read_text()
        # Find the line with ssl=False
        for line in source.splitlines():
            if "ssl=False" in line:
                assert "Disable" in line or "disable" in line, (
                    f"ssl=False comment should mention disabling, got: {line.strip()}"
                )
                break
        else:
            raise AssertionError("ssl=False not found in web.py")


# ---------------------------------------------------------------------------
# 6. No stale egg-info artifact
# ---------------------------------------------------------------------------
class TestNoStaleEggInfo:
    def test_no_ansiblex_egg_info(self):
        egg_dir = SRC / "ansiblex.egg-info"
        assert not egg_dir.exists(), "ansiblex.egg-info should be deleted"

    def test_egg_info_gitignored(self):
        gitignore = (SRC.parent / ".gitignore").read_text()
        assert "*.egg-info" in gitignore


# ---------------------------------------------------------------------------
# 7. Module import sanity — all changed files still import cleanly
# ---------------------------------------------------------------------------
_CHANGED_MODULES = [
    "src.llm.haiku_classifier",
    "src.permissions.manager",
    "src.sessions.manager",
    "src.learning.reflector",
    "src.scheduler.scheduler",
    "src.discord.voice",
    "src.monitoring.watcher",
    "src.health.server",
    "src.tools.skill_context",
    "src.tools.skill_manager",
    "src.tools.web",
]


class TestChangedModulesImport:
    def test_all_changed_modules_import(self):
        for modname in _CHANGED_MODULES:
            mod = importlib.import_module(modname)
            assert mod is not None, f"Failed to import {modname}"
