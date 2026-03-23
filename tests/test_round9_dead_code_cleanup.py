"""Round 9: Verify dead code cleanup is complete.

Checks:
- routing.py deleted (resolve_claude_code_target, CLAUDE_CODE_DEFAULTS gone)
- _init_routing_defaults removed from LokiBot
- No imports of routing module anywhere in src/
- Unused conftest mock_ssh fixture removed
- Test files have clean imports (no unused patch/json/call)
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402


class TestRoutingModuleRemoved:
    """routing.py should be completely deleted."""

    def test_routing_py_not_on_disk(self):
        assert not Path("src/discord/routing.py").exists()

    def test_routing_module_not_importable(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("src.discord.routing")

    def test_no_routing_import_in_client(self):
        source = Path("src/discord/client.py").read_text()
        assert "from .routing" not in source
        assert "import routing" not in source

    def test_no_resolve_claude_code_target_in_source(self):
        for p in Path("src").rglob("*.py"):
            text = p.read_text()
            assert "resolve_claude_code_target" not in text, f"Found in {p}"

    def test_no_CLAUDE_CODE_DEFAULTS_in_source(self):
        for p in Path("src").rglob("*.py"):
            text = p.read_text()
            assert "CLAUDE_CODE_DEFAULTS" not in text, f"Found in {p}"


class TestInitRoutingDefaultsRemoved:
    """_init_routing_defaults method should be gone from LokiBot."""

    def test_no_init_routing_defaults_in_client(self):
        source = Path("src/discord/client.py").read_text()
        assert "_init_routing_defaults" not in source

    def test_lokibot_has_no_init_routing_defaults(self):
        from src.discord.client import LokiBot
        assert not hasattr(LokiBot, "_init_routing_defaults")


class TestUnusedImportsCleanedUp:
    """Test files should not have unused imports from removed features."""

    def test_chat_path_no_unused_imports(self):
        source = Path("tests/test_chat_path_optimization.py").read_text()
        assert "import json" not in source
        assert "call" not in source.split("from unittest.mock import")[1].split("\n")[0] if "from unittest.mock import" in source else True

    def test_claude_code_routing_no_unused_patch(self):
        source = Path("tests/test_claude_code_routing.py").read_text()
        mock_line = [l for l in source.split("\n") if "from unittest.mock import" in l]
        if mock_line:
            assert "patch" not in mock_line[0]

    def test_integration_scenarios_no_unused_patch(self):
        source = Path("tests/test_integration_scenarios.py").read_text()
        mock_line = [l for l in source.split("\n") if "from unittest.mock import" in l]
        if mock_line:
            assert "patch" not in mock_line[0]


class TestConftestCleanedUp:
    """conftest.py should not have unused fixtures."""

    def test_no_mock_ssh_fixture(self):
        source = Path("tests/conftest.py").read_text()
        assert "def mock_ssh" not in source

    def test_conftest_fixtures_minimal(self):
        """conftest.py should only define actively-used fixtures."""
        tree = ast.parse(Path("tests/conftest.py").read_text())
        fixture_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Attribute) and dec.attr == "fixture":
                        fixture_names.append(node.name)
                    elif isinstance(dec, ast.Name) and dec.id == "fixture":
                        fixture_names.append(node.name)
        # These are the expected fixtures
        expected = {"tmp_dir", "tools_config", "config"}
        assert set(fixture_names) == expected


class TestNoDuplicateFixtures:
    """No test file should shadow conftest fixtures unnecessarily."""

    def test_r6_final_fixes_no_duplicate_tmp_dir(self):
        source = Path("tests/test_r6_final_fixes.py").read_text()
        assert source.count("def tmp_dir") == 0


class TestHostRoutingTestsRemoved:
    """test_claude_code_host_routing.py should be deleted (tested dead code)."""

    def test_host_routing_test_file_gone(self):
        assert not Path("tests/test_claude_code_host_routing.py").exists()

    def test_full_verification_no_routing_class(self):
        source = Path("tests/test_full_verification.py").read_text()
        assert "class TestRouting" not in source
        assert "CLAUDE_CODE_DEFAULTS" not in source
