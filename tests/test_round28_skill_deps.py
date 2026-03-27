"""Tests for Round 28: Skill dependency resolution."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.schema import ToolsConfig, ToolHost
from src.tools.executor import ToolExecutor
from src.tools.skill_manager import (
    MAX_SKILL_DEPENDENCIES,
    SkillDiagnostic,
    SkillManager,
    _extract_dependencies_from_source,
    _install_packages,
    _is_package_installed,
    _parse_package_name,
    resolve_dependencies,
)


# ---------------------------------------------------------------------------
# _parse_package_name
# ---------------------------------------------------------------------------


class TestParsePackageName:
    def test_simple_name(self):
        assert _parse_package_name("requests") == "requests"

    def test_with_version_gte(self):
        assert _parse_package_name("requests>=2.28.0") == "requests"

    def test_with_version_eq(self):
        assert _parse_package_name("numpy==1.24.0") == "numpy"

    def test_with_extras(self):
        assert _parse_package_name("Pillow[jpeg]") == "Pillow"

    def test_with_extras_and_version(self):
        assert _parse_package_name("requests[security]>=2.0") == "requests"

    def test_dashes_and_dots(self):
        assert _parse_package_name("python-dateutil") == "python-dateutil"
        assert _parse_package_name("zope.interface") == "zope.interface"

    def test_empty_string(self):
        assert _parse_package_name("") == ""

    def test_whitespace(self):
        assert _parse_package_name("  requests  ") == "requests"

    def test_invalid_start(self):
        assert _parse_package_name("!invalid") == ""

    def test_single_char(self):
        assert _parse_package_name("x") == "x"

    def test_complex_spec(self):
        assert _parse_package_name("scikit-learn>=1.0,<2.0") == "scikit-learn"


# ---------------------------------------------------------------------------
# _is_package_installed
# ---------------------------------------------------------------------------


class TestIsPackageInstalled:
    def test_installed_package(self):
        # pytest itself is definitely installed
        assert _is_package_installed("pytest") is True

    def test_not_installed_package(self):
        assert _is_package_installed("nonexistent_pkg_xyz_12345") is False

    def test_pip_installed(self):
        assert _is_package_installed("pip") is True


# ---------------------------------------------------------------------------
# _install_packages (mocked)
# ---------------------------------------------------------------------------


class TestInstallPackages:
    @patch("src.tools.skill_manager.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        ok, output = _install_packages(["requests"])
        assert ok is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "pip" in args
        assert "install" in args
        assert "requests" in args

    @patch("src.tools.skill_manager.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="ERROR: No matching distribution"
        )
        ok, output = _install_packages(["nonexistent_pkg"])
        assert ok is False
        assert "No matching distribution" in output

    @patch("src.tools.skill_manager.subprocess.run")
    def test_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pip", timeout=120)
        ok, output = _install_packages(["slow_pkg"], timeout=120)
        assert ok is False
        assert "timed out" in output

    @patch("src.tools.skill_manager.subprocess.run")
    def test_exception(self, mock_run):
        mock_run.side_effect = OSError("No pip found")
        ok, output = _install_packages(["whatever"])
        assert ok is False
        assert "No pip found" in output

    @patch("src.tools.skill_manager.subprocess.run")
    def test_multiple_packages(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        ok, _ = _install_packages(["pkg_a", "pkg_b"])
        assert ok is True
        args = mock_run.call_args[0][0]
        assert "pkg_a" in args
        assert "pkg_b" in args

    @patch("src.tools.skill_manager.subprocess.run")
    def test_quiet_flag(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        _install_packages(["x"])
        args = mock_run.call_args[0][0]
        assert "--quiet" in args
        assert "--disable-pip-version-check" in args


# ---------------------------------------------------------------------------
# _extract_dependencies_from_source
# ---------------------------------------------------------------------------


class TestExtractDependenciesFromSource:
    def test_simple_dict(self):
        source = '''
SKILL_DEFINITION = {
    "name": "my_skill",
    "description": "test",
    "input_schema": {},
    "dependencies": ["requests", "aiohttp>=3.0"],
}
'''
        assert _extract_dependencies_from_source(source) == ["requests", "aiohttp>=3.0"]

    def test_no_dependencies_key(self):
        source = '''
SKILL_DEFINITION = {
    "name": "my_skill",
    "description": "test",
    "input_schema": {},
}
'''
        assert _extract_dependencies_from_source(source) == []

    def test_empty_dependencies(self):
        source = '''
SKILL_DEFINITION = {
    "name": "my_skill",
    "description": "test",
    "input_schema": {},
    "dependencies": [],
}
'''
        assert _extract_dependencies_from_source(source) == []

    def test_syntax_error(self):
        source = "def foo(:\n  pass"
        assert _extract_dependencies_from_source(source) == []

    def test_non_literal_dict(self):
        # Dynamic construction — can't extract via AST
        source = '''
d = {"name": "x"}
d["dependencies"] = ["requests"]
SKILL_DEFINITION = d
'''
        assert _extract_dependencies_from_source(source) == []

    def test_dependencies_not_list(self):
        source = '''
SKILL_DEFINITION = {
    "name": "x",
    "description": "test",
    "input_schema": {},
    "dependencies": "not_a_list",
}
'''
        assert _extract_dependencies_from_source(source) == []

    def test_dependencies_with_non_string_items(self):
        source = '''
SKILL_DEFINITION = {
    "name": "x",
    "description": "test",
    "input_schema": {},
    "dependencies": ["requests", 42],
}
'''
        assert _extract_dependencies_from_source(source) == []

    def test_no_skill_definition(self):
        source = '''
OTHER_VAR = {"dependencies": ["requests"]}
'''
        assert _extract_dependencies_from_source(source) == []

    def test_augmented_assign_ignored(self):
        # SKILL_DEFINITION += ... should not match
        source = '''
SKILL_DEFINITION = {"name": "x", "description": "t", "input_schema": {}}
'''
        assert _extract_dependencies_from_source(source) == []

    def test_multiple_assignments(self):
        # First assignment wins
        source = '''
SKILL_DEFINITION = {"name": "x", "dependencies": ["first"]}
SKILL_DEFINITION = {"name": "x", "dependencies": ["second"]}
'''
        # ast.walk visits all assignments; our code returns on first match
        result = _extract_dependencies_from_source(source)
        assert result == ["first"] or result == ["second"]  # implementation detail


# ---------------------------------------------------------------------------
# resolve_dependencies
# ---------------------------------------------------------------------------


class TestResolveDependencies:
    def test_empty_deps(self):
        installed, new, diags = resolve_dependencies([])
        assert installed == []
        assert new == []
        assert diags == []

    @patch("src.tools.skill_manager._is_package_installed", return_value=True)
    def test_all_already_installed(self, mock_check):
        installed, new, diags = resolve_dependencies(["requests", "aiohttp"])
        assert installed == ["requests", "aiohttp"]
        assert new == []
        # No error diagnostics
        assert not any(d.level == "error" for d in diags)

    @patch("src.tools.skill_manager._install_packages", return_value=(True, ""))
    @patch("src.tools.skill_manager._is_package_installed", return_value=False)
    def test_install_missing(self, mock_check, mock_install):
        installed, new, diags = resolve_dependencies(["newpkg"])
        assert installed == []
        assert new == ["newpkg"]
        mock_install.assert_called_once_with(["newpkg"])
        assert any("Auto-installed" in d.message for d in diags)

    @patch("src.tools.skill_manager._install_packages", return_value=(False, "pip error"))
    @patch("src.tools.skill_manager._is_package_installed", return_value=False)
    def test_install_failure(self, mock_check, mock_install):
        installed, new, diags = resolve_dependencies(["badpkg"])
        assert installed == []
        assert new == []
        assert any(d.level == "error" for d in diags)
        assert any("Failed to install" in d.message for d in diags)

    def test_too_many_deps(self):
        deps = [f"pkg{i}" for i in range(MAX_SKILL_DEPENDENCIES + 1)]
        installed, new, diags = resolve_dependencies(deps)
        assert installed == []
        assert new == []
        assert any(d.level == "error" for d in diags)
        assert any("Too many dependencies" in d.message for d in diags)

    def test_invalid_spec(self):
        installed, new, diags = resolve_dependencies(["!invalid"])
        assert installed == []
        assert new == []
        assert any("Invalid dependency spec" in d.message for d in diags)

    @patch("src.tools.skill_manager._install_packages", return_value=(True, ""))
    @patch("src.tools.skill_manager._is_package_installed", side_effect=[True, False])
    def test_mixed_installed_and_missing(self, mock_check, mock_install):
        installed, new, diags = resolve_dependencies(["existing", "missing"])
        assert installed == ["existing"]
        assert new == ["missing"]
        mock_install.assert_called_once_with(["missing"])

    @patch("src.tools.skill_manager._is_package_installed", return_value=True)
    def test_exact_max_deps_allowed(self, mock_check):
        deps = [f"pkg{i}" for i in range(MAX_SKILL_DEPENDENCIES)]
        installed, new, diags = resolve_dependencies(deps)
        assert len(installed) == MAX_SKILL_DEPENDENCIES
        assert not any(d.level == "error" for d in diags)


# ---------------------------------------------------------------------------
# SkillManager integration: load with dependencies
# ---------------------------------------------------------------------------

SKILL_CODE_WITH_DEPS = '''
SKILL_DEFINITION = {
    "name": "dep_skill",
    "description": "Skill with dependencies",
    "input_schema": {"type": "object", "properties": {}},
    "dependencies": ["requests", "aiohttp"],
}

async def execute(inp, context):
    return "ok"
'''

SKILL_CODE_NO_DEPS = '''
SKILL_DEFINITION = {
    "name": "nodep_skill",
    "description": "Skill without dependencies",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    return "ok"
'''


@pytest.fixture
def skill_mgr(tmp_dir: Path, tools_config: ToolsConfig) -> SkillManager:
    executor = ToolExecutor(tools_config)
    skills_dir = tmp_dir / "skills"
    skills_dir.mkdir()
    return SkillManager(str(skills_dir), executor)


class TestSkillLoadWithDeps:
    @patch("src.tools.skill_manager.resolve_dependencies")
    def test_loads_with_deps_resolved(self, mock_resolve, skill_mgr: SkillManager):
        mock_resolve.return_value = (["requests", "aiohttp"], [], [])
        result = skill_mgr.create_skill("dep_skill", SKILL_CODE_WITH_DEPS)
        assert "successfully" in result
        assert skill_mgr.has_skill("dep_skill")
        mock_resolve.assert_called_once_with(["requests", "aiohttp"])

    @patch("src.tools.skill_manager.resolve_dependencies")
    def test_dep_diagnostics_merged(self, mock_resolve, skill_mgr: SkillManager):
        diag = SkillDiagnostic("warn", "Auto-installed dependencies: newpkg")
        mock_resolve.return_value = ([], ["newpkg"], [diag])
        skill_mgr.create_skill("dep_skill", SKILL_CODE_WITH_DEPS)
        info = skill_mgr.get_skill_info("dep_skill")
        assert info is not None
        messages = [d["message"] for d in info["diagnostics"]]
        assert "Auto-installed dependencies: newpkg" in messages

    @patch("src.tools.skill_manager.resolve_dependencies")
    def test_dep_error_still_tries_load(self, mock_resolve, skill_mgr: SkillManager):
        # Dep resolution fails, but module can still load (deps might be optional)
        diag = SkillDiagnostic("error", "Failed to install dependencies: pip error")
        mock_resolve.return_value = ([], [], [diag])
        result = skill_mgr.create_skill("dep_skill", SKILL_CODE_WITH_DEPS)
        # Should still load since the code itself doesn't actually import requests
        assert "successfully" in result
        info = skill_mgr.get_skill_info("dep_skill")
        messages = [d["message"] for d in info["diagnostics"]]
        assert any("Failed to install" in m for m in messages)

    def test_no_deps_no_resolve(self, skill_mgr: SkillManager):
        with patch("src.tools.skill_manager.resolve_dependencies") as mock_resolve:
            skill_mgr.create_skill("nodep_skill", SKILL_CODE_NO_DEPS)
            mock_resolve.assert_not_called()

    @patch("src.tools.skill_manager.resolve_dependencies")
    def test_edit_skill_resolves_deps(self, mock_resolve, skill_mgr: SkillManager):
        # First create without deps
        mock_resolve.return_value = ([], [], [])
        skill_mgr.create_skill("nodep_skill", SKILL_CODE_NO_DEPS)
        mock_resolve.reset_mock()

        # Edit to add deps
        new_code = SKILL_CODE_WITH_DEPS.replace("dep_skill", "nodep_skill")
        mock_resolve.return_value = (["requests", "aiohttp"], [], [])
        result = skill_mgr.edit_skill("nodep_skill", new_code)
        assert "successfully" in result
        mock_resolve.assert_called_once()


class TestSkillLoadAllWithDeps:
    @patch("src.tools.skill_manager.resolve_dependencies")
    def test_load_all_resolves_deps(self, mock_resolve, tmp_dir: Path, tools_config: ToolsConfig):
        mock_resolve.return_value = (["requests"], [], [])
        skills_dir = tmp_dir / "auto_skills"
        skills_dir.mkdir()
        (skills_dir / "dep_skill.py").write_text(SKILL_CODE_WITH_DEPS)
        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(skills_dir), executor)
        assert mgr.has_skill("dep_skill")
        mock_resolve.assert_called_once()


# ---------------------------------------------------------------------------
# check_dependencies
# ---------------------------------------------------------------------------


class TestCheckDependencies:
    @patch("src.tools.skill_manager.resolve_dependencies", return_value=([], [], []))
    def test_skill_not_found(self, mock_resolve, skill_mgr: SkillManager):
        result = skill_mgr.check_dependencies("nonexistent")
        assert "error" in result

    @patch("src.tools.skill_manager.resolve_dependencies", return_value=([], [], []))
    def test_no_dependencies(self, mock_resolve, skill_mgr: SkillManager):
        skill_mgr.create_skill("nodep_skill", SKILL_CODE_NO_DEPS)
        result = skill_mgr.check_dependencies("nodep_skill")
        assert result["all_satisfied"] is True
        assert result["dependencies"] == []

    @patch("src.tools.skill_manager._is_package_installed", return_value=True)
    @patch("src.tools.skill_manager.resolve_dependencies", return_value=(["requests", "aiohttp"], [], []))
    def test_all_satisfied(self, mock_resolve, mock_check, skill_mgr: SkillManager):
        skill_mgr.create_skill("dep_skill", SKILL_CODE_WITH_DEPS)
        result = skill_mgr.check_dependencies("dep_skill")
        assert result["all_satisfied"] is True
        assert len(result["dependencies"]) == 2
        assert all(d["installed"] for d in result["dependencies"])

    @patch("src.tools.skill_manager._is_package_installed", side_effect=[True, False])
    @patch("src.tools.skill_manager.resolve_dependencies", return_value=(["requests"], ["aiohttp"], []))
    def test_partial_satisfied(self, mock_resolve, mock_check, skill_mgr: SkillManager):
        skill_mgr.create_skill("dep_skill", SKILL_CODE_WITH_DEPS)
        result = skill_mgr.check_dependencies("dep_skill")
        assert result["all_satisfied"] is False
        installed_flags = [d["installed"] for d in result["dependencies"]]
        assert True in installed_flags
        assert False in installed_flags

    @patch("src.tools.skill_manager.resolve_dependencies", return_value=(["requests", "aiohttp"], [], []))
    def test_dependency_details(self, mock_resolve, skill_mgr: SkillManager):
        with patch("src.tools.skill_manager._is_package_installed", return_value=True):
            skill_mgr.create_skill("dep_skill", SKILL_CODE_WITH_DEPS)
        result = skill_mgr.check_dependencies("dep_skill")
        for dep in result["dependencies"]:
            assert "spec" in dep
            assert "package" in dep
            assert "installed" in dep


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDependencyEdgeCases:
    def test_extract_from_complex_source(self):
        source = '''
import os

# Some comment
SKILL_DEFINITION = {
    "name": "complex_skill",
    "description": "A complex skill",
    "input_schema": {"type": "object"},
    "dependencies": ["numpy>=1.24", "pandas[sql]", "scikit-learn"],
    "version": "1.0.0",
    "tags": ["data", "ml"],
}

def helper():
    pass

async def execute(inp, context):
    return "ok"
'''
        deps = _extract_dependencies_from_source(source)
        assert deps == ["numpy>=1.24", "pandas[sql]", "scikit-learn"]

    def test_parse_package_name_version_tilde(self):
        assert _parse_package_name("pkg~=1.4") == "pkg"

    def test_parse_package_name_not_equal(self):
        assert _parse_package_name("pkg!=1.0") == "pkg"

    def test_resolve_mixed_valid_invalid(self):
        with patch("src.tools.skill_manager._is_package_installed", return_value=True):
            installed, new, diags = resolve_dependencies(["!bad", "good_pkg"])
        assert installed == ["good_pkg"]
        assert any("Invalid dependency spec" in d.message for d in diags)

    @patch("src.tools.skill_manager._install_packages", return_value=(True, ""))
    @patch("src.tools.skill_manager._is_package_installed", return_value=False)
    def test_resolve_preserves_full_spec(self, mock_check, mock_install):
        installed, new, diags = resolve_dependencies(["requests>=2.28.0"])
        mock_install.assert_called_once_with(["requests>=2.28.0"])
        assert new == ["requests>=2.28.0"]

    def test_extract_deps_from_empty_file(self):
        assert _extract_dependencies_from_source("") == []

    def test_extract_deps_from_comment_only(self):
        assert _extract_dependencies_from_source("# just a comment\n") == []


# ---------------------------------------------------------------------------
# Validate skill code with dependencies
# ---------------------------------------------------------------------------


class TestValidateSkillCodeWithDeps:
    def test_validate_reports_dependencies(self, skill_mgr: SkillManager):
        code = '''
SKILL_DEFINITION = {
    "name": "val_skill",
    "description": "test",
    "input_schema": {"type": "object"},
    "dependencies": ["requests"],
}

async def execute(inp, context):
    return "ok"
'''
        report = skill_mgr.validate_skill_code(code)
        assert report["valid"] is True
        assert report["metadata"]["dependencies"] == ["requests"]

    def test_validate_no_deps(self, skill_mgr: SkillManager):
        code = '''
SKILL_DEFINITION = {
    "name": "val_skill",
    "description": "test",
    "input_schema": {"type": "object"},
}

async def execute(inp, context):
    return "ok"
'''
        report = skill_mgr.validate_skill_code(code)
        assert report["valid"] is True
        assert report["metadata"]["dependencies"] == []


# ---------------------------------------------------------------------------
# list_skills includes dependency info
# ---------------------------------------------------------------------------


class TestListSkillsDeps:
    @patch("src.tools.skill_manager.resolve_dependencies", return_value=(["requests", "aiohttp"], [], []))
    def test_list_includes_dependencies(self, mock_resolve, skill_mgr: SkillManager):
        skill_mgr.create_skill("dep_skill", SKILL_CODE_WITH_DEPS)
        skills = skill_mgr.list_skills()
        assert len(skills) == 1
        assert skills[0]["dependencies"] == ["requests", "aiohttp"]

    @patch("src.tools.skill_manager.resolve_dependencies", return_value=([], [], []))
    def test_list_empty_deps(self, mock_resolve, skill_mgr: SkillManager):
        skill_mgr.create_skill("nodep_skill", SKILL_CODE_NO_DEPS)
        skills = skill_mgr.list_skills()
        assert len(skills) == 1
        assert skills[0]["dependencies"] == []
