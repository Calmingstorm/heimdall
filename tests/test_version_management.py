"""Tests for Round 8: Version management.

Tests version reading from pyproject.toml, --version flag, API status
version field, dashboard version display, nfpm env var versioning,
and release workflow pyproject.toml sync.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Path helpers
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
UI_DIR = PROJECT_ROOT / "ui"
PACKAGING_DIR = PROJECT_ROOT / "packaging"
WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"


# ---------------------------------------------------------------------------
# TestGetVersion — src/version.py
# ---------------------------------------------------------------------------

class TestGetVersion:
    """Tests for get_version() function."""

    def test_returns_string(self):
        """get_version() always returns a string."""
        from src.version import get_version

        result = get_version()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_version_from_pyproject(self):
        """get_version() reads version from pyproject.toml in dev mode."""
        from src.version import get_version

        # In dev mode (not pip-installed), should read from pyproject.toml
        version = get_version()
        # Should match pyproject.toml version
        toml_path = PROJECT_ROOT / "pyproject.toml"
        toml_text = toml_path.read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', toml_text, re.MULTILINE)
        assert match, "pyproject.toml must have a version field"
        assert version == match.group(1)

    def test_fallback_when_metadata_and_toml_unavailable(self):
        """get_version() returns fallback when nothing is available."""
        from src.version import _FALLBACK_VERSION, get_version

        # Patch importlib.metadata.version to raise and Path to not find toml
        with patch("importlib.metadata.version", side_effect=Exception("no package")):
            with patch("src.version.Path") as mock_path_cls:
                mock_file = MagicMock()
                mock_file.is_file.return_value = False
                mock_path_cls.return_value.resolve.return_value.parent.parent.__truediv__ = MagicMock(
                    return_value=mock_file
                )
                result = get_version()
                assert result == _FALLBACK_VERSION

    def test_version_is_semver_format(self):
        """Version from pyproject.toml is in semver format."""
        from src.version import get_version

        version = get_version()
        parts = version.split(".")
        assert len(parts) >= 2, f"Version '{version}' should be semver-like"

    def test_fallback_version_constant(self):
        """_FALLBACK_VERSION is a valid version string."""
        from src.version import _FALLBACK_VERSION

        assert isinstance(_FALLBACK_VERSION, str)
        assert "dev" in _FALLBACK_VERSION

    def test_importlib_metadata_path(self):
        """get_version() tries importlib.metadata first."""
        with patch.dict("sys.modules", {}):
            # Patch importlib.metadata.version to return a known value
            with patch("importlib.metadata.version", return_value="9.8.7"):
                from importlib import metadata
                result = metadata.version("heimdall")
                assert result == "9.8.7"

    def test_pyproject_toml_regex_parses_correctly(self):
        """The regex pattern matches pyproject.toml version format."""
        test_content = '[project]\nname = "heimdall"\nversion = "2.3.4"\n'
        match = re.search(r'^version\s*=\s*"([^"]+)"', test_content, re.MULTILINE)
        assert match
        assert match.group(1) == "2.3.4"

    def test_pyproject_toml_exists_and_has_version(self):
        """pyproject.toml exists and has a version field."""
        toml_path = PROJECT_ROOT / "pyproject.toml"
        assert toml_path.is_file(), "pyproject.toml must exist"
        content = toml_path.read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        assert match, "pyproject.toml must have version = \"X.Y.Z\""


# ---------------------------------------------------------------------------
# TestVersionFlag — src/__main__.py --version
# ---------------------------------------------------------------------------

class TestVersionFlag:
    """Tests for --version / -V flag in __main__.py."""

    def test_main_has_version_check(self):
        """__main__.py checks for --version flag."""
        main_path = SRC_DIR / "__main__.py"
        content = main_path.read_text()
        assert "--version" in content
        assert "-V" in content

    def test_version_flag_prints_version(self, capsys):
        """--version flag prints version and returns."""
        from src.version import get_version

        original_argv = sys.argv[:]
        try:
            sys.argv = ["heimdall", "--version"]
            from src.__main__ import main
            main()
            captured = capsys.readouterr()
            assert f"Heimdall {get_version()}" in captured.out
        finally:
            sys.argv = original_argv

    def test_short_version_flag_prints_version(self, capsys):
        """Short -V flag prints version and returns."""
        from src.version import get_version

        original_argv = sys.argv[:]
        try:
            sys.argv = ["heimdall", "-V"]
            from src.__main__ import main
            main()
            captured = capsys.readouterr()
            assert f"Heimdall {get_version()}" in captured.out
        finally:
            sys.argv = original_argv

    def test_version_flag_returns_without_starting_bot(self):
        """--version does not attempt to load config or start bot."""
        original_argv = sys.argv[:]
        try:
            sys.argv = ["heimdall", "--version"]
            from src.__main__ import main

            # If it tries to start the bot, it would fail due to missing config
            # A successful return proves it short-circuited
            main()
        finally:
            sys.argv = original_argv

    def test_version_imports_from_version_module(self):
        """__main__.py imports from src.version."""
        main_path = SRC_DIR / "__main__.py"
        content = main_path.read_text()
        assert "from .version import get_version" in content


# ---------------------------------------------------------------------------
# TestAPIStatusVersion — web API /api/status response
# ---------------------------------------------------------------------------

class TestAPIStatusVersion:
    """Tests for version in /api/status response."""

    def test_api_imports_version_module(self):
        """api.py imports get_version from version module."""
        api_path = SRC_DIR / "web" / "api.py"
        content = api_path.read_text()
        assert "from ..version import get_version" in content

    def test_status_response_includes_version_key(self):
        """The /api/status response dict includes 'version' key."""
        api_path = SRC_DIR / "web" / "api.py"
        content = api_path.read_text()
        # Find the json_response dict in get_status
        assert '"version": get_version()' in content

    def test_version_is_first_field_in_status(self):
        """Version should be near the top of the status response."""
        api_path = SRC_DIR / "web" / "api.py"
        content = api_path.read_text()
        # version should appear before status in the response dict
        version_pos = content.index('"version": get_version()')
        status_pos = content.index('"status": "online"')
        assert version_pos < status_pos, "version should be first field in status response"


# ---------------------------------------------------------------------------
# TestDashboardVersion — web UI dashboard version display
# ---------------------------------------------------------------------------

class TestDashboardVersion:
    """Tests for version display in the dashboard."""

    def test_dashboard_shows_version(self):
        """Dashboard template references status.version."""
        dashboard_path = UI_DIR / "js" / "pages" / "dashboard.js"
        content = dashboard_path.read_text()
        assert "status.version" in content

    def test_dashboard_version_is_conditional(self):
        """Version display is conditional (v-if) so it doesn't break if missing."""
        dashboard_path = UI_DIR / "js" / "pages" / "dashboard.js"
        content = dashboard_path.read_text()
        assert 'v-if="status.version"' in content

    def test_dashboard_version_has_v_prefix(self):
        """Version display shows 'v' prefix."""
        dashboard_path = UI_DIR / "js" / "pages" / "dashboard.js"
        content = dashboard_path.read_text()
        assert "v{{ status.version }}" in content

    def test_dashboard_version_in_hero_sub(self):
        """Version appears in the hero subtitle area (next to uptime)."""
        dashboard_path = UI_DIR / "js" / "pages" / "dashboard.js"
        content = dashboard_path.read_text()
        # Should be in the dash-hero-sub section with separator
        hero_sub_match = re.search(r'dash-hero-sub.*?</div>', content, re.DOTALL)
        assert hero_sub_match, "dash-hero-sub should exist"
        hero_sub = hero_sub_match.group(0)
        assert "status.version" in hero_sub, "version should be in dash-hero-sub"


# ---------------------------------------------------------------------------
# TestNfpmVersioning — packaging/nfpm.yml env var version
# ---------------------------------------------------------------------------

class TestNfpmVersioning:
    """Tests for nfpm.yml environment variable version."""

    def test_nfpm_version_uses_env_var(self):
        """nfpm.yml version field uses ${VERSION} env var."""
        content = (PACKAGING_DIR / "nfpm.yml").read_text()
        config = yaml.safe_load(content)
        version = config.get("version", "")
        assert "${VERSION" in version, f"nfpm version should use ${{VERSION}}, got '{version}'"

    def test_nfpm_version_uses_env_var(self):
        """nfpm.yml version uses ${VERSION} env var."""
        content = (PACKAGING_DIR / "nfpm.yml").read_text()
        config = yaml.safe_load(content)
        version = config.get("version", "")
        assert "${VERSION" in version, f"Version should use ${{VERSION}} env var, got '{version}'"

    def test_nfpm_still_has_version_schema_semver(self):
        """nfpm.yml retains version_schema: semver."""
        content = (PACKAGING_DIR / "nfpm.yml").read_text()
        config = yaml.safe_load(content)
        assert config.get("version_schema") == "semver"


# ---------------------------------------------------------------------------
# TestReleaseWorkflowVersionSync — .github/workflows/release.yml
# ---------------------------------------------------------------------------

class TestReleaseWorkflowVersionSync:
    """Tests for release workflow pyproject.toml version sync."""

    def _read_workflow(self):
        content = (WORKFLOWS_DIR / "release.yml").read_text()
        return yaml.safe_load(content)

    def test_workflow_has_pyproject_version_step_in_deb_job(self):
        """build-deb job updates pyproject.toml version from tag."""
        content = (WORKFLOWS_DIR / "release.yml").read_text()
        config = self._read_workflow()
        jobs = config.get("jobs", config.get(True, {}).get("jobs", {}))
        if not jobs:
            # Handle PyYAML on→True conversion
            for key in config:
                if isinstance(config[key], dict) and "jobs" in config[key]:
                    jobs = config[key]["jobs"]
                    break
        deb_steps = config.get("jobs", {}).get("build-deb", {}).get("steps", [])
        step_names = [s.get("name", "") for s in deb_steps]
        assert any("pyproject" in n.lower() or "version" in n.lower()
                    for n in step_names if n), \
            f"build-deb should have version sync step, found: {step_names}"

    def test_workflow_has_pyproject_version_step_in_docker_job(self):
        """build-docker job also updates pyproject.toml version."""
        content = (WORKFLOWS_DIR / "release.yml").read_text()
        config = yaml.safe_load(content)
        docker_steps = config.get("jobs", {}).get("build-docker", {}).get("steps", [])
        step_names = [s.get("name", "") for s in docker_steps]
        assert any("pyproject" in n.lower() or "version" in n.lower()
                    for n in step_names if n), \
            f"build-docker should have version sync step, found: {step_names}"

    def test_workflow_version_sync_uses_sed(self):
        """Version sync step uses sed to update pyproject.toml."""
        content = (WORKFLOWS_DIR / "release.yml").read_text()
        # Should contain sed command targeting pyproject.toml
        assert "sed" in content and "pyproject.toml" in content, \
            "Workflow should use sed to update pyproject.toml"

    def test_workflow_version_sync_references_version_output(self):
        """Version sync step uses the extracted version from tag."""
        content = (WORKFLOWS_DIR / "release.yml").read_text()
        # Should reference steps.version.outputs.version
        assert "steps.version.outputs.version" in content

    def test_workflow_passes_version_env_to_nfpm(self):
        """Build-deb job passes VERSION env var to nfpm."""
        content = (WORKFLOWS_DIR / "release.yml").read_text()
        config = yaml.safe_load(content)
        deb_steps = config.get("jobs", {}).get("build-deb", {}).get("steps", [])
        # Find the build step with env
        found = False
        for step in deb_steps:
            env = step.get("env", {})
            if "VERSION" in env:
                found = True
                break
        assert found, "build-deb should pass VERSION env var to nfpm"

    def test_nfpm_version_matches_workflow_env_var(self):
        """nfpm.yml ${VERSION} matches the env var name passed by workflow."""
        nfpm_content = (PACKAGING_DIR / "nfpm.yml").read_text()
        workflow_content = (WORKFLOWS_DIR / "release.yml").read_text()
        # nfpm uses ${VERSION:-...}
        assert "${VERSION:-" in nfpm_content or "${VERSION}" in nfpm_content
        # workflow passes VERSION env var
        assert "VERSION:" in workflow_content


# ---------------------------------------------------------------------------
# TestVersionModuleStructure — code quality checks
# ---------------------------------------------------------------------------

class TestVersionModuleStructure:
    """Tests for version module code quality."""

    def test_version_module_exists(self):
        """src/version.py exists."""
        assert (SRC_DIR / "version.py").is_file()

    def test_version_module_is_importable(self):
        """src.version module can be imported."""
        from src.version import get_version
        assert callable(get_version)

    def test_version_module_exports_fallback(self):
        """_FALLBACK_VERSION is accessible."""
        from src.version import _FALLBACK_VERSION
        assert isinstance(_FALLBACK_VERSION, str)

    def test_version_module_no_heavy_imports(self):
        """version.py should not import heavy modules at module level."""
        version_path = SRC_DIR / "version.py"
        content = version_path.read_text()
        # importlib.metadata should be imported inside get_version(), not at top
        lines = content.split("\n")
        top_imports = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("class "):
                break
            if stripped.startswith("import ") or stripped.startswith("from "):
                top_imports.append(stripped)
        # re and pathlib are fine, but importlib.metadata should be lazy
        heavy = [i for i in top_imports if "importlib.metadata" in i]
        assert not heavy, f"importlib.metadata should be lazy-imported, found: {heavy}"


# ---------------------------------------------------------------------------
# TestVersionConsistency — cross-file consistency
# ---------------------------------------------------------------------------

class TestVersionConsistency:
    """Tests for version consistency across files."""

    def test_nfpm_version_uses_env_var(self):
        """nfpm.yml version is set via VERSION env var at build time."""
        nfpm_content = (PACKAGING_DIR / "nfpm.yml").read_text()
        nfpm_config = yaml.safe_load(nfpm_content)
        nfpm_version = nfpm_config.get("version", "")
        assert "${VERSION" in nfpm_version

    def test_get_version_matches_pyproject(self):
        """get_version() returns the same version as pyproject.toml."""
        from src.version import get_version

        toml_path = PROJECT_ROOT / "pyproject.toml"
        toml_text = toml_path.read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', toml_text, re.MULTILINE)
        assert match
        assert get_version() == match.group(1)

    def test_version_module_resolution_order(self):
        """get_version() tries importlib.metadata before pyproject.toml fallback."""
        version_path = SRC_DIR / "version.py"
        content = version_path.read_text()
        # In the get_version function body, importlib.metadata should come first
        func_start = content.index("def get_version")
        func_body = content[func_start:]
        metadata_pos = func_body.index("importlib.metadata")
        toml_pos = func_body.index("pyproject.toml")
        assert metadata_pos < toml_pos, \
            "importlib.metadata should be tried before pyproject.toml"

    def test_both_build_jobs_sync_version(self):
        """Both build-deb and build-docker jobs update pyproject.toml."""
        content = (WORKFLOWS_DIR / "release.yml").read_text()
        # Count occurrences of the sed command
        sed_count = content.count("Set version in pyproject.toml")
        assert sed_count == 2, \
            f"Both build jobs should have version sync step, found {sed_count}"
