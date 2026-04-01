"""Tests for Heimdall packaging validation — systemd service, postinstall, preremove."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.packaging.validate import (
    DATA_SUBDIRS,
    FHS_PATHS,
    REQUIRED_SERVICE_DIRECTIVES,
    check_script_syntax,
    extract_script_operations,
    parse_systemd_unit,
    validate_postinstall,
    validate_preremove,
    validate_service_file,
)

PACKAGING_DIR = Path(__file__).parent.parent / "packaging"


# ---------------------------------------------------------------------------
# parse_systemd_unit
# ---------------------------------------------------------------------------

class TestParseSystemdUnit:
    """Tests for the systemd unit file parser."""

    def test_parses_sections(self):
        """Parser extracts section headers correctly."""
        content = "[Unit]\nDescription=Test\n\n[Service]\nType=simple\n"
        result = parse_systemd_unit(content)
        assert "Unit" in result
        assert "Service" in result

    def test_parses_key_value_pairs(self):
        """Parser extracts key=value within sections."""
        content = "[Service]\nType=simple\nUser=heimdall\n"
        result = parse_systemd_unit(content)
        assert result["Service"]["Type"] == "simple"
        assert result["Service"]["User"] == "heimdall"

    def test_handles_values_with_equals(self):
        """Parser handles values containing '=' characters."""
        content = "[Service]\nExecStart=/usr/bin/prog --flag=value\n"
        result = parse_systemd_unit(content)
        assert result["Service"]["ExecStart"] == "/usr/bin/prog --flag=value"

    def test_ignores_comments(self):
        """Parser skips comment lines."""
        content = "[Unit]\n# This is a comment\nDescription=Test\n; Another comment\n"
        result = parse_systemd_unit(content)
        assert len(result["Unit"]) == 1
        assert result["Unit"]["Description"] == "Test"

    def test_ignores_blank_lines(self):
        """Parser skips blank lines without error."""
        content = "[Unit]\n\n\nDescription=Test\n\n"
        result = parse_systemd_unit(content)
        assert result["Unit"]["Description"] == "Test"

    def test_empty_input(self):
        """Parser returns empty dict for empty input."""
        assert parse_systemd_unit("") == {}

    def test_strips_whitespace(self):
        """Parser strips whitespace from keys and values."""
        content = "[Service]\n  Type  =  simple  \n"
        result = parse_systemd_unit(content)
        assert result["Service"]["Type"] == "simple"


# ---------------------------------------------------------------------------
# validate_service_file
# ---------------------------------------------------------------------------

class TestValidateServiceFile:
    """Tests for systemd service file validation."""

    def test_valid_service_file(self):
        """Real heimdall.service passes validation."""
        content = (PACKAGING_DIR / "heimdall.service").read_text()
        errors = validate_service_file(content)
        assert errors == [], f"Validation errors: {errors}"

    def test_missing_section(self):
        """Validation catches missing [Service] section."""
        content = "[Unit]\nDescription=Test\n[Install]\nWantedBy=multi-user.target\n"
        errors = validate_service_file(content)
        assert any("Missing [Service] section" in e for e in errors)

    def test_missing_directive(self):
        """Validation catches missing required directive."""
        content = "[Unit]\nDescription=Test\n[Service]\nType=simple\n[Install]\nWantedBy=multi-user.target\n"
        errors = validate_service_file(content)
        # Should flag missing User, Group, WorkingDirectory, etc.
        assert len(errors) > 0
        assert any("Missing directive" in e for e in errors)

    def test_wrong_directive_value(self):
        """Validation catches incorrect directive value."""
        content = (
            "[Unit]\nDescription=Test\n"
            "[Service]\nType=forking\nUser=heimdall\nGroup=heimdall\n"
            "WorkingDirectory=/opt/heimdall\n"
            "ExecStart=/opt/heimdall/.venv/bin/python -m src\n"
            "EnvironmentFile=/etc/heimdall/.env\n"
            "Restart=on-failure\n"
            "[Install]\nWantedBy=multi-user.target\n"
        )
        errors = validate_service_file(content)
        assert any("Type" in e and "forking" in e for e in errors)

    def test_missing_wantedby(self):
        """Validation catches missing WantedBy."""
        content = (
            "[Unit]\nDescription=Test\n"
            "[Service]\nType=simple\nUser=heimdall\nGroup=heimdall\n"
            "WorkingDirectory=/opt/heimdall\n"
            "ExecStart=/opt/heimdall/.venv/bin/python -m src\n"
            "EnvironmentFile=/etc/heimdall/.env\n"
            "Restart=on-failure\n"
            "[Install]\n"
        )
        errors = validate_service_file(content)
        assert any("WantedBy" in e for e in errors)

    def test_service_file_has_security_hardening(self):
        """Service file includes security directives."""
        content = (PACKAGING_DIR / "heimdall.service").read_text()
        parsed = parse_systemd_unit(content)
        service = parsed["Service"]
        assert service.get("NoNewPrivileges") == "yes"
        assert service.get("ProtectSystem") == "strict"
        assert service.get("ProtectHome") == "yes"
        assert service.get("PrivateTmp") == "yes"

    def test_service_file_has_readwrite_paths(self):
        """Service file grants write access to data and log dirs."""
        content = (PACKAGING_DIR / "heimdall.service").read_text()
        parsed = parse_systemd_unit(content)
        rw_paths = parsed["Service"].get("ReadWritePaths", "")
        assert "/var/lib/heimdall" in rw_paths
        assert "/var/log/heimdall" in rw_paths

    def test_service_file_restart_delay(self):
        """Service file uses a restart delay to avoid crash loops."""
        content = (PACKAGING_DIR / "heimdall.service").read_text()
        parsed = parse_systemd_unit(content)
        assert parsed["Service"]["RestartSec"] == "5"

    def test_required_directives_match_spec(self):
        """REQUIRED_SERVICE_DIRECTIVES contains the spec from BUILD_STATUS.md."""
        assert REQUIRED_SERVICE_DIRECTIVES["Type"] == "simple"
        assert REQUIRED_SERVICE_DIRECTIVES["User"] == "heimdall"
        assert REQUIRED_SERVICE_DIRECTIVES["Restart"] == "on-failure"
        assert REQUIRED_SERVICE_DIRECTIVES["ExecStart"] == "/opt/heimdall/.venv/bin/python -m src"


# ---------------------------------------------------------------------------
# extract_script_operations
# ---------------------------------------------------------------------------

class TestExtractScriptOperations:
    """Tests for shell script operation extraction."""

    def test_extracts_user_creation(self):
        """Extractor finds groupadd/useradd."""
        content = "#!/bin/bash\ngroupadd --system heimdall\nuseradd --system heimdall\n"
        ops = extract_script_operations(content)
        assert len(ops["user_creation"]) == 2

    def test_extracts_directory_creation(self):
        """Extractor finds mkdir commands."""
        content = "#!/bin/bash\nmkdir -p /var/lib/heimdall\nmkdir -p /etc/heimdall\n"
        ops = extract_script_operations(content)
        assert len(ops["directory_creation"]) == 2

    def test_extracts_symlinks(self):
        """Extractor finds ln commands."""
        content = "#!/bin/bash\nln -sf /etc/heimdall/config.yml /opt/heimdall/config.yml\n"
        ops = extract_script_operations(content)
        assert len(ops["symlinks"]) == 1

    def test_extracts_service_ops(self):
        """Extractor finds systemctl commands."""
        content = "#!/bin/bash\nsystemctl daemon-reload\nsystemctl enable heimdall\n"
        ops = extract_script_operations(content)
        assert len(ops["service_ops"]) == 2

    def test_skips_comments(self):
        """Extractor ignores commented-out commands."""
        content = "#!/bin/bash\n# mkdir -p /tmp/test\nmkdir -p /real/dir\n"
        ops = extract_script_operations(content)
        assert len(ops["directory_creation"]) == 1

    def test_extracts_venv_ops(self):
        """Extractor finds venv and pip commands."""
        content = "#!/bin/bash\npython3.12 -m venv /opt/heimdall/.venv\npip install pkg\n"
        ops = extract_script_operations(content)
        assert len(ops["venv"]) == 2

    def test_extracts_ownership(self):
        """Extractor finds chown commands."""
        content = "#!/bin/bash\nchown -R heimdall:heimdall /var/lib/heimdall\n"
        ops = extract_script_operations(content)
        assert len(ops["ownership"]) == 1

    def test_extracts_permissions(self):
        """Extractor finds chmod commands."""
        content = "#!/bin/bash\nchmod 600 /etc/heimdall/.env\n"
        ops = extract_script_operations(content)
        assert len(ops["permissions"]) == 1


# ---------------------------------------------------------------------------
# validate_postinstall
# ---------------------------------------------------------------------------

class TestValidatePostinstall:
    """Tests for postinstall script validation."""

    def test_valid_postinstall(self):
        """Real postinstall.sh passes validation."""
        content = (PACKAGING_DIR / "postinstall.sh").read_text()
        errors = validate_postinstall(content)
        assert errors == [], f"Validation errors: {errors}"

    def test_missing_shebang(self):
        """Validation catches missing shebang."""
        content = "set -e\nmkdir -p /etc/heimdall\n"
        errors = validate_postinstall(content)
        assert any("shebang" in e for e in errors)

    def test_missing_set_e(self):
        """Validation catches missing set -e."""
        content = "#!/bin/bash\nmkdir -p /etc/heimdall\n"
        errors = validate_postinstall(content)
        assert any("set -e" in e for e in errors)

    def test_missing_user_creation(self):
        """Validation catches missing user creation."""
        content = "#!/bin/bash\nset -e\nmkdir -p /etc/heimdall\n"
        errors = validate_postinstall(content)
        assert any("user" in e.lower() or "group" in e.lower() for e in errors)

    def test_missing_data_subdir(self):
        """Validation catches missing data subdirectory."""
        content = (
            "#!/bin/bash\nset -e\n"
            "groupadd --system heimdall\nuseradd --system heimdall\n"
            "mkdir -p /etc/heimdall\nmkdir -p /var/lib/heimdall\n"
            "mkdir -p /var/log/heimdall\n"
            "ln -sf /etc/heimdall/config.yml /opt/heimdall/config.yml\n"
            "python3.12 -m venv /opt/heimdall/.venv\n"
            "chown heimdall:heimdall /var/lib/heimdall\n"
            "systemctl enable heimdall\n"
        )
        errors = validate_postinstall(content)
        # Should flag missing data subdirs (sessions, context, etc.)
        assert any("data subdir" in e for e in errors)

    def test_missing_symlinks(self):
        """Validation catches missing symlink commands."""
        content = (
            "#!/bin/bash\nset -e\n"
            "groupadd --system heimdall\nuseradd --system heimdall\n"
            "mkdir -p /etc/heimdall\n"
            "mkdir -p /var/lib/heimdall/{sessions,context,skills,search,knowledge}\n"
            "mkdir -p /var/log/heimdall\n"
            "python3.12 -m venv /opt/heimdall/.venv\n"
            "chown heimdall:heimdall /var/lib/heimdall\n"
            "systemctl enable heimdall\n"
        )
        errors = validate_postinstall(content)
        assert any("symlink" in e.lower() for e in errors)

    def test_postinstall_creates_all_fhs_dirs(self):
        """Postinstall creates config, data, and log directories."""
        content = (PACKAGING_DIR / "postinstall.sh").read_text()
        ops = extract_script_operations(content)
        for path_name, path_val in FHS_PATHS.items():
            if path_name == "install_dir":
                continue
            assert any(path_val in line for line in ops["directory_creation"]), \
                f"Missing mkdir for {path_val}"

    def test_postinstall_creates_data_subdirs(self):
        """Postinstall creates all required data subdirectories."""
        content = (PACKAGING_DIR / "postinstall.sh").read_text()
        ops = extract_script_operations(content)
        for subdir in DATA_SUBDIRS:
            assert any(subdir in line for line in ops["directory_creation"]), \
                f"Missing mkdir for data subdir: {subdir}"

    def test_postinstall_sets_env_permissions(self):
        """Postinstall restricts .env file permissions to 600."""
        content = (PACKAGING_DIR / "postinstall.sh").read_text()
        ops = extract_script_operations(content)
        assert any("600" in line and ".env" in line for line in ops["permissions"]), \
            "Must chmod 600 the .env file"

    def test_postinstall_preserves_existing_config(self):
        """Postinstall only copies config if not already present (upgrade safety)."""
        content = (PACKAGING_DIR / "postinstall.sh").read_text()
        # Should have conditional: if [ ! -f ... ]; then cp ...
        assert "! -f" in content, "Must check for existing config before overwriting"

    def test_postinstall_creates_venv_with_python312(self):
        """Postinstall creates venv using python3.12."""
        content = (PACKAGING_DIR / "postinstall.sh").read_text()
        assert "python3.12 -m venv" in content

    def test_postinstall_installs_package(self):
        """Postinstall runs pip install for the package."""
        content = (PACKAGING_DIR / "postinstall.sh").read_text()
        ops = extract_script_operations(content)
        assert any("pip" in line and "install" in line for line in ops["venv"])

    def test_postinstall_enables_service(self):
        """Postinstall enables the systemd service."""
        content = (PACKAGING_DIR / "postinstall.sh").read_text()
        ops = extract_script_operations(content)
        assert any("enable" in line and "heimdall" in line for line in ops["service_ops"])

    def test_postinstall_reloads_daemon(self):
        """Postinstall runs systemctl daemon-reload."""
        content = (PACKAGING_DIR / "postinstall.sh").read_text()
        assert "daemon-reload" in content


# ---------------------------------------------------------------------------
# validate_preremove
# ---------------------------------------------------------------------------

class TestValidatePreremove:
    """Tests for preremove script validation."""

    def test_valid_preremove(self):
        """Real preremove.sh passes validation."""
        content = (PACKAGING_DIR / "preremove.sh").read_text()
        errors = validate_preremove(content)
        assert errors == [], f"Validation errors: {errors}"

    def test_missing_stop(self):
        """Validation catches missing systemctl stop."""
        content = "#!/bin/bash\nset -e\nsystemctl disable heimdall\n"
        errors = validate_preremove(content)
        assert any("stop" in e for e in errors)

    def test_missing_disable(self):
        """Validation catches missing systemctl disable."""
        content = "#!/bin/bash\nset -e\nsystemctl stop heimdall\n"
        errors = validate_preremove(content)
        assert any("disable" in e for e in errors)

    def test_preremove_checks_service_active(self):
        """Preremove checks if service is active before stopping."""
        content = (PACKAGING_DIR / "preremove.sh").read_text()
        assert "is-active" in content, "Should check if service is active before stopping"

    def test_preremove_checks_service_enabled(self):
        """Preremove checks if service is enabled before disabling."""
        content = (PACKAGING_DIR / "preremove.sh").read_text()
        assert "is-enabled" in content, "Should check if service is enabled before disabling"

    def test_preremove_reloads_daemon(self):
        """Preremove runs daemon-reload after disabling."""
        content = (PACKAGING_DIR / "preremove.sh").read_text()
        assert "daemon-reload" in content


# ---------------------------------------------------------------------------
# check_script_syntax (bash -n)
# ---------------------------------------------------------------------------

class TestCheckScriptSyntax:
    """Tests for bash syntax checking."""

    def test_valid_postinstall_syntax(self):
        """postinstall.sh passes bash -n syntax check."""
        ok, err = check_script_syntax(PACKAGING_DIR / "postinstall.sh")
        assert ok, f"Syntax error: {err}"

    def test_valid_preremove_syntax(self):
        """preremove.sh passes bash -n syntax check."""
        ok, err = check_script_syntax(PACKAGING_DIR / "preremove.sh")
        assert ok, f"Syntax error: {err}"

    def test_invalid_syntax_detected(self, tmp_path):
        """Syntax checker catches invalid bash."""
        bad_script = tmp_path / "bad.sh"
        bad_script.write_text("#!/bin/bash\nif true; then\n")  # Missing fi
        ok, err = check_script_syntax(bad_script)
        assert not ok
        assert err  # Should have error message

    def test_nonexistent_file(self, tmp_path):
        """Syntax checker handles missing file gracefully."""
        ok, err = check_script_syntax(tmp_path / "nonexistent.sh")
        assert not ok


# ---------------------------------------------------------------------------
# FHS_PATHS and DATA_SUBDIRS constants
# ---------------------------------------------------------------------------

class TestPackagingConstants:
    """Tests for packaging constants."""

    def test_fhs_paths_complete(self):
        """FHS_PATHS contains all required directory definitions."""
        assert "install_dir" in FHS_PATHS
        assert "config_dir" in FHS_PATHS
        assert "data_dir" in FHS_PATHS
        assert "log_dir" in FHS_PATHS

    def test_data_subdirs_complete(self):
        """DATA_SUBDIRS contains all required subdirectories."""
        required = {"sessions", "context", "skills", "search", "knowledge"}
        assert set(DATA_SUBDIRS) == required

    def test_fhs_paths_are_absolute(self):
        """All FHS paths are absolute paths."""
        for name, path in FHS_PATHS.items():
            assert path.startswith("/"), f"{name} must be absolute: {path}"

    def test_service_directives_match_spec(self):
        """Required service directives match the BUILD_STATUS.md specification."""
        assert REQUIRED_SERVICE_DIRECTIVES["ExecStart"].endswith("python -m src")
        assert REQUIRED_SERVICE_DIRECTIVES["EnvironmentFile"] == "/etc/heimdall/.env"
        assert REQUIRED_SERVICE_DIRECTIVES["WorkingDirectory"] == "/opt/heimdall"


# ---------------------------------------------------------------------------
# Integration: cross-file consistency
# ---------------------------------------------------------------------------

class TestPackagingConsistency:
    """Tests that packaging files are consistent with each other."""

    def test_service_user_matches_postinstall(self):
        """Service file user matches the user created by postinstall."""
        service_content = (PACKAGING_DIR / "heimdall.service").read_text()
        postinstall_content = (PACKAGING_DIR / "postinstall.sh").read_text()

        parsed = parse_systemd_unit(service_content)
        service_user = parsed["Service"]["User"]

        assert service_user in postinstall_content, \
            f"postinstall must create user '{service_user}' referenced in service file"

    def test_service_paths_match_postinstall(self):
        """Service file paths are created by postinstall."""
        service_content = (PACKAGING_DIR / "heimdall.service").read_text()
        postinstall_content = (PACKAGING_DIR / "postinstall.sh").read_text()

        parsed = parse_systemd_unit(service_content)
        work_dir = parsed["Service"]["WorkingDirectory"]
        env_file = parsed["Service"]["EnvironmentFile"]

        # WorkingDirectory parent should be referenced
        assert work_dir in postinstall_content, \
            f"postinstall must reference {work_dir}"
        # EnvironmentFile directory should be created
        env_dir = str(Path(env_file).parent)
        assert env_dir in postinstall_content, \
            f"postinstall must create {env_dir}"

    def test_preremove_references_same_service(self):
        """Preremove stops the same service name as the unit file."""
        preremove_content = (PACKAGING_DIR / "preremove.sh").read_text()
        assert "heimdall.service" in preremove_content or "heimdall" in preremove_content

    def test_postinstall_symlinks_match_service_expectations(self):
        """Postinstall creates symlinks for config.yml and .env."""
        content = (PACKAGING_DIR / "postinstall.sh").read_text()
        ops = extract_script_operations(content)
        symlink_text = " ".join(ops["symlinks"])
        assert "config.yml" in symlink_text, "Must symlink config.yml"
        assert ".env" in symlink_text, "Must symlink .env"

    def test_postinstall_symlinks_data_directory(self):
        """Postinstall symlinks data directory from /var/lib/heimdall."""
        content = (PACKAGING_DIR / "postinstall.sh").read_text()
        ops = extract_script_operations(content)
        symlink_text = " ".join(ops["symlinks"])
        assert "/var/lib/heimdall" in symlink_text, "Must symlink data dir"
