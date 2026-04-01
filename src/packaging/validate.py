"""
Validate Heimdall packaging artifacts (systemd unit, shell scripts).

Used by tests and by the release workflow to catch packaging errors early.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

# FHS paths that postinstall must create/reference
FHS_PATHS = {
    "install_dir": "/opt/heimdall",
    "config_dir": "/etc/heimdall",
    "data_dir": "/var/lib/heimdall",
    "log_dir": "/var/log/heimdall",
}

# Subdirectories postinstall must create under data_dir
DATA_SUBDIRS = ["sessions", "context", "skills", "search", "knowledge"]

# Required systemd directives and their expected values
REQUIRED_SERVICE_DIRECTIVES = {
    "Type": "simple",
    "User": "heimdall",
    "Group": "heimdall",
    "WorkingDirectory": "/opt/heimdall",
    "ExecStart": "/opt/heimdall/.venv/bin/python -m src",
    "EnvironmentFile": "/etc/heimdall/.env",
    "Restart": "on-failure",
}


def parse_systemd_unit(content: str) -> dict[str, dict[str, str]]:
    """Parse a systemd unit file into {section: {key: value}} dict.

    Handles multi-value keys by keeping the last value.
    Ignores comments and blank lines.
    """
    sections: dict[str, dict[str, str]] = {}
    current_section: str | None = None

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue

        section_match = re.match(r"^\[(.+)\]$", line)
        if section_match:
            current_section = section_match.group(1)
            sections.setdefault(current_section, {})
            continue

        if current_section and "=" in line:
            key, _, value = line.partition("=")
            sections[current_section][key.strip()] = value.strip()

    return sections


def validate_service_file(content: str) -> list[str]:
    """Validate a systemd service file, return list of error strings.

    Returns empty list if valid.
    """
    errors: list[str] = []
    parsed = parse_systemd_unit(content)

    # Must have required sections
    for section in ("Unit", "Service", "Install"):
        if section not in parsed:
            errors.append(f"Missing [{section}] section")

    if "Service" not in parsed:
        return errors  # Can't check directives without [Service]

    service = parsed["Service"]
    for key, expected in REQUIRED_SERVICE_DIRECTIVES.items():
        actual = service.get(key)
        if actual is None:
            errors.append(f"Missing directive: {key}")
        elif actual != expected:
            errors.append(f"{key}: expected '{expected}', got '{actual}'")

    # Install section should have WantedBy
    install = parsed.get("Install", {})
    if "WantedBy" not in install:
        errors.append("Missing WantedBy in [Install]")

    return errors


def _resolve_shell_vars(content: str) -> dict[str, str]:
    """Extract VAR=value assignments from a shell script.

    Returns {var_name: value} for simple assignments like:
        VAR="value"
        VAR='/value'
        VAR=value
    """
    variables: dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        match = re.match(r'^([A-Z_][A-Z_0-9]*)=["\']?([^"\']*)["\']?\s*$', stripped)
        if match:
            variables[match.group(1)] = match.group(2)
    return variables


def _expand_vars(line: str, variables: dict[str, str]) -> str:
    """Expand $VAR and ${VAR} references in a line using known variables."""
    result = line
    for name, value in variables.items():
        result = result.replace(f'"${name}"', value)
        result = result.replace(f"${{{name}}}", value)
        result = result.replace(f"${name}", value)
    return result


def extract_script_operations(content: str) -> dict[str, list[str]]:
    """Extract key operations from a shell script.

    Resolves shell variable assignments (VAR="/path") and expands
    references in commands so that validation can match literal paths.

    Returns a dict with operation categories and their occurrences:
    - 'user_creation': groupadd/useradd commands
    - 'directory_creation': mkdir commands
    - 'symlinks': ln commands
    - 'service_ops': systemctl commands
    - 'ownership': chown commands
    - 'permissions': chmod commands
    - 'venv': venv/pip commands
    """
    variables = _resolve_shell_vars(content)

    ops: dict[str, list[str]] = {
        "user_creation": [],
        "directory_creation": [],
        "symlinks": [],
        "service_ops": [],
        "ownership": [],
        "permissions": [],
        "venv": [],
    }

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue

        expanded = _expand_vars(stripped, variables)

        if "groupadd" in expanded or "useradd" in expanded:
            ops["user_creation"].append(expanded)
        if "mkdir" in expanded:
            ops["directory_creation"].append(expanded)
        if expanded.startswith("ln ") or "ln -s" in expanded:
            ops["symlinks"].append(expanded)
        if "systemctl" in expanded:
            ops["service_ops"].append(expanded)
        if "chown" in expanded:
            ops["ownership"].append(expanded)
        if "chmod" in expanded:
            ops["permissions"].append(expanded)
        if "venv" in expanded or "pip" in expanded:
            ops["venv"].append(expanded)

    return ops


def validate_postinstall(content: str) -> list[str]:
    """Validate postinstall script has all required operations.

    Returns list of error strings. Empty = valid.
    """
    errors: list[str] = []
    ops = extract_script_operations(content)

    if not content.startswith("#!/bin/bash"):
        errors.append("Missing bash shebang")

    if "set -e" not in content:
        errors.append("Missing 'set -e' (exit on error)")

    # Must create system user
    if not ops["user_creation"]:
        errors.append("No user/group creation commands")

    # Must create FHS directories
    for path_name, path_val in FHS_PATHS.items():
        if path_name == "install_dir":
            continue  # Install dir is created by package manager
        if not any(path_val in line for line in ops["directory_creation"]):
            errors.append(f"Missing mkdir for {path_val}")

    # Must create data subdirectories
    for subdir in DATA_SUBDIRS:
        if not any(subdir in line for line in ops["directory_creation"]):
            errors.append(f"Missing mkdir for data subdir: {subdir}")

    # Must create symlinks from app dir to FHS locations
    if not ops["symlinks"]:
        errors.append("No symlink creation commands")

    # Must set up venv and install deps
    if not ops["venv"]:
        errors.append("No venv/pip commands")

    # Must set ownership
    if not ops["ownership"]:
        errors.append("No chown commands")

    # Must enable the service
    if not any("enable" in line for line in ops["service_ops"]):
        errors.append("Missing systemctl enable")

    return errors


def validate_preremove(content: str) -> list[str]:
    """Validate preremove script has required operations.

    Returns list of error strings. Empty = valid.
    """
    errors: list[str] = []
    ops = extract_script_operations(content)

    if not content.startswith("#!/bin/bash"):
        errors.append("Missing bash shebang")

    if "set -e" not in content:
        errors.append("Missing 'set -e' (exit on error)")

    # Must stop the service
    if not any("stop" in line for line in ops["service_ops"]):
        errors.append("Missing systemctl stop")

    # Must disable the service
    if not any("disable" in line for line in ops["service_ops"]):
        errors.append("Missing systemctl disable")

    return errors


def check_script_syntax(script_path: Path) -> tuple[bool, str]:
    """Run bash -n to check shell script syntax.

    Returns (ok, error_message). ok=True if syntax is valid.
    """
    try:
        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0, result.stderr.strip()
    except FileNotFoundError:
        return False, "bash not found"
    except subprocess.TimeoutExpired:
        return False, "syntax check timed out"
