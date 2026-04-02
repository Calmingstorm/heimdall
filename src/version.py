"""Version management for Heimdall.

Reads version from package metadata (installed via pip) with fallback
to parsing pyproject.toml directly (development / uninstalled mode).
"""
from __future__ import annotations

import re
from pathlib import Path

_FALLBACK_VERSION = "0.0.0-dev"


def get_version() -> str:
    """Return the Heimdall version string.

    Resolution order:
    1. importlib.metadata (works when installed via pip / .deb)
    2. Parse pyproject.toml in the project root (development mode)
    3. Fallback to "0.0.0-dev"
    """
    # Try installed package metadata first
    try:
        from importlib.metadata import version

        return version("heimdall")
    except Exception:
        pass

    # Fallback: parse pyproject.toml directly
    try:
        toml_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if toml_path.is_file():
            text = toml_path.read_text(encoding="utf-8")
            match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
            if match:
                return match.group(1)
    except Exception:
        pass

    return _FALLBACK_VERSION
