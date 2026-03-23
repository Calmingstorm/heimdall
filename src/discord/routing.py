"""Message routing helpers — lightweight, no heavy dependencies."""
from __future__ import annotations

import re

# --- Claude Code host/directory routing ---
# Routes to the first configured host with claude CLI by default.
# Route to a secondary host when message indicates production/server-specific files.
_SERVER_INDICATORS = re.compile(
    r"|".join([
        # Explicit "on server" / "on the server"
        r"\bon\s+(?:the\s+)?server\b",
        # "server config/logs/files/setup/version"
        r"\bserver\s+(?:config|logs?|files?|setup|version)\b",
        # Server-specific absolute paths
        r"/opt/",
        # Server-hosted service + file/config analysis context (either order)
        r"\b(?:grafana|prometheus|loki|gitea|nginx)\s+(?:config|conf|setup|rules?|dashboard)\b",
        r"\b(?:config|conf|setup)\s+(?:for\s+)?(?:grafana|prometheus|loki|gitea|nginx)\b",
        # Explicit production/deployed references
        r"\b(?:production|deployed|running)\s+(?:config|version|code|files?)\b",
    ]),
    re.IGNORECASE,
)

# Default targets for claude -p routing.
# "primary" is the default host; "secondary" is used for server/production context.
# Populated from config at startup by LokiBot._init_routing_defaults().
CLAUDE_CODE_DEFAULTS: dict[str, tuple[str, str]] = {
    "primary": ("", "/opt/project"),
    "secondary": ("", "/opt/project"),
}


def resolve_claude_code_target(message: str) -> tuple[str, str]:
    """Determine host and working directory for claude -p routing.

    Routes to the primary host by default.
    Routes to secondary when message indicates production/server-specific analysis.

    Returns (host, working_directory) tuple.
    """
    if _SERVER_INDICATORS.search(message):
        return CLAUDE_CODE_DEFAULTS["secondary"]
    return CLAUDE_CODE_DEFAULTS["primary"]
