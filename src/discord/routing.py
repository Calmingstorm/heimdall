"""Message routing helpers — lightweight, no heavy dependencies."""
from __future__ import annotations

import re

# Word-boundary regex patterns for task keyword pre-check.
# These bypass the classifier to save a round-trip for obvious tasks.
# Single ambiguous words (check, log, memory, service, find, run, status)
# are omitted — the classifier handles those.
# Multi-word phrases that are unambiguous in context ARE included.
_TASK_KEYWORD_PATTERNS = re.compile(
    r"|".join([
        # Infrastructure-specific (never appear in casual chat)
        r"\brestart\b", r"\bdeploy\b", r"\bansible\b", r"\bssh\b",
        r"\bplaybook\b", r"\bdocker\b", r"\bprometheus\b", r"\bgrafana\b",
        r"\bsiglos\b", r"\bcpu\b", r"\bdisk\b", r"\bincus\b",
        r"\bjournalctl\b", r"\bsystemctl\b",
        # Multi-word infra phrases (unambiguous even though individual words are not)
        r"\bcheck disk\b", r"\bcheck memory\b", r"\bcheck service\b",
        r"\bcheck docker\b", r"\bcheck logs?\b", r"\bcheck status\b",
        r"\bdocker logs?\b", r"\bdocker stats\b", r"\bdocker compose\b",
        r"\bgit pull\b", r"\bgit diff\b", r"\bgit log\b", r"\bgit status\b",
        # File operations
        r"\bwrite file\b", r"\bread file\b", r"\bcreate file\b",
        # Bot operations
        r"\bclear chat\b", r"\bsave note\b", r"\bpurge\b", r"\bwipe\b",
        # "remember" — only match command-like forms, not casual reminiscence
        r"\bremember\s+(?:this|that)\b",
        r"\brecall\b", r"\bforget\b",
        # Action directives (re-execute previous action)
        r"\btry again\b", r"\bgo ahead\b", r"\bdo it\b", r"\bproceed\b", r"\bretry\b",
        # Search / lookup — require multi-word to avoid "I did some search"
        r"\bweb\s+search\b", r"\bsearch\s+for\b", r"\bsearch\s+(?:the|my|this|about)\b",
        r"\blook up\b",
        # News / current events — require multi-word to avoid "that's not news"
        r"\bnews\s+(?:about|on|from|today|for)\b",
        r"\bwhat.{0,10}news\b", r"\bany\s+news\b", r"\blatest\s+news\b",
        r"\bheadline\b", r"\bcurrent events?\b",
        # Audit / digest / skill
        r"\baudit\b", r"\bdigest\b",
    ]),
    re.IGNORECASE,
)


def is_task_by_keyword(content: str) -> bool:
    """Fast pre-check: return True if the message contains unambiguous task keywords.

    Uses word-boundary regex to avoid false positives from substring matching.
    Single ambiguous words are left to the classifier.
    Multi-word infrastructure phrases are safe to bypass on.
    """
    return bool(_TASK_KEYWORD_PATTERNS.search(content))


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

# Default targets for claude -p on each host.
# Users should override these via config or context files.
CLAUDE_CODE_DEFAULTS = {
    "desktop": "/root/project",
    "server": "/opt/project",
}


def resolve_claude_code_target(message: str) -> tuple[str, str]:
    """Determine host and working directory for claude -p routing.

    Routes to the first configured host by default.
    Routes to server when message indicates production/server-specific analysis.

    Returns (host, working_directory) tuple.
    """
    if _SERVER_INDICATORS.search(message):
        return ("server", CLAUDE_CODE_DEFAULTS["server"])
    return ("desktop", CLAUDE_CODE_DEFAULTS["desktop"])
