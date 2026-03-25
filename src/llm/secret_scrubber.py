"""Secret scrubbing for tool output and LLM responses.

Provides pattern-based detection and redaction of sensitive values
(passwords, API keys, tokens, private keys, database URIs) to prevent
them from leaking through LLM responses or tool output.
"""
from __future__ import annotations

import re

# Patterns to scrub from tool output before it reaches the LLM
OUTPUT_SECRET_PATTERNS = [
    re.compile(r"(?<![a-zA-Z])(password|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"(api[_-]?key|apikey)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"(secret|token|access_token|auth_token)\s*[:=]\s*['\"]?\S{16,}", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"BEGIN\s+(RSA|EC|OPENSSH|DSA)?\s*PRIVATE\s+KEY", re.IGNORECASE),
    re.compile(r"(mysql|postgres|mongodb(\+srv)?)://\S+:\S+@", re.IGNORECASE),
    # GitHub tokens (ghp_, gho_, ghu_, ghs_, ghr_)
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}"),
    # AWS access key IDs
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # Stripe live/test secret keys
    re.compile(r"[sr]k_(live|test)_[A-Za-z0-9]{20,}"),
    # Slack tokens (xoxb, xoxp, xoxa, xoxo, xoxr, xoxs)
    re.compile(r"xox[boaprs]-[a-zA-Z0-9-]+"),
]


def scrub_output_secrets(text: str) -> str:
    """Replace detected secrets in tool output with [REDACTED]."""
    for pattern in OUTPUT_SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text
