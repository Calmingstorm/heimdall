"""Secret scrubbing for tool output and LLM responses.

Provides pattern-based detection and redaction of sensitive values
(passwords, API keys, tokens, private keys, database URIs) to prevent
them from leaking through LLM responses or tool output.
"""
from __future__ import annotations

import re

# Patterns to scrub from tool output before it reaches the LLM
OUTPUT_SECRET_PATTERNS = [
    re.compile(r"(password|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"(api[_-]?key|apikey)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"(secret|token|access_token|auth_token)\s*[:=]\s*['\"]?\S{16,}", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"BEGIN\s+(RSA|EC|OPENSSH)\s+PRIVATE\s+KEY", re.IGNORECASE),
    re.compile(r"(mysql|postgres|mongodb(\+srv)?)://\S+:\S+@", re.IGNORECASE),
]


def scrub_output_secrets(text: str) -> str:
    """Replace detected secrets in tool output with [REDACTED]."""
    for pattern in OUTPUT_SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text
