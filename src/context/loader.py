from __future__ import annotations

import re
from pathlib import Path

from ..logging import get_logger

log = get_logger("context")

SECRET_PATTERNS = [
    re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(secret|token)\s*[:=]\s*['\"]?\S{16,}"),
    re.compile(r"(?i)(access_token|auth_token)\s*[:=]\s*\S+"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"(?i)BEGIN\s+(RSA|EC|OPENSSH)\s+PRIVATE\s+KEY"),
]


class ContextLoader:
    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self._context: str = ""

    def load(self) -> str:
        if not self.directory.is_dir():
            log.warning("Context directory %s does not exist", self.directory)
            self._context = ""
            return self._context

        parts: list[str] = []
        for md_file in sorted(self.directory.glob("*.md")):
            content = md_file.read_text()
            self._scan_secrets(md_file.name, content)
            parts.append(f"# {md_file.stem}\n\n{content}")

        self._context = "\n\n---\n\n".join(parts)
        log.info(
            "Loaded %d context files (%d chars)",
            len(parts),
            len(self._context),
        )
        return self._context

    def reload(self) -> str:
        log.info("Reloading context files")
        return self.load()

    @property
    def context(self) -> str:
        return self._context

    def _scan_secrets(self, filename: str, content: str) -> None:
        for pattern in SECRET_PATTERNS:
            matches = pattern.findall(content)
            if matches:
                log.warning(
                    "SECURITY: Potential secret detected in %s "
                    "(pattern: %s, %d match(es)). "
                    "Remove credentials from context files!",
                    filename,
                    pattern.pattern[:40],
                    len(matches),
                )
