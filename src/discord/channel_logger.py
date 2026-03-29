"""Passive channel logger — writes ALL Discord messages to JSONL files.

Zero LLM tokens. Pure file I/O. One JSON line per message, appended to
``data/channel_logs/{channel_id}.jsonl``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..logging import get_logger

log = get_logger("channel_logger")


class ChannelLogger:
    """Append-only JSONL logger for Discord channel messages.

    Parameters
    ----------
    log_dir:
        Directory where per-channel JSONL files are stored.
        Created automatically if it does not exist.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        # Track last indexed timestamp per channel (in-memory, reset on restart)
        self._last_indexed_ts: dict[str, float] = {}

    def log_message(self, message: object) -> None:
        """Append a single message to the appropriate channel JSONL file.

        Skips DMs (no guild).  Tolerant of missing attributes so it never
        raises and never blocks the caller.
        """
        try:
            # Skip DMs — no guild means no channel log
            channel = getattr(message, "channel", None)
            if channel is None:
                return
            guild = getattr(channel, "guild", None)
            if guild is None:
                return

            channel_id = str(channel.id)
            author = getattr(message, "author", None)

            record = {
                "ts": message.created_at.timestamp() if hasattr(message, "created_at") and message.created_at else 0.0,
                "author_id": str(author.id) if author else "0",
                "author": str(getattr(author, "display_name", getattr(author, "name", "Unknown"))),
                "bot": bool(getattr(author, "bot", False)),
                "content": getattr(message, "content", "") or "",
                "attachments": [a.filename for a in getattr(message, "attachments", [])],
                "channel_id": channel_id,
                "guild_id": str(guild.id),
            }

            path = self._log_dir / f"{channel_id}.jsonl"
            # Ensure directory exists (handles deletion while running)
            self._log_dir.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, separators=(",", ":")) + "\n"
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            # Never let logging failures propagate — the message handler must not break
            log.debug("Failed to log channel message", exc_info=True)
