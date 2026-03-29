"""Passive channel logger — writes ALL Discord messages to JSONL files.

Zero LLM tokens. Pure file I/O. One JSON line per message, appended to
``data/channel_logs/{channel_id}.jsonl``.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from ..logging import get_logger

if TYPE_CHECKING:
    from ..search.fts import FullTextIndex

log = get_logger("channel_logger")


class ChannelLogger:
    """Append-only JSONL logger for Discord channel messages.

    Parameters
    ----------
    log_dir:
        Directory where per-channel JSONL files are stored.
        Created automatically if it does not exist.
    """

    # Batch size cap for FTS indexing to limit memory on huge JSONL files
    FTS_BATCH_LIMIT = 5000

    def __init__(self, log_dir: str | Path) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._dir_exists = True  # track dir state to avoid per-message stat()
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
            line = json.dumps(record, separators=(",", ":")) + "\n"
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
            except FileNotFoundError:
                # Directory was deleted while running — recreate and retry once
                self._log_dir.mkdir(parents=True, exist_ok=True)
                self._dir_exists = True
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            # Never let logging failures propagate — the message handler must not break
            log.debug("Failed to log channel message", exc_info=True)

    def index_to_fts(self, fts: FullTextIndex) -> int:
        """Batch-index new messages into the FTS5 channel_log_fts table.

        Reads JSONL files, finds lines newer than the last indexed timestamp
        per channel, and inserts them into FTS. Returns total rows indexed.

        On first call (no channels indexed yet), clears the FTS table to
        prevent duplicates after a restart.
        """
        if not fts or not fts.available:
            return 0
        total = 0
        try:
            if not self._log_dir.exists():
                return 0
            # On fresh start, clear stale FTS data to prevent duplicates
            if not self._last_indexed_ts:
                fts.clear_channel_logs()
            for path in self._log_dir.glob("*.jsonl"):
                channel_id = path.stem
                cutoff = self._last_indexed_ts.get(channel_id, 0.0)
                batch: list[dict] = []
                max_ts = cutoff
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            ts = record.get("ts", 0.0)
                            if ts > cutoff:
                                batch.append(record)
                                if ts > max_ts:
                                    max_ts = ts
                                if len(batch) >= self.FTS_BATCH_LIMIT:
                                    break  # cap memory; remainder indexed next cycle
                except Exception:
                    log.debug("Failed to read %s for indexing", path, exc_info=True)
                    continue
                if batch:
                    indexed = fts.index_channel_messages(batch)
                    total += indexed
                    if max_ts > cutoff:
                        self._last_indexed_ts[channel_id] = max_ts
        except Exception:
            log.debug("FTS channel log indexing failed", exc_info=True)
        if total:
            log.info("Indexed %d channel log messages into FTS", total)
        return total

    def search(self, query: str, limit: int = 20, channel_id: str | None = None) -> list[dict]:
        """Keyword search on JSONL files (fallback when FTS is unavailable).

        Returns dicts with content, author, channel_id, timestamp, type="channel".
        Reads files in reverse (newest messages first) for better relevance.
        """
        results: list[dict] = []
        query_lower = query.lower()
        if not query_lower:
            return results
        try:
            if not self._log_dir.exists():
                return results
            for path in self._log_dir.glob("*.jsonl"):
                if channel_id and path.stem != channel_id:
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        # Use deque to read lines newest-first without loading all into memory
                        lines = deque(f, maxlen=50000)  # cap at 50K most recent lines
                    for line in reversed(lines):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content = record.get("content", "")
                        if query_lower in content.lower():
                            results.append({
                                "content": content[:500],
                                "author": record.get("author", "Unknown"),
                                "channel_id": record.get("channel_id", ""),
                                "timestamp": record.get("ts", 0.0),
                                "type": "channel",
                            })
                            if len(results) >= limit:
                                return results
                except Exception:
                    continue
        except Exception:
            log.debug("Channel log keyword search failed", exc_info=True)
        return results
