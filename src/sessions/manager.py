from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING

from ..logging import get_logger
if TYPE_CHECKING:
    from ..learning.reflector import ConversationReflector
    from ..search.embedder import LocalEmbedder
    from ..search.vectorstore import SessionVectorStore

# Type alias for the compaction callable:
#   async (messages: list[dict], system: str) -> str
CompactionFn = Callable[[list[dict], str], Awaitable[str]]

log = get_logger("sessions")
COMPACTION_THRESHOLD = 40  # compact when history exceeds this
CONTINUITY_MAX_AGE = 48 * 3600  # carry forward summaries from archives < 48 hours old

# Relevance scoring constants
RELEVANCE_KEEP_RECENT = 3  # always include the most recent N messages
RELEVANCE_MIN_SCORE = 0.15  # minimum overlap score to include an older message
RELEVANCE_MAX_OLDER = 7  # max older messages to include beyond recent window

# Common stop words to ignore when scoring relevance
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "to", "of", "and", "or",
    "for", "that", "this", "with", "was", "are", "be", "has", "have",
    "had", "do", "does", "did", "but", "not", "you", "i", "me", "my",
    "we", "he", "she", "they", "what", "how", "can", "will", "just",
    "so", "if", "no", "yes", "at", "by", "from", "up", "out", "as",
})

_TOKEN_RE = re.compile(r"[a-z0-9_./:-]+")


def _tokenize(text: str) -> set[str]:
    """Extract meaningful lowercase tokens from text, filtering stop words."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS and len(t) > 1}


def score_relevance(query: str, message_content: str) -> float:
    """Score how relevant a message is to the current query.

    Returns a float between 0.0 and 1.0 based on keyword overlap.
    Uses Jaccard-like scoring: |intersection| / |query_tokens|.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    msg_tokens = _tokenize(message_content)
    if not msg_tokens:
        return 0.0
    overlap = query_tokens & msg_tokens
    return len(overlap) / len(query_tokens)


@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    user_id: str | None = None


@dataclass
class Session:
    channel_id: str
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    summary: str = ""  # compacted summary of older messages
    last_user_id: str | None = None  # Discord user ID of most recent human message


class SessionManager:
    def __init__(
        self,
        max_history: int,
        max_age_hours: int,
        persist_dir: str,
        reflector: ConversationReflector | None = None,
        vector_store: SessionVectorStore | None = None,
        embedder: LocalEmbedder | None = None,
    ) -> None:
        self.max_history = max_history
        self.max_age_seconds = max_age_hours * 3600
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._dirty: set[str] = set()
        self._reflector = reflector
        self._reflection_tasks: set[asyncio.Task] = set()
        self._vector_store = vector_store
        self._embedder = embedder
        self._indexing_tasks: set[asyncio.Task] = set()
        self._compaction_fn: CompactionFn | None = None

    def set_compaction_fn(self, fn: CompactionFn) -> None:
        """Register an async callable for LLM-based compaction.

        The callable signature is ``async (messages, system) -> str`` where
        *messages* is a single-element list ``[{"role": "user", "content": ...}]``
        and *system* is the summarisation instruction.
        """
        self._compaction_fn = fn

    def get_or_create(self, channel_id: str) -> Session:
        if channel_id not in self._sessions:
            session = Session(channel_id=channel_id)
            summary = self._find_recent_summary(channel_id)
            if summary:
                session.summary = f"[Continuing from previous conversation] {summary}"
                log.info("Carried forward summary for channel %s", channel_id)
            self._sessions[channel_id] = session
            self._dirty.add(channel_id)
        session = self._sessions[channel_id]
        session.last_active = time.time()
        return session

    def _find_recent_summary(self, channel_id: str) -> str:
        """Find the most recent archived summary for a channel within the continuity window."""
        archive_dir = self.persist_dir / "archive"
        if not archive_dir.exists():
            return ""
        now = time.time()
        best_summary = ""
        best_time = 0.0
        for path in archive_dir.glob(f"{channel_id}_*.json"):
            try:
                data = json.loads(path.read_text())
                last_active = data.get("last_active", 0)
                summary = data.get("summary", "")
                if summary and now - last_active < CONTINUITY_MAX_AGE and last_active > best_time:
                    best_summary = summary
                    best_time = last_active
            except Exception:
                continue
        return best_summary

    def add_message(
        self, channel_id: str, role: str, content: str,
        *, user_id: str | None = None,
    ) -> None:
        session = self.get_or_create(channel_id)
        session.messages.append(Message(
            role=role, content=content,
            user_id=user_id if role == "user" else None,
        ))
        if role == "user" and user_id:
            session.last_user_id = user_id
        self._dirty.add(channel_id)

    def remove_last_message(self, channel_id: str, role: str) -> bool:
        """Remove the most recent message if it matches *role*.

        Used to clean up orphaned user messages when processing fails
        (e.g. API errors, budget exceeded) so they don't persist in
        history and waste tokens on subsequent requests.
        """
        session = self._sessions.get(channel_id)
        if not session or not session.messages:
            return False
        if session.messages[-1].role == role:
            session.messages.pop()
            self._dirty.add(channel_id)
            return True
        return False

    def get_history(self, channel_id: str) -> list[dict[str, str]]:
        session = self.get_or_create(channel_id)
        messages = [{"role": m.role, "content": m.content} for m in session.messages]

        # Prepend summary if we have one
        if session.summary:
            messages.insert(0, {
                "role": "user",
                "content": f"[Previous conversation summary: {session.summary}]",
            })
            messages.insert(1, {
                "role": "assistant",
                "content": "Understood, I have context from our previous conversation.",
            })

        return messages

    async def get_history_with_compaction(
        self, channel_id: str,
    ) -> list[dict[str, str]]:
        """Get history, compacting old messages if threshold is exceeded.

        A ``compaction_fn`` must be registered via :meth:`set_compaction_fn`
        before compaction can run.
        """
        session = self.get_or_create(channel_id)

        if len(session.messages) > COMPACTION_THRESHOLD:
            await self._compact(session)

        return self.get_history(channel_id)

    async def get_task_history(
        self, channel_id: str, max_messages: int = 10,
        current_query: str | None = None,
    ) -> list[dict[str, str]]:
        """Get abbreviated history for the tool-calling path.

        Returns fewer messages than full history to reduce the influence of
        potentially stale or poisoned older exchanges. The summary (if any)
        still provides broader context.

        When *current_query* is provided, older messages (beyond the most
        recent ``RELEVANCE_KEEP_RECENT``) are scored for keyword relevance
        and only the most relevant ones are included.  This prevents stale
        context from unrelated earlier conversations from bleeding in.
        """
        session = self.get_or_create(channel_id)

        # Compact first if needed
        if len(session.messages) > COMPACTION_THRESHOLD:
            await self._compact(session)

        # Take only the most recent messages as the candidate pool
        candidate_msgs = session.messages[-max_messages:]

        if current_query and len(candidate_msgs) > RELEVANCE_KEEP_RECENT:
            # Always include the most recent messages unconditionally
            recent = candidate_msgs[-RELEVANCE_KEEP_RECENT:]
            older = candidate_msgs[:-RELEVANCE_KEEP_RECENT]

            # Score older messages for relevance
            scored: list[tuple[float, Message]] = []
            for msg in older:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                s = score_relevance(current_query, content)
                scored.append((s, msg))

            # Keep messages above the minimum score threshold, up to the cap
            relevant = [(s, m) for s, m in scored if s >= RELEVANCE_MIN_SCORE]
            relevant.sort(key=lambda x: x[0], reverse=True)
            relevant = relevant[:RELEVANCE_MAX_OLDER]

            dropped = len(older) - len(relevant)
            if dropped > 0:
                log.info(
                    "Relevance filter: dropped %d/%d older messages for channel %s",
                    dropped, len(older), channel_id,
                )

            # Reconstruct in original order: relevant older + recent
            # Preserve original ordering among the kept older messages
            kept_set = {id(m) for _, m in relevant}
            filtered = [m for m in older if id(m) in kept_set] + list(recent)
        else:
            filtered = list(candidate_msgs)

        messages = [{"role": m.role, "content": m.content} for m in filtered]

        # Prepend summary if available
        if session.summary:
            messages.insert(0, {
                "role": "user",
                "content": f"[Previous conversation summary: {session.summary}]",
            })
            messages.insert(1, {
                "role": "assistant",
                "content": "Understood, I have context from our previous conversation.",
            })

        return messages

    async def _compact(self, session: Session) -> None:
        """Summarize older messages and keep only recent ones."""
        # Keep the most recent messages, summarize the rest
        keep_count = self.max_history // 2
        to_summarize = session.messages[:-keep_count]
        to_keep = session.messages[-keep_count:]

        if not to_summarize:
            return

        # Build conversation text for summarization
        convo_text = "\n".join(
            f"{m.role}: {m.content[:500]}" for m in to_summarize
        )

        # If there's an existing summary, include it so the LLM merges
        # everything into one concise summary instead of concatenating
        if session.summary:
            convo_text = (
                f"[Previous summary]: {session.summary[:1000]}\n\n"
                f"[New messages to incorporate]:\n{convo_text}"
            )

        system_instruction = (
            "Summarize the following conversation into a concise context summary. "
            "If a previous summary is provided, merge it with the new messages.\n\n"
            "RULES:\n"
            "1. PRESERVE: User preferences, names, topics discussed, decisions made, "
            "successful task outcomes (what tools accomplished and on which hosts), "
            "infrastructure state changes, and which tool names were used.\n"
            "2. OMIT: Error messages, API failures, 'I can\\'t' or 'unable to' statements, "
            "partial completion reports, and any response where the assistant said it could not do something.\n"
            "3. OMIT: Any data not confirmed by actual tool results.\n"
            "4. Keep under 300 words.\n"
            "5. Start with a one-line topic summary, then bullet key facts."
        )

        try:
            if not self._compaction_fn:
                raise RuntimeError("No compaction backend configured")
            summary_text = await self._compaction_fn(
                [{"role": "user", "content": convo_text}],
                system_instruction,
            )
            session.summary = summary_text.strip()

            # Trigger reflection on discarded messages before replacing
            discarded = list(to_summarize)
            summary_snapshot = session.summary

            session.messages = to_keep
            self._dirty.add(session.channel_id)
            log.info(
                "Compacted %d messages into summary for channel %s",
                len(to_summarize),
                session.channel_id,
            )

            if self._reflector and len(discarded) >= 5:
                # Collect all distinct user_ids from discarded messages
                participant_ids = list(dict.fromkeys(
                    m.user_id for m in discarded if m.user_id
                ))
                task = asyncio.create_task(
                    self._safe_reflect_compacted(
                        discarded, summary_snapshot,
                        user_ids=participant_ids,
                    )
                )
                self._reflection_tasks.add(task)
                task.add_done_callback(self._reflection_tasks.discard)
        except Exception as e:
            log.error("Failed to compact session: %s", e)
            # Fallback: just trim without LLM summary.  Preserve the
            # existing summary — it describes older context that is still
            # relevant even though the detailed messages are gone.
            session.messages = session.messages[-self.max_history:]
            self._dirty.add(session.channel_id)

    def reset(self, channel_id: str) -> None:
        self._sessions.pop(channel_id, None)
        log.info("Session reset for channel %s", channel_id)

    def prune(self) -> int:
        now = time.time()
        expired = [
            cid
            for cid, s in self._sessions.items()
            if now - s.last_active > self.max_age_seconds
        ]
        for cid in expired:
            self._archive_session(cid)
            # Delete the session file so it won't be reloaded on next startup.
            # Data is preserved in the archive directory.
            session_file = self.persist_dir / f"{cid}.json"
            if session_file.exists():
                session_file.unlink()
            del self._sessions[cid]
        if expired:
            log.info("Pruned and archived %d expired sessions", len(expired))
        return len(expired)

    def _archive_session(self, channel_id: str) -> None:
        """Save a session to the archive before pruning."""
        session = self._sessions.get(channel_id)
        if not session or not session.messages:
            return
        archive_dir = self.persist_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        timestamp = int(session.last_active)
        path = archive_dir / f"{channel_id}_{timestamp}.json"
        data = asdict(session)
        path.write_text(json.dumps(data, indent=2))
        log.info("Archived session %s (%d messages)", channel_id, len(session.messages))

        # Trigger full reflection on the completed session
        if self._reflector and len(session.messages) >= 3:
            session_copy = copy.deepcopy(session)
            # Collect all distinct user_ids from session messages
            participant_ids = list(dict.fromkeys(
                m.user_id for m in session.messages if m.user_id
            ))
            task = asyncio.create_task(
                self._safe_reflect(session_copy, user_ids=participant_ids)
            )
            self._reflection_tasks.add(task)
            task.add_done_callback(self._reflection_tasks.discard)

        # Index for semantic + FTS search
        if self._vector_store and self._vector_store.available:
            task = asyncio.create_task(self._safe_index(path))
            self._indexing_tasks.add(task)
            task.add_done_callback(self._indexing_tasks.discard)

    async def _safe_index(self, archive_path: Path) -> None:
        """Index an archived session for semantic search, catching all errors."""
        try:
            await self._vector_store.index_session(archive_path, self._embedder)
        except Exception as e:
            log.error("Session indexing failed for %s: %s", archive_path, e)

    def _search_archives(self, query_lower: str, limit: int) -> list[dict]:
        """Search archived session files for keyword matches (sync, for use in thread)."""
        results: list[dict] = []
        archive_dir = self.persist_dir / "archive"
        if not archive_dir.exists():
            return results
        for path in sorted(archive_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text())
                summary = data.get("summary", "")
                if summary and query_lower in summary.lower():
                    results.append({
                        "type": "summary",
                        "content": summary[:500],
                        "timestamp": data.get("last_active", 0),
                        "channel_id": data.get("channel_id", "unknown"),
                    })
                for msg in reversed(data.get("messages", [])):
                    content = msg.get("content", "")
                    if query_lower in content.lower():
                        results.append({
                            "type": msg["role"],
                            "content": content[:500],
                            "timestamp": msg.get("timestamp", 0),
                            "channel_id": data.get("channel_id", "unknown"),
                        })
                        if len(results) >= limit:
                            return results
            except Exception:
                continue
        return results

    async def search_history(self, query: str, limit: int = 10) -> list[dict]:
        """Search current and archived sessions for matching messages."""
        query_lower = query.lower()
        results = []

        # Step 1: keyword search on current sessions
        for session in self._sessions.values():
            if session.summary and query_lower in session.summary.lower():
                results.append({
                    "type": "summary",
                    "content": session.summary[:500],
                    "timestamp": session.last_active,
                    "channel_id": session.channel_id,
                })
            for msg in reversed(session.messages):
                if query_lower in msg.content.lower():
                    results.append({
                        "type": msg.role,
                        "content": msg.content[:500],
                        "timestamp": msg.timestamp,
                        "channel_id": session.channel_id,
                    })
                    if len(results) >= limit:
                        return results

        # Step 2: keyword search on archives (most recent first)
        # Run in thread to avoid blocking the event loop with file I/O
        archive_results = await asyncio.to_thread(
            self._search_archives, query_lower, limit - len(results),
        )
        results.extend(archive_results)
        if len(results) >= limit:
            return results[:limit]

        # Step 3: hybrid search (FTS5 + semantic) fills remaining slots
        if len(results) < limit and self._vector_store:
            try:
                hybrid_results = await self._vector_store.search_hybrid(
                    query, self._embedder, limit=limit,
                )
                # De-duplicate by (channel_id, timestamp)
                seen = {(r["channel_id"], r["timestamp"]) for r in results}
                for hr in hybrid_results:
                    key = (hr["channel_id"], hr["timestamp"])
                    if key not in seen:
                        results.append(hr)
                        seen.add(key)
                        if len(results) >= limit:
                            break
            except Exception as e:
                log.warning("Hybrid search failed, returning keyword-only results: %s", e)

        return results

    def scrub_secrets(self, channel_id: str, content: str) -> bool:
        """Remove a message containing secrets from history."""
        session = self._sessions.get(channel_id)
        if not session:
            return False
        before = len(session.messages)
        session.messages = [
            m for m in session.messages if content not in m.content
        ]
        removed = before - len(session.messages)
        if removed:
            self._dirty.add(channel_id)
            log.warning(
                "Scrubbed %d message(s) containing secrets from channel %s",
                removed,
                channel_id,
            )
        return removed > 0

    async def _safe_reflect(
        self, session: Session,
        user_ids: list[str] | None = None,
    ) -> None:
        """Reflect on a completed session, catching all errors."""
        try:
            await self._reflector.reflect_on_session(
                session, user_ids=user_ids or ([session.last_user_id] if session.last_user_id else []),
            )
        except Exception as e:
            log.error("Session reflection failed: %s", e)

    async def _safe_reflect_compacted(
        self, messages: list[Message], summary: str,
        user_ids: list[str] | None = None,
    ) -> None:
        """Reflect on compacted messages, catching all errors."""
        try:
            await self._reflector.reflect_on_compacted(
                messages, summary, user_ids=user_ids or [],
            )
        except Exception as e:
            log.error("Compaction reflection failed: %s", e)

    def save(self) -> None:
        """Persist only sessions that changed since the last save."""
        for cid in self._dirty:
            session = self._sessions.get(cid)
            if session is None:
                continue
            path = self.persist_dir / f"{cid}.json"
            data = asdict(session)
            path.write_text(json.dumps(data, indent=2))
        self._dirty.clear()

    def save_all(self) -> None:
        """Persist every session (used during shutdown)."""
        for cid, session in self._sessions.items():
            path = self.persist_dir / f"{cid}.json"
            data = asdict(session)
            path.write_text(json.dumps(data, indent=2))
        self._dirty.clear()

    def load(self) -> None:
        for path in self.persist_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                messages = [Message(**m) for m in data.get("messages", [])]
                self._sessions[data["channel_id"]] = Session(
                    channel_id=data["channel_id"],
                    messages=messages,
                    created_at=data.get("created_at", time.time()),
                    last_active=data.get("last_active", time.time()),
                    summary=data.get("summary", ""),
                    last_user_id=data.get("last_user_id"),
                )
            except Exception as e:
                log.error("Failed to load session from %s: %s", path, e)
