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
COMPACTION_MAX_CHARS = 800  # target max chars per compacted summary block

# Topic change detection constants
TOPIC_CHANGE_SCORE_THRESHOLD = 0.05  # below this overlap = topic change
TOPIC_CHANGE_TIME_GAP = 300  # 5 minutes in seconds
TOPIC_CHANGE_RECENT_WINDOW = 5  # check overlap against this many recent messages

# Relevance scoring constants
RELEVANCE_KEEP_RECENT = 5  # always include the most recent N messages
RELEVANCE_MIN_SCORE = 0.10  # minimum overlap score to include an older message
RELEVANCE_MAX_OLDER = 10  # max older messages to include beyond recent window

# Tool output summarization constants
TOOL_SUMMARY_THRESHOLD = 10  # summarize when this many tool calls occurred
TOOL_SUMMARY_MAX_CHARS = 500  # max chars for summarized tool response in history
CHAT_RESPONSE_MAX_CHARS = 1500  # max chars for text-only (no-tool) response in history

# Context budget constants
CONTEXT_TOKEN_BUDGET = 16000  # max estimated tokens for history sent to LLM
CHARS_PER_TOKEN = 4  # rough estimate: 1 token ≈ 4 chars
BUDGET_KEEP_RECENT = 5  # always keep the most recent N messages regardless of budget

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


def summarize_tool_response(
    response: str,
    tools_used: list[str],
    threshold: int = TOOL_SUMMARY_THRESHOLD,
) -> str:
    """Compress a verbose tool-loop response for history storage.

    When a request used *threshold* or more tool calls, the LLM's final
    response can be very long (describing each intermediate step).  This
    function extracts the key outcome and produces a compact summary that
    lists which tools were used and what the final result was.

    Returns the original response unchanged if fewer than *threshold*
    tools were used or the response is already short enough.
    """
    if len(tools_used) < threshold:
        return response
    if len(response) <= TOOL_SUMMARY_MAX_CHARS:
        return response

    # Deduplicate tools while preserving first-occurrence order
    seen: set[str] = set()
    unique_tools: list[str] = []
    for t in tools_used:
        if t not in seen:
            seen.add(t)
            unique_tools.append(t)

    tool_list = ", ".join(unique_tools[:15])  # cap display at 15 unique
    if len(unique_tools) > 15:
        tool_list += f" (+{len(unique_tools) - 15} more)"

    header = f"[Task used {len(tools_used)} tool calls ({tool_list})]\n"

    # Extract outcome: take the last paragraph or last few sentences
    # Split on double-newline for paragraphs, or single-newline for lines
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    if paragraphs:
        # Take the last paragraph as the outcome
        outcome = paragraphs[-1]
        # If there's a second-to-last that looks like a result summary, include it
        if len(paragraphs) >= 2 and len(outcome) < 100:
            outcome = paragraphs[-2] + "\n\n" + outcome
    else:
        # Single block — take the last 400 chars
        outcome = response[-400:]

    # Budget: header + outcome must fit in TOOL_SUMMARY_MAX_CHARS
    budget = TOOL_SUMMARY_MAX_CHARS - len(header)
    if len(outcome) > budget:
        # Reserve 3 chars for "..." prefix
        outcome = outcome[-(budget - 3):]
        # Clean up — don't start mid-word
        first_space = outcome.find(" ")
        if first_space > 0 and first_space < 50:
            outcome = "..." + outcome[first_space:]
        else:
            outcome = "..." + outcome

    result = header + outcome
    log.info(
        "Summarized tool response: %d chars → %d chars (%d tool calls)",
        len(response), len(result), len(tools_used),
    )
    return result


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length (~4 chars per token)."""
    return max(1, len(text) // CHARS_PER_TOKEN)


_SUMMARY_PREFIX = "[Previous conversation summary:"


def _content_text(m: dict) -> str:
    """Extract text from a message dict, handling non-string content."""
    c = m["content"]
    return c if isinstance(c, str) else str(c)


def apply_token_budget(
    messages: list[dict[str, str]],
    budget: int = CONTEXT_TOKEN_BUDGET,
) -> tuple[list[dict[str, str]], int]:
    """Trim message list to fit within a token budget.

    Drops oldest messages first, always keeping the most recent
    ``BUDGET_KEEP_RECENT`` messages.  Returns the trimmed list and the
    number of messages dropped.

    The summary pair (if present at the start) is protected — dropped
    last, only after all other non-recent messages are gone.
    """
    if not messages:
        return messages, 0

    # Calculate total tokens
    total = sum(estimate_tokens(_content_text(m)) for m in messages)
    if total <= budget:
        return messages, 0

    # Identify protected recent messages (tail)
    keep_n = min(BUDGET_KEEP_RECENT, len(messages))
    recent = messages[-keep_n:]
    older = messages[:-keep_n] if keep_n < len(messages) else []

    # Detect summary pair at the start of older messages
    has_summary = (
        len(older) >= 2
        and _content_text(older[0]).startswith(_SUMMARY_PREFIX)
    )
    summary_pair = older[:2] if has_summary else []
    droppable = older[2:] if has_summary else list(older)

    def _older_tokens() -> int:
        return sum(estimate_tokens(_content_text(m)) for m in summary_pair + droppable)

    recent_tokens = sum(estimate_tokens(_content_text(m)) for m in recent)

    # Drop oldest droppable (non-summary, non-recent) first
    dropped = 0
    while droppable and recent_tokens + _older_tokens() > budget:
        droppable.pop(0)
        dropped += 1

    # If still over budget, drop summary pair
    if summary_pair and recent_tokens + _older_tokens() > budget:
        summary_pair.clear()
        dropped += 2

    if dropped > 0:
        log.info(
            "Context budget: trimmed %d older message(s) to fit %d-token budget",
            dropped, budget,
        )

    return summary_pair + droppable + recent, dropped


@dataclass(slots=True)
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    user_id: str | None = None


@dataclass(slots=True)
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
        self._channel_logger: object | None = None
        self._fts_index: object | None = None

    def set_channel_search(self, channel_logger: object, fts_index: object | None = None) -> None:
        """Register channel logger and FTS index for search_history integration."""
        self._channel_logger = channel_logger
        self._fts_index = fts_index

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

    def detect_topic_change(
        self, channel_id: str, current_query: str,
    ) -> dict:
        """Detect whether the current query represents a topic change.

        Returns a dict with:
        - ``is_topic_change``: whether a topic change was detected
        - ``time_gap``: seconds since last message (0 if no history)
        - ``has_time_gap``: whether the time gap exceeds TOPIC_CHANGE_TIME_GAP
        - ``max_overlap``: highest relevance score against recent messages

        A topic change is detected when the keyword overlap between the current
        query and recent history messages is below TOPIC_CHANGE_SCORE_THRESHOLD.
        A time gap >5 min combined with low overlap strengthens the signal.
        """
        session = self._sessions.get(channel_id)
        if not session or not session.messages:
            return {
                "is_topic_change": False,
                "time_gap": 0.0,
                "has_time_gap": False,
                "max_overlap": 0.0,
            }

        # Time gap from last message
        last_msg = session.messages[-1]
        time_gap = time.time() - last_msg.timestamp

        # Score overlap against recent messages
        recent = session.messages[-TOPIC_CHANGE_RECENT_WINDOW:]
        scores = []
        for msg in recent:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            s = score_relevance(current_query, content)
            scores.append(s)

        max_overlap = max(scores) if scores else 0.0
        has_time_gap = time_gap > TOPIC_CHANGE_TIME_GAP

        # Topic change: low overlap with all recent messages AND a time gap.
        # Both conditions required — prevents false triggers on casual follow-ups
        # like "thanks" or "what about X" that have low keyword overlap but are
        # clearly part of the same conversation.
        is_topic_change = (
            max_overlap < TOPIC_CHANGE_SCORE_THRESHOLD
            and has_time_gap
            and len(recent) >= 2
        )

        if is_topic_change:
            log.info(
                "Topic change detected for channel %s (max_overlap=%.3f, time_gap=%.0fs)",
                channel_id, max_overlap, time_gap,
            )

        return {
            "is_topic_change": is_topic_change,
            "time_gap": time_gap,
            "has_time_gap": has_time_gap,
            "max_overlap": max_overlap,
        }

    async def get_task_history(
        self, channel_id: str, max_messages: int = 10,
        current_query: str | None = None,
        topic_change: bool = False,
    ) -> list[dict[str, str]]:
        """Get abbreviated history for the tool-calling path.

        Returns fewer messages than full history to reduce the influence of
        potentially stale or poisoned older exchanges. The summary (if any)
        still provides broader context.

        When *current_query* is provided, older messages (beyond the most
        recent ``RELEVANCE_KEEP_RECENT``) are scored for keyword relevance
        and only the most relevant ones are included.  This prevents stale
        context from unrelated earlier conversations from bleeding in.

        When *topic_change* is True, the history window is reduced to only
        the most recent message (the current one) — previous context is
        mostly irrelevant for a new topic.  The summary is still included
        for broad continuity.
        """
        session = self.get_or_create(channel_id)

        # Compact first if needed
        if len(session.messages) > COMPACTION_THRESHOLD:
            await self._compact(session)

        # On topic change, shrink to just the most recent message
        if topic_change:
            candidate_msgs = session.messages[-1:] if session.messages else []
            log.info(
                "Topic change: reduced history to %d message(s) for channel %s",
                len(candidate_msgs), channel_id,
            )
        else:
            # Take only the most recent messages as the candidate pool
            candidate_msgs = session.messages[-max_messages:]

        if current_query and not topic_change and len(candidate_msgs) > RELEVANCE_KEEP_RECENT:
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

        # Enforce token budget — drop oldest first, keep recent BUDGET_KEEP_RECENT
        messages, budget_dropped = apply_token_budget(messages)
        if budget_dropped > 0:
            log.info(
                "Token budget: dropped %d message(s) for channel %s",
                budget_dropped, channel_id,
            )

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
            "FORMAT:\n"
            "Line 1: [Topics: comma-separated topic tags, e.g. nginx, dns, server-a]\n"
            "Line 2+: Bullet points of key facts.\n\n"
            "RULES:\n"
            "1. PRESERVE VERBATIM: Hostnames, IPs, UUIDs, file paths, container names, "
            "service names, port numbers, usernames. Never paraphrase identifiers.\n"
            "2. PRESERVE: User preferences, decisions made, successful task outcomes "
            "(what tools accomplished and on which hosts), infrastructure state changes, "
            "and which tool names were used.\n"
            "3. OMIT: Intermediate steps, retries, tool iteration details, Error messages, "
            "API failures, 'I can\\'t' or 'unable to' statements, partial completion reports, "
            "and any response where the assistant could not do something.\n"
            "4. OMIT: Any data not confirmed by actual tool results.\n"
            "5. OMIT: Conversational filler, greetings, acknowledgments.\n"
            f"6. Keep the ENTIRE summary under {COMPACTION_MAX_CHARS} characters.\n"
            "7. Each bullet: WHAT happened → OUTCOME (host/path/service if applicable)."
        )

        try:
            if not self._compaction_fn:
                raise RuntimeError("No compaction backend configured")
            summary_text = await self._compaction_fn(
                [{"role": "user", "content": convo_text}],
                system_instruction,
            )
            summary_text = summary_text.strip()

            # Enforce max summary length — truncate at last complete line
            if len(summary_text) > COMPACTION_MAX_CHARS:
                truncated = summary_text[:COMPACTION_MAX_CHARS]
                last_newline = truncated.rfind("\n")
                if last_newline > 0:
                    truncated = truncated[:last_newline]
                else:
                    # No newline — fall back to last space to avoid mid-word cut
                    last_space = truncated.rfind(" ")
                    if last_space > 0:
                        truncated = truncated[:last_space]
                summary_text = truncated.rstrip()
                log.info(
                    "Compaction summary truncated to %d chars for channel %s",
                    len(summary_text), session.channel_id,
                )

            session.summary = summary_text

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

        # Step 4: channel log search (full channel history from all users)
        if len(results) < limit and self._channel_logger:
            try:
                remaining = limit - len(results)
                fts = self._fts_index
                channel_results = []
                if fts and hasattr(fts, "search_channel_logs"):
                    channel_results = fts.search_channel_logs(query, limit=remaining)
                if not channel_results and hasattr(self._channel_logger, "search"):
                    channel_results = await asyncio.to_thread(
                        self._channel_logger.search, query, remaining,
                    )
                # De-duplicate against existing results
                seen = {(r.get("channel_id", ""), r.get("timestamp", 0)) for r in results}
                for cr in channel_results:
                    key = (cr.get("channel_id", ""), cr.get("timestamp", 0))
                    if key not in seen:
                        results.append(cr)
                        seen.add(key)
                        if len(results) >= limit:
                            break
            except Exception as e:
                log.warning("Channel log search failed: %s", e)

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
