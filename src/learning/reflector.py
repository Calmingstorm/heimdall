from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ..logging import get_logger

if TYPE_CHECKING:
    from ..sessions.manager import Message, Session

# Type alias: async (messages: list[dict], system: str) -> str
TextFn = Callable[[list[dict], str], Awaitable[str]]

log = get_logger("learning")

_REFLECTION_HEADER = """\
Extract clear, explicit lessons from this conversation. Return a JSON array.
Each element: {"key": "snake_case_id", "category": "correction|preference|operational|fact", "content": "max 150 chars"}
Rules:
- Max 5 insights per reflection
- Reuse existing keys when updating a known fact
- Return [] if nothing worth learning
- Categories: correction (user corrected the bot), preference (how user likes things done), operational (system/infra knowledge), fact (general truth)
Anti-hallucination rules:
- ONLY record preferences the user EXPLICITLY stated — never infer unstated preferences
- Never generalize a specific correction into a broad prohibition (e.g. user corrects a refusal to discuss earthquakes → record "do not refuse news requests", NOT "avoid political topics")
- If a user corrects the bot's refusal to do something, the lesson is "do not refuse [specific thing]" — never "avoid [broad topic]"
- Never invent behavioral rules the user did not ask for
- When in doubt, return [] — a missed insight is better than a hallucinated one"""

_CONSOLIDATION_HEADER = """\
Consolidate these learned entries to """


class ConversationReflector:
    """Reviews conversations after they end and extracts reusable insights."""

    def __init__(
        self,
        learned_path: str,
        *,
        max_entries: int = 30,
        consolidation_target: int = 20,
        enabled: bool = True,
    ) -> None:
        self._path = Path(learned_path)
        self._lock = asyncio.Lock()
        self._max_entries = max_entries
        self._consolidation_target = consolidation_target
        self._enabled = enabled
        self._text_fn: TextFn | None = None

    def set_text_fn(self, fn: TextFn) -> None:
        """Register an async callable for LLM text generation.

        The callable signature is ``async (messages, system) -> str``.
        When set, ``_reflect()`` and ``_consolidate()`` use this instead
        of direct API calls.
        """
        self._text_fn = fn

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                log.error("Failed to load learned.json: %s", e)
        return {"version": 1, "last_reflection": None, "entries": []}

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2))

    def get_prompt_section(self, user_id: str | None = None) -> str:
        """Format learned entries for injection into the system prompt.

        If *user_id* is provided, includes global entries (no user_id) plus
        entries tagged for that specific user.  Entries tagged for *other*
        users are excluded.
        """
        data = self._load()
        entries = data.get("entries", [])
        if not entries:
            return ""
        filtered = []
        for e in entries:
            entry_uid = e.get("user_id")
            if entry_uid is None:
                # Global entry — always include
                filtered.append(e)
            elif user_id and entry_uid == user_id:
                # Entry belongs to the requesting user
                filtered.append(e)
            # else: entry belongs to another user — skip
        if not filtered:
            return ""
        lines = [f"- [{e['category']}] {e['content']}" for e in filtered]
        return "## Learned Context\n" + "\n".join(lines)

    async def reflect_on_session(
        self, session: Session, *, user_id: str | None = None,
        user_ids: list[str] | None = None,
    ) -> None:
        """Full reflection on a completed session."""
        if not self._enabled:
            return
        messages = session.messages
        if len(messages) < 3:
            return

        # Prefer explicit user_ids list; fall back to legacy single user_id
        effective_ids = user_ids if user_ids is not None else ([user_id] if user_id else [])
        conversation = self._format_conversation(messages, session.summary)
        await self._reflect(conversation, full=True, user_ids=effective_ids)

    async def reflect_on_compacted(
        self, messages: list[Message], summary: str,
        *, user_id: str | None = None,
        user_ids: list[str] | None = None,
    ) -> None:
        """Lighter reflection on messages about to be discarded during compaction."""
        if not self._enabled:
            return
        if len(messages) < 5:
            return

        effective_ids = user_ids if user_ids is not None else ([user_id] if user_id else [])
        conversation = self._format_conversation(messages, summary)
        await self._reflect(conversation, full=False, user_ids=effective_ids)

    def _format_conversation(
        self, messages: list[Message], summary: str = "",
    ) -> str:
        parts = []
        if summary:
            parts.append(f"[Summary of earlier conversation]: {summary[:500]}")
        for m in messages:
            uid = getattr(m, "user_id", None)
            if uid:
                parts.append(f"{m.role} [user_id={uid}]: {m.content[:500]}")
            else:
                parts.append(f"{m.role}: {m.content[:500]}")
        return "\n".join(parts)

    async def _reflect(
        self, conversation: str, *, full: bool,
        user_ids: list[str] | None = None,
    ) -> None:
        async with self._lock:
            data = await asyncio.to_thread(self._load)
            existing = data.get("entries", [])

            existing_text = "\n".join(
                f"- [{e['category']}] {e['key']}: {e['content']}"
                for e in existing
            ) if existing else "(none)"

            # When multiple users participated, instruct the LLM to attribute entries
            user_hint = ""
            if user_ids and len(user_ids) > 1:
                user_hint = (
                    "\nMultiple users participated. For preference/correction entries, "
                    "include a \"user_id\" field with the user's ID from the conversation. "
                    "Participant IDs: " + ", ".join(user_ids) + "\n"
                )

            prompt = (
                _REFLECTION_HEADER + user_hint
                + "\n\nCurrently known:\n" + existing_text
                + "\n\nConversation:\n" + conversation
            )

            system_instruction = (
                "You extract explicit lessons from conversations. "
                "Only record what the user clearly stated or corrected. "
                "Never infer unstated preferences. Return only valid JSON."
            )

            try:
                if not self._text_fn:
                    log.warning("No text completion backend configured for reflection")
                    return
                raw_text = await self._text_fn(
                    [{"role": "user", "content": prompt}],
                    system_instruction,
                )
            except Exception as e:
                log.error("Reflection API call failed: %s", e)
                return

            raw = raw_text.strip()
            new_entries = self._parse_entries(raw)
            if not new_entries:
                log.debug("Reflection produced no new insights")
                return

            # If not full reflection, only keep corrections and operational
            if not full:
                new_entries = [
                    e for e in new_entries
                    if e["category"] in ("correction", "operational")
                ]
                if not new_entries:
                    return

            # Tag user-specific entries with user_id.
            # If the LLM already assigned a user_id (multi-user case), keep it.
            # If only one user participated, tag preference/correction entries.
            effective_ids = user_ids or []
            single_user = effective_ids[0] if len(effective_ids) == 1 else None
            for entry in new_entries:
                if entry["category"] in ("preference", "correction"):
                    if single_user and "user_id" not in entry:
                        entry["user_id"] = single_user
                    # If multi-user, the LLM should have set user_id via _parse_entries
                    # If it didn't and we have multiple users, leave untagged (global)
                # operational and fact entries stay global (no user_id)

            # Merge by key
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            existing_by_key = {e["key"]: e for e in existing}
            for entry in new_entries:
                entry["content"] = entry["content"][:150]
                if entry["key"] in existing_by_key:
                    existing_by_key[entry["key"]]["content"] = entry["content"]
                    existing_by_key[entry["key"]]["category"] = entry["category"]
                    existing_by_key[entry["key"]]["updated_at"] = now
                    # Update user_id if set
                    if "user_id" in entry:
                        existing_by_key[entry["key"]]["user_id"] = entry["user_id"]
                else:
                    entry["created_at"] = now
                    entry["updated_at"] = now
                    existing_by_key[entry["key"]] = entry

            merged = list(existing_by_key.values())

            # Consolidate if over limit
            if len(merged) > self._max_entries:
                merged = await self._consolidate(merged)

            data["entries"] = merged
            data["last_reflection"] = now
            await asyncio.to_thread(self._save, data)
            log.info(
                "Reflection complete: %d new insights, %d total entries",
                len(new_entries), len(merged),
            )

    async def _consolidate(self, entries: list[dict]) -> list[dict]:
        """Ask the LLM to merge entries down to the consolidation target."""
        entries_text = json.dumps(entries, indent=2)
        prompt = (
            _CONSOLIDATION_HEADER + str(self._consolidation_target)
            + ' or fewer. Merge duplicates, drop stale or low-value items.'
            ' Return a JSON array with the same schema:'
            ' [{"key": ..., "category": ..., "content": ..., "user_id": ...}]'
            ' Preserve the user_id field exactly as-is for each entry (null if absent).'
            "\n\nEntries:\n" + entries_text
        )

        system_instruction = "You consolidate learned entries. Return only valid JSON array."

        try:
            if not self._text_fn:
                log.warning("No text completion backend configured for consolidation")
                entries.sort(key=lambda e: e.get("updated_at", ""), reverse=True)
                return entries[: self._consolidation_target]
            raw_text = await self._text_fn(
                [{"role": "user", "content": prompt}],
                system_instruction,
            )
        except Exception as e:
            log.error("Consolidation API call failed: %s", e)
            # Fallback: just keep the most recently updated entries
            entries.sort(key=lambda e: e.get("updated_at", ""), reverse=True)
            return entries[: self._consolidation_target]

        raw = raw_text.strip()
        consolidated = self._parse_entries(raw)
        if not consolidated:
            log.warning("Consolidation returned no entries, keeping originals trimmed")
            entries.sort(key=lambda e: e.get("updated_at", ""), reverse=True)
            return entries[: self._consolidation_target]

        # Preserve timestamps from originals where possible
        orig_by_key = {e["key"]: e for e in entries}
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for entry in consolidated:
            entry["content"] = entry["content"][:150]
            if entry["key"] in orig_by_key:
                orig = orig_by_key[entry["key"]]
                entry["created_at"] = orig.get("created_at", now)
                entry["updated_at"] = now
                # Preserve user_id from original if consolidation dropped it
                if "user_id" not in entry and "user_id" in orig:
                    entry["user_id"] = orig["user_id"]
            else:
                entry["created_at"] = now
                entry["updated_at"] = now

        log.info(
            "Consolidated %d entries down to %d",
            len(entries), len(consolidated),
        )
        return consolidated

    @staticmethod
    def _parse_entries(raw: str) -> list[dict]:
        """Parse JSON array from LLM response, tolerating markdown fences."""
        # Strip markdown code fences if present
        if "```" in raw:
            lines = raw.split("\n")
            filtered = []
            inside = False
            for line in lines:
                if line.strip().startswith("```"):
                    inside = not inside
                    continue
                if inside:
                    filtered.append(line)
            raw = "\n".join(filtered)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find a JSON array in the response
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    log.warning("Could not parse reflection response: %s", raw[:200])
                    return []
            else:
                log.warning("No JSON array found in reflection response: %s", raw[:200])
                return []

        if not isinstance(parsed, list):
            return []

        valid = []
        for item in parsed:
            if (
                isinstance(item, dict)
                and "key" in item
                and "category" in item
                and "content" in item
                and item["category"] in ("correction", "preference", "operational", "fact")
            ):
                entry = {
                    "key": str(item["key"]),
                    "category": item["category"],
                    "content": str(item["content"]),
                }
                if item.get("user_id"):
                    entry["user_id"] = str(item["user_id"])
                valid.append(entry)
        return valid
