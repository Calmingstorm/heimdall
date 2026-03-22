"""Haiku-based message classifier for routing Discord messages.

Uses the Anthropic Messages API (raw HTTP, no SDK) with claude-haiku-4-5
for fast, accurate single-word classification responses.
"""
from __future__ import annotations

import asyncio

import aiohttp

from .circuit_breaker import CircuitBreaker, CircuitOpenError
from ..logging import get_logger

log = get_logger("llm.haiku")

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
_VALID_CATEGORIES = frozenset(("chat", "claude_code", "task"))
_RETRYABLE_STATUSES = frozenset((408, 429, 500, 502, 503, 504, 529))


class HaikuClassifier:
    """Classify messages using Anthropic's Haiku model via raw HTTP.

    Drop-in replacement for OllamaClassifier. Same interface:
    classify(content, has_recent_tool_use, skill_hints) -> str

    Features:
    - Raw HTTP POST to Anthropic Messages API (no SDK, no streaming)
    - Reusable aiohttp session (not created per request)
    - Single retry for transient failures
    - Independent CircuitBreaker("haiku_classify") from Codex

    Fallback behavior:
    - CircuitOpenError (Haiku known down) -> return "chat"
    - Other exceptions (after retry) -> record failure, return "task"
    """

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001") -> None:
        self.api_key = api_key
        self.model = model
        self.breaker = CircuitBreaker("haiku_classify")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a reusable aiohttp session, creating one if needed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _build_system_prompt(
        self,
        has_recent_tool_use: bool = False,
        skill_hints: str = "",
    ) -> str:
        """Build the classification system prompt.

        IDENTICAL to OllamaClassifier._build_system_prompt() — same
        classification logic, same results.
        """
        system = (
            "Classify the user message as 'chat', 'claude_code', or 'task'. "
            "Reply with ONLY one word.\n"
            "- 'chat' = casual conversation, greetings, opinions, advice, "
            "simple questions answerable from general knowledge\n"
            "- 'claude_code' = code analysis, code review, explaining code or functions, "
            "debugging code, summarizing code changes, reading or "
            "searching through source files — ONLY when the task is pure code analysis "
            "with no need to post files, deploy, or interact with Discord/infrastructure\n"
            "- 'task' = ALWAYS 'task' if the message involves ANY of: "
            "git operations (commits, diffs, logs, reviews of specific commits), "
            "running commands on remote hosts, checking system status (disk/memory/CPU/services), "
            "deployments, Docker operations, Prometheus queries, restarting services, "
            "news, headlines, current events, what's happening, what's in the news, "
            "remembering/recalling/forgetting things, saving notes, anything needing "
            "SSH access, real-time monitoring data, or external APIs, "
            "generating/writing code AND posting/attaching/deploying it\n"
            "- Action directives like 'try again', 'go ahead', 'do it', 'proceed', "
            "'retry', 'run it', 'yes do that' = ALWAYS 'task' "
            "(they imply re-executing a previous action)"
        )
        if skill_hints:
            system += (
                f"\n- The bot has these tools available: {skill_hints}\n"
                "If the user is REQUESTING information these tools provide (e.g. 'what's the weather', "
                "'pronounce this word', 'any news about X', 'generate an image of'), classify as 'task'.\n"
                "If the user is just TALKING ABOUT the topic casually (e.g. 'the weather sucks', "
                "'I was practicing pronunciation'), classify as 'chat'."
            )
        if has_recent_tool_use:
            system += (
                "\n- CONTEXT: The user recently ran tool commands in this conversation. "
                "Short follow-ups like 'and the desktop?', 'what about X?', "
                "'same for Y', 'now check Z', 'how about memory?' are 'task' "
                "(they refer to the previous action). Only pure pleasantries "
                "like 'thanks', 'cool', 'ok' are 'chat'."
            )
        return system

    async def _send_request(
        self, session: aiohttp.ClientSession, content: str, system: str,
    ) -> str:
        """Send a single classification request to Anthropic Messages API.

        Returns the category string. Raises on HTTP errors so caller can retry.
        """
        async with session.post(
            ANTHROPIC_MESSAGES_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            json={
                "model": self.model,
                "max_tokens": 5,
                "temperature": 0.0,
                "system": system,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status in _RETRYABLE_STATUSES:
                body = await resp.text()
                raise _RetryableError(
                    f"Haiku returned HTTP {resp.status}: {body[:200]}"
                )
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"Haiku returned HTTP {resp.status}: {body[:200]}"
                )
            data = await resp.json()
            return data["content"][0]["text"].strip().lower()

    async def classify(
        self,
        content: str,
        has_recent_tool_use: bool = False,
        skill_hints: str = "",
    ) -> str:
        """Classify a message as 'chat', 'claude_code', or 'task'.

        Uses Anthropic Messages API (non-streaming) for a single-word response.
        Retries once on transient failures (429, 503, 529, timeouts).

        Returns
        -------
        str
            One of 'chat', 'claude_code', or 'task'.
        """
        try:
            self.breaker.check()

            system = self._build_system_prompt(
                has_recent_tool_use=has_recent_tool_use,
                skill_hints=skill_hints,
            )

            session = await self._get_session()

            # Try once, retry on transient errors
            try:
                result = await self._send_request(session, content, system)
            except (_RetryableError, asyncio.TimeoutError, aiohttp.ClientError) as first_err:
                log.info("Haiku classify attempt 1 failed (%s), retrying...", first_err)
                await asyncio.sleep(1)
                result = await self._send_request(session, content, system)

            self.breaker.record_success()
            log.info("Classified %r as %r via Haiku (skills=%s, recent_tools=%s)",
                     content[:80], result, bool(skill_hints), has_recent_tool_use)
            return result if result in _VALID_CATEGORIES else "task"

        except CircuitOpenError:
            log.debug("Haiku circuit open — defaulting to 'chat'")
            return "chat"
        except Exception:
            log.warning("Haiku classify failed — defaulting to 'task'", exc_info=True)
            self.breaker.record_failure()
            return "task"


class _RetryableError(Exception):
    """Transient Haiku API error that should be retried."""
