"""Web chat processing — virtual message and shared chat logic.

Provides a WebMessage class that mimics discord.Message with no-op Discord
operations, and a process_web_chat() function that runs a message through the
same Codex tool loop used by Discord messages.
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from ..llm.secret_scrubber import scrub_output_secrets
from ..logging import get_logger

if TYPE_CHECKING:
    from ..discord.client import HeimdallBot

log = get_logger("web.chat")

# Monotonic counter for virtual message IDs
_next_msg_id = int(time.time() * 1000)

# Max content length for a single chat message
MAX_CHAT_CONTENT_LEN = 4000


class _NoOpContextManager:
    """Async context manager that does nothing (replaces channel.typing)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _WebSentMessage:
    """Minimal sent-message stand-in (returned by channel.send)."""

    async def edit(self, **kwargs):
        pass


class _WebChannel:
    """Minimal channel-like object for web messages."""

    def __init__(self, channel_id: str):
        self.id = channel_id

    def typing(self):
        return _NoOpContextManager()

    async def send(self, content=None, **kwargs) -> _WebSentMessage:
        return _WebSentMessage()


class _WebAuthor:
    """Minimal author-like object for web messages."""

    def __init__(self, user_id: str, username: str):
        self.id = user_id
        self.bot = False
        self.display_name = username
        self.name = username
        self.mention = f"@{username}"

    def __str__(self):
        return self.display_name


class WebMessage:
    """Virtual Discord-like message for web chat.

    Provides the minimum interface that _process_with_tools expects
    from a discord.Message object.
    """

    def __init__(self, channel_id: str, user_id: str, username: str):
        global _next_msg_id
        _next_msg_id += 1
        self.id = _next_msg_id
        self.channel = _WebChannel(channel_id)
        self.author = _WebAuthor(user_id, username)
        self.webhook_id = None
        self.attachments = []


# Re-use the same scrubbing function applied to Discord responses.
# It lives in client.py as a module-level function but depends only on
# scrub_output_secrets plus a few compiled regexes.  To avoid importing
# the entire client module (which pulls in discord.py), we apply
# scrub_output_secrets here — the extra natural-language patterns are a
# nice-to-have but not critical for the web endpoint.
_scrub = scrub_output_secrets


async def process_web_chat(
    bot: HeimdallBot,
    content: str,
    channel_id: str,
    user_id: str = "web-user",
    username: str = "WebUser",
) -> dict:
    """Process a web chat message through the Codex tool loop.

    Returns dict with:
      - response: str — the LLM response text
      - tools_used: list[str] — tool names called during processing
      - is_error: bool — whether an error occurred
    """
    msg = WebMessage(channel_id=channel_id, user_id=user_id, username=username)
    tagged = f"[{username}]: {content}"
    bot.sessions.add_message(channel_id, "user", tagged, user_id=user_id)

    if not bot.codex_client:
        bot.sessions.remove_last_message(channel_id, "user")
        return {
            "response": "No LLM backend available.",
            "tools_used": [],
            "is_error": True,
        }

    try:
        sp = bot._build_system_prompt(channel=None, user_id=user_id, query=content)
        sp = await bot._inject_tool_hints(sp, content, user_id)
        history = await bot.sessions.get_task_history(channel_id, max_messages=20)

        response, _already_sent, is_error, tools_used, handoff = (
            await bot._process_with_tools(msg, history, system_prompt_override=sp)
        )
        response = _scrub(response)

        if not is_error:
            # Match client.py behaviour: only save if tools were used or handoff
            if tools_used or handoff:
                bot.sessions.add_message(channel_id, "assistant", response)
            bot.sessions.prune()
            try:
                await asyncio.to_thread(bot.sessions.save)
            except Exception:
                log.warning("Failed to save session %s", channel_id, exc_info=True)
        else:
            if tools_used:
                sanitized = (
                    f"[Previous request used tools ({', '.join(tools_used[:5])}) "
                    f"but encountered an error. The user may ask to retry.]"
                )
            else:
                sanitized = "[Previous request encountered an error before tool execution.]"
            bot.sessions.add_message(channel_id, "assistant", sanitized)
            bot.sessions.prune()
            try:
                await asyncio.to_thread(bot.sessions.save)
            except Exception:
                log.warning("Failed to save session %s", channel_id, exc_info=True)

        return {
            "response": response,
            "tools_used": tools_used,
            "is_error": is_error,
        }
    except Exception as e:
        log.error("Web chat error: %s", e, exc_info=True)
        bot.sessions.remove_last_message(channel_id, "user")
        return {
            "response": f"Error processing message: {_scrub(str(e))}",
            "tools_used": [],
            "is_error": True,
        }
