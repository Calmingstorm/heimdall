from __future__ import annotations

import asyncio
import base64
import collections
import io
import os
import re
import time
from typing import Callable

import discord
from discord import app_commands

from ..audit import AuditLogger
from ..config.schema import Config
from ..context import ContextLoader
from ..knowledge import KnowledgeStore
from ..monitoring import InfraWatcher
from .background_task import (
    BackgroundTask, run_background_task, create_task_id, MAX_STEPS,
)
from ..learning import ConversationReflector
from ..llm import CircuitOpenError, CodexAuth, CodexChatClient
from ..llm.haiku_classifier import HaikuClassifier
from ..llm.secret_scrubber import scrub_output_secrets
from ..llm.types import LLMResponse
from ..llm.system_prompt import build_system_prompt, build_chat_system_prompt
from ..logging import get_logger
from ..scheduler import Scheduler
from ..sessions import SessionManager
from ..tools import ToolExecutor, SkillManager, get_tool_definitions
from ..tools.registry import requires_approval
from ..tools.tool_memory import ToolMemory
from ..search import OllamaEmbedder, SessionVectorStore
from ..permissions import PermissionManager
from .approval import request_approval
from .routing import is_task_by_keyword, resolve_claude_code_target
from .voice import VoiceManager, VoiceMessageProxy

log = get_logger("discord")

# Friendly fallback when Codex returns an empty response after retries
_EMPTY_RESPONSE_FALLBACK = "I couldn't generate a response. Please try again."

# Webhook IDs allowed to bypass the bot-author check (for testing)
_ALLOWED_WEBHOOK_IDS = {"1485046995650482406"}

# Patterns that might indicate a secret was pasted
SECRET_SCRUB_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S{8,}"),
    re.compile(r"xox[boaprs]-[a-zA-Z0-9-]+"),
    # Natural language: "my password is ...", "password for gmail is ..."
    re.compile(r"(?i)(?:my\s+)?(?:password|passwd|pwd)\s+(?:\S+\s+){0,4}(?:is|was)\s+\S{6,}"),
]

DISCORD_MAX_LEN = 2000
MAX_TOOL_ITERATIONS = 20
TOOL_OUTPUT_MAX_CHARS = 12000  # ~3000 tokens; cap tool results to prevent context bloat
SEND_MAX_RETRIES = 3


class ToolLoopCancelView(discord.ui.View):
    """Cancel button attached to the tool loop progress embed.

    When pressed by an allowed user, sets an asyncio.Event that the tool loop
    checks between iterations.  This bypasses the per-channel message lock
    because Discord button interactions are handled out-of-band.
    """

    def __init__(self, allowed_user_ids: list[str], timeout: float = 600) -> None:
        super().__init__(timeout=timeout)
        self._allowed = set(allowed_user_ids)
        self._cancel_event = asyncio.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.grey, emoji="\u274c")
    async def cancel_button(
        self, interaction: discord.Interaction, button: discord.ui.Button,
    ) -> None:
        if str(interaction.user.id) not in self._allowed:
            await interaction.response.send_message(
                "You are not authorized to cancel this task.", ephemeral=True,
            )
            return
        self._cancel_event.set()
        button.disabled = True
        button.label = "Cancelled"
        self.stop()
        await interaction.response.edit_message(view=self)

    def disable(self) -> None:
        for item in self.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True
        self.stop()  # Unregister from discord.py event listener

# Additional patterns for scrubbing LLM responses before Discord delivery.
# These extend OUTPUT_SECRET_PATTERNS (applied via scrub_output_secrets) with
# patterns more likely to appear in natural-language LLM output.
_RESPONSE_EXTRA_PATTERNS = [
    re.compile(r"xox[boaprs]-[a-zA-Z0-9-]+"),  # Slack tokens
    # Natural language: "the password is ...", "my password is hunter2"
    re.compile(r"(?i)(?:my\s+)?(?:password|passwd|pwd)\s+(?:\S+\s+){0,4}(?:is|was)\s+\S{6,}"),
]


def scrub_response_secrets(text: str) -> str:
    """Scrub potential secrets from LLM responses before sending to Discord.

    Applies the tool-output patterns (passwords, API keys, private keys,
    database URLs) plus additional patterns for secrets that LLMs might
    express in natural language.
    """
    text = scrub_output_secrets(text)
    for pattern in _RESPONSE_EXTRA_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def truncate_tool_output(text: str, max_chars: int = TOOL_OUTPUT_MAX_CHARS) -> str:
    """Truncate large tool output, preserving the start and end for context.

    Tool results stay in the messages list and are re-sent as input tokens
    on every subsequent iteration of the tool loop.  Capping output prevents
    a single large result (Prometheus JSON, file contents, long command output)
    from ballooning costs across iterations.
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return (
        text[:half]
        + f"\n\n[... {omitted} characters omitted ...]\n\n"
        + text[-half:]
    )


class LokiBot(discord.Client):
    def __init__(self, config: Config) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        intents.members = True
        intents.voice_states = True
        super().__init__(intents=intents)

        self.config = config
        self.tree = app_commands.CommandTree(self)

        # Per-channel lock to prevent concurrent processing of the same message
        self._channel_locks: dict[str, asyncio.Lock] = {}
        # Pending file attachments from skills — per-channel to avoid cross-channel leaks
        self._pending_files: dict[str, list[tuple[bytes, str]]] = {}
        # Track recently processed message IDs to prevent duplicate handling
        self._processed_messages: collections.OrderedDict[int, None] = collections.OrderedDict()
        self._processed_messages_max = 100
        # Recent tool executions for conversational context (injected into system prompt)
        # Per-channel: {channel_id: [(timestamp, entry_text), ...]}
        self._recent_actions: dict[str, list[tuple[float, str]]] = {}
        self._recent_actions_max = 10
        self._recent_actions_expiry = 3600  # seconds (1 hour)
        # Per-channel timestamp of last tool use (monotonic), for classifier context
        self._last_tool_use: dict[str, float] = {}
        # Background task tracking
        self._background_tasks: dict[str, BackgroundTask] = {}
        self._background_tasks_max = 20

        self.context_loader = ContextLoader(config.context.directory)
        self.context_loader.load()

        self.reflector = ConversationReflector(
            learned_path="./data/learned.json",
            max_entries=config.learning.max_entries,
            consolidation_target=config.learning.consolidation_target,
            enabled=config.learning.enabled,
        )

        # Semantic search + FTS5 components
        self._vector_store: SessionVectorStore | None = None
        self._embedder: OllamaEmbedder | None = None
        self._knowledge_store: KnowledgeStore | None = None
        self._fts_index: FullTextIndex | None = None
        if config.search.enabled:
            self._embedder = OllamaEmbedder(
                base_url=config.search.ollama_url,
                model=config.search.embed_model,
            )
            # Initialize FTS5 index (SQLite, no external deps)
            from pathlib import Path
            fts_db_path = str(Path(config.search.chromadb_path).parent / "fts.db")
            from ..search.fts import FullTextIndex
            self._fts_index = FullTextIndex(fts_db_path)
            if not self._fts_index.available:
                self._fts_index = None

            self._vector_store = SessionVectorStore(
                config.search.chromadb_path, fts_index=self._fts_index,
            )
            if not self._vector_store.available:
                self._vector_store = None
            self._knowledge_store = KnowledgeStore(
                config.search.chromadb_path, fts_index=self._fts_index,
            )
            if not self._knowledge_store.available:
                self._knowledge_store = None

        # Message classifier — Haiku via Anthropic Messages API (raw HTTP)
        self.classifier = HaikuClassifier(
            api_key=config.anthropic.api_key,
        )

        self.sessions = SessionManager(
            max_history=config.sessions.max_history,
            max_age_hours=config.sessions.max_age_hours,
            persist_dir=config.sessions.persist_directory,
            reflector=self.reflector,
            vector_store=self._vector_store,
            embedder=self._embedder,
        )
        self.sessions.load()

        self._memory_path = "./data/memory.json"

        # Browser automation
        self.browser_manager = None
        if config.browser.enabled:
            from ..tools.browser import BrowserManager
            self.browser_manager = BrowserManager(
                cdp_url=config.browser.cdp_url,
                default_timeout_ms=config.browser.default_timeout_ms,
                viewport_width=config.browser.viewport_width,
                viewport_height=config.browser.viewport_height,
            )

        self.tool_executor = ToolExecutor(
            config.tools, memory_path=self._memory_path,
            browser_manager=self.browser_manager,
        )
        self.skill_manager = SkillManager(
            skills_dir="./data/skills",
            tool_executor=self.tool_executor,
            memory_path=self._memory_path,
        )

        # Initialize Codex client if configured
        self.codex_client: CodexChatClient | None = None
        if config.openai_codex.enabled:
            codex_auth = CodexAuth(config.openai_codex.credentials_path)
            if codex_auth.is_configured():
                self.codex_client = CodexChatClient(
                    auth=codex_auth,
                    model=config.openai_codex.model,
                    max_tokens=config.openai_codex.max_tokens,
                )
                log.info("Codex backend enabled (model: %s)", config.openai_codex.model)

                # Use Codex for session compaction
                async def _codex_compaction(messages: list[dict], system: str) -> str:
                    return await self.codex_client.chat(
                        messages=messages, system=system, max_tokens=300,
                    )
                self.sessions.set_compaction_fn(_codex_compaction)

                # Use Codex for learning reflection
                async def _codex_reflection(messages: list[dict], system: str) -> str:
                    return await self.codex_client.chat(
                        messages=messages, system=system, max_tokens=500,
                    )
                self.reflector.set_text_fn(_codex_reflection)
            else:
                log.warning("Codex enabled in config but no credentials found. Run scripts/codex_login.py")

        self.scheduler = Scheduler(data_path="./data/schedules.json")
        self.audit = AuditLogger(path="./data/audit.jsonl")
        self.permissions = PermissionManager(
            config_tiers=config.permissions.tiers,
            default_tier=config.permissions.default_tier,
            overrides_path=config.permissions.overrides_path,
        )

        self.tool_memory = ToolMemory("./data/tool_memory.json")

        # Wire optional services into skill manager for expanded skill context
        self.skill_manager.set_services(
            knowledge_store=self._knowledge_store,
            embedder=self._embedder,
            session_manager=self.sessions,
            scheduler=self.scheduler,
        )

        # Voice support
        self.voice_manager: VoiceManager | None = None
        if config.voice.enabled:
            self.voice_manager = VoiceManager(config.voice, self)
            self.voice_manager.on_transcription = self._on_voice_transcription

        # Proactive infrastructure monitoring
        self.infra_watcher: InfraWatcher | None = None
        if config.monitoring.enabled and config.monitoring.checks:
            self.infra_watcher = InfraWatcher(
                config=config.monitoring,
                executor=self.tool_executor,
                alert_callback=self._on_monitor_alert,
            )

        self._system_prompt = self._build_system_prompt()
        self._register_commands()

    def _build_system_prompt(
        self, channel: discord.abc.GuildChannel | None = None,
        user_id: str | None = None,
        query: str | None = None,
    ) -> str:
        hosts = {
            alias: f"{h.ssh_user}@{h.address}"
            for alias, h in self.config.tools.hosts.items()
        }

        voice_info = "Voice support is not enabled."
        if self.voice_manager:
            if self.voice_manager.is_connected:
                ch = self.voice_manager.current_channel
                ch_name = ch.name if ch else "unknown"
                voice_info = (
                    f"You are currently in voice channel **{ch_name}**. "
                    "You can hear users via speech-to-text and respond via text-to-speech. "
                    "Transcribed voice input appears as regular messages. Your text responses "
                    "are spoken aloud in the voice channel AND posted as text. "
                    "Keep voice responses concise and conversational — they will be spoken."
                )
            else:
                voice_info = (
                    "Voice support is enabled but you are not in a voice channel. "
                    "Users can ask you to join with '/voice join' or 'join voice'."
                )

        prompt = build_system_prompt(
            context=self.context_loader.context,
            hosts=hosts,
            services=self.config.tools.allowed_services,
            playbooks=self.config.tools.allowed_playbooks,
            voice_info=voice_info,
        )

        # Inject persistent memory into the system prompt (per-user + global)
        memory = self.tool_executor._load_memory_for_user(user_id)
        if memory:
            memory_text = "\n".join(f"- **{k}**: {v}" for k, v in memory.items())
            prompt += f"\n\n## Persistent Memory\n{memory_text}"

        # Inject learned context from cross-conversation reflection (per-user filtered)
        if hasattr(self, "reflector"):
            learned = self.reflector.get_prompt_section(user_id=user_id)
            if learned:
                prompt += f"\n\n{learned}"

        # Inject user-created skills list
        if hasattr(self, "skill_manager"):
            skills = self.skill_manager.list_skills()
            if skills:
                skills_text = "\n".join(f"- `{s['name']}`: {s['description']}" for s in skills)
                prompt += f"\n\n## User-Created Skills\n{skills_text}"

        # Inject recent tool executions for this channel only
        if channel is not None:
            channel_id = str(channel.id)
            now = time.time()
            expiry = getattr(self, "_recent_actions_expiry", 3600)
            channel_actions = [
                entry for ts, entry in self._recent_actions.get(channel_id, [])
                if now - ts < expiry
            ]
            if channel_actions:
                actions_text = "\n".join(channel_actions[-10:])
                prompt += f"\n\n## Recent Actions\n{actions_text}"

        # Tool use pattern hints injected separately via _inject_tool_hints()
        # because format_hints is async (needs embedder).

        # Channel-based personality: if the channel has a topic/description set,
        # inject it as a personality directive. All other rules still apply.
        if channel is not None:
            topic = getattr(channel, "topic", None)
            if topic and topic.strip():
                prompt += (
                    f"\n\n## Channel Personality\n"
                    f"For this channel, adopt the following personality while keeping all other rules intact:\n"
                    f"{topic.strip()}"
                )

        return prompt

    async def _inject_tool_hints(self, system_prompt: str, query: str, user_id: str | None = None) -> str:
        """Return system_prompt with tool use pattern hints appended (async, needs embedder).

        Returns the original prompt unchanged if hints are unavailable or an error occurs.
        """
        try:
            tm = getattr(self, "tool_memory", None)
            if not tm or not hasattr(tm, "format_hints"):
                return system_prompt
            perms = getattr(self, "permissions", None)
            allowed = perms.allowed_tool_names(user_id) if perms and user_id else None
            embedder = getattr(self, "_embedder", None)
            hints = await tm.format_hints(
                query, allowed_tools=allowed, embedder=embedder,
            )
            if hints:
                return system_prompt + f"\n\n{hints}"
        except Exception:
            pass  # Non-critical — hints are optional optimization
        return system_prompt

    def _build_chat_system_prompt(
        self, channel: discord.abc.GuildChannel | None = None,
        user_id: str | None = None,
    ) -> str:
        """Build a lightweight system prompt for chat-routed messages.

        Includes identity, rules, memory, and personality but omits
        infrastructure details, tool docs, host lists, and PromQL to
        save input tokens on casual conversation.
        """
        voice_info = "Voice support is not enabled."
        if self.voice_manager:
            if self.voice_manager.is_connected:
                ch = self.voice_manager.current_channel
                ch_name = ch.name if ch else "unknown"
                voice_info = (
                    f"You are currently in voice channel **{ch_name}**. "
                    "Transcribed voice input appears as regular messages. Your text responses "
                    "are spoken aloud. Keep voice responses concise and conversational."
                )

        prompt = build_chat_system_prompt(voice_info=voice_info)

        # Inject persistent memory (per-user + global, personalization matters for chat)
        memory = self.tool_executor._load_memory_for_user(user_id)
        if memory:
            memory_text = "\n".join(f"- **{k}**: {v}" for k, v in memory.items())
            prompt += f"\n\n## Persistent Memory\n{memory_text}"

        # Inject learned context (per-user filtered, personality from past conversations)
        if hasattr(self, "reflector"):
            learned = self.reflector.get_prompt_section(user_id=user_id)
            if learned:
                prompt += f"\n\n{learned}"

        # Channel personality
        if channel is not None:
            topic = getattr(channel, "topic", None)
            if topic and topic.strip():
                prompt += (
                    f"\n\n## Channel Personality\n"
                    f"For this channel, adopt the following personality while keeping all other rules intact:\n"
                    f"{topic.strip()}"
                )

        return prompt

    def _merged_tool_definitions(self) -> list[dict]:
        """Merge built-in and skill tool definitions, deduplicating by name.

        Built-in tools take priority over skills with the same name.
        """
        builtin = get_tool_definitions()
        builtin_names = {t["name"] for t in builtin}
        skill_defs = [
            t for t in self.skill_manager.get_tool_definitions()
            if t["name"] not in builtin_names
        ]
        return builtin + skill_defs

    def _track_recent_action(
        self, tool_name: str, tool_input: dict, result_preview: str,
        elapsed_ms: int, channel_id: str | None = None,
    ) -> None:
        """Record a tool execution for conversational context injection.

        Actions are stored per-channel so that channel A's tool results
        don't leak into channel B's system prompt.  Each entry carries a
        real timestamp for time-based expiry (1 hour).
        """
        if not channel_id:
            return  # No channel context — nothing to inject later

        from datetime import datetime
        ts = datetime.now().strftime("%H:%M")
        inp_summary = ", ".join(f"{k}={v}" for k, v in tool_input.items() if isinstance(v, str))
        if len(inp_summary) > 100:
            inp_summary = inp_summary[:100] + "..."
        status = "OK" if "error" not in result_preview.lower()[:50] else "ERROR"
        entry = f"- [{ts}] `{tool_name}`({inp_summary}) → {status} ({elapsed_ms}ms)"

        actions = self._recent_actions.setdefault(channel_id, [])
        actions.append((time.time(), entry))
        # Cap per-channel list
        if len(actions) > self._recent_actions_max:
            self._recent_actions[channel_id] = actions[-self._recent_actions_max:]

        # Track per-channel tool use time for classifier context
        self._last_tool_use[channel_id] = time.monotonic()

    def _register_commands(self) -> None:
        @self.tree.command(name="status", description="Show Loki bot status")
        async def cmd_status(interaction: discord.Interaction) -> None:
            if not self._is_allowed_user(interaction.user):
                await interaction.response.send_message("Access denied.", ephemeral=True)
                return
            voice_status = ""
            if self.voice_manager:
                if self.voice_manager.is_connected:
                    ch = self.voice_manager.current_channel
                    voice_status = f"\nVoice: Connected to **{ch.name}**" if ch else "\nVoice: Connected"
                else:
                    voice_status = "\nVoice: Not connected"
            codex_status = "Codex: " + ("enabled" if self.codex_client else "not configured")
            classifier_status = "Classifier: Haiku"
            await interaction.response.send_message(
                f"**Loki Status**\n"
                f"{codex_status}\n"
                f"{classifier_status}"
                f"{voice_status}"
            )

        @self.tree.command(name="reset", description="Reset conversation history")
        async def cmd_reset(interaction: discord.Interaction) -> None:
            if not self._is_allowed_user(interaction.user):
                await interaction.response.send_message("Access denied.", ephemeral=True)
                return
            self.sessions.reset(str(interaction.channel_id))
            await interaction.response.send_message("Conversation history cleared.")

        @self.tree.command(name="reload", description="Reload context files")
        async def cmd_reload(interaction: discord.Interaction) -> None:
            if not self._is_allowed_user(interaction.user):
                await interaction.response.send_message("Access denied.", ephemeral=True)
                return
            self.context_loader.reload()
            self._system_prompt = self._build_system_prompt()
            await interaction.response.send_message("Context files reloaded.")

        @self.tree.command(name="purge", description="Delete recent messages in this channel")
        @app_commands.describe(count="Number of messages to delete (default 100, max 500)")
        async def cmd_purge(interaction: discord.Interaction, count: int = 100) -> None:
            if not self._is_allowed_user(interaction.user):
                await interaction.response.send_message("Access denied.", ephemeral=True)
                return
            count = min(count, 500)
            await interaction.response.defer(ephemeral=True)
            deleted = await interaction.channel.purge(limit=count)
            self.sessions.reset(str(interaction.channel_id))
            await interaction.followup.send(
                f"Deleted {len(deleted)} messages and reset conversation history.",
                ephemeral=True,
            )

        @self.tree.command(name="usage", description="Show token usage details")
        async def cmd_usage(interaction: discord.Interaction) -> None:
            if not self._is_allowed_user(interaction.user):
                await interaction.response.send_message("Access denied.", ephemeral=True)
                return
            await interaction.response.send_message(
                "**Usage**\n"
                "All backends are subscription-based (free).\n"
                "Codex: ChatGPT subscription\n"
                "Classifier: Haiku (Anthropic API)\n"
                "Claude Code: Max subscription"
            )


    def _is_allowed_user(self, user: discord.User | discord.Member) -> bool:
        if not self.config.discord.allowed_users:
            return True
        return str(user.id) in self.config.discord.allowed_users

    def _is_allowed_channel(self, channel_id: int) -> bool:
        if not self.config.discord.channels:
            return True
        return str(channel_id) in self.config.discord.channels

    def _check_for_secrets(self, content: str) -> bool:
        return any(p.search(content) for p in SECRET_SCRUB_PATTERNS)

    async def on_ready(self) -> None:
        log.info("Logged in as %s (ID: %s)", self.user, self.user.id)
        # Prune stale sessions loaded from disk.  load() reads ALL persisted
        # session files regardless of age; pruning here removes expired ones
        # immediately instead of waiting for the first user message.
        pruned = self.sessions.prune()
        if pruned:
            log.info("Startup: pruned %d stale sessions", pruned)
        # Sync commands to each guild (instant) instead of global (up to 1hr)
        for guild in self.guilds:
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            log.info("Slash commands synced to guild: %s", guild.name)
        self.scheduler.start(self._on_scheduled_task)
        if self._vector_store and self._embedder:
            asyncio.create_task(self._backfill_archives())
        # Start proactive monitoring if configured
        if hasattr(self, "infra_watcher") and self.infra_watcher:
            self.infra_watcher.start()

    async def _backfill_archives(self) -> None:
        """Backfill semantic search index and FTS5 with existing archive files."""
        try:
            archive_dir = self.sessions.persist_dir / "archive"
            count = await self._vector_store.backfill(archive_dir, self._embedder)
            if count:
                log.info("Backfilled %d archive sessions into vector store", count)
            else:
                log.info("Vector store up to date")
            # Backfill knowledge FTS from existing ChromaDB data
            if self._knowledge_store and self._fts_index:
                kb_count = self._knowledge_store.backfill_fts()
                if kb_count:
                    log.info("Backfilled %d knowledge chunks into FTS index", kb_count)
        except Exception as e:
            log.error("Archive backfill failed: %s", e)

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Auto-join voice channel when an allowed user joins."""
        if not self.voice_manager or not self.config.voice.auto_join:
            return
        if member.bot:
            return
        if not self._is_allowed_user(member):
            return
        # User joined a voice channel (was not in one before)
        if before.channel is None and after.channel is not None:
            if not self.voice_manager.is_connected:
                log.info("Auto-joining voice channel %s (user: %s)", after.channel.name, member)
                await self.voice_manager.join_channel(after.channel)
        # User left — if we're in that channel and it's now empty (minus bot), leave
        elif before.channel is not None and after.channel is None:
            if self.voice_manager.is_connected and self.voice_manager.current_channel == before.channel:
                humans = [m for m in before.channel.members if not m.bot]
                if not humans:
                    log.info("All users left voice channel, disconnecting")
                    await self.voice_manager.leave_channel()

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            # Allow specific webhooks for testing
            if not (message.webhook_id and str(message.webhook_id) in _ALLOWED_WEBHOOK_IDS):
                return
        is_test_webhook = message.webhook_id and str(message.webhook_id) in _ALLOWED_WEBHOOK_IDS
        if not is_test_webhook and not self._is_allowed_user(message.author):
            return
        if not self._is_allowed_channel(message.channel.id):
            return

        log.info(
            "on_message fired: msg_id=%s channel=%s content=%r",
            message.id, message.channel.id, message.content[:80],
        )

        # Dedup: skip if we've already processed this exact message
        if message.id in self._processed_messages:
            log.warning("Duplicate on_message for msg_id=%s, skipping", message.id)
            return
        self._processed_messages[message.id] = None
        # Keep bounded — remove oldest entries (OrderedDict preserves insertion order)
        while len(self._processed_messages) > self._processed_messages_max:
            self._processed_messages.popitem(last=False)

        content = message.content
        # Strip the bot mention from the message if present
        if self.user and self.user.mentioned_in(message):
            content = content.replace(f"<@{self.user.id}>", "").strip()
            content = content.replace(f"<@!{self.user.id}>", "").strip()

        # Handle file attachments — append file contents to the message
        attachment_text, image_blocks = await self._process_attachments(message)
        if attachment_text:
            content = f"{content}\n\n{attachment_text}" if content else attachment_text

        if not content and not image_blocks:
            return

        if not content:
            content = "(see attached image)"

        # Check for secrets, scrub from history and delete the message
        if self._check_for_secrets(content):
            self.sessions.scrub_secrets(str(message.channel.id), content)
            try:
                await message.delete()
                await message.channel.send(
                    f"{message.author.mention} I detected a secret/credential in your message. "
                    "I've deleted it and scrubbed it from my history."
                )
            except discord.Forbidden:
                await message.channel.send(
                    f"{message.author.mention} I detected a secret/credential in your message. "
                    "I've scrubbed it from my history. "
                    "I couldn't delete the message — please delete it manually."
                )
            return

        # Voice commands via natural language (short, direct commands only)
        if self.voice_manager:
            _voice_lower = content.lower().strip()
            _voice_words = _voice_lower.split()
            # Only treat short messages (≤8 words) as voice commands to avoid
            # false positives on pasted changelogs or longer messages
            if len(_voice_words) <= 8:
                _join_words = {"join", "hop", "get in", "come to", "connect", "enter", "hop in", "come in"}
                _leave_words = {"leave", "disconnect", "get out", "exit", "go away", "hop out"}
                _voice_context = {"voice", "vc", "channel", "call", "chat"}
                _has_voice_context = any(w in _voice_lower for w in _voice_context)

                if _has_voice_context and any(w in _voice_lower for w in _join_words):
                    if isinstance(message.author, discord.Member) and message.author.voice:
                        result = await self.voice_manager.join_channel(message.author.voice.channel)
                        await message.reply(result)
                    else:
                        await message.reply("You need to be in a voice channel first.")
                    return
                if _has_voice_context and any(w in _voice_lower for w in _leave_words):
                    result = await self.voice_manager.leave_channel()
                    await message.reply(result)
                    return

        # If bot is in a voice channel, auto-attach voice callback for TTS
        vc_callback = None
        if self.voice_manager and self.voice_manager.is_connected:
            async def vc_callback(response: str) -> None:
                await self.voice_manager.speak(response)

        await self._handle_message(message, content, image_blocks=image_blocks, voice_callback=vc_callback)

    async def _process_attachments(self, message: discord.Message) -> tuple[str, list[dict]]:
        """Download attachments and return (text_parts, image_blocks).

        Text files are returned as formatted strings.
        Images are returned as image content blocks (base64).
        """
        text_parts = []
        image_blocks = []

        _image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        _image_media_types = {"image/png", "image/jpeg", "image/gif", "image/webp"}

        for att in message.attachments:
            ext = "." + att.filename.rsplit(".", 1)[-1].lower() if "." in att.filename else ""

            # Image attachments — send to Claude as vision content blocks
            is_image = ext in _image_extensions or (att.content_type and att.content_type in _image_media_types)
            if is_image:
                # Image size limit: 5MB per image (base64-encoded)
                if att.size > 5 * 1024 * 1024:
                    text_parts.append(f"[Image: {att.filename} ({att.size / 1024 / 1024:.1f} MB, exceeds 5 MB limit)]")
                    continue
                try:
                    data = await att.read()
                    b64 = base64.b64encode(data).decode("ascii")
                    # Detect actual media type from file magic bytes, don't trust Discord's content_type
                    media_type = self._detect_image_type(data) or att.content_type or f"image/{ext.lstrip('.')}"
                    if media_type == "image/jpg":
                        media_type = "image/jpeg"
                    image_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    })
                    text_parts.append(f"[Image attached: {att.filename}]")
                    log.info("Processed image attachment: %s (%d KB)", att.filename, att.size // 1024)
                except Exception as e:
                    text_parts.append(f"[Image: {att.filename} (failed to read: {e})]")
                continue

            # Text files — read and inline
            text_extensions = {
                ".txt", ".md", ".yml", ".yaml", ".json", ".toml", ".ini",
                ".cfg", ".conf", ".py", ".sh", ".bash", ".js", ".ts",
                ".html", ".css", ".xml", ".csv", ".log", ".env",
                ".service", ".timer", ".sql", ".php", ".rb", ".go",
                ".rs", ".java", ".c", ".h", ".cpp", ".hpp",
            }

            is_text = ext in text_extensions or (att.content_type and "text" in att.content_type)

            # Large text files (>100KB) — preview + suggest knowledge base ingestion
            if att.size > 100_000 and is_text:
                try:
                    data = await att.read()
                    text = data.decode("utf-8", errors="replace")
                    preview = text[:2000]
                    text_parts.append(
                        f"**Attached file: {att.filename}** ({att.size:,} bytes — too large to fully inline, showing first 2KB)\n"
                        f"```\n{preview}\n```\n"
                        f"[This file is {att.size:,} bytes. You should offer to ingest it into the knowledge base "
                        f"using the ingest_document tool so it can be searched later.]"
                    )
                except Exception as e:
                    text_parts.append(f"[Attachment: {att.filename} (failed to read: {e})]")
                continue

            if att.size > 100_000:
                text_parts.append(f"[Attachment: {att.filename} ({att.size} bytes, too large to read)]")
                continue

            if is_text:
                try:
                    data = await att.read()
                    text = data.decode("utf-8", errors="replace")
                    text_parts.append(f"**Attached file: {att.filename}**\n```\n{text}\n```")
                    # Append smart context hints based on file type and size
                    hint = self._get_attachment_hint(att.filename, ext, att.size)
                    if hint:
                        text_parts.append(hint)
                except Exception as e:
                    text_parts.append(f"[Attachment: {att.filename} (failed to read: {e})]")
            else:
                text_parts.append(f"[Attachment: {att.filename} ({att.content_type or 'binary'}, {att.size} bytes)]")

        return "\n\n".join(text_parts) if text_parts else "", image_blocks

    @staticmethod
    def _get_attachment_hint(filename: str, ext: str, size: int) -> str:
        """Return a context hint for the LLM based on file type and size.

        Hints guide the LLM to suggest appropriate actions (ingest, create skill, deploy).
        Returns empty string if no special hint applies.
        """
        hints = []

        # Python files — suggest creating as a skill
        if ext == ".py":
            hints.append(
                "[This is a Python file. If it looks like a bot skill/tool "
                "(has SKILL_DEFINITION and execute function), offer to create it as a skill "
                "using the create_skill tool. Otherwise, offer to ingest it as documentation.]"
            )

        # YAML/JSON config files — suggest deploying or ingesting
        elif ext in {".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".conf"}:
            hints.append(
                "[This is a configuration file. You can offer to: "
                "(1) deploy it to a host using write_file, "
                "(2) ingest it into the knowledge base as documentation using ingest_document, "
                "or (3) analyze it and suggest improvements.]"
            )

        # Shell scripts — suggest deploying or reviewing
        elif ext in {".sh", ".bash"}:
            hints.append(
                "[This is a shell script. You can offer to: "
                "(1) deploy it to a host using write_file, "
                "(2) review it for issues, or "
                "(3) run it on a specific host using run_command.]"
            )

        # Systemd units — suggest deploying
        elif ext in {".service", ".timer"}:
            hints.append(
                "[This is a systemd unit file. You can offer to deploy it to a host "
                "using write_file (to /etc/systemd/system/) and enable it.]"
            )

        # Documentation files — suggest knowledge base ingestion
        elif ext in {".md", ".txt"}:
            hints.append(
                "[This is a documentation file. You can offer to ingest it into the "
                "knowledge base using the ingest_document tool so it can be searched later.]"
            )

        # Large files (>50KB) — always suggest knowledge base ingestion
        if size > 50_000:
            hints.append(
                "[This file is large. You should offer to ingest it into the knowledge base "
                "using the ingest_document tool for future reference.]"
            )

        return "\n".join(hints)

    async def _on_voice_transcription(
        self, text: str, member: discord.Member, transcript_channel: discord.TextChannel,
    ) -> None:
        """Handle transcribed voice input — route through message pipeline."""
        log.info("Voice transcription from %s: %r", member, text[:80])

        # Post the transcription to the transcript channel
        await transcript_channel.send(f"**{member.display_name}** (voice): {text}")

        # Create a proxy message for the pipeline
        proxy = VoiceMessageProxy(
            author=member,
            channel=transcript_channel,
            id=int(time.time() * 1000),
            guild=member.guild,
        )

        # Define voice callback for dual output (speak + text)
        async def voice_callback(response: str) -> None:
            if self.voice_manager:
                await self.voice_manager.speak(response)

        await self._handle_message(
            proxy, text,
            voice_callback=voice_callback,
        )

    async def _handle_message(
        self, message: discord.Message, content: str, *, image_blocks: list[dict] | None = None,
        voice_callback: Callable | None = None,
    ) -> None:
        channel_id = str(message.channel.id)

        # Acquire per-channel lock — messages queue naturally via the lock
        lock = self._channel_locks.setdefault(channel_id, asyncio.Lock())

        async with lock:
            # Thread context inheritance: if this is a thread with no session yet,
            # seed it with the parent channel's summary so context carries over.
            # Must be inside the lock to prevent two concurrent messages from
            # both seeding the thread and to safely access parent session state.
            if isinstance(message.channel, discord.Thread) and message.channel.parent:
                parent_id = str(message.channel.parent.id)
                thread_session = self.sessions.get_or_create(channel_id)
                if not thread_session.messages:
                    parent_session = self.sessions.get_or_create(parent_id)
                    if parent_session.messages or parent_session.summary:
                        # Copy the parent's summary and last few messages for context
                        thread_session.summary = parent_session.summary or ""
                        # Include recent parent messages as additional context
                        recent = parent_session.messages[-6:]
                        if recent:
                            parent_context = "\n".join(
                                f"{m.role}: {m.content[:300]}" for m in recent
                            )
                            if thread_session.summary:
                                thread_session.summary += f"\n[Parent channel context]:\n{parent_context}"
                            else:
                                thread_session.summary = f"[Parent channel context]:\n{parent_context}"
                        log.info("Thread %s inherited context from parent %s", channel_id, parent_id)

            await self._handle_message_inner(
                message, content, channel_id,
                image_blocks=image_blocks or [],
                voice_callback=voice_callback,
            )

    async def _handle_message_inner(
        self, message: discord.Message, content: str, channel_id: str,
        *, image_blocks: list[dict] | None = None,
        voice_callback: Callable | None = None,
    ) -> None:
        user_id = str(message.author.id)
        # Prefix with display name so the LLM knows who's talking
        display_name = message.author.display_name or message.author.name
        tagged_content = f"[{display_name}]: {content}"
        self.sessions.add_message(channel_id, "user", tagged_content, user_id=user_id)

        # Compact history if needed before sending to Claude
        history = await self.sessions.get_history_with_compaction(channel_id)

        # If images are attached, inject them into the last user message for Claude
        # (stored as text-only in session to avoid bloating history with base64)
        if image_blocks:
            history = list(history)  # don't mutate the original
            if history and history[-1]["role"] == "user":
                last_msg = history[-1]
                text_content = last_msg["content"] if isinstance(last_msg["content"], str) else str(last_msg["content"])
                # Build multimodal content: images first, then text
                history[-1] = {
                    "role": "user",
                    "content": image_blocks + [{"type": "text", "text": text_content}],
                }
            log.info("Attached %d image(s) to message for Claude vision", len(image_blocks))

        try:
            # Images force the "task" route (vision requires tool-capable backend)
            if image_blocks:
                msg_type = "task"
                log.info("Message has images, forcing task route (vision)")
            elif is_task_by_keyword(content):
                msg_type = "task"
                log.info("Message matched task keyword, forcing 'task' route")
            else:
                # Classify message into chat/claude_code/task via Haiku.
                # All backends are free (subscription-based), so always classify.
                _recent_tool = (
                    channel_id in self._last_tool_use
                    and time.monotonic() - self._last_tool_use[channel_id] < 300
                )
                # Build skill hints for the classifier so it knows what tools exist
                _skill_hints = ", ".join(
                    f"{s['name']} ({s['description'][:60]})"
                    for s in self.skill_manager.list_skills()
                ) if self.skill_manager else ""
                msg_type = await self.classifier.classify(
                    content, has_recent_tool_use=_recent_tool,
                    skill_hints=_skill_hints,
                )

            # Guest tier: force chat route (no tools, no code execution)
            if self.permissions.is_guest(str(message.author.id)):
                msg_type = "chat"
                log.info("Guest tier user %s, forcing chat route", message.author.id)

            already_sent = False
            is_error = False
            tools_used: list[str] = []
            if msg_type == "chat" and self.codex_client:
                # Route chat to OpenAI Codex (ChatGPT subscription, $0)
                log.info("Message classified as 'chat', routing to Codex")
                chat_prompt = self._build_chat_system_prompt(channel=message.channel, user_id=user_id)
                try:
                    response = await self.codex_client.chat(
                        messages=history,
                        system=chat_prompt,
                    )
                    if not response:
                        response = _EMPTY_RESPONSE_FALLBACK
                    log.info("Codex response: %r", response[:200])
                except Exception as e:
                    log.warning("Codex chat failed: %s", e)
                    response = "Chat is temporarily unavailable. Please try again in a moment."
                    is_error = True
            elif msg_type == "chat":
                # No Codex configured
                log.info("Message classified as 'chat', no chat backend configured")
                response = "Chat backend is not configured."
                is_error = True
            elif msg_type == "claude_code":
                # Route to Claude Code CLI (claude -p) — free, reads files directly.
                # Ideal for code analysis, review, script writing, explaining code.
                log.info("Message classified as 'claude_code', routing to Claude Code CLI")
                # Include recent conversation context so claude -p can understand
                # follow-up references (e.g. "what about the error handling?").
                # claude -p is stateless — without context, follow-ups would fail.
                claude_prompt = content
                if len(history) > 1:
                    # history includes the current message as last item;
                    # take up to 6 preceding messages (3 exchanges) for context
                    context_msgs = history[:-1][-6:]
                    context_parts = []
                    for m in context_msgs:
                        role = "User" if m["role"] == "user" else "Assistant"
                        text = m["content"] if isinstance(m["content"], str) else str(m["content"])
                        if len(text) > 500:
                            text = text[:500] + "..."
                        context_parts.append(f"{role}: {text}")
                    claude_prompt = (
                        "Previous conversation (for context):\n---\n"
                        + "\n".join(context_parts)
                        + "\n---\n\nCurrent request:\n"
                        + content
                    )
                cc_host, cc_dir = resolve_claude_code_target(content)
                log.info("Claude Code target: host=%s dir=%s", cc_host, cc_dir)
                try:
                    async with message.channel.typing():
                        response = await self.tool_executor._handle_claude_code({
                            "host": cc_host,
                            "working_directory": cc_dir,
                            "prompt": claude_prompt,
                            "allow_edits": False,
                            "max_output_chars": 8000,
                        })
                    # If claude -p returned an error, fall back to Codex chat
                    if response.startswith("Claude Code failed") or response.startswith("Unknown"):
                        log.warning("Claude Code CLI failed, falling back to Codex: %s", response[:200])
                        if self.codex_client:
                            chat_prompt = self._build_chat_system_prompt(channel=message.channel, user_id=user_id)
                            try:
                                response = await self.codex_client.chat(messages=history, system=chat_prompt)
                                if not response:
                                    response = _EMPTY_RESPONSE_FALLBACK
                            except Exception as codex_err:
                                log.warning("Codex fallback also failed: %s", codex_err)
                                response = "Claude Code CLI and fallback both failed. Try again later."
                                is_error = True
                        else:
                            is_error = True
                except Exception as e:
                    log.warning("Claude Code routing failed, falling back to Codex: %s", e)
                    if self.codex_client:
                        chat_prompt = self._build_chat_system_prompt(channel=message.channel, user_id=user_id)
                        try:
                            response = await self.codex_client.chat(messages=history, system=chat_prompt)
                            if not response:
                                response = _EMPTY_RESPONSE_FALLBACK
                        except Exception as codex_err:
                            log.warning("Codex fallback also failed: %s", codex_err)
                            response = f"Claude Code CLI failed: {e}"
                            is_error = True
                    else:
                        response = f"Claude Code CLI failed: {e}"
                        is_error = True
            else:
                # Task — use Codex with tools (free via subscription)
                if not self.codex_client:
                    await self._send_with_retry(
                        message,
                        "No tool backend available. Please try again later.",
                    )
                    self.sessions.remove_last_message(channel_id, "user")
                    return
                # Build full system prompt only for task messages that need it.
                # Chat paths use _build_chat_system_prompt() instead, so building
                # the full prompt eagerly for every message wastes disk I/O
                # (memory.json + learned.json reads) and string formatting.
                _sp = self._build_system_prompt(channel=message.channel, user_id=user_id, query=content)
                _sp = await self._inject_tool_hints(_sp, content, user_id)
                log.info("Task route: using Codex with tools")
                # Use abbreviated history to reduce poisoning from stale responses
                task_history = await self.sessions.get_task_history(channel_id, max_messages=10)
                # Re-inject images into task history (they were added to `history` above
                # but get_task_history returns fresh history without them)
                if image_blocks and task_history and task_history[-1]["role"] == "user":
                    last = task_history[-1]
                    text = last["content"] if isinstance(last["content"], str) else str(last["content"])
                    task_history[-1] = {
                        "role": "user",
                        "content": image_blocks + [{"type": "text", "text": text}],
                    }
                try:
                    response, already_sent, is_error, tools_used, handoff = await self._process_with_tools(
                        message, task_history, system_prompt_override=_sp,
                    )
                except Exception as codex_err:
                    log.warning("Codex tool loop failed: %s", codex_err)
                    response = f"Tool execution failed: {codex_err}"
                    is_error = True
                    handoff = False
                # Skill requested Codex handoff — route skill result to Codex for response
                if handoff and self.codex_client and not is_error:
                    log.info("Skill handoff to Codex for response")
                    _skill_response = response  # Save before overwriting
                    chat_prompt = self._build_chat_system_prompt(channel=message.channel, user_id=user_id)
                    codex_messages = list(history) + [
                        {"role": "assistant", "content": f"[Tool result: {response}]"},
                        {"role": "user", "content": "Respond to the user based on the tool result above. Be conversational and helpful."},
                    ]
                    try:
                        response = await self.codex_client.chat(
                            messages=codex_messages,
                            system=chat_prompt,
                        )
                        if not response:
                            log.warning("Codex handoff returned empty, using skill result directly")
                            response = _skill_response
                        already_sent = False
                    except Exception as e:
                        log.warning("Codex handoff failed, using skill result directly: %s", e)
                        response = _skill_response
                        already_sent = False
                # Track task route usage so follow-ups classify as "task"
                self._last_tool_use[channel_id] = time.monotonic()
        except Exception as e:
            log.error("Error processing message: %s", e, exc_info=True)
            await self._send_with_retry(message, f"Something went wrong: {e}")
            self.sessions.remove_last_message(channel_id, "user")
            return

        # Scrub secrets from LLM response before logging, saving, or sending.
        # Tool output is already scrubbed (scrub_output_secrets in _run_tool),
        # but the LLM may echo, reconstruct, or hallucinate secrets in its
        # natural-language response text.
        response = scrub_response_secrets(response)

        log.info("Final response to send: %r", response[:200])
        if not is_error:
            # On the task route, if no tools were used and no skill handoff,
            # the response may be a fabrication. Save a neutral marker instead
            # to prevent poisoning future requests.
            if msg_type == "task" and not tools_used and not handoff:
                sanitized = "[Response provided without tool use.]"
                self.sessions.add_message(channel_id, "assistant", sanitized)
            else:
                self.sessions.add_message(channel_id, "assistant", response)
            self.sessions.prune()
            await asyncio.to_thread(self.sessions.save)

            # Record tool use pattern for future hints
            if tools_used:
                try:
                    await self.tool_memory.record(
                        content, tools_used, success=True, embedder=self._embedder,
                    )
                except Exception:
                    pass  # Non-critical
        else:
            # Save a sanitized error marker instead of the full error response.
            # The user sees the full error on Discord, but raw refusals and
            # fabrications are NOT persisted to prevent context poisoning.
            if tools_used:
                sanitized = (
                    f"[Previous request used tools ({', '.join(tools_used[:5])}) "
                    f"but encountered an error. The user may ask to retry.]"
                )
            else:
                sanitized = "[Previous request encountered an error before tool execution.]"
            self.sessions.add_message(channel_id, "assistant", sanitized)
            self.sessions.prune()
            await asyncio.to_thread(self.sessions.save)

        if voice_callback:
            await voice_callback(response)
        if not already_sent:
            # _send_chunked picks up pending files and attaches them to the
            # first message — text + file arrive as one Discord message.
            await self._send_chunked(message, response)
        else:
            # Streamed response already on Discord — post pending files separately
            pending = self._pending_files.pop(channel_id, [])
            if pending:
                discord_files = [
                    discord.File(io.BytesIO(data), filename=fname)
                    for data, fname in pending
                ]
                try:
                    await message.channel.send(files=discord_files)
                except Exception as e:
                    log.warning("Failed to send pending skill files: %s", e)

    async def _process_with_tools(
        self,
        message: discord.Message,
        history: list[dict],
        system_prompt_override: str | None = None,
    ) -> tuple[str, bool, bool, list[str], bool]:
        """Process a message with Codex tool loop.

        Returns (text, already_sent, is_error, tools_used, handoff):
        - text: the response text
        - already_sent: True if the response was streamed to Discord already
        - is_error: True if an error occurred (API failed, max iterations,
          circuit breaker). Error responses are saved to history for
          continuation ("keep going"). Tool memory is not recorded.
        - tools_used: list of tool names called during this loop
        - handoff: True if the response should be handed off to another handler
        """
        system_prompt = system_prompt_override or self._system_prompt
        tools = self._merged_tool_definitions() if self.config.tools.enabled else None
        messages = list(history)

        # Insert context separator between history and the current user request
        # so Codex evaluates tools fresh instead of repeating patterns from history
        if len(messages) > 1:
            separator = {
                "role": "developer",
                "content": (
                    "=== CURRENT REQUEST ===\n"
                    "The messages above are conversation history for context only. "
                    "For the user's new request below, evaluate your CURRENTLY AVAILABLE "
                    "tools and use them. Do not repeat prior refusals or text-only responses. "
                    "If a tool exists for the requested action, call it."
                ),
            }
            messages.insert(-1, separator)

        # Track which tools are used during this loop for tool memory
        # Local variable (not instance attr) to avoid cross-channel contamination
        tools_used_in_loop: list[str] = []

        # Progress embed tracking — single editable embed replaces scattered messages
        progress_embed_msg: discord.Message | None = None
        progress_steps: list[dict] = []
        cancel_view: ToolLoopCancelView | None = None

        # Session approval cache — once a tool is approved, skip re-prompting in this loop
        approved_tools: set[str] = set()

        user_id = str(message.author.id)

        # Filter tools based on user permission tier (skip for test webhooks)
        is_test_wh = message.webhook_id and str(message.webhook_id) in _ALLOWED_WEBHOOK_IDS
        if tools is not None and not is_test_wh:
            tools = self.permissions.filter_tools(user_id, tools)
            # filter_tools returns None for guest tier (no tools)
            if tools is not None and not tools:
                tools = None  # empty list → None (no tools)

        log.info("Tool loop starting: %d tools available, %d messages in history",
                 len(tools) if tools else 0, len(messages))

        for iteration in range(MAX_TOOL_ITERATIONS):
            # Check if user pressed the cancel button on the progress embed
            if cancel_view is not None and cancel_view.is_cancelled:
                if progress_embed_msg and progress_steps:
                    try:
                        cancel_view.disable()
                        embed = self._build_tool_progress_embed(progress_steps, "error",
                                                                footer="Cancelled by user.")
                        await progress_embed_msg.edit(embed=embed, view=cancel_view)
                    except Exception:
                        pass
                report = self._build_partial_completion_report(progress_steps)
                error_msg = "Task cancelled by user."
                if report:
                    error_msg = f"{report}\n\n{error_msg}"
                return error_msg, False, True, tools_used_in_loop, False
            # Show typing indicator while waiting for LLM response
            try:
                async with message.channel.typing():
                    llm_resp = await self.codex_client.chat_with_tools(
                        messages=messages,
                        system=system_prompt,
                        tools=tools or [],
                    )
            except CircuitOpenError as coe:
                # Circuit breaker open — wait for recovery, then retry once
                wait_secs = min(coe.retry_after, 90.0)
                log.info("Circuit breaker open for %s, waiting %.0fs for recovery", coe.provider, wait_secs)
                try:
                    if progress_embed_msg:
                        embed = self._build_tool_progress_embed(
                            progress_steps, "running",
                            footer=f"API recovering — retrying in {wait_secs:.0f}s...",
                        )
                        await progress_embed_msg.edit(embed=embed, view=cancel_view)
                except Exception:
                    pass
                await asyncio.sleep(wait_secs)
                try:
                    async with message.channel.typing():
                        llm_resp = await self.codex_client.chat_with_tools(
                            messages=messages,
                            system=system_prompt,
                            tools=tools or [],
                        )
                except Exception as retry_err:
                    # Recovery failed — fall through to error reporting
                    if progress_embed_msg and progress_steps:
                        try:
                            if cancel_view:
                                cancel_view.disable()
                            embed = self._build_tool_progress_embed(progress_steps, "error")
                            await progress_embed_msg.edit(embed=embed, view=cancel_view)
                        except Exception:
                            pass
                    report = self._build_partial_completion_report(progress_steps)
                    error_msg = f"LLM API error (circuit breaker recovery failed): {retry_err}"
                    if report:
                        error_msg = f"{report}\n\n{error_msg}"
                    return error_msg, False, True, tools_used_in_loop, False
            except Exception as api_err:
                # LLM API failed — report what was already completed
                if progress_embed_msg and progress_steps:
                    try:
                        if cancel_view:
                            cancel_view.disable()
                        embed = self._build_tool_progress_embed(progress_steps, "error")
                        await progress_embed_msg.edit(embed=embed, view=cancel_view)
                    except Exception:
                        pass
                report = self._build_partial_completion_report(progress_steps)
                error_msg = f"LLM API error: {api_err}"
                if report:
                    error_msg = f"{report}\n\n{error_msg}"
                return error_msg, False, True, tools_used_in_loop, False
            if not llm_resp.is_tool_use:
                # Update progress embed to show completion and disable cancel button
                if progress_embed_msg and progress_steps:
                    try:
                        if cancel_view:
                            cancel_view.disable()
                        embed = self._build_tool_progress_embed(progress_steps, "complete")
                        await progress_embed_msg.edit(embed=embed, view=cancel_view)
                    except Exception:
                        pass
                return llm_resp.text or _EMPTY_RESPONSE_FALLBACK, False, False, tools_used_in_loop, False

            # Build internal-format assistant content from LLMResponse
            assistant_content: list[dict] = []
            if llm_resp.text:
                assistant_content.append({"type": "text", "text": llm_resp.text})
            for tc in llm_resp.tool_calls:
                assistant_content.append({
                    "type": "tool_use", "id": tc.id,
                    "name": tc.name, "input": tc.input,
                })
            messages.append({"role": "assistant", "content": assistant_content})

            tool_calls_this_iter = llm_resp.tool_calls

            # Execute tools
            tool_calls = tool_calls_this_iter

            # Build progress step and send/update progress embed
            tool_names = [t.name for t in tool_calls]
            tools_used_in_loop.extend(tool_names)
            reasoning = None
            if llm_resp.text:
                reasoning = llm_resp.text if len(llm_resp.text) <= 200 else llm_resp.text[:200] + "..."
            progress_steps.append({
                "tools": tool_names,
                "reasoning": reasoning,
                "status": "running",
            })
            try:
                embed = self._build_tool_progress_embed(progress_steps, "running")
                if progress_embed_msg is None:
                    cancel_view = ToolLoopCancelView(
                        allowed_user_ids=self.config.discord.allowed_users,
                    )
                    progress_embed_msg = await message.channel.send(
                        embed=embed, view=cancel_view,
                    )
                else:
                    await progress_embed_msg.edit(embed=embed)
            except Exception:
                pass

            # Execute tools in parallel
            async def _run_tool(block):
                nonlocal system_prompt
                tool_name = block.name
                tool_input = block.input
                log.info("Tool call: %s(%s)", tool_name, tool_input)

                approved = True
                # Auto-approve for test webhooks
                is_test_wh = message.webhook_id and str(message.webhook_id) in _ALLOWED_WEBHOOK_IDS
                # Check skill manager first for approval, fall through to registry
                skill_approval = self.skill_manager.requires_approval(tool_name)
                needs_approval = skill_approval if skill_approval is not None else requires_approval(tool_name)
                if needs_approval and tool_name not in approved_tools and not is_test_wh:
                    approved = await request_approval(
                        bot=self,
                        channel=message.channel,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        allowed_users=self.config.discord.allowed_users,
                        timeout=self.config.tools.approval_timeout_seconds,
                    )
                    if approved:
                        approved_tools.add(tool_name)
                    if not approved:
                        await self.audit.log_execution(
                            user_id=str(message.author.id),
                            user_name=str(message.author),
                            channel_id=str(message.channel.id),
                            tool_name=tool_name,
                            tool_input=tool_input,
                            approved=False,
                            result_summary="Denied or timed out",
                            execution_time_ms=0,
                        )
                        return {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Action was denied or timed out.",
                        }

                t0 = time.monotonic()
                error = None
                # Handle Discord-native tools
                try:
                    if tool_name == "purge_messages":
                        result = await self._handle_purge(message, tool_input)
                    elif tool_name == "browser_screenshot":
                        result = await self._handle_browser_screenshot(message, tool_input)
                    elif tool_name == "generate_file":
                        result = await self._handle_generate_file(message, tool_input)
                    elif tool_name == "post_file":
                        result = await self._handle_post_file(message, tool_input)
                    elif tool_name == "schedule_task":
                        result = self._handle_schedule_task(message, tool_input)
                    elif tool_name == "list_schedules":
                        result = self._handle_list_schedules()
                    elif tool_name == "delete_schedule":
                        result = self._handle_delete_schedule(tool_input)
                    elif tool_name == "parse_time":
                        result = self._handle_parse_time(tool_input)
                    elif tool_name == "search_history":
                        result = await self._handle_search_history(tool_input)
                    elif tool_name == "delegate_task":
                        result = await self._handle_delegate_task(message, tool_input)
                    elif tool_name == "list_tasks":
                        result = self._handle_list_tasks(tool_input)
                    elif tool_name == "cancel_task":
                        result = self._handle_cancel_task(tool_input)
                    elif tool_name == "search_knowledge":
                        result = await self._handle_search_knowledge(tool_input)
                    elif tool_name == "ingest_document":
                        result = await self._handle_ingest_document(tool_input, str(message.author))
                    elif tool_name == "list_knowledge":
                        result = self._handle_list_knowledge()
                    elif tool_name == "delete_knowledge":
                        result = self._handle_delete_knowledge(tool_input)
                    elif tool_name == "set_permission":
                        result = self._handle_set_permission(
                            str(message.author.id), tool_input,
                        )
                    elif tool_name == "search_audit":
                        result = await self._handle_search_audit(tool_input)
                    elif tool_name == "create_digest":
                        result = self._handle_create_digest(message, tool_input)
                    elif tool_name == "create_skill":
                        result = await asyncio.to_thread(
                            self.skill_manager.create_skill, tool_input["name"], tool_input["code"],
                        )
                        system_prompt = self._build_system_prompt(channel=message.channel, user_id=user_id)
                    elif tool_name == "edit_skill":
                        result = await asyncio.to_thread(
                            self.skill_manager.edit_skill, tool_input["name"], tool_input["code"],
                        )
                        system_prompt = self._build_system_prompt(channel=message.channel, user_id=user_id)
                    elif tool_name == "delete_skill":
                        result = await asyncio.to_thread(
                            self.skill_manager.delete_skill, tool_input["name"],
                        )
                        system_prompt = self._build_system_prompt(channel=message.channel, user_id=user_id)
                    elif tool_name == "list_skills":
                        skills = self.skill_manager.list_skills()
                        if not skills:
                            result = "No user-created skills."
                        else:
                            lines = []
                            for s in skills:
                                appr = " [requires approval]" if s["requires_approval"] else ""
                                lines.append(f"**{s['name']}**: {s['description']}{appr}")
                            result = f"**User-created skills ({len(skills)}):**\n" + "\n".join(lines)
                    elif self.skill_manager.has_skill(tool_name):
                        async def _skill_msg(text: str) -> None:
                            await message.channel.send(text)
                        async def _skill_file(data: bytes, filename: str, caption: str = "") -> None:
                            channel_id_key = str(message.channel.id)
                            self._pending_files.setdefault(channel_id_key, []).append((data, filename))
                        result = await self.skill_manager.execute(
                            tool_name, tool_input,
                            message_callback=_skill_msg,
                            file_callback=_skill_file,
                        )
                    else:
                        result = await self.tool_executor.execute(tool_name, tool_input, user_id=user_id)
                except Exception as e:
                    error = str(e)
                    result = f"Error executing {tool_name}: {e}"

                elapsed_ms = int((time.monotonic() - t0) * 1000)

                # Scrub secrets from tool output
                result = scrub_output_secrets(result)

                # Audit log
                await self.audit.log_execution(
                    user_id=str(message.author.id),
                    user_name=str(message.author),
                    channel_id=str(message.channel.id),
                    tool_name=tool_name,
                    tool_input=tool_input,
                    approved=approved,
                    result_summary=result,
                    execution_time_ms=elapsed_ms,
                    error=error,
                )

                # Track for conversational context
                self._track_recent_action(
                    tool_name, tool_input, result[:200], elapsed_ms,
                    channel_id=str(message.channel.id),
                )

                # Truncate large outputs before sending back to the LLM.
                # Tool results stay in messages and are re-sent every iteration,
                # so capping here prevents a single large result from ballooning
                # input token costs across the tool loop.
                tool_content = truncate_tool_output(result)

                return {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_content,
                }

            # Run all tool calls concurrently with per-tool timeout
            tool_timeout = self.config.tools.tool_timeout_seconds

            async def _run_tool_with_timeout(block):
                try:
                    return await asyncio.wait_for(
                        _run_tool(block), timeout=tool_timeout,
                    )
                except asyncio.TimeoutError:
                    error_msg = (
                        f"Tool '{block.name}' timed out after {tool_timeout}s"
                    )
                    try:
                        await self.audit.log_execution(
                            user_id=str(message.author.id),
                            user_name=str(message.author),
                            channel_id=str(message.channel.id),
                            tool_name=block.name,
                            tool_input=block.input,
                            approved=True,
                            result_summary=error_msg,
                            execution_time_ms=int(tool_timeout * 1000),
                            error=error_msg,
                        )
                    except Exception:
                        pass
                    return {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": error_msg,
                    }

            step_t0 = time.monotonic()
            tool_results = await asyncio.gather(
                *[_run_tool_with_timeout(b) for b in tool_calls],
            )
            messages.append({"role": "user", "content": list(tool_results)})

            # Update progress step to done with elapsed time
            step_elapsed_ms = int((time.monotonic() - step_t0) * 1000)
            progress_steps[-1]["status"] = "done"
            progress_steps[-1]["elapsed_ms"] = step_elapsed_ms
            if progress_embed_msg:
                try:
                    embed = self._build_tool_progress_embed(progress_steps, "running")
                    await progress_embed_msg.edit(embed=embed)
                except Exception:
                    pass

            # Check if all tool calls in this iteration are skills that want
            # Codex to handle the response instead of another tool-loop iteration.
            tool_names_this_round = [b.name for b in tool_calls]
            if (
                getattr(self, "codex_client", None)
                and all(self.skill_manager.should_handoff_to_codex(n) is True for n in tool_names_this_round)
            ):
                # Update progress embed to complete on handoff
                if progress_embed_msg and progress_steps:
                    try:
                        if cancel_view:
                            cancel_view.disable()
                        embed = self._build_tool_progress_embed(progress_steps, "complete")
                        await progress_embed_msg.edit(embed=embed, view=cancel_view)
                    except Exception:
                        pass
                # Collect skill results as context for Codex
                skill_output = "\n".join(
                    r["content"] for r in tool_results if isinstance(r, dict)
                )
                return skill_output, False, False, tools_used_in_loop, True  # handoff=True

        # Update progress embed to error on max iterations
        if progress_embed_msg and progress_steps:
            try:
                if cancel_view:
                    cancel_view.disable()
                embed = self._build_tool_progress_embed(progress_steps, "error")
                await progress_embed_msg.edit(embed=embed, view=cancel_view)
            except Exception:
                pass
        report = self._build_partial_completion_report(progress_steps)
        error_msg = "Too many tool calls in sequence. Please try a simpler request."
        if report:
            error_msg = f"{report}\n\n{error_msg}"
        return error_msg, False, True, tools_used_in_loop, False

    @staticmethod
    def _build_tool_progress_embed(
        steps: list[dict],
        status: str = "running",
        footer: str | None = None,
    ) -> discord.Embed:
        """Build a Discord embed showing tool loop progress.

        Args:
            steps: list of dicts with keys: tools (list[str]), reasoning (str|None),
                   status ("running"|"done"), elapsed_ms (int|None)
            status: overall status — "running", "complete", or "error"
            footer: optional footer text (e.g. recovery wait message)
        """
        colors = {
            "running": discord.Color.blue(),
            "complete": discord.Color.green(),
            "error": discord.Color.red(),
        }

        lines = []
        for i, step in enumerate(steps, 1):
            tool_str = ", ".join(f"`{t}`" for t in step["tools"])
            if step["status"] == "done":
                secs = step.get("elapsed_ms", 0) / 1000
                lines.append(f"\u2713 Step {i}: {tool_str} ({secs:.1f}s)")
            else:
                lines.append(f"\u25b6 Step {i}: {tool_str}...")

        # Show reasoning from the latest running step
        latest = steps[-1] if steps else None
        if latest and latest.get("reasoning") and latest["status"] == "running":
            lines.append(f"\n*{latest['reasoning']}*")

        if footer:
            lines.append(f"\n{footer}")

        description = "\n".join(lines)
        if len(description) > 4000:
            description = description[:4000] + "\n*(truncated)*"

        return discord.Embed(
            description=description or "Starting...",
            color=colors.get(status, discord.Color.blue()),
        )

    @staticmethod
    def _build_partial_completion_report(steps: list[dict]) -> str:
        """Build a human-readable summary of completed steps for partial failure.

        Used when the tool loop exits early (API error, max iterations) so the
        user can see what was already accomplished before the failure.
        """
        done = [s for s in steps if s.get("status") == "done"]
        if not done:
            return ""
        lines = [f"**Partial completion ({len(done)}/{len(steps)} steps):**"]
        for i, step in enumerate(done, 1):
            tool_str = ", ".join(f"`{t}`" for t in step["tools"])
            secs = step.get("elapsed_ms", 0) / 1000
            lines.append(f"\u2713 Step {i}: {tool_str} ({secs:.1f}s)")
        return "\n".join(lines)

    @staticmethod
    def _detect_image_type(data: bytes) -> str | None:
        """Detect image media type from file magic bytes."""
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if data[:2] == b"\xff\xd8":
            return "image/jpeg"
        if data[:4] == b"GIF8":
            return "image/gif"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        return None

    async def _handle_purge(self, message: discord.Message, inp: dict) -> str:
        """Delete recent messages in the channel."""
        count = min(inp.get("count", 100), 500)
        try:
            deleted = await message.channel.purge(limit=count)
            self.sessions.reset(str(message.channel.id))
            return f"Deleted {len(deleted)} messages and reset conversation history."
        except discord.Forbidden:
            return "I don't have permission to delete messages in this channel."
        except Exception as e:
            return f"Failed to purge messages: {e}"

    async def _handle_browser_screenshot(self, message: discord.Message, inp: dict) -> str:
        """Take a browser screenshot and post it as a Discord image."""
        if not self.browser_manager:
            return "Browser automation is not enabled. Set browser.enabled=true in config."
        from ..tools.browser import handle_browser_screenshot
        try:
            text, screenshot_bytes = await handle_browser_screenshot(self.browser_manager, inp)
            if screenshot_bytes:
                discord_file = discord.File(io.BytesIO(screenshot_bytes), filename="screenshot.png")
                await message.channel.send(file=discord_file)
            return text
        except Exception as e:
            return f"Browser screenshot failed: {e}"

    async def _handle_generate_file(self, message: discord.Message, inp: dict) -> str:
        """Generate a file from content and post it as a Discord attachment."""
        filename = inp.get("filename", "output.txt")
        content = inp.get("content", "")
        caption = inp.get("caption", "")

        file_bytes = content.encode("utf-8")
        discord_file = discord.File(io.BytesIO(file_bytes), filename=filename)
        try:
            await message.channel.send(content=caption or None, file=discord_file)
            return f"File `{filename}` ({len(file_bytes)} bytes) posted to channel."
        except Exception as e:
            return f"Failed to post file: {e}"

    async def _handle_post_file(self, message: discord.Message, inp: dict) -> str:
        """Fetch a file from a remote host via SSH and post it to Discord."""
        host_alias = inp.get("host")
        path = inp.get("path")
        caption = inp.get("caption", "")

        if not host_alias or not path:
            return "Both 'host' and 'path' are required."

        resolved = self.tool_executor._resolve_host(host_alias)
        if not resolved:
            return f"Unknown or disallowed host: {host_alias}"
        address, ssh_user, _os = resolved

        # Fetch file as base64 via SSH (handles binary safely)
        import shlex
        safe_path = shlex.quote(path)
        ssh_args = [
            "ssh",
            "-i", self.config.tools.ssh_key_path,
            "-o", f"UserKnownHostsFile={self.config.tools.ssh_known_hosts_path}",
            "-o", "StrictHostKeyChecking=yes",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            f"{ssh_user}@{address}",
            f"base64 {safe_path}",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                return f"Failed to fetch file: {stderr.decode('utf-8', errors='replace').strip()}"
            file_bytes = base64.b64decode(stdout)
        except asyncio.TimeoutError:
            return "File fetch timed out (30s)."
        except Exception as e:
            return f"Failed to fetch file: {e}"

        if not file_bytes:
            return f"File not found or empty: {path}"

        # Size check (Discord limit: 25MB for non-boosted servers)
        if len(file_bytes) > 25 * 1024 * 1024:
            return f"File too large to post ({len(file_bytes) / 1024 / 1024:.1f} MB). Discord limit is 25 MB."

        filename = os.path.basename(path)
        try:
            file = discord.File(io.BytesIO(file_bytes), filename=filename)
            await message.channel.send(content=caption or None, file=file)
            return f"Posted `{filename}` ({len(file_bytes) / 1024:.1f} KB) to the channel."
        except discord.HTTPException as e:
            return f"Failed to upload to Discord: {e}"

    # Scheduling intent patterns — the user's current message must contain
    # at least one of these to allow schedule_task.  Prevents the LLM from
    # proactively scheduling things based on conversation history alone.
    _SCHEDULE_INTENT_RE = re.compile(
        r"|".join([
            r"\bschedule\b", r"\bremind\b", r"\balarm\b", r"\btimer\b",
            r"\bevery\s+\d", r"\bcron\b", r"\bat\s+\d",
            r"\bin\s+\d+\s+(?:min|hour|day|sec)",
            r"\brecurring\b", r"\bdaily\b", r"\bweekly\b", r"\bhourly\b",
            r"\btomorrow\b", r"\btonight\b",
            r"\bnext\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month)\b",
        ]),
        re.IGNORECASE,
    )

    def _handle_schedule_task(self, message: discord.Message, inp: dict) -> str:
        """Create a scheduled task."""
        # Guard: only schedule if the user's current message expresses intent
        user_text = getattr(message, "content", "") or ""
        if not self._SCHEDULE_INTENT_RE.search(user_text):
            log.warning(
                "schedule_task blocked: no scheduling intent in user message %r",
                user_text[:120],
            )
            return (
                "I considered scheduling a task, but your message doesn't appear to "
                "request scheduling. If you'd like me to schedule something, please "
                "ask explicitly (e.g. 'remind me to…', 'schedule a check every…')."
            )
        try:
            schedule = self.scheduler.add(
                description=inp.get("description", "Unnamed task"),
                action=inp.get("action", "reminder"),
                channel_id=str(message.channel.id),
                cron=inp.get("cron"),
                run_at=inp.get("run_at"),
                message=inp.get("message"),
                tool_name=inp.get("tool_name"),
                tool_input=inp.get("tool_input"),
                steps=inp.get("steps"),
                trigger=inp.get("trigger"),
            )
            if schedule.get("trigger"):
                trigger_desc = ", ".join(
                    f"{k}={v}" for k, v in schedule["trigger"].items()
                )
                return (
                    f"Scheduled webhook-triggered task (ID: {schedule['id']}): "
                    f"{schedule['description']}. Trigger: {trigger_desc}"
                )
            next_run = schedule.get("next_run", "unknown")
            stype = "recurring" if schedule.get("cron") else "one-time"
            return (
                f"Scheduled {stype} task (ID: {schedule['id']}): "
                f"{schedule['description']}. Next run: {next_run}"
            )
        except ValueError as e:
            return f"Failed to create schedule: {e}"
        except Exception as e:
            return f"Error creating schedule: {e}"

    def _handle_list_schedules(self) -> str:
        """List all scheduled tasks."""
        schedules = self.scheduler.list_all()
        if not schedules:
            return "No scheduled tasks."
        lines = []
        for s in schedules:
            if s.get("trigger"):
                trigger_desc = ", ".join(
                    f"{k}={v}" for k, v in s["trigger"].items()
                )
                stype = f"trigger: {trigger_desc}"
            elif s.get("cron"):
                stype = f"cron `{s['cron']}`"
            else:
                stype = "one-time"
            next_run = s.get("next_run", "on trigger" if s.get("trigger") else "N/A")
            last_run = s.get("last_run", "never")
            lines.append(
                f"- **{s['id']}**: {s['description']} ({stype}) "
                f"| next: {next_run} | last: {last_run}"
            )
        return f"**Scheduled tasks ({len(schedules)}):**\n" + "\n".join(lines)

    def _handle_delete_schedule(self, inp: dict) -> str:
        """Delete a scheduled task."""
        schedule_id = inp.get("schedule_id", "")
        if self.scheduler.delete(schedule_id):
            return f"Deleted schedule {schedule_id}."
        return f"Schedule {schedule_id} not found."

    def _handle_parse_time(self, inp: dict) -> str:
        """Parse a natural language time expression to ISO datetime."""
        expression = inp.get("expression", "")
        if not expression:
            return "Error: 'expression' is required (e.g. 'in 2 hours', 'tomorrow at 9am')"
        from ..tools.time_parser import parse_time

        try:
            result = parse_time(expression)
            return f"Parsed '{expression}' → {result}"
        except ValueError as e:
            return f"Error: {e}"

    async def _handle_search_history(self, inp: dict) -> str:
        """Search past conversation history."""
        query = inp.get("query", "")
        limit = min(inp.get("limit", 10), 20)
        if not query:
            return "A search query is required."

        results = await self.sessions.search_history(query, limit=limit)
        if not results:
            return f"No past conversations found matching '{query}'."

        lines = []
        for r in results:
            from datetime import datetime
            ts = datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M")
            role = r["type"]
            content = r["content"].replace("\n", " ")[:300]
            lines.append(f"[{ts}] ({role}): {content}")

        return f"**Found {len(results)} result(s) for '{query}':**\n" + "\n".join(lines)

    async def _handle_search_knowledge(self, inp: dict) -> str:
        """Semantic search over the knowledge base."""
        if not self._knowledge_store or not self._embedder:
            return "Knowledge base is not available (search not enabled or ChromaDB not initialized)."

        query = inp.get("query", "")
        limit = min(inp.get("limit", 5), 10)
        if not query:
            return "A search query is required."

        results = await self._knowledge_store.search_hybrid(query, self._embedder, limit=limit)
        if not results:
            return f"No knowledge base results for '{query}'. Try web_search for external information."

        lines = []
        for r in results:
            source = r["source"]
            score = r["score"]
            content = r["content"].replace("\n", " ")[:500]
            lines.append(f"**[{source}]** (score: {score})\n{content}")

        return f"**Found {len(results)} result(s) for '{query}':**\n\n" + "\n\n".join(lines)

    async def _handle_ingest_document(self, inp: dict, uploader: str) -> str:
        """Ingest a document into the knowledge base."""
        if not self._knowledge_store or not self._embedder:
            return "Knowledge base is not available (search not enabled or ChromaDB not initialized)."

        source = inp.get("source", "")
        content = inp.get("content", "")
        if not source or not content:
            return "Both 'source' and 'content' are required."

        count = await self._knowledge_store.ingest(
            content=content,
            source=source,
            embedder=self._embedder,
            uploader=uploader,
        )
        if count == 0:
            return f"Failed to ingest '{source}' — no chunks could be embedded."
        return f"Ingested '{source}' into knowledge base ({count} chunks indexed)."

    def _handle_list_knowledge(self) -> str:
        """List all documents in the knowledge base."""
        if not self._knowledge_store:
            return "Knowledge base is not available."

        sources = self._knowledge_store.list_sources()
        if not sources:
            return "Knowledge base is empty. Use ingest_document to add documents."

        lines = []
        for s in sources:
            lines.append(f"- **{s['source']}** ({s['chunks']} chunks, by {s['uploader']}, {s['ingested_at'][:10]})")
        total = sum(s["chunks"] for s in sources)
        return f"**Knowledge base: {len(sources)} document(s), {total} total chunks**\n" + "\n".join(lines)

    def _handle_delete_knowledge(self, inp: dict) -> str:
        """Delete a document from the knowledge base."""
        if not self._knowledge_store:
            return "Knowledge base is not available."

        source = inp.get("source", "")
        if not source:
            return "'source' is required."

        count = self._knowledge_store.delete_source(source)
        if count == 0:
            return f"No document found with source '{source}'."
        return f"Deleted '{source}' from knowledge base ({count} chunks removed)."

    def _handle_set_permission(self, caller_id: str, inp: dict) -> str:
        """Set a user's permission tier. Only admins can call this."""
        if not self.permissions.is_admin(caller_id):
            return "Permission denied. Only admins can change permission tiers."
        target_user_id = inp["user_id"]
        tier = inp["tier"]
        try:
            self.permissions.set_tier(target_user_id, tier)
        except ValueError as e:
            return str(e)
        return f"Permission tier for user {target_user_id} set to **{tier}**."

    async def _handle_search_audit(self, inp: dict) -> str:
        """Search the audit log."""
        results = await self.audit.search(
            tool_name=inp.get("tool_name"),
            user=inp.get("user"),
            host=inp.get("host"),
            keyword=inp.get("keyword"),
            date=inp.get("date"),
            limit=min(inp.get("limit", 20), 50),
        )
        if not results:
            return "No audit log entries found matching the criteria."

        lines = []
        for entry in results:
            ts = entry.get("timestamp", "?")[:19]
            tool = entry.get("tool_name", "?")
            user = entry.get("user_name", "?")
            approved = "approved" if entry.get("approved") else "denied"
            elapsed = entry.get("execution_time_ms", 0)
            summary = entry.get("result_summary", "")[:200]
            err = entry.get("error")
            status = f"ERROR: {err}" if err else summary
            lines.append(
                f"[{ts}] **{tool}** by {user} ({approved}, {elapsed}ms)\n  {status}"
            )
        return f"**Audit log ({len(results)} entries):**\n" + "\n".join(lines)

    def _handle_create_digest(self, message: discord.Message, inp: dict) -> str:
        """Create a scheduled daily digest."""
        cron = inp.get("cron", "0 8 * * *")
        description = inp.get("description", "Daily Infrastructure Digest")
        try:
            schedule = self.scheduler.add(
                description=description,
                action="digest",
                channel_id=str(message.channel.id),
                cron=cron,
            )
            return (
                f"Created digest schedule (ID: {schedule['id']}): {description}\n"
                f"Cron: `{cron}` | Next run: {schedule['next_run']}"
            )
        except ValueError as e:
            return f"Failed to create digest: {e}"

    async def _on_scheduled_digest(self, schedule: dict) -> None:
        """Run the daily infrastructure digest and post results."""
        channel_id = schedule.get("channel_id")
        if not channel_id:
            log.warning("Digest has no channel_id: %s", schedule["id"])
            return

        channel = self.get_channel(int(channel_id))
        if not channel:
            log.warning("Digest channel %s not found", channel_id)
            return

        log.info("Running daily digest for channel %s", channel_id)
        try:
            raw = await self._format_digest_raw()
        except Exception as e:
            log.error("Digest data collection failed: %s", e)
            await channel.send(f"**Daily Infrastructure Digest**\n\nFailed to collect data: {e}")
            return

        # Summarize the digest — prefer Codex (free), fall back to raw truncation
        digest_messages = [{"role": "user", "content": f"Summarize this infrastructure status report concisely. Highlight any issues, warnings, or anomalies. If everything looks healthy, say so briefly.\n\n{raw}"}]
        digest_system = "You are a concise infrastructure report summarizer. Output a short summary with key findings."
        try:
            if self.codex_client:
                summary = await self.codex_client.chat(
                    messages=digest_messages, system=digest_system, max_tokens=500,
                )
            else:
                log.warning("No Codex client for digest summary, using raw")
                summary = raw[:3000]
        except Exception as e:
            log.warning("Digest summary failed, using raw: %s", e)
            summary = raw[:3000]

        await channel.send(f"**Daily Infrastructure Digest**\n\n{summary}")

        # Audit log the digest
        await self.audit.log_execution(
            user_id="system",
            user_name="scheduler",
            channel_id=channel_id,
            tool_name="digest",
            tool_input={"schedule_id": schedule.get("id")},
            approved=True,
            result_summary=summary[:500],
            execution_time_ms=0,
        )

    async def _format_digest_raw(self) -> str:
        """Collect raw infrastructure data for the digest."""
        tasks = []
        labels = []

        # Disk + memory checks on all hosts
        for host_alias in self.config.tools.hosts:
            tasks.append(self.tool_executor.execute("check_disk", {"host": host_alias}))
            labels.append(f"Disk ({host_alias})")
            tasks.append(self.tool_executor.execute("check_memory", {"host": host_alias}))
            labels.append(f"Memory ({host_alias})")

        # Prometheus firing alerts
        tasks.append(self.tool_executor.execute(
            "query_prometheus", {"query": "ALERTS{alertstate='firing'}"}
        ))
        labels.append("Prometheus Alerts")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        sections = []
        for label, result in zip(labels, results):
            if isinstance(result, Exception):
                sections.append(f"### {label}\nERROR: {result}")
            else:
                sections.append(f"### {label}\n{result[:800]}")

        return "\n\n".join(sections)

    def _resolve_mentions(self, text: str) -> str:
        """Replace @username with proper Discord <@ID> mentions."""
        def _replace(match: re.Match) -> str:
            name = match.group(1).lower()
            for guild in self.guilds:
                for member in guild.members:
                    if member.name.lower() == name or (member.nick and member.nick.lower() == name):
                        return f"<@{member.id}>"
            return match.group(0)  # leave unchanged if not found
        return re.sub(r"@(\w+)", _replace, text)

    async def _on_monitor_alert(self, message: str) -> None:
        """Callback fired by the infrastructure watcher when a threshold is crossed."""
        channel_id = self.config.monitoring.alert_channel_id
        if not channel_id:
            # Fall back to first configured channel
            if self.config.discord.channels:
                channel_id = self.config.discord.channels[0]
            else:
                log.warning("Monitor alert has no channel to send to: %s", message[:100])
                return

        channel = self.get_channel(int(channel_id))
        if not channel:
            log.warning("Monitor alert channel %s not found", channel_id)
            return

        try:
            await channel.send(message)
            log.info("Sent monitor alert to channel %s", channel_id)
        except Exception as e:
            log.error("Failed to send monitor alert: %s", e)

    # --- Background task delegation ---

    async def _handle_delegate_task(self, message: discord.Message, inp: dict) -> str:
        """Create and start a background task."""
        description = inp.get("description", "Background task")
        steps = inp.get("steps", [])

        if not steps or not isinstance(steps, list):
            return "No steps provided."
        if len(steps) > MAX_STEPS:
            return f"Too many steps ({len(steps)}). Maximum is {MAX_STEPS}."

        # Validate all steps have tool_name
        for i, step in enumerate(steps):
            if not isinstance(step, dict) or "tool_name" not in step:
                return f"Step {i}: must have 'tool_name'."

        task = BackgroundTask(
            task_id=create_task_id(),
            description=description,
            steps=steps,
            channel=message.channel,
            requester=str(message.author),
            requester_id=str(message.author.id),
        )

        # Prune old completed tasks
        completed = [
            tid for tid, t in self._background_tasks.items()
            if t.status in ("completed", "failed", "cancelled")
        ]
        while len(completed) > self._background_tasks_max:
            old = completed.pop(0)
            del self._background_tasks[old]

        self._background_tasks[task.task_id] = task

        # Launch in background
        async def _run():
            try:
                await run_background_task(
                    task, self.tool_executor, self.skill_manager,
                    knowledge_store=self._knowledge_store,
                    embedder=self._embedder,
                    audit_logger=self.audit,
                )
            except Exception as e:
                log.error("Background task %s crashed: %s", task.task_id, e, exc_info=True)
                task.status = "failed"

        task._asyncio_task = asyncio.create_task(_run())

        return (
            f"Background task started (ID: `{task.task_id}`): **{description}** "
            f"({len(steps)} steps). Progress will be posted to this channel."
        )

    def _handle_list_tasks(self, inp: dict | None = None) -> str:
        """List background tasks, or get detailed results for a specific task."""
        if not self._background_tasks:
            return "No background tasks."

        task_id = (inp or {}).get("task_id")

        # Detailed view for a specific task
        if task_id:
            task = self._background_tasks.get(task_id)
            if not task:
                return f"No task found with ID `{task_id}`."
            lines = [
                f"**{task.description}** [{task.status}]",
                f"ID: `{task.task_id}` | {len(task.results)}/{len(task.steps)} steps",
                "",
            ]
            for r in task.results:
                icon = {"ok": "+", "error": "!", "skipped": "-", "cancelled": "x"}.get(r.status, "?")
                lines.append(f"[{icon}] **Step {r.index + 1} ({r.description})** ({r.elapsed_ms}ms):")
                lines.append(r.output if r.output else "(no output)")
                lines.append("")
            text = "\n".join(lines)
            if len(text) > 3800:
                text = text[:3800] + "\n... (truncated, full results were posted in the progress message)"
            return text

        # Overview of all tasks
        lines = []
        for tid, t in self._background_tasks.items():
            done = len(t.results)
            total = len(t.steps)
            ok = sum(1 for r in t.results if r.status == "ok")
            errors = sum(1 for r in t.results if r.status == "error")
            lines.append(
                f"- `{tid}` [{t.status}] **{t.description}** "
                f"({done}/{total} steps, {ok} ok, {errors} errors)"
            )
        return "\n".join(lines)

    def _handle_cancel_task(self, inp: dict) -> str:
        """Cancel a running background task."""
        task_id = inp.get("task_id", "")
        task = self._background_tasks.get(task_id)
        if not task:
            return f"No task found with ID `{task_id}`."
        if task.status != "running":
            return f"Task `{task_id}` is not running (status: {task.status})."
        task.cancel()
        return f"Cancellation requested for task `{task_id}`."

    async def _run_scheduled_workflow(
        self, channel: discord.abc.Messageable, schedule: dict,
    ) -> None:
        """Execute a multi-step workflow from a scheduled task."""
        steps = schedule.get("steps", [])
        desc = schedule.get("description", "Workflow")
        results: list[str] = []
        prev_output = ""

        for i, step in enumerate(steps):
            tool_name = step["tool_name"]
            tool_input = step.get("tool_input", {})
            condition = step.get("condition")
            step_desc = step.get("description", tool_name)

            # Evaluate condition against previous step's output
            if condition and prev_output:
                if condition.startswith("!"):
                    # Negated condition: skip if substring IS present
                    if condition[1:].lower() in prev_output.lower():
                        results.append(f"**Step {i+1}** (`{step_desc}`): skipped (condition `{condition}` met)")
                        continue
                else:
                    # Normal condition: skip if substring is NOT present
                    if condition.lower() not in prev_output.lower():
                        results.append(f"**Step {i+1}** (`{step_desc}`): skipped (condition `{condition}` not met)")
                        continue

            try:
                # Check if this is a skill or built-in tool
                if self.skill_manager.has_skill(tool_name):
                    output = await self.skill_manager.execute(tool_name, tool_input)
                else:
                    output = await self.tool_executor.execute(tool_name, tool_input)
                prev_output = output
                results.append(f"**Step {i+1}** (`{step_desc}`): OK\n```\n{output[:400]}\n```")
            except Exception as e:
                results.append(f"**Step {i+1}** (`{step_desc}`): FAILED — {e}")
                on_failure = step.get("on_failure", "abort")
                if on_failure == "abort":
                    results.append("Workflow aborted due to step failure.")
                    break
                # "continue" keeps going

        summary = "\n".join(results)
        text = f"**Workflow: {desc}**\n{summary}"
        # Truncate if needed
        if len(text) > 1900:
            text = text[:1900] + "\n... (truncated)"

        try:
            await channel.send(text)
        except Exception as e:
            log.error("Failed to post workflow results: %s", e)

    async def _on_scheduled_task(self, schedule: dict) -> None:
        """Callback fired by the scheduler when a task is due."""
        channel_id = schedule.get("channel_id")
        if not channel_id:
            log.warning("Scheduled task has no channel_id: %s", schedule["id"])
            return

        channel = self.get_channel(int(channel_id))
        if not channel:
            log.warning("Scheduled task channel %s not found", channel_id)
            return

        if schedule["action"] == "digest":
            await self._on_scheduled_digest(schedule)
            return

        if schedule["action"] == "reminder":
            msg = schedule.get("message", schedule["description"])
            # Resolve @username mentions to proper Discord <@ID> mentions
            msg = self._resolve_mentions(msg)
            await channel.send(f"**Scheduled reminder:** {msg}")

        elif schedule["action"] == "check":
            tool_name = schedule.get("tool_name")
            tool_input = schedule.get("tool_input", {})
            try:
                result = await self.tool_executor.execute(tool_name, tool_input)
                text = f"**Scheduled: {schedule['description']}**\n```\n{result[:1800]}\n```"
                await channel.send(text)
            except Exception as e:
                await channel.send(
                    f"**Scheduled task failed:** {schedule['description']}\nError: {e}"
                )

        elif schedule["action"] == "workflow":
            await self._run_scheduled_workflow(channel, schedule)

        else:
            log.warning("Unknown scheduled action type: %s (schedule %s)", schedule["action"], schedule.get("id"))

    async def _send_with_retry(
        self,
        message: discord.Message,
        text: str,
        as_reply: bool = True,
        files: list[discord.File] | None = None,
    ) -> discord.Message | None:
        """Send a message with retry on failure. Optionally attach files."""
        for attempt in range(SEND_MAX_RETRIES):
            try:
                log.info("Sending message (attempt %d, reply=%s): %r", attempt + 1, as_reply, text[:100])
                kwargs: dict = {}
                if files:
                    kwargs["files"] = files
                if as_reply:
                    sent = await message.reply(text, **kwargs)
                else:
                    sent = await message.channel.send(text, **kwargs)
                log.info("Message sent successfully: msg_id=%s", sent.id if sent else "None")
                return sent
            except discord.HTTPException as e:
                if attempt < SEND_MAX_RETRIES - 1:
                    log.warning("Discord send failed (attempt %d): %s", attempt + 1, e)
                    await asyncio.sleep(1 + attempt)
                else:
                    log.error("Discord send failed after %d retries: %s", SEND_MAX_RETRIES, e)
        return None

    async def _send_chunked(self, message: discord.Message, text: str) -> None:
        """Send a response, splitting into chunks if it exceeds Discord's limit.
        If the response is very long, send as a file attachment instead.
        Attaches any pending skill files to the first message."""
        # Collect pending file attachments from skills (per-channel)
        pending = self._pending_files.pop(str(message.channel.id), [])

        discord_files = [
            discord.File(io.BytesIO(data), filename=fname)
            for data, fname in pending
        ]

        # If the response is extremely long, send as file
        if len(text) > DISCORD_MAX_LEN * 4:
            text_file = discord.File(
                io.BytesIO(text.encode("utf-8")),
                filename="response.md",
            )
            discord_files.append(text_file)
            await self._send_with_retry(message, "Response too long for chat, attached as file:", files=discord_files)
            return

        if len(text) <= DISCORD_MAX_LEN:
            if discord_files:
                await self._send_with_retry(message, text, files=discord_files)
            else:
                await self._send_with_retry(message, text)
            return

        chunks: list[str] = []
        current = ""
        in_code_block = False

        # Pre-split any lines longer than the chunk limit so the chunker
        # never encounters a single line that can't fit in one chunk.
        max_line_len = DISCORD_MAX_LEN - 20
        lines: list[str] = []
        for raw_line in text.split("\n"):
            while len(raw_line) > max_line_len:
                lines.append(raw_line[:max_line_len])
                raw_line = raw_line[max_line_len:]
            lines.append(raw_line)

        for line in lines:
            if line.startswith("```"):
                in_code_block = not in_code_block

            if len(current) + len(line) + 1 > DISCORD_MAX_LEN - 10:
                if in_code_block:
                    current += "\n```"
                if current.strip():
                    chunks.append(current)
                current = ""
                if in_code_block:
                    current = "```\n"
            current += line + "\n"

        if current.strip():
            chunks.append(current)

        for i, chunk in enumerate(chunks):
            if i == 0 and discord_files:
                await self._send_with_retry(message, chunk, files=discord_files)
            elif i == 0:
                await self._send_with_retry(message, chunk)
            else:
                await self._send_with_retry(message, chunk, as_reply=False)
