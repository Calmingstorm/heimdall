from __future__ import annotations

import asyncio
import base64
import collections
import io
import os
import re
import time
from collections.abc import Callable

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
from ..tools.autonomous_loop import LoopManager
from ..learning import ConversationReflector
from ..llm import CircuitOpenError, CodexAuth, CodexChatClient
from ..llm.secret_scrubber import scrub_output_secrets
from ..llm.system_prompt import build_system_prompt, build_chat_system_prompt
from ..logging import get_logger
from ..scheduler import Scheduler
from ..sessions import SessionManager
from ..tools import ToolExecutor, SkillManager, get_tool_definitions
from ..tools.tool_memory import ToolMemory
from ..search import LocalEmbedder, SessionVectorStore
from ..permissions import PermissionManager
from .voice import VoiceManager, VoiceMessageProxy

log = get_logger("discord")

# Friendly fallback when Codex returns an empty response after retries
_EMPTY_RESPONSE_FALLBACK = "I couldn't generate a response. Please try again."

# Webhook IDs allowed to bypass the bot-author check.
# Populated from ALLOWED_WEBHOOK_IDS env var (comma-separated) at startup.
_ALLOWED_WEBHOOK_IDS: set[str] = set()

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

class _LoopMessageProxy:
    """Lightweight proxy providing a discord.Message-like interface for loop iterations.

    Allows Discord-native tool handlers to be called from autonomous loop
    iterations without a real Discord message object.
    """

    def __init__(self, channel: object, user_id: str, user_name: str = "loop") -> None:
        self.channel = channel
        self.id = 0  # No triggering message
        self.webhook_id = None
        self.author = _LoopAuthorProxy(user_id, user_name)


class _LoopAuthorProxy:
    """Lightweight proxy for message.author in loop context."""

    def __init__(self, user_id: str, name: str) -> None:
        self.id = int(user_id) if user_id.isdigit() else 0
        self.bot = False
        self._name = name

    def __str__(self) -> str:
        return self._name


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


# Patterns that suggest fabricated tool output when no tools were actually called.
# Each is (compiled_regex, description) for testability.
_FABRICATION_PATTERNS: list[re.Pattern[str]] = [
    # Claims of running/executing/investigating commands
    re.compile(
        r"(?i)\b(?:I\s+(?:ran|executed|checked|performed|ran\s+a|"
        r"looked\s+at|reviewed|inspected|examined|verified|confirmed|"
        r"tested|scanned|monitored|queried)|"
        r"running|executing|here(?:'s| is) the (?:output|result)|"
        r"the (?:command|output|result) (?:returned|shows?|is)|"
        r"I (?:can see|found) (?:that )?(?:the |your )?)"
    ),
    # Fake command output patterns (``` followed by lines that look like terminal output)
    re.compile(
        r"```(?:bash|shell|console|text|output)?\s*\n"
        r"(?:[\$#>].*\n|(?:total |drwx|Filesystem|CONTAINER|NAME |PID |USER ))",
    ),
    # Claims of completed actions without tool calls (generated, posted, created, saved, etc.)
    re.compile(
        r"(?i)\b(?:generated|posted|created|saved|uploaded|deployed|installed|"
        r"started|stopped|deleted|removed|wrote|written|sent|fetched|downloaded)"
        r"(?:\s+(?:and\s+)?(?:posted|uploaded|saved|sent|attached|delivered))?"
        r"\b.{0,40}\b(?:image|file|script|server|container|process|document|skill)"
    ),
    # Claims referencing data sources without having checked them
    re.compile(
        r"(?i)\b(?:according to (?:the )?(?:logs?|output|results?|data|metrics|dashboard)|"
        r"based on (?:the )?(?:output|logs?|results?|metrics))\b"
    ),
]


def detect_fabrication(text: str, tools_used: list[str]) -> bool:
    """Detect if a text-only response fabricates tool results.

    Returns True if the response contains patterns suggesting the LLM claimed
    to run commands or check systems without actually calling any tools.

    Only meaningful when tools_used is empty — if tools were called, the
    response is based on real results.
    """
    if tools_used:
        return False
    if not text or len(text) < 20:
        return False
    return any(p.search(text) for p in _FABRICATION_PATTERNS)


# Developer message injected when fabrication is detected, prompting a retry.
_FABRICATION_RETRY_MSG = {
    "role": "developer",
    "content": (
        "STOP. Your previous response claimed results but you did NOT call any tools. "
        "That is a fabrication. You MUST call the appropriate tool to get real results. "
        "Do NOT respond with text only — call the tool NOW."
    ),
}


# ---------------------------------------------------------------------------
# Tool-unavailability fabrication — catches Codex claiming tools are disabled
# without actually trying them. Only fires when no tools were called.
# ---------------------------------------------------------------------------

_TOOL_UNAVAIL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?i)\b(?:not (?:enabled|available|configured)|"
        r"is(?:n't| not) (?:enabled|available|configured|supported)|"
        r"is disabled|cannot be used)\b"
    ),
    re.compile(
        r"(?i)\bcan(?:'t|not)\b.{0,30}\b(?:generate|create|produce|render)\b.{0,20}"
        r"\b(?:image|photo|picture|screenshot)"
    ),
    re.compile(
        r"(?i)\b(?:image|photo) generation.{0,20}\b(?:not|isn't|unavailable|disabled)\b"
    ),
    # Claims of lacking access or capability
    re.compile(
        r"(?i)\b(?:(?:don't|do not) have (?:access|the ability) to|"
        r"no (?:tool|way) (?:to |for )(?:do )?(?:that|this)|"
        r"that(?:'s| is) not something I can)\b"
    ),
]


def detect_tool_unavailable(text: str, tools_used: list[str]) -> bool:
    """Detect if a response falsely claims a tool is unavailable.

    Returns True if the response claims a tool is not enabled/available/etc.
    without actually trying to call it.  Only meaningful when tools_used is
    empty — if tools were called and returned a real error, that's legitimate.
    """
    if tools_used:
        return False
    if not text or len(text) < 15:
        return False
    return any(p.search(text) for p in _TOOL_UNAVAIL_PATTERNS)


_TOOL_UNAVAIL_RETRY_MSG = {
    "role": "developer",
    "content": (
        "Every tool in your tool list is available. Call the "
        "tool instead of claiming it's unavailable."
    ),
}


# ---------------------------------------------------------------------------
# Hedging detection — catches "shall I", "if you want", etc.
# Used for bot-to-bot interactions where hedging is never appropriate.
# ---------------------------------------------------------------------------

_HEDGING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?i)\b(?:if you(?:'d| would)? (?:like|want|prefer)|"
        r"shall I|should I|would you like(?: me to)?|"
        r"ready (?:when|on) you|let me know (?:if|when)|"
        r"I can (?:do|help|run|execute|set up) (?:that|this|it) (?:for you|if)|"
        r"just (?:say|tell) (?:the word|me when|me if)|"
        r"want me to)\b"
    ),
    re.compile(
        r"(?i)\b(?:here(?:'s| is) (?:a |the )?plan|"
        r"I(?:'d| would) (?:suggest|recommend)|"
        r"before (?:I |we )(?:proceed|go ahead|start)|"
        r"I'll wait for (?:your|the) (?:go[- ]ahead|confirmation|approval)|"
        r"awaiting (?:your|the) (?:confirmation|input|response|approval|go[- ]ahead)|"
        r"once you (?:confirm|approve|give the go[- ]ahead)|"
        r"(?:your call|up to you|your decision))\b"
    ),
    re.compile(
        r"(?i)^Plan:|"
        r"I can(?:'t| not) directly|"
        r"I (?:need|have) to .{0,30} (?:first|before)|"
        r"I'm (?:going to|about to|proceeding to)"
    ),
]


def detect_hedging(text: str, tools_used: list[str]) -> bool:
    """Detect if a response hedges instead of executing.

    Returns True if the response contains hedging language and no tools
    were called — meaning the LLM asked for permission instead of acting.
    """
    if tools_used:
        return False
    if not text or len(text) < 15:
        return False
    return any(p.search(text) for p in _HEDGING_PATTERNS)


# Developer message injected when hedging is detected on a bot message.
_HEDGING_RETRY_MSG = {
    "role": "developer",
    "content": (
        "STOP. The user is another bot — it cannot confirm, approve, or choose. "
        "Do NOT ask permission, suggest plans, or hedge. EXECUTE the requested "
        "action NOW by calling the appropriate tools. Never say 'if you want', "
        "'shall I', or 'would you like' — just DO IT."
    ),
}


# ---------------------------------------------------------------------------
# Code-block hedging — catches Codex showing a bash/shell command instead
# of executing it via run_command.  Only fires when no tools were called.
# ---------------------------------------------------------------------------

_CODE_BLOCK_HEDGING_PATTERN: re.Pattern[str] = re.compile(
    r"```(?:bash|sh|shell|zsh)\s*\n",
)


def detect_code_hedging(text: str, tools_used: list[str]) -> bool:
    """Detect if a response shows a bash code block instead of executing it.

    Returns True if the response contains a bash/sh code block but no tools
    were called — meaning the LLM showed what it should have run.
    """
    if tools_used:
        return False
    if not text or len(text) < 15:
        return False
    return bool(_CODE_BLOCK_HEDGING_PATTERN.search(text))


_CODE_HEDGING_RETRY_MSG = {
    "role": "developer",
    "content": (
        "Execute the command using run_command instead of showing it. "
        "You are an executor, not a manual."
    ),
}


# ---------------------------------------------------------------------------
# Premature failure detection — catches when Codex gives up too early
# instead of exhausting fallback chains.
# ---------------------------------------------------------------------------

_FAILURE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?i)(?:couldn'?t (?:get|resolve|find|fetch|retrieve|determine|complete|"
        r"access|connect)|"
        r"(?:failed|unable) to (?:get|resolve|find|fetch|retrieve|connect|access)|"
        r"(?:no|zero) (?:results?|matches?|data) (?:found|returned|available)|"
        r"(?:is|was|currently) (?:blocked|unavailable|down|broken|failing)|"
        r"(?:error|Error):)"
    ),
    re.compile(
        r"(?i)(?:workaround|fallback|alternative|try (?:this|these|instead)|"
        r"use this .{0,20} instead|if you want .{0,30} workaround)"
    ),
    # Connection/execution failure patterns
    re.compile(
        r"(?i)(?:timed?\s*out|connection (?:refused|failed|reset|closed)|"
        r"(?:doesn't|does not|isn't|is not) (?:seem to be )?(?:work(?:ing)?|respond(?:ing)?))"
    ),
]


def detect_premature_failure(text: str, tools_used: list[str]) -> bool:
    """Detect if a response reports failure without exhausting alternatives.

    Returns True if the response describes a failure/error AND tools were
    called (partial execution) — meaning the LLM tried something, hit an
    error, and gave up instead of trying a different approach.

    Only fires when tools WERE used (partial attempt). Pure fabrication
    (no tools) is handled by detect_fabrication instead.
    """
    if not tools_used:
        return False  # No tools called — fabrication detector handles this
    if not text or len(text) < 30:
        return False
    return any(p.search(text) for p in _FAILURE_PATTERNS)


_FAILURE_RETRY_MSG = {
    "role": "developer",
    "content": (
        "STOP. You hit an error but gave up too early. You MUST exhaust ALL "
        "alternative approaches before reporting failure. Try: different APIs, "
        "different search terms, different tools, hardcoded IDs if you know them, "
        "web search for the answer, or any other creative approach. Only report "
        "failure after ALL options are genuinely exhausted."
    ),
}


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


def combine_bot_messages(parts: list[str]) -> str:
    """Combine buffered bot messages, intelligently merging code blocks.

    Handles:
    - Split code blocks (open in one message, close in later one) — joined
      with a single newline so no extra blank lines appear inside the block.
    - Adjacent code blocks (close fence then immediately open fence) — merged
      into one continuous block by removing the redundant fence pair.
    - Regular text between code blocks — joined with double newline as usual.
    """
    if len(parts) <= 1:
        return parts[0] if parts else ""

    # Join parts, using \n (not \n\n) when the previous part has an unclosed
    # code block — meaning the next part is a continuation of the same block.
    result = parts[0]
    for i in range(1, len(parts)):
        fence_count = result.count("```")
        if fence_count % 2 == 1:
            # Inside an unclosed code block — continuation, single newline
            result += "\n" + parts[i]
        else:
            result += "\n\n" + parts[i]

    # Merge adjacent code blocks: \n```<ws>\n\n```<lang>\n → \n
    # This collapses e.g. "\n```\n\n```bash\n" into a single block.
    result = re.sub(r"\n```[ \t]*\n\n```(\w*)[ \t]*\n", "\n", result)

    return result


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
        self._start_time = time.monotonic()

        # Configure timezone for time parser module
        from ..tools.time_parser import set_default_timezone
        set_default_timezone(config.timezone)

        # Per-channel lock to prevent concurrent processing of the same message
        self._channel_locks: dict[str, asyncio.Lock] = {}
        # Pending file attachments from skills — per-channel to avoid cross-channel leaks
        self._pending_files: dict[str, list[tuple[bytes, str]]] = {}
        # Track recently processed message IDs to prevent duplicate handling
        self._processed_messages: collections.OrderedDict[int, None] = collections.OrderedDict()
        self._processed_messages_max = 100
        # Bot message buffer: accumulate rapid-fire bot messages before processing
        # Key: (channel_id, author_id) → list of content strings
        self._bot_msg_buffer: dict[tuple[str, str], list[str]] = {}
        self._bot_msg_tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._bot_msg_buffer_delay: float = 2.0  # seconds to wait for more bot messages
        # Recent tool executions for conversational context (injected into system prompt)
        # Per-channel: {channel_id: [(timestamp, entry_text), ...]}
        self._recent_actions: dict[str, list[tuple[float, str]]] = {}
        self._recent_actions_max = 10
        self._recent_actions_expiry = 3600  # seconds (1 hour)
        # Background task tracking
        self._background_tasks: dict[str, BackgroundTask] = {}
        self._background_tasks_max = 20
        # Autonomous loop manager
        self.loop_manager = LoopManager()
        # Cached merged tool definitions — invalidated on skill create/edit/delete
        self._cached_merged_tools: list[dict] | None = None
        # Cached host string dict — invalidated on context reload
        self._cached_hosts: dict[str, str] | None = None
        # Cached skills list text — invalidated on skill create/edit/delete
        self._cached_skills_text: str | None = None
        # TTL cache for per-user memory (avoids file I/O per message)
        self._memory_cache: dict[str | None, tuple[float, dict[str, str]]] = {}
        self._memory_cache_ttl: float = 60.0  # seconds
        # TTL cache for reflector prompt section (avoids file I/O per message)
        self._reflector_cache: dict[str | None, tuple[float, str]] = {}
        self._reflector_cache_ttl: float = 60.0  # seconds
        # Throttled cache cleanup
        self._last_cache_cleanup: float = 0.0
        self._cache_cleanup_interval: float = 300.0  # every 5 minutes

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
        self._embedder: LocalEmbedder | None = None
        self._knowledge_store: KnowledgeStore | None = None
        self._fts_index: FullTextIndex | None = None
        if config.search.enabled:
            self._embedder = LocalEmbedder()
            # Initialize FTS5 index (SQLite, no external deps)
            from pathlib import Path
            search_db_path = config.search.search_db_path
            fts_db_path = str(Path(search_db_path).parent / "fts.db")
            from ..search.fts import FullTextIndex
            self._fts_index = FullTextIndex(fts_db_path)
            if not self._fts_index.available:
                self._fts_index = None

            # Always initialize stores — they work in FTS-only mode even without
            # sqlite-vec or embedder. Don't null them out on vec init failure.
            session_db = str(Path(search_db_path) / "sessions.db") if Path(search_db_path).is_dir() else search_db_path + "_sessions.db"
            knowledge_db = str(Path(search_db_path) / "knowledge.db") if Path(search_db_path).is_dir() else search_db_path + "_knowledge.db"
            # Ensure the directory exists
            Path(search_db_path).mkdir(parents=True, exist_ok=True)
            session_db = str(Path(search_db_path) / "sessions.db")
            knowledge_db = str(Path(search_db_path) / "knowledge.db")

            self._vector_store = SessionVectorStore(
                session_db, fts_index=self._fts_index,
            )
            if not self._vector_store.available:
                self._vector_store = None
            self._knowledge_store = KnowledgeStore(
                knowledge_db, fts_index=self._fts_index,
            )
            if not self._knowledge_store.available:
                self._knowledge_store = None

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
        self._init_allowed_webhook_ids()
        self._log_startup_config()

    def _init_allowed_webhook_ids(self) -> None:
        """Populate _ALLOWED_WEBHOOK_IDS from ALLOWED_WEBHOOK_IDS env var."""
        global _ALLOWED_WEBHOOK_IDS
        raw = os.environ.get("ALLOWED_WEBHOOK_IDS", "")
        if raw:
            _ALLOWED_WEBHOOK_IDS = {wid.strip() for wid in raw.split(",") if wid.strip()}

    def _log_startup_config(self) -> None:
        """Log configuration summary at startup to help users verify setup."""
        cfg = self.config
        if not cfg.tools.hosts:
            log.warning("No hosts configured — SSH tools will not work until hosts are added to config.yml")
        else:
            log.info("Configured hosts: %s", ", ".join(cfg.tools.hosts.keys()))
        if not cfg.tools.claude_code_host:
            log.info("claude_code_host not set — claude -p code generation requires a configured host")
        if cfg.openai_codex.enabled and not self.codex_client:
            log.warning("Codex enabled but not configured — session compaction and learning reflection disabled")
        if cfg.discord.respond_to_bots:
            log.info("Bot interaction enabled — will respond to other bots")
        if cfg.discord.require_mention:
            log.info("Mention-only mode — will only respond when @mentioned")

    def _get_cached_hosts(self) -> dict[str, str]:
        """Return cached host string dict. Rebuilt on config reload."""
        if self._cached_hosts is None:
            self._cached_hosts = {
                alias: f"{h.ssh_user}@{h.address}"
                for alias, h in self.config.tools.hosts.items()
            }
        return self._cached_hosts

    def _get_cached_skills_text(self) -> str:
        """Return cached skills list text. Invalidated on skill create/edit/delete."""
        if self._cached_skills_text is None:
            if hasattr(self, "skill_manager"):
                skills = self.skill_manager.list_skills()
                if skills:
                    self._cached_skills_text = "\n".join(
                        f"- `{s['name']}`: {s['description']}" for s in skills
                    )
                else:
                    self._cached_skills_text = ""
            else:
                self._cached_skills_text = ""
        return self._cached_skills_text

    def _get_cached_memory(self, user_id: str | None) -> dict[str, str]:
        """Return cached per-user memory with TTL to avoid file I/O per message."""
        now = time.time()
        cached = self._memory_cache.get(user_id)
        if cached and now - cached[0] < self._memory_cache_ttl:
            return cached[1]
        memory = self.tool_executor._load_memory_for_user(user_id)
        self._memory_cache[user_id] = (now, memory)
        return memory

    def _get_cached_reflector(self, user_id: str | None) -> str:
        """Return cached reflector prompt section with TTL to avoid file I/O per message."""
        if not hasattr(self, "reflector"):
            return ""
        now = time.time()
        cached = self._reflector_cache.get(user_id)
        if cached and now - cached[0] < self._reflector_cache_ttl:
            return cached[1]
        learned = self.reflector.get_prompt_section(user_id=user_id)
        self._reflector_cache[user_id] = (now, learned)
        return learned

    def _invalidate_prompt_caches(self) -> None:
        """Invalidate all prompt-related caches. Called on config/context reload."""
        self._cached_hosts = None
        self._cached_skills_text = None
        self._memory_cache.clear()
        self._reflector_cache.clear()

    def _build_system_prompt(
        self, channel: discord.abc.GuildChannel | None = None,
        user_id: str | None = None,
        query: str | None = None,
    ) -> str:
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
            hosts=self._get_cached_hosts(),
            services=self.config.tools.allowed_services,
            playbooks=self.config.tools.allowed_playbooks,
            voice_info=voice_info,
            tz=self.config.timezone,
            claude_code_dir=self.config.tools.claude_code_dir,
        )

        # Inject persistent memory into the system prompt (per-user + global)
        memory = self._get_cached_memory(user_id)
        if memory:
            memory_text = "\n".join(f"- **{k}**: {v}" for k, v in memory.items())
            prompt += f"\n\n## Persistent Memory\n{memory_text}"

        # Inject learned context from cross-conversation reflection (per-user filtered)
        learned = self._get_cached_reflector(user_id)
        if learned:
            prompt += f"\n\n{learned}"

        # Inject user-created skills list (cached, invalidated on skill CRUD)
        skills_text = self._get_cached_skills_text()
        if skills_text:
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

        prompt = build_chat_system_prompt(voice_info=voice_info, tz=self.config.timezone)

        # Inject persistent memory (per-user + global, personalization matters for chat)
        memory = self._get_cached_memory(user_id)
        if memory:
            memory_text = "\n".join(f"- **{k}**: {v}" for k, v in memory.items())
            prompt += f"\n\n## Persistent Memory\n{memory_text}"

        # Inject learned context (per-user filtered, personality from past conversations)
        learned = self._get_cached_reflector(user_id)
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
        Cached — invalidated on skill create/edit/delete.
        """
        if self._cached_merged_tools is not None:
            return self._cached_merged_tools
        builtin = get_tool_definitions(enabled_packs=self.config.tools.tool_packs)
        builtin_names = {t["name"] for t in builtin}
        skill_defs = [
            t for t in self.skill_manager.get_tool_definitions()
            if t["name"] not in builtin_names
        ]
        self._cached_merged_tools = builtin + skill_defs
        return self._cached_merged_tools

    def _cleanup_stale_caches(self) -> None:
        """Remove stale entries from per-channel caches to prevent memory leaks.

        Called periodically (every _cache_cleanup_interval seconds) after session prune.
        Removes expired entries from _recent_actions and _channel_locks for
        channels that no longer have active sessions.
        """
        now = time.time()
        # Clean up _recent_actions: remove channels with all expired entries
        expired_channels = []
        for channel_id, actions in self._recent_actions.items():
            actions[:] = [(ts, entry) for ts, entry in actions if now - ts < self._recent_actions_expiry]
            if not actions:
                expired_channels.append(channel_id)
        for channel_id in expired_channels:
            del self._recent_actions[channel_id]

        # Clean up _channel_locks for channels no longer in active sessions
        active_channels = set(self.sessions._sessions.keys())
        stale_locks = [cid for cid in self._channel_locks if cid not in active_channels]
        for cid in stale_locks:
            del self._channel_locks[cid]

        # Clean up expired TTL cache entries for memory and reflector
        ttl = getattr(self, "_memory_cache_ttl", 60.0)
        self._memory_cache = {
            k: v for k, v in getattr(self, "_memory_cache", {}).items()
            if now - v[0] < ttl
        }
        ttl = getattr(self, "_reflector_cache_ttl", 60.0)
        self._reflector_cache = {
            k: v for k, v in getattr(self, "_reflector_cache", {}).items()
            if now - v[0] < ttl
        }

    def _maybe_cleanup_caches(self) -> None:
        """Run cache cleanup if enough time has passed since the last run."""
        try:
            now = time.time()
            interval = getattr(self, "_cache_cleanup_interval", 300.0)
            last = getattr(self, "_last_cache_cleanup", 0.0)
            if now - last > interval:
                self._cleanup_stale_caches()
                self._last_cache_cleanup = now
        except Exception:
            pass  # Non-critical — don't break message processing

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
            await interaction.response.send_message(
                f"**Loki Status**\n"
                f"{codex_status}"
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
            self._invalidate_prompt_caches()
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
        packs = self.config.tools.tool_packs
        if packs:
            log.info("Tool packs enabled: %s", ", ".join(packs))
        else:
            log.info("Tool packs: all tools loaded (no filtering)")
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
        if self._vector_store:
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
            # Backfill knowledge FTS from existing data
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
        # Never respond to our own messages
        if message.author == self.user:
            return

        if message.author.bot:
            # Allow specific webhooks (via ALLOWED_WEBHOOK_IDS env var)
            is_allowed_webhook = message.webhook_id and str(message.webhook_id) in _ALLOWED_WEBHOOK_IDS
            if not is_allowed_webhook and not self.config.discord.respond_to_bots:
                return

        is_test_webhook = message.webhook_id and str(message.webhook_id) in _ALLOWED_WEBHOOK_IDS
        if not is_test_webhook and not self._is_allowed_user(message.author):
            return
        if not self._is_allowed_channel(message.channel.id):
            return

        # require_mention: only respond when the bot is @mentioned (or in DMs)
        if self.config.discord.require_mention:
            is_dm = not hasattr(message.channel, "guild") or message.channel.guild is None
            is_mentioned = self.user and self.user.mentioned_in(message)
            if not is_dm and not is_mentioned:
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

        # Buffer rapid-fire bot messages (e.g. code blocks split across messages)
        # Wait 2s after each bot message to see if more follow, then process all at once
        if message.author.bot and self.config.discord.respond_to_bots:
            buf_key = (str(message.channel.id), str(message.author.id))
            if buf_key not in self._bot_msg_buffer:
                self._bot_msg_buffer[buf_key] = []
            self._bot_msg_buffer[buf_key].append(message.content)

            # Cancel previous timer for this bot+channel
            if buf_key in self._bot_msg_tasks:
                self._bot_msg_tasks[buf_key].cancel()

            # Set new timer — process after delay of silence
            async def _flush_bot_buffer(key, orig_msg):
                await asyncio.sleep(self._bot_msg_buffer_delay)
                parts = self._bot_msg_buffer.pop(key, [])
                self._bot_msg_tasks.pop(key, None)
                if not parts:
                    return
                combined = combine_bot_messages(parts)
                log.info("Bot buffer flushed: %d messages from %s combined", len(parts), orig_msg.author)
                # Strip mention from combined content
                if self.user:
                    combined = combined.replace(f"<@{self.user.id}>", "").strip()
                    combined = combined.replace(f"<@!{self.user.id}>", "").strip()
                if combined:
                    await self._handle_message(orig_msg, combined, image_blocks=[])

            self._bot_msg_tasks[buf_key] = asyncio.create_task(
                _flush_bot_buffer(buf_key, message)
            )
            return

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

            # PDF attachments — extract text inline
            if ext == ".pdf":
                if att.size > 25 * 1024 * 1024:
                    text_parts.append(f"[PDF: {att.filename} ({att.size / 1024 / 1024:.1f} MB, exceeds 25 MB limit)]")
                    continue
                try:
                    import fitz
                    data = await att.read()
                    doc = fitz.open(stream=data, filetype="pdf")
                    try:
                        pages_text = []
                        for i, page in enumerate(doc):
                            pages_text.append(f"Page {i + 1}: {page.get_text()}")
                        full_text = "\n".join(pages_text)
                        if len(full_text) > 8000:
                            full_text = full_text[:8000] + "\n[... truncated ...]"
                        text_parts.append(
                            f"**Attached PDF: {att.filename}** ({doc.page_count} pages)\n```\n{full_text}\n```\n"
                            f"[This is a PDF. Text has been extracted. For detailed analysis, use analyze_pdf tool.]"
                        )
                    finally:
                        doc.close()
                except Exception as e:
                    text_parts.append(f"[PDF: {att.filename} (failed to extract text: {e})]")
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

        try:
            is_guest = self.permissions.is_guest(str(message.author.id))
            already_sent = False
            is_error = False
            tools_used: list[str] = []
            handoff = False

            if is_guest:
                # Guest tier: chat only, no tools
                log.info("Guest tier user %s, chat route (no tools)", message.author.id)
                # Guests use full history (with compaction)
                history = await self.sessions.get_history_with_compaction(channel_id)
                if image_blocks:
                    history = list(history)
                    if history and history[-1]["role"] == "user":
                        last_msg = history[-1]
                        text_content = last_msg["content"] if isinstance(last_msg["content"], str) else str(last_msg["content"])
                        history[-1] = {
                            "role": "user",
                            "content": image_blocks + [{"type": "text", "text": text_content}],
                        }
                    log.info("Attached %d image(s) to message for Claude vision", len(image_blocks))
                if self.codex_client:
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
                else:
                    log.info("No chat backend configured for guest user")
                    response = "Chat backend is not configured."
                    is_error = True
            else:
                # Everyone else: Codex with ALL tools
                if not self.codex_client:
                    await self._send_with_retry(
                        message,
                        "No tool backend available. Please try again later.",
                    )
                    self.sessions.remove_last_message(channel_id, "user")
                    return
                _sp = self._build_system_prompt(channel=message.channel, user_id=user_id, query=content)
                _sp = await self._inject_tool_hints(_sp, content, user_id)
                log.info("Routing to Codex with tools")
                # Use abbreviated history to reduce poisoning from stale responses
                # (get_task_history handles compaction internally)
                task_history = await self.sessions.get_task_history(channel_id, max_messages=20)
                if image_blocks and task_history and task_history[-1]["role"] == "user":
                    last = task_history[-1]
                    text = last["content"] if isinstance(last["content"], str) else str(last["content"])
                    task_history[-1] = {
                        "role": "user",
                        "content": image_blocks + [{"type": "text", "text": text}],
                    }
                    log.info("Attached %d image(s) to message for Claude vision", len(image_blocks))
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
                    # Fetch full history for handoff (compaction already ran in get_task_history)
                    history = self.sessions.get_history(channel_id)
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
        except Exception as e:
            log.error("Error processing message: %s", e, exc_info=True)
            await self._send_with_retry(message, scrub_response_secrets(f"Something went wrong: {e}"))
            self.sessions.remove_last_message(channel_id, "user")
            return

        # Scrub secrets from LLM response before logging, saving, or sending.
        # Tool output is already scrubbed (scrub_output_secrets in _run_tool),
        # but the LLM may echo, reconstruct, or hallucinate secrets in its
        # natural-language response text.
        response = scrub_response_secrets(response)

        log.info("Final response to send: %r", response[:200])
        if not is_error:
            # If tools were available but not used (and no skill handoff),
            # don't save the response — text-only replies pollute history
            # and teach the model that answering without tools is acceptable.
            if not is_guest and not tools_used and not handoff:
                pass
            else:
                self.sessions.add_message(channel_id, "assistant", response)
            self.sessions.prune()
            self._maybe_cleanup_caches()
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
        is_bot_message = getattr(message.author, "bot", False) and self.config.discord.respond_to_bots
        # Always inject message ID so the LLM can reference it (e.g. add_reaction)
        msg_id_note = f"Current message ID: {message.id}"
        if len(messages) > 1:
            sep_text = (
                "=== CURRENT REQUEST ===\n"
                f"{msg_id_note}\n"
                "The messages above are conversation history for context only. "
                "For the user's new request below, evaluate your CURRENTLY AVAILABLE "
                "tools and use them. Do not repeat prior refusals or text-only responses. "
                "If a tool exists for the requested action, call it."
            )
            if is_bot_message:
                sep_text += (
                    "\n\nIMPORTANT: This message is from ANOTHER BOT. "
                    "Bots cannot confirm, choose, or approve. "
                    "EXECUTE immediately — never hedge, ask permission, or say "
                    "'if you want' / 'shall I' / 'would you like'. "
                    "If the message contains code, use run_script to execute it. "
                    "If it asks for output, call the tool and paste raw results."
                )
            separator = {"role": "developer", "content": sep_text}
            messages.insert(-1, separator)
        else:
            # No history — still provide message ID context
            messages.insert(0, {"role": "developer", "content": msg_id_note})

        # Track which tools are used during this loop for tool memory
        # Local variable (not instance attr) to avoid cross-channel contamination
        tools_used_in_loop: list[str] = []

        # Progress embed tracking — single editable embed replaces scattered messages
        progress_embed_msg: discord.Message | None = None
        progress_steps: list[dict] = []
        cancel_view: ToolLoopCancelView | None = None

        user_id = str(message.author.id)

        # Filter tools based on user permission tier (skip for test webhooks)
        is_test_wh = message.webhook_id and str(message.webhook_id) in _ALLOWED_WEBHOOK_IDS
        if tools is not None and not is_test_wh:
            tools = self.permissions.filter_tools(user_id, tools)
            # Normalize empty tool list to None
            if tools is not None and not tools:
                tools = None

        # Collect image blocks from analyze_image calls for vision injection
        pending_image_blocks: list[dict] = []

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
                # Fabrication detection: if no tools were called on the FIRST
                # iteration and the response looks like it fabricated results,
                # inject a correction and retry once.
                if (
                    iteration == 0
                    and not tools_used_in_loop
                    and detect_fabrication(llm_resp.text or "", tools_used_in_loop)
                ):
                    log.warning(
                        "Fabrication detected on first response — retrying with correction"
                    )
                    # Add the fabricated response and correction to messages
                    messages.append({"role": "assistant", "content": llm_resp.text})
                    messages.append(_FABRICATION_RETRY_MSG)
                    continue  # retry the loop — iteration increments

                # Tool-unavailability fabrication: Codex claims a tool is
                # "not enabled" / "not available" without trying it.
                if (
                    iteration == 0
                    and not tools_used_in_loop
                    and detect_tool_unavailable(llm_resp.text or "", tools_used_in_loop)
                ):
                    log.warning(
                        "Tool-unavailability fabrication detected — retrying with correction"
                    )
                    messages.append({"role": "assistant", "content": llm_resp.text})
                    messages.append(_TOOL_UNAVAIL_RETRY_MSG)
                    continue  # retry the loop — iteration increments

                # Hedging detection for bot messages: if no tools were called
                # and the response hedges ("shall I", "if you want"), retry once.
                if (
                    iteration == 0
                    and not tools_used_in_loop
                    and detect_hedging(llm_resp.text or "", tools_used_in_loop)
                ):
                    log.warning(
                        "Hedging detected — retrying with correction"
                    )
                    messages.append({"role": "assistant", "content": llm_resp.text})
                    messages.append(_HEDGING_RETRY_MSG)
                    continue  # retry the loop — iteration increments

                # Code-block hedging: Codex shows a bash/shell command instead
                # of executing it via run_command.
                if (
                    iteration == 0
                    and not tools_used_in_loop
                    and detect_code_hedging(llm_resp.text or "", tools_used_in_loop)
                ):
                    log.warning(
                        "Code-block hedging detected — retrying with correction"
                    )
                    messages.append({"role": "assistant", "content": llm_resp.text})
                    messages.append(_CODE_HEDGING_RETRY_MSG)
                    continue  # retry the loop — iteration increments

                # Premature failure detection: if tools were called but the response
                # reports failure without exhausting alternatives, retry once.
                if (
                    iteration == 0
                    and tools_used_in_loop
                    and detect_premature_failure(llm_resp.text or "", tools_used_in_loop)
                ):
                    log.warning(
                        "Premature failure detected — retrying with correction"
                    )
                    messages.append({"role": "assistant", "content": llm_resp.text})
                    messages.append(_FAILURE_RETRY_MSG)
                    continue

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
                nonlocal system_prompt, pending_image_blocks
                tool_name = block.name
                tool_input = block.input
                log.info("Tool call: %s(%s)", tool_name, tool_input)

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
                    elif tool_name == "start_loop":
                        result = self._handle_start_loop(message, tool_input)
                    elif tool_name == "stop_loop":
                        result = self._handle_stop_loop(tool_input)
                    elif tool_name == "list_loops":
                        result = self._handle_list_loops()
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
                        self._cached_merged_tools = None  # invalidate tool cache
                        self._cached_skills_text = None  # invalidate skills text cache
                        system_prompt = self._build_system_prompt(channel=message.channel, user_id=user_id)
                    elif tool_name == "edit_skill":
                        result = await asyncio.to_thread(
                            self.skill_manager.edit_skill, tool_input["name"], tool_input["code"],
                        )
                        self._cached_merged_tools = None  # invalidate tool cache
                        self._cached_skills_text = None  # invalidate skills text cache
                        system_prompt = self._build_system_prompt(channel=message.channel, user_id=user_id)
                    elif tool_name == "delete_skill":
                        result = await asyncio.to_thread(
                            self.skill_manager.delete_skill, tool_input["name"],
                        )
                        self._cached_merged_tools = None  # invalidate tool cache
                        self._cached_skills_text = None  # invalidate skills text cache
                        system_prompt = self._build_system_prompt(channel=message.channel, user_id=user_id)
                    elif tool_name == "add_reaction":
                        result = await self._handle_add_reaction(message, tool_input)
                    elif tool_name == "create_poll":
                        result = await self._handle_create_poll(message, tool_input)
                    elif tool_name == "broadcast":
                        result = await self._handle_broadcast(message, tool_input)
                    elif tool_name == "analyze_image":
                        result = await self._handle_analyze_image(message, tool_input)
                    elif tool_name == "generate_image":
                        result = await self._handle_generate_image(message, tool_input)
                    elif tool_name == "list_skills":
                        skills = self.skill_manager.list_skills()
                        if not skills:
                            result = "No user-created skills."
                        else:
                            lines = []
                            for s in skills:
                                lines.append(f"**{s['name']}**: {s['description']}")
                            result = f"**User-created skills ({len(skills)}):**\n" + "\n".join(lines)
                    elif self.skill_manager.has_skill(tool_name):
                        async def _skill_msg(text: str) -> None:
                            await message.channel.send(scrub_response_secrets(text))
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

                # Handle special image block return from analyze_image
                if isinstance(result, dict) and "__image_block__" in result:
                    pending_image_blocks.append(result["__image_block__"])
                    result = f"[Image loaded. Analyze it with this instruction: {result['__prompt__']}]"

                # Scrub secrets from tool output
                result = scrub_output_secrets(result)

                # Audit log
                await self.audit.log_execution(
                    user_id=str(message.author.id),
                    user_name=str(message.author),
                    channel_id=str(message.channel.id),
                    tool_name=tool_name,
                    tool_input=tool_input,
                    approved=True,
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

            # Inject pending image blocks as vision content for the next LLM call.
            # This reuses the same base64 image block format as _process_attachments.
            if pending_image_blocks:
                vision_content: list[dict] = list(pending_image_blocks)
                vision_content.append({
                    "type": "text",
                    "text": "The image(s) above were fetched by analyze_image. Describe and analyze them.",
                })
                messages.append({"role": "user", "content": vision_content})
                log.info("Injected %d image block(s) into tool loop messages", len(pending_image_blocks))
                pending_image_blocks.clear()

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

    def _handle_schedule_task(self, message: discord.Message, inp: dict) -> str:
        """Create a scheduled task."""
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
        """Semantic + FTS search over the knowledge base."""
        if not self._knowledge_store:
            return "Knowledge base is not available (search not enabled or not initialized)."

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
            score = r.get("score", r.get("rrf_score", r.get("rank", 0)))
            content = scrub_output_secrets(r["content"].replace("\n", " ")[:500])
            lines.append(f"**[{source}]** (score: {score})\n{content}")

        return f"**Found {len(results)} result(s) for '{query}':**\n\n" + "\n\n".join(lines)

    async def _handle_ingest_document(self, inp: dict, uploader: str) -> str:
        """Ingest a document into the knowledge base."""
        if not self._knowledge_store:
            return "Knowledge base is not available (search not enabled or not initialized)."

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
            return f"Failed to ingest '{source}' — no chunks could be indexed."
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
            await channel.send(scrub_response_secrets(f"**Daily Infrastructure Digest**\n\nFailed to collect data: {e}"))
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

        await channel.send(scrub_response_secrets(f"**Daily Infrastructure Digest**\n\n{summary}"))

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
            await channel.send(scrub_response_secrets(message))
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

        # Build Codex callback for conversational follow-up
        codex_cb = None
        if self.codex_client:
            async def _codex_followup(messages: list[dict], system: str, max_tokens: int) -> str:
                return await self.codex_client.chat(
                    messages=messages, system=system, max_tokens=max_tokens,
                )
            codex_cb = _codex_followup

        # Launch in background
        async def _run():
            try:
                await run_background_task(
                    task, self.tool_executor, self.skill_manager,
                    knowledge_store=self._knowledge_store,
                    embedder=self._embedder,
                    audit_logger=self.audit,
                    codex_callback=codex_cb,
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

    def _handle_start_loop(self, message: discord.Message, inp: dict) -> str:
        """Start an autonomous loop."""
        goal = inp.get("goal", "")
        if not goal:
            return "A 'goal' is required to start a loop."

        interval = inp.get("interval_seconds", 60)
        mode = inp.get("mode", "notify")
        stop_condition = inp.get("stop_condition")
        max_iterations = inp.get("max_iterations", 50)

        # Build iteration callback that runs through Codex with tools
        async def _iteration_cb(
            prompt: str, channel: object, prev_context: str | None,
        ) -> str:
            return await self._run_loop_iteration(
                prompt, channel, prev_context, str(message.author.id),
            )

        result = self.loop_manager.start_loop(
            goal=goal,
            channel=message.channel,
            requester_id=str(message.author.id),
            requester_name=str(message.author),
            iteration_callback=_iteration_cb,
            interval_seconds=interval,
            mode=mode,
            stop_condition=stop_condition,
            max_iterations=max_iterations,
        )

        # If result is a loop ID (short hex), format success message
        if result.startswith("Error"):
            return result
        return (
            f"Loop started (ID: `{result}`): **{goal[:100]}** "
            f"(every {max(10, interval)}s, mode={mode}, max {max_iterations} iterations)"
        )

    def _handle_stop_loop(self, inp: dict) -> str:
        """Stop an autonomous loop."""
        loop_id = inp.get("loop_id", "")
        if not loop_id:
            return "A 'loop_id' is required."
        return self.loop_manager.stop_loop(loop_id)

    def _handle_list_loops(self) -> str:
        """List all autonomous loops."""
        return self.loop_manager.list_loops()

    async def _run_loop_iteration(
        self,
        prompt: str,
        channel: object,
        prev_context: str | None,
        user_id: str,
    ) -> str:
        """Run a single loop iteration through Codex with full tool access.

        Simplified version of _process_with_tools for autonomous loops:
        same Codex + tool execution pipeline but without progress embeds,
        cancel buttons, or detection retries.
        """
        if not self.codex_client:
            return "Codex client not available."

        # Resolve requester name for audit logging and message proxy
        requester_name = "loop"
        for loop_info in self.loop_manager._loops.values():
            if loop_info.requester_id == user_id:
                requester_name = loop_info.requester_name
                break
        msg_proxy = _LoopMessageProxy(channel, user_id, requester_name)

        # Build messages for the iteration
        messages: list[dict] = []
        if prev_context:
            messages.append({
                "role": "user",
                "content": f"Previous iteration results:\n{prev_context}",
            })
            messages.append({
                "role": "assistant",
                "content": "Understood, I have the context from previous iterations.",
            })
        messages.append({"role": "user", "content": prompt})

        # Build system prompt and tool definitions
        system_prompt = self._build_system_prompt(channel=channel, user_id=user_id)
        tools = self._merged_tool_definitions() if self.config.tools.enabled else None

        final_text = ""
        tool_timeout = self.config.tools.tool_timeout_seconds
        channel_id_str = str(getattr(channel, "id", ""))

        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                response = await self.codex_client.chat_with_tools(
                    messages=messages, system=system_prompt, tools=tools or [],
                )
            except Exception as e:
                log.warning("Loop iteration Codex call failed: %s", e)
                return f"LLM call failed: {e}"

            if response.text:
                final_text = response.text

            if not response.tool_calls:
                break

            # Build assistant content with tool_use blocks (matches _process_with_tools format)
            assistant_content: list[dict] = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use", "id": tc.id,
                    "name": tc.name, "input": tc.input,
                })
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute tools concurrently with per-tool timeout
            async def _run_loop_tool(block):
                nonlocal system_prompt
                tool_name = block.name
                tool_input = block.input
                log.info("Loop tool call: %s(%s)", tool_name, tool_input)

                t0 = time.monotonic()
                error = None
                try:
                    raw = await asyncio.wait_for(
                        self._dispatch_loop_tool(
                            tool_name, tool_input, msg_proxy, user_id,
                        ),
                        timeout=tool_timeout,
                    )
                    # Skill CRUD invalidates caches
                    if tool_name in ("create_skill", "edit_skill", "delete_skill"):
                        self._cached_merged_tools = None
                        self._cached_skills_text = None
                        system_prompt = self._build_system_prompt(
                            channel=channel, user_id=user_id,
                        )
                except asyncio.TimeoutError:
                    error = f"Tool '{tool_name}' timed out after {tool_timeout}s"
                    raw = error
                except Exception as e:
                    error = str(e)
                    raw = f"Error executing {tool_name}: {e}"

                elapsed_ms = int((time.monotonic() - t0) * 1000)

                # Handle image block returns from analyze_image
                if isinstance(raw, dict) and "__image_block__" in raw:
                    raw = f"[Image loaded: {raw.get('__prompt__', '')}]"

                result = truncate_tool_output(scrub_output_secrets(str(raw)))

                # Audit log
                try:
                    await self.audit.log_execution(
                        user_id=user_id,
                        user_name=requester_name,
                        channel_id=channel_id_str,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        approved=True,
                        result_summary=result,
                        execution_time_ms=elapsed_ms,
                        error=error,
                    )
                except Exception:
                    pass

                return {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }

            tool_results = await asyncio.gather(
                *[_run_loop_tool(tc) for tc in response.tool_calls],
            )
            messages.append({"role": "user", "content": list(tool_results)})

        # Scrub final text; posting is handled by _post_response in LoopManager
        if final_text:
            final_text = scrub_output_secrets(final_text)
            if len(final_text) > DISCORD_MAX_LEN:
                final_text = final_text[:DISCORD_MAX_LEN - 50] + "\n... (truncated)"

        return final_text or "(no response)"

    async def _dispatch_loop_tool(
        self,
        tool_name: str,
        tool_input: dict,
        msg_proxy: _LoopMessageProxy,
        user_id: str,
    ) -> str | dict:
        """Dispatch a tool call to the correct handler within a loop iteration.

        Mirrors the Discord-native tool dispatch in _process_with_tools, using
        a lightweight message proxy instead of a real Discord message.
        """
        # --- Discord-native tools (message + input) ---
        if tool_name == "purge_messages":
            return await self._handle_purge(msg_proxy, tool_input)
        if tool_name == "browser_screenshot":
            return await self._handle_browser_screenshot(msg_proxy, tool_input)
        if tool_name == "generate_file":
            return await self._handle_generate_file(msg_proxy, tool_input)
        if tool_name == "post_file":
            return await self._handle_post_file(msg_proxy, tool_input)
        if tool_name == "schedule_task":
            return self._handle_schedule_task(msg_proxy, tool_input)
        if tool_name == "delegate_task":
            return await self._handle_delegate_task(msg_proxy, tool_input)
        if tool_name == "start_loop":
            return self._handle_start_loop(msg_proxy, tool_input)
        if tool_name == "add_reaction":
            return await self._handle_add_reaction(msg_proxy, tool_input)
        if tool_name == "create_poll":
            return await self._handle_create_poll(msg_proxy, tool_input)
        if tool_name == "broadcast":
            return await self._handle_broadcast(msg_proxy, tool_input)
        if tool_name == "analyze_image":
            return await self._handle_analyze_image(msg_proxy, tool_input)
        if tool_name == "generate_image":
            return await self._handle_generate_image(msg_proxy, tool_input)
        if tool_name == "create_digest":
            return self._handle_create_digest(msg_proxy, tool_input)

        # --- Discord-native tools (input only) ---
        if tool_name == "list_schedules":
            return self._handle_list_schedules()
        if tool_name == "delete_schedule":
            return self._handle_delete_schedule(tool_input)
        if tool_name == "parse_time":
            return self._handle_parse_time(tool_input)
        if tool_name == "search_history":
            return await self._handle_search_history(tool_input)
        if tool_name == "list_tasks":
            return self._handle_list_tasks(tool_input)
        if tool_name == "cancel_task":
            return self._handle_cancel_task(tool_input)
        if tool_name == "stop_loop":
            return self._handle_stop_loop(tool_input)
        if tool_name == "list_loops":
            return self._handle_list_loops()
        if tool_name == "search_knowledge":
            return await self._handle_search_knowledge(tool_input)
        if tool_name == "ingest_document":
            return await self._handle_ingest_document(tool_input, str(msg_proxy.author))
        if tool_name == "list_knowledge":
            return self._handle_list_knowledge()
        if tool_name == "delete_knowledge":
            return self._handle_delete_knowledge(tool_input)
        if tool_name == "set_permission":
            return self._handle_set_permission(user_id, tool_input)
        if tool_name == "search_audit":
            return await self._handle_search_audit(tool_input)

        # --- Skill CRUD ---
        if tool_name == "create_skill":
            return await asyncio.to_thread(
                self.skill_manager.create_skill, tool_input["name"], tool_input["code"],
            )
        if tool_name == "edit_skill":
            return await asyncio.to_thread(
                self.skill_manager.edit_skill, tool_input["name"], tool_input["code"],
            )
        if tool_name == "delete_skill":
            return await asyncio.to_thread(
                self.skill_manager.delete_skill, tool_input["name"],
            )
        if tool_name == "list_skills":
            skills = self.skill_manager.list_skills()
            if not skills:
                return "No user-created skills."
            lines = [f"**{s['name']}**: {s['description']}" for s in skills]
            return f"**User-created skills ({len(skills)}):**\n" + "\n".join(lines)

        # --- User-created skills ---
        if self.skill_manager.has_skill(tool_name):
            ch = msg_proxy.channel

            async def _skill_msg(text: str) -> None:
                await ch.send(scrub_response_secrets(text))

            async def _skill_file(data: bytes, filename: str, caption: str = "") -> None:
                ch_id_key = str(getattr(ch, "id", ""))
                self._pending_files.setdefault(ch_id_key, []).append((data, filename))

            return await self.skill_manager.execute(
                tool_name, tool_input,
                message_callback=_skill_msg,
                file_callback=_skill_file,
            )

        # --- Executor-routed tools (run_command, check_disk, SSH, etc.) ---
        return await self.tool_executor.execute(tool_name, tool_input, user_id=user_id)

    async def _handle_add_reaction(self, message: discord.Message, inp: dict) -> str:
        """Add an emoji reaction to a message."""
        message_id = inp.get("message_id")
        emoji = inp.get("emoji")
        if not emoji:
            return "'emoji' is required."
        # Resolve "this"/"current"/empty to the triggering message
        if not message_id or str(message_id).lower() in ("this", "current", "self"):
            message_id = str(message.id)
        try:
            msg = await message.channel.fetch_message(int(message_id))
            await msg.add_reaction(emoji)
            return "Reaction added."
        except discord.NotFound:
            return f"Message {message_id} not found in this channel."
        except discord.Forbidden:
            return "Permission denied to add reaction."
        except Exception as e:
            return f"Failed to add reaction: {e}"

    async def _handle_create_poll(self, message: discord.Message, inp: dict) -> str:
        """Create a Discord native poll in the current channel."""
        from datetime import timedelta

        question = inp.get("question")
        options = inp.get("options", [])
        if not question or not options:
            return "Both 'question' and 'options' are required."
        if len(options) > 10:
            return "Discord polls support a maximum of 10 options."
        # Scrub secrets from poll content before sending to Discord
        question = scrub_response_secrets(str(question))
        options = [scrub_response_secrets(str(opt)) for opt in options]
        duration_hours = min(inp.get("duration_hours", 24), 168)
        multiple = inp.get("multiple", False)
        try:
            poll = discord.Poll(
                question=question,
                duration=timedelta(hours=duration_hours),
                multiple=multiple,
            )
            for opt in options:
                poll.add_answer(text=opt)
            await message.channel.send(poll=poll)
            return "Poll created."
        except Exception as e:
            return f"Failed to create poll: {e}"

    async def _handle_broadcast(self, message: discord.Message, inp: dict) -> str:
        """Send a message with optional rich embed to the current channel."""
        text = inp.get("text")
        embed_data = inp.get("embed")
        embed_obj = None

        if embed_data and isinstance(embed_data, dict):
            color_str = embed_data.get("color", "#000000")
            try:
                color_val = int(color_str.lstrip("#"), 16)
            except (ValueError, AttributeError):
                color_val = 0
            embed_obj = discord.Embed(
                title=embed_data.get("title"),
                description=embed_data.get("description"),
                color=color_val,
            )
            for field in embed_data.get("fields", []):
                embed_obj.add_field(
                    name=field.get("name", "\u200b"),
                    value=field.get("value", "\u200b"),
                    inline=field.get("inline", False),
                )

        if not text and not embed_obj:
            return "Provide 'text' and/or 'embed' content."

        # Scrub secrets before sending to Discord (broadcast bypasses LLM response scrubbing)
        if text:
            text = scrub_response_secrets(text)
        await message.channel.send(content=text, embed=embed_obj)
        return "Message sent."

    async def _handle_analyze_image(self, message: discord.Message, inp: dict) -> str | dict:
        """Fetch an image and return a vision block for the LLM to analyze.

        Returns either an error string or a dict with ``__image_block__`` key
        that the tool loop injects as a vision content block.
        """
        import aiohttp

        url = inp.get("url")
        host = inp.get("host")
        path = inp.get("path")
        prompt = inp.get("prompt", "Describe this image in detail.")

        image_bytes: bytes | None = None

        if url:
            # Validate URL scheme to prevent SSRF via file://, ftp://, etc.
            if not url.startswith(("http://", "https://")):
                return "Only http:// and https:// URLs are supported."
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status != 200:
                            return f"Failed to fetch image from URL (HTTP {resp.status})"
                        ct = resp.headers.get("Content-Type", "")
                        if not ct.startswith("image/"):
                            return f"URL does not point to an image (Content-Type: {ct})"
                        image_bytes = await resp.read()
            except Exception as e:
                return f"Failed to fetch image from URL: {e}"
        elif host and path:
            # Use executor to fetch from host via base64
            import shlex
            resolved = self.tool_executor._resolve_host(host)
            if not resolved:
                return f"Unknown or disallowed host: {host}"
            address, ssh_user, _os = resolved
            safe_path = shlex.quote(path)
            code, output = await self.tool_executor._exec_command(
                address, f"base64 -w0 {safe_path}", ssh_user,
            )
            if code != 0:
                return f"Failed to read image from host: {output}"
            try:
                image_bytes = base64.b64decode(output.strip())
            except Exception as e:
                return f"Failed to decode image data: {e}"
        else:
            return "Provide either 'url' or both 'host' and 'path'."

        if not image_bytes:
            return "No image data retrieved."

        # Enforce 5MB limit (same as Discord attachment limit)
        if len(image_bytes) > 5 * 1024 * 1024:
            return "Image exceeds 5MB size limit."

        media_type = self._detect_image_type(image_bytes)
        if not media_type:
            return "Unsupported image format. Supported: PNG, JPEG, GIF, WEBP."

        b64 = base64.b64encode(image_bytes).decode("ascii")

        # Return a special marker dict that the tool loop will inject as a
        # vision content block.  The tool result text sent to the LLM will be
        # the prompt, while the image block gets appended to the next user
        # message so Codex can see it.
        return {
            "__image_block__": {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            },
            "__prompt__": prompt,
        }

    async def _handle_generate_image(self, message: discord.Message, inp: dict) -> str:
        """Generate an image via ComfyUI and post as Discord attachment."""
        if not self.config.comfyui.enabled:
            return "Image generation is disabled. Enable ComfyUI in config to use this tool."

        prompt_text = inp.get("prompt", "")
        if not prompt_text:
            return "A 'prompt' describing the image is required."

        negative = inp.get("negative", "")
        width = inp.get("width", 1024)
        height = inp.get("height", 1024)

        # Clamp dimensions to reasonable range
        width = max(64, min(2048, width))
        height = max(64, min(2048, height))

        from ..tools.comfyui import ComfyUIClient

        client = ComfyUIClient(self.config.comfyui.url)
        image_bytes = await client.generate(
            prompt=prompt_text,
            negative=negative,
            width=width,
            height=height,
        )

        if not image_bytes:
            return "Image generation failed. ComfyUI may be unavailable or the request timed out."

        try:
            file = discord.File(io.BytesIO(image_bytes), filename="generated.png")
            caption = scrub_response_secrets(f"Generated: {prompt_text[:100]}")
            await message.channel.send(content=caption, file=file)
            return f"Image generated and posted ({len(image_bytes) / 1024:.1f} KB)."
        except discord.HTTPException as e:
            return f"Failed to upload generated image to Discord: {e}"

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
            await channel.send(scrub_response_secrets(text))
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
                await channel.send(scrub_response_secrets(text))
            except Exception as e:
                await channel.send(
                    scrub_response_secrets(f"**Scheduled task failed:** {schedule['description']}\nError: {e}")
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
        code_block_lang = ""

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
            # Track code block state (toggle on ``` lines)
            if line.startswith("```"):
                if in_code_block:
                    in_code_block = False
                    code_block_lang = ""
                else:
                    in_code_block = True
                    code_block_lang = line[3:].strip()

            if len(current) + len(line) + 1 > DISCORD_MAX_LEN - 10:
                if in_code_block:
                    current += "\n```"
                if current.strip():
                    chunks.append(current)
                current = ""
                if in_code_block:
                    current = f"```{code_block_lang}\n"
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
