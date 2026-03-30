from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import aiohttp

from ..logging import get_logger

if TYPE_CHECKING:
    from ..knowledge.store import KnowledgeStore
    from ..scheduler.scheduler import Scheduler
    from ..search.embedder import LocalEmbedder
    from ..sessions.manager import SessionManager
    from .executor import ToolExecutor


# ---------------------------------------------------------------------------
# Resource tracking
# ---------------------------------------------------------------------------

@dataclass
class ResourceTracker:
    """Tracks resource usage during a single skill execution."""
    tool_calls: int = 0
    http_requests: int = 0
    messages_sent: int = 0
    files_sent: int = 0
    bytes_downloaded: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_calls": self.tool_calls,
            "http_requests": self.http_requests,
            "messages_sent": self.messages_sent,
            "files_sent": self.files_sent,
            "bytes_downloaded": self.bytes_downloaded,
        }


# ---------------------------------------------------------------------------
# Sandbox limits
# ---------------------------------------------------------------------------

# Maximum number of tool calls per skill execution.
MAX_SKILL_TOOL_CALLS = 50
# Maximum number of HTTP requests per skill execution.
MAX_SKILL_HTTP_REQUESTS = 20
# Maximum number of messages per skill execution.
MAX_SKILL_MESSAGES = 10
MAX_SKILL_FILES = 10

# File path patterns that skills are NOT allowed to read.
_DENIED_PATH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(^|/)\.env($|\.)"),       # .env, .env.local, etc.
    re.compile(r"(^|/)config\.ya?ml$"),     # config.yml / config.yaml
    re.compile(r"/etc/shadow$"),            # system shadow passwords
    re.compile(r"(^|/)id_(rsa|ed25519|ecdsa|dsa)$"),  # SSH private keys
    re.compile(r"(^|/)\.ssh/"),             # entire .ssh directory
    re.compile(r"(^|/)credentials\.json$"), # service credentials
    re.compile(r"(^|/)\.kube/config$"),     # kubernetes config
]


def is_path_denied(path: str) -> bool:
    """Return True if a file path matches a denied pattern."""
    for pat in _DENIED_PATH_PATTERNS:
        if pat.search(path):
            return True
    return False


# Operator-configured URLs that skills are allowed to access despite being
# local/private. Set via config: skills.allowed_urls: ["http://localhost:8188"]
_SKILL_ALLOWED_URLS: set[str] = set()


def set_skill_allowed_urls(urls: list[str]) -> None:
    """Populate the skill URL allowlist from config."""
    _SKILL_ALLOWED_URLS.clear()
    for u in urls:
        _SKILL_ALLOWED_URLS.add(u.rstrip("/"))


def is_url_blocked(url: str) -> bool:
    """Return True if a URL targets localhost, private IPs, or metadata endpoints."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        return True  # malformed → block

    # Block empty/missing hostname
    if not host:
        return True

    # Check operator allowlist — matches if the URL starts with any allowed prefix
    url_base = f"{parsed.scheme}://{host}:{parsed.port}" if parsed.port else f"{parsed.scheme}://{host}"
    if any(url_base.rstrip("/") == allowed or url.startswith(allowed) for allowed in _SKILL_ALLOWED_URLS):
        return False

    # Block common localhost names
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True

    # Block cloud metadata endpoints
    if host in ("169.254.169.254", "metadata.google.internal"):
        return True

    # Block private IP ranges
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            return True
    except ValueError:
        pass  # hostname, not IP — allow

    return False


# Tools that skills are allowed to call via execute_tool().
# Only read-only / non-destructive tools are included.
SKILL_SAFE_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "search_history",
    "search_audit",
    "search_knowledge",
    "list_knowledge",
    "list_schedules",
    "list_skills",
    "list_tasks",
    "memory_manage",
    "parse_time",
    "web_search",
    "fetch_url",
    "browser_screenshot",
    "browser_read_page",
    "browser_read_table",
})


class SkillContext:
    """API surface passed to user-created skills.

    Provides SSH execution, HTTP helpers, Prometheus queries, file reading,
    persistent memory, channel messaging, config access, knowledge base,
    conversation history search, scheduler, and generic tool execution.
    """

    def __init__(
        self,
        tool_executor: ToolExecutor,
        skill_name: str,
        memory_path: str | None = None,
        message_callback: Callable[[str], Awaitable[None]] | None = None,
        file_callback: Callable[[bytes, str, str], Awaitable[None]] | None = None,
        knowledge_store: KnowledgeStore | None = None,
        embedder: LocalEmbedder | None = None,
        session_manager: SessionManager | None = None,
        scheduler: Scheduler | None = None,
        skill_config: dict[str, Any] | None = None,
        resource_tracker: ResourceTracker | None = None,
    ) -> None:
        self._executor = tool_executor
        self._log = get_logger(f"skills.{skill_name}")
        self._memory_path = Path(memory_path) if memory_path else None
        self._message_callback = message_callback
        self._file_callback = file_callback
        self._knowledge_store = knowledge_store
        self._embedder = embedder
        self._session_manager = session_manager
        self._scheduler = scheduler
        self._config: dict[str, Any] = skill_config or {}
        self._tracker: ResourceTracker = resource_tracker or ResourceTracker()

    async def run_on_host(self, alias: str, command: str) -> str:
        """Run a shell command on a managed host via SSH. Returns output string."""
        return await self._executor._run_on_host(alias, command)

    async def query_prometheus(self, query: str) -> str:
        """Run a PromQL instant query against Prometheus via curl.

        Requires Prometheus to be reachable from a configured host.
        """
        # Use run_command with curl since the dedicated query_prometheus tool was removed.
        hosts = list(self._executor.config.hosts.keys())
        if not hosts:
            return "No hosts configured to reach Prometheus."
        host = hosts[0]
        from urllib.parse import quote as url_quote
        encoded_query = url_quote(query)
        return await self._executor.execute("run_command", {
            "host": host,
            "command": f"curl -sf 'http://localhost:9090/api/v1/query?query={encoded_query}'",
        })

    async def read_file(self, host: str, path: str, lines: int = 200) -> str:
        """Read a file from a managed host. Returns file content."""
        if is_path_denied(path):
            self._log.warning("Skill attempted to read denied path: %s", path)
            return f"Access denied: '{path}' is a restricted path."
        return await self._executor.execute("read_file", {
            "host": host, "path": path, "lines": lines,
        })

    async def post_message(self, text: str) -> None:
        """Send a message to the channel that invoked this skill."""
        if self._tracker.messages_sent >= MAX_SKILL_MESSAGES:
            self._log.warning("Skill exceeded message limit (%d)", MAX_SKILL_MESSAGES)
            return
        if self._message_callback:
            await self._message_callback(text)
            self._tracker.messages_sent += 1
        else:
            self._log.warning("post_message called but no channel callback available")

    async def post_file(self, data: bytes, filename: str, caption: str = "") -> None:
        """Send a binary file to the channel that invoked this skill."""
        if self._tracker.files_sent >= MAX_SKILL_FILES:
            self._log.warning("Skill exceeded file send limit (%d)", MAX_SKILL_FILES)
            return
        if self._file_callback:
            await self._file_callback(data, filename, caption)
            self._tracker.files_sent += 1
        else:
            self._log.warning("post_file called but no channel callback available")

    def remember(self, key: str, value: str) -> None:
        """Save a key/value pair to persistent memory."""
        if not self._memory_path:
            return
        memory = self._load_memory()
        memory[key] = value
        self._save_memory(memory)

    def recall(self, key: str) -> str | None:
        """Retrieve a value from persistent memory. Returns None if not found."""
        memory = self._load_memory()
        return memory.get(key)

    def get_hosts(self) -> list[str]:
        """List available host aliases."""
        return list(self._executor.config.hosts.keys())

    def get_services(self) -> list[str]:
        """List allowed systemd service names.

        Returns an empty list since systemd tools were removed.
        """
        return []

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a single skill config value. Returns default if not set."""
        return self._config.get(key, default)

    def get_all_config(self) -> dict[str, Any]:
        """Get all skill config values (with defaults applied)."""
        return dict(self._config)

    async def http_get(
        self,
        url: str,
        params: dict | None = None,
        timeout: int = 15,
        headers: dict[str, str] | None = None,
    ) -> dict | list | str | bytes:
        """Perform an HTTP GET request. Auto-parses JSON, returns bytes for binary content.

        Custom headers can be passed via *headers*. By default ``Accept: application/json``
        is included unless overridden. Binary content types (image/*, video/*) return raw bytes.
        """
        if is_url_blocked(url):
            self._log.warning("Skill attempted blocked URL: %s", url)
            return "Access denied: internal/private URLs are not allowed from skills."
        if self._tracker.http_requests >= MAX_SKILL_HTTP_REQUESTS:
            return f"HTTP request limit ({MAX_SKILL_HTTP_REQUESTS}) exceeded."
        self._tracker.http_requests += 1
        merged = {"Accept": "application/json"}
        if headers:
            merged.update(headers)
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, params=params, headers=merged,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                ct = resp.content_type or ""
                if "json" in ct:
                    return await resp.json()
                # Return raw bytes for binary content (images, gifs, etc.)
                if ct.startswith(("image/", "application/octet-stream", "video/")):
                    data = await resp.read()
                    self._tracker.bytes_downloaded += len(data)
                    return data
                text = await resp.text()
                self._tracker.bytes_downloaded += len(text.encode())
                try:
                    return json.loads(text)
                except (ValueError, TypeError):
                    return text

    async def http_post(
        self,
        url: str,
        json: dict | None = None,
        data: str | None = None,
        timeout: int = 15,
        headers: dict[str, str] | None = None,
    ) -> dict | list | str:
        """Perform an HTTP POST request. Auto-parses JSON responses, otherwise returns string.

        Custom headers can be passed via *headers*.
        """
        if is_url_blocked(url):
            self._log.warning("Skill attempted blocked URL: %s", url)
            return "Access denied: internal/private URLs are not allowed from skills."
        if self._tracker.http_requests >= MAX_SKILL_HTTP_REQUESTS:
            return f"HTTP request limit ({MAX_SKILL_HTTP_REQUESTS}) exceeded."
        self._tracker.http_requests += 1
        merged: dict[str, str] = {}
        if headers:
            merged.update(headers)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=json, data=data, headers=merged or None,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                ct = resp.content_type or ""
                if "json" in ct:
                    return await resp.json()
                text = await resp.text()
                self._tracker.bytes_downloaded += len(text.encode())
                try:
                    return json.loads(text)
                except (ValueError, TypeError):
                    return text

    async def search_knowledge(self, query: str, limit: int = 5) -> list[dict]:
        """Search the knowledge base. Returns list of {content, source, score}."""
        if not self._knowledge_store or not self._embedder:
            return []
        return await self._knowledge_store.search_hybrid(query, self._embedder, limit=limit)

    async def ingest_document(self, content: str, source: str) -> int:
        """Ingest text into the knowledge base. Returns number of chunks indexed."""
        if not self._knowledge_store or not self._embedder:
            return 0
        return await self._knowledge_store.ingest(content, source, self._embedder)

    async def search_history(self, query: str, limit: int = 10) -> list[dict]:
        """Search conversation history. Returns list of {type, content, timestamp, channel_id}."""
        if not self._session_manager:
            return []
        return await self._session_manager.search_history(query, limit=limit)

    def schedule_task(
        self,
        description: str,
        action: str,
        channel_id: str,
        **kwargs: Any,
    ) -> dict | None:
        """Add a scheduled task. Returns the schedule dict, or None if scheduler unavailable.

        Keyword args are passed to Scheduler.add() — e.g. cron, run_at, trigger,
        tool_name, tool_input, steps, message.
        """
        if not self._scheduler:
            return None
        return self._scheduler.add(description, action, channel_id, **kwargs)

    def list_schedules(self) -> list[dict]:
        """List all scheduled tasks."""
        if not self._scheduler:
            return []
        return self._scheduler.list_all()

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a scheduled task by ID. Returns True if deleted."""
        if not self._scheduler:
            return False
        return self._scheduler.delete(schedule_id)

    async def execute_tool(self, tool_name: str, tool_input: dict | None = None) -> str:
        """Execute a safe built-in tool by name. Returns the tool's output string.

        Only tools listed in SKILL_SAFE_TOOLS are allowed. Destructive
        tools (run_command, write_file, etc.) are blocked from skill context.
        """
        if tool_name not in SKILL_SAFE_TOOLS:
            self._log.warning("Skill attempted blocked tool: %s", tool_name)
            return f"Tool '{tool_name}' is not allowed from skills. Only read-only tools are permitted."
        if self._tracker.tool_calls >= MAX_SKILL_TOOL_CALLS:
            return f"Tool call limit ({MAX_SKILL_TOOL_CALLS}) exceeded."
        self._tracker.tool_calls += 1
        # Apply file path restriction for read_file
        if tool_name == "read_file":
            path = (tool_input or {}).get("path", "")
            if is_path_denied(path):
                self._log.warning("Skill attempted to read denied path via tool: %s", path)
                return f"Access denied: '{path}' is a restricted path."
        return await self._executor.execute(tool_name, tool_input or {})

    def log(self, msg: str) -> None:
        """Write a log message under the skill's namespace."""
        self._log.info("%s", msg)

    def _load_memory(self) -> dict[str, str]:
        if not self._memory_path or not self._memory_path.exists():
            return {}
        try:
            return json.loads(self._memory_path.read_text())
        except Exception:
            return {}

    def _save_memory(self, data: dict[str, str]) -> None:
        if not self._memory_path:
            return
        self._memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_path.write_text(json.dumps(data, indent=2))
