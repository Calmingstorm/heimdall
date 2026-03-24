from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import aiohttp

from ..logging import get_logger

if TYPE_CHECKING:
    from ..knowledge.store import KnowledgeStore
    from ..scheduler.scheduler import Scheduler
    from ..search.embedder import LocalEmbedder
    from ..sessions.manager import SessionManager
    from .executor import ToolExecutor

# Tools that skills are allowed to call via execute_tool().
# Only read-only / non-destructive tools are included.
SKILL_SAFE_TOOLS: frozenset[str] = frozenset({
    "check_service",
    "check_docker",
    "check_disk",
    "check_memory",
    "check_logs",
    "query_prometheus",
    "query_prometheus_range",
    "read_file",
    "docker_logs",
    "docker_compose_status",
    "docker_compose_logs",
    "docker_stats",
    "git_status",
    "git_log",
    "git_diff",
    "git_show",
    "search_history",
    "search_audit",
    "search_knowledge",
    "list_knowledge",
    "list_schedules",
    "list_skills",
    "list_tasks",
    "memory_manage",
    "parse_time",
    "incus_list",
    "incus_info",
    "incus_snapshot_list",
    "incus_logs",
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

    async def run_on_host(self, alias: str, command: str) -> str:
        """Run a shell command on a managed host via SSH. Returns output string."""
        return await self._executor._run_on_host(alias, command)

    async def query_prometheus(self, query: str) -> str:
        """Run a PromQL instant query against Prometheus. Returns raw JSON."""
        return await self._executor.execute("query_prometheus", {"query": query})

    async def read_file(self, host: str, path: str, lines: int = 200) -> str:
        """Read a file from a managed host. Returns file content."""
        return await self._executor.execute("read_file", {
            "host": host, "path": path, "lines": lines,
        })

    async def post_message(self, text: str) -> None:
        """Send a message to the channel that invoked this skill."""
        if self._message_callback:
            await self._message_callback(text)
        else:
            self._log.warning("post_message called but no channel callback available")

    async def post_file(self, data: bytes, filename: str, caption: str = "") -> None:
        """Send a binary file to the channel that invoked this skill."""
        if self._file_callback:
            await self._file_callback(data, filename, caption)
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
        """List allowed systemd service names."""
        return list(self._executor.config.allowed_services)

    async def http_get(
        self, url: str, params: dict | None = None, timeout: int = 15,
    ) -> dict | list | str:
        """Perform an HTTP GET request. Auto-parses JSON responses, otherwise returns string."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                ct = resp.content_type or ""
                if "json" in ct:
                    return await resp.json()
                text = await resp.text()
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
    ) -> dict | list | str:
        """Perform an HTTP POST request. Auto-parses JSON responses, otherwise returns string."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=json, data=data, timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                ct = resp.content_type or ""
                if "json" in ct:
                    return await resp.json()
                text = await resp.text()
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
