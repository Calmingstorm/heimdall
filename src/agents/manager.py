"""Agent manager — spawn, track, and coordinate autonomous agents.

Each agent runs as an independent asyncio task with its own LLM session,
isolated message history, and full tool access. Agents cannot spawn sub-agents.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..llm.secret_scrubber import scrub_output_secrets
from ..logging import get_logger

log = get_logger("agents")

# --- Constants ---
MAX_CONCURRENT_AGENTS = 5        # per channel
MAX_AGENT_LIFETIME = 3600        # 1 hour
MAX_AGENT_ITERATIONS = 30        # LLM turns per agent
STALE_WARN_SECONDS = 120         # 2 min no activity → log warning
CLEANUP_DELAY = 300              # 5 min after terminal state → remove
WAIT_DEFAULT_TIMEOUT = 300       # default timeout for wait_for_agents
WAIT_POLL_INTERVAL = 2           # poll interval for wait_for_agents
ITERATION_CB_TIMEOUT = 120       # 2 min timeout per LLM call
TOOL_EXEC_TIMEOUT = 300          # 5 min timeout per tool execution

_TERMINAL_STATUSES = frozenset({"completed", "failed", "timeout", "killed"})

# Tools agents are NOT allowed to call (prevents nesting and cross-agent interference)
AGENT_BLOCKED_TOOLS = frozenset({
    "spawn_agent",
    "send_to_agent",
    "list_agents",
    "kill_agent",
    "get_agent_results",
    "wait_for_agents",
})


def filter_agent_tools(tools: list[dict]) -> list[dict]:
    """Remove agent-management tools from a tool list for agent isolation."""
    return [t for t in tools if t.get("name") not in AGENT_BLOCKED_TOOLS]

# Callback types
# iteration_callback: (messages, system_prompt, tools) → LLMResponse-like dict
#   dict with keys: "text" (str), "tool_calls" (list[dict]), "stop_reason" (str)
IterationCallback = Callable[
    [list[dict], str, list[dict]],
    Awaitable[dict],
]

# tool_executor_callback: (tool_name, tool_input) → result string
ToolExecutorCallback = Callable[
    [str, dict],
    Awaitable[str],
]

# announce_callback: DEPRECATED — agents no longer post directly to Discord.
# Kept as optional parameter for API compat (loop_bridge passes it through).
AnnounceCallback = Callable[
    [str, str],
    Awaitable[None],
]


@dataclass
class AgentInfo:
    """Metadata and state for a running agent."""
    id: str
    label: str
    goal: str
    channel_id: str
    requester_id: str
    requester_name: str
    status: str = "running"         # running | completed | failed | timeout | killed
    created_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    result: str = ""
    error: str = ""
    messages: list[dict] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    iteration_count: int = 0
    last_activity: float = field(default_factory=time.time)
    _task: asyncio.Task | None = field(default=None, repr=False)
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    _inbox: asyncio.Queue = field(default_factory=asyncio.Queue)


class AgentManager:
    """Manages autonomous agent lifecycle — spawn, message, list, kill, cleanup."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentInfo] = {}
        self._cleanup_tasks: dict[str, asyncio.Task] = {}

    def spawn(
        self,
        label: str,
        goal: str,
        channel_id: str,
        requester_id: str,
        requester_name: str,
        iteration_callback: IterationCallback,
        tool_executor_callback: ToolExecutorCallback,
        announce_callback: AnnounceCallback | None = None,
        tools: list[dict] | None = None,
        system_prompt: str = "",
    ) -> str:
        """Spawn a new agent. Returns agent_id on success, or 'Error: ...' string."""
        # Check per-channel limit
        channel_count = sum(
            1 for a in self._agents.values()
            if a.channel_id == channel_id and a.status == "running"
        )
        if channel_count >= MAX_CONCURRENT_AGENTS:
            return f"Error: Maximum concurrent agents ({MAX_CONCURRENT_AGENTS}) reached for this channel."

        if not label or not goal:
            return "Error: Both 'label' and 'goal' are required."

        agent_id = uuid.uuid4().hex[:8]
        agent = AgentInfo(
            id=agent_id,
            label=label,
            goal=goal,
            channel_id=channel_id,
            requester_id=requester_id,
            requester_name=requester_name,
        )

        # Build agent system prompt
        agent_system = system_prompt
        if agent_system:
            agent_system += "\n\n"
        else:
            agent_system = ""
        agent_system += (
            f"AGENT CONTEXT: You are agent '{label}' working on a specific task. "
            f"Focus ONLY on this task. Do NOT spawn sub-agents. "
            f"When done, provide a clear summary of results."
        )

        # Seed messages with the goal
        agent.messages = [{"role": "user", "content": goal}]

        # Start the async task
        task = asyncio.ensure_future(
            _run_agent(
                agent=agent,
                system_prompt=agent_system,
                tools=tools or [],
                iteration_callback=iteration_callback,
                tool_executor_callback=tool_executor_callback,
                announce_callback=announce_callback,
            )
        )
        agent._task = task
        # Schedule cleanup when the agent task finishes (any exit path)
        task.add_done_callback(lambda _t: self._schedule_cleanup(agent_id))
        self._agents[agent_id] = agent

        log.info(
            "Spawned agent %s (%s) for channel %s by %s: %s",
            agent_id, label, channel_id, requester_name, goal[:100],
        )
        return agent_id

    def send(self, agent_id: str, message: str) -> str:
        """Inject a message into a running agent's inbox."""
        agent = self._agents.get(agent_id)
        if not agent:
            return f"Error: Agent '{agent_id}' not found."
        if agent.status != "running":
            return f"Error: Agent '{agent_id}' is not running (status: {agent.status})."
        if not message:
            return "Error: Message cannot be empty."

        agent._inbox.put_nowait(message)
        log.info("Sent message to agent %s (%s): %s", agent_id, agent.label, message[:80])
        return f"Message delivered to agent '{agent.label}'."

    def list(self, channel_id: str | None = None) -> list[dict]:
        """List agents, optionally filtered by channel."""
        result = []
        for agent in self._agents.values():
            if channel_id and agent.channel_id != channel_id:
                continue
            runtime = (agent.ended_at or time.time()) - agent.created_at
            result.append({
                "id": agent.id,
                "label": agent.label,
                "status": agent.status,
                "iteration_count": agent.iteration_count,
                "runtime_seconds": round(runtime, 1),
                "tools_used": len(agent.tools_used),
                "goal": agent.goal[:100],
            })
        return result

    def kill(self, agent_id: str) -> str:
        """Cancel a running agent."""
        agent = self._agents.get(agent_id)
        if not agent:
            return f"Error: Agent '{agent_id}' not found."
        if agent.status != "running":
            return f"Agent '{agent_id}' already in terminal state: {agent.status}."

        agent._cancel_event.set()
        log.info("Kill signal sent to agent %s (%s)", agent_id, agent.label)
        return f"Kill signal sent to agent '{agent.label}'."

    def get_results(self, agent_id: str) -> dict | None:
        """Get structured results of an agent."""
        agent = self._agents.get(agent_id)
        if not agent:
            return None

        runtime = (agent.ended_at or time.time()) - agent.created_at
        return {
            "id": agent.id,
            "label": agent.label,
            "status": agent.status,
            "result": agent.result,
            "error": agent.error,
            "iteration_count": agent.iteration_count,
            "tools_used": agent.tools_used,
            "runtime_seconds": round(runtime, 1),
            "goal": agent.goal,
        }

    async def wait_for_agents(
        self,
        agent_ids: list[str],
        timeout: float = WAIT_DEFAULT_TIMEOUT,
        poll_interval: float = WAIT_POLL_INTERVAL,
    ) -> dict[str, dict]:
        """Wait for all specified agents to reach terminal state.

        Returns {agent_id: results_dict} for each agent. Agents not found
        are reported as {"status": "not_found", "error": "..."}.
        """
        if not agent_ids:
            return {}

        deadline = time.time() + timeout
        while time.time() < deadline:
            all_done = True
            for aid in agent_ids:
                agent = self._agents.get(aid)
                if agent and agent.status not in _TERMINAL_STATUSES:
                    all_done = False
                    break
            if all_done:
                break
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(poll_interval, remaining))

        # Collect results
        results: dict[str, dict] = {}
        for aid in agent_ids:
            r = self.get_results(aid)
            if r:
                results[aid] = r
            else:
                results[aid] = {
                    "id": aid,
                    "status": "not_found",
                    "error": f"Agent '{aid}' not found.",
                }

        still_running = [
            aid for aid, r in results.items() if r.get("status") == "running"
        ]
        if still_running:
            log.warning(
                "wait_for_agents timed out with %d still running: %s",
                len(still_running), still_running,
            )

        return results

    def spawn_group(
        self,
        tasks: list[dict],
        channel_id: str,
        requester_id: str,
        requester_name: str,
        iteration_callback: IterationCallback,
        tool_executor_callback: ToolExecutorCallback,
        announce_callback: AnnounceCallback | None = None,
        tools: list[dict] | None = None,
        system_prompt: str = "",
    ) -> list[str]:
        """Spawn multiple agents at once. Returns list of agent_ids (or error strings).

        Each task dict must have 'label' and 'goal' keys.
        """
        ids: list[str] = []
        for task in tasks:
            label = task.get("label", "")
            goal = task.get("goal", "")
            aid = self.spawn(
                label=label,
                goal=goal,
                channel_id=channel_id,
                requester_id=requester_id,
                requester_name=requester_name,
                iteration_callback=iteration_callback,
                tool_executor_callback=tool_executor_callback,
                announce_callback=announce_callback,
                tools=tools,
                system_prompt=system_prompt,
            )
            ids.append(aid)
        return ids

    async def cleanup(self) -> int:
        """Remove agents that have been in terminal state for > CLEANUP_DELAY. Returns count removed."""
        now = time.time()
        to_remove = []
        for agent_id, agent in self._agents.items():
            if agent.status in ("completed", "failed", "timeout", "killed"):
                if agent.ended_at and (now - agent.ended_at) > CLEANUP_DELAY:
                    to_remove.append(agent_id)

        for agent_id in to_remove:
            del self._agents[agent_id]
            # Cancel cleanup task if one exists
            ct = self._cleanup_tasks.pop(agent_id, None)
            if ct and not ct.done():
                ct.cancel()

        if to_remove:
            log.info("Cleaned up %d finished agents", len(to_remove))
        return len(to_remove)

    def _schedule_cleanup(self, agent_id: str) -> None:
        """Schedule cleanup of an agent after CLEANUP_DELAY."""
        async def _delayed_cleanup():
            await asyncio.sleep(CLEANUP_DELAY)
            agent = self._agents.pop(agent_id, None)
            self._cleanup_tasks.pop(agent_id, None)
            if agent:
                log.debug("Auto-cleaned agent %s (%s)", agent_id, agent.label)

        task = asyncio.ensure_future(_delayed_cleanup())
        self._cleanup_tasks[agent_id] = task

    def check_health(self) -> dict:
        """Check agent health: force-kill stuck agents, log stale ones.

        Safety net for agents stuck in long tool calls that bypass the
        per-iteration lifetime check. Returns {"killed": N, "stale": N}.
        """
        now = time.time()
        killed = 0
        stale = 0
        for agent in list(self._agents.values()):
            if agent.status != "running":
                continue
            elapsed = now - agent.created_at
            idle = now - agent.last_activity
            if elapsed > MAX_AGENT_LIFETIME:
                agent._cancel_event.set()
                killed += 1
                log.warning(
                    "Force-killed stuck agent %s (%s): lifetime exceeded (%ds)",
                    agent.id, agent.label, int(elapsed),
                )
            elif idle > STALE_WARN_SECONDS:
                stale += 1
                log.warning(
                    "Agent %s (%s) appears stale: %ds idle",
                    agent.id, agent.label, int(idle),
                )
        return {"killed": killed, "stale": stale}

    @property
    def active_count(self) -> int:
        return sum(1 for a in self._agents.values() if a.status == "running")

    @property
    def total_count(self) -> int:
        return len(self._agents)


async def _run_agent(
    agent: AgentInfo,
    system_prompt: str,
    tools: list[dict],
    iteration_callback: IterationCallback,
    tool_executor_callback: ToolExecutorCallback,
    announce_callback: AnnounceCallback | None = None,
) -> None:
    """Execute an agent's tool loop until completion, error, or timeout.

    Agents are silent internal workers — they do NOT post directly to Discord.
    Results are stored in agent.result/agent.error for the parent to collect
    via wait_for_agents or get_agent_results.
    """

    try:
        for iteration in range(MAX_AGENT_ITERATIONS):
            # Check cancellation
            if agent._cancel_event.is_set():
                agent.status = "killed"
                agent.ended_at = time.time()
                log.info("Agent %s (%s) killed after %ds", agent.id, agent.label, int(time.time() - agent.created_at))
                return

            # Check lifetime
            elapsed = time.time() - agent.created_at
            if elapsed > MAX_AGENT_LIFETIME:
                agent.status = "timeout"
                agent.result = _get_last_progress(agent)
                agent.ended_at = time.time()
                log.warning("Agent %s (%s) timed out after %ds, %d iterations", agent.id, agent.label, int(elapsed), agent.iteration_count)
                return

            # Check inbox for injected messages
            while not agent._inbox.empty():
                try:
                    msg = agent._inbox.get_nowait()
                    agent.messages.append({
                        "role": "user",
                        "content": f"[Message from parent] {msg}",
                    })
                    log.debug("Agent %s received inbox message", agent.id)
                except asyncio.QueueEmpty:
                    break

            # Call LLM
            agent.last_activity = time.time()
            agent.iteration_count = iteration + 1

            try:
                response = await asyncio.wait_for(
                    iteration_callback(agent.messages, system_prompt, tools),
                    timeout=ITERATION_CB_TIMEOUT,
                )
            except asyncio.TimeoutError:
                log.error("Agent %s LLM call timed out at iteration %d", agent.id, iteration)
                agent.status = "failed"
                agent.error = f"LLM call timed out after {ITERATION_CB_TIMEOUT}s"
                agent.ended_at = time.time()
                return
            except Exception as e:
                log.error("Agent %s LLM call failed at iteration %d: %s", agent.id, iteration, e)
                agent.status = "failed"
                agent.error = f"LLM call failed: {e}"
                agent.ended_at = time.time()
                return

            text = response.get("text", "")
            tool_calls = response.get("tool_calls", [])

            # Append assistant response to messages
            agent.messages.append({"role": "assistant", "content": text})

            # No tool calls = agent is done
            if not tool_calls:
                agent.status = "completed"
                agent.result = text
                agent.ended_at = time.time()
                elapsed = time.time() - agent.created_at
                log.info("Agent %s (%s) completed in %ds, %d tool calls", agent.id, agent.label, int(elapsed), len(agent.tools_used))
                return

            # Execute tool calls
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_input = tc.get("input", {})

                if tool_name not in agent.tools_used:
                    agent.tools_used.append(tool_name)

                agent.last_activity = time.time()

                try:
                    result = await asyncio.wait_for(
                        tool_executor_callback(tool_name, tool_input),
                        timeout=TOOL_EXEC_TIMEOUT,
                    )
                    result = scrub_output_secrets(str(result))
                except asyncio.TimeoutError:
                    result = f"Error: Tool '{tool_name}' timed out after {TOOL_EXEC_TIMEOUT}s"
                    log.warning("Agent %s tool %s timed out", agent.id, tool_name)
                except Exception as e:
                    result = f"Error: {e}"
                    log.warning("Agent %s tool %s failed: %s", agent.id, tool_name, e)

                # Append tool result to messages
                agent.messages.append({
                    "role": "user",
                    "content": f"[Tool result: {tool_name}]\n{result}",
                })

            # Check stale warning
            if time.time() - agent.last_activity > STALE_WARN_SECONDS:
                log.warning(
                    "Agent %s (%s) has been idle for >%ds",
                    agent.id, agent.label, STALE_WARN_SECONDS,
                )

        # Exhausted iterations
        agent.status = "completed"
        agent.result = _get_last_progress(agent)
        agent.ended_at = time.time()
        elapsed = time.time() - agent.created_at
        log.info("Agent %s (%s) completed in %ds after %d iterations (max reached), %d tool calls", agent.id, agent.label, int(elapsed), MAX_AGENT_ITERATIONS, len(agent.tools_used))

    except asyncio.CancelledError:
        agent.status = "killed"
        agent.ended_at = time.time()
        log.info("Agent %s (%s) was cancelled", agent.id, agent.label)

    except Exception as e:
        agent.status = "failed"
        agent.error = str(e)
        agent.ended_at = time.time()
        log.error("Agent %s (%s) crashed: %s", agent.id, agent.label, e)


def _get_last_progress(agent: AgentInfo) -> str:
    """Extract the last meaningful text from agent messages."""
    for msg in reversed(agent.messages):
        if msg["role"] == "assistant" and msg.get("content"):
            return msg["content"]
    return "(no output)"


