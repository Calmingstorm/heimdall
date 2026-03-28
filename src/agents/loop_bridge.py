"""Loop-Agent bridge — integrates autonomous loops with multi-agent spawning.

When an autonomous loop detects a situation that needs parallel sub-tasks
(e.g., "monitor X, when Y happens spawn agents to fix it"), this bridge
provides the glue between LoopManager iterations and AgentManager.spawn().

The bridge:
- Injects agent awareness into loop iteration prompts
- Spawns agents with loop context (loop ID, iteration number, goal)
- Tracks which agents belong to which loop
- Collects agent results back into loop iteration context
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from ..logging import get_logger

log = get_logger("loop_bridge")

# Max agents spawned per single loop iteration (prevent runaway)
MAX_AGENTS_PER_ITERATION = 3
# Max total agents spawned by a single loop across its lifetime
MAX_AGENTS_PER_LOOP = 10
# Default timeout waiting for agents spawned from a loop
LOOP_AGENT_WAIT_TIMEOUT = 300  # 5 minutes


@dataclass
class LoopAgentRecord:
    """Tracks an agent spawned from a loop iteration."""
    agent_id: str
    loop_id: str
    iteration: int
    label: str
    spawned_at: float = field(default_factory=time.time)
    collected: bool = False


class LoopAgentBridge:
    """Bridges LoopManager and AgentManager for agent-aware loops."""

    def __init__(self, agent_manager: object) -> None:
        self._agent_manager = agent_manager
        # loop_id → list of LoopAgentRecords
        self._loop_agents: dict[str, list[LoopAgentRecord]] = {}

    def get_loop_agent_ids(self, loop_id: str) -> list[str]:
        """Return all agent IDs spawned by a given loop."""
        records = self._loop_agents.get(loop_id, [])
        return [r.agent_id for r in records]

    def get_loop_agent_count(self, loop_id: str) -> int:
        """Return count of agents spawned by a loop (lifetime total)."""
        return len(self._loop_agents.get(loop_id, []))

    def spawn_agents_for_loop(
        self,
        loop_id: str,
        iteration: int,
        loop_goal: str,
        tasks: list[dict],
        channel_id: str,
        requester_id: str,
        requester_name: str,
        iteration_callback: object,
        tool_executor_callback: object,
        announce_callback: object = None,
        tools: list[dict] | None = None,
        system_prompt: str = "",
    ) -> list[str]:
        """Spawn agents for a loop iteration.

        Each task dict must have 'label' and 'goal' keys.
        Returns list of agent_ids (or error strings starting with 'Error:').
        """
        if not tasks:
            return []

        # Enforce per-iteration limit
        if len(tasks) > MAX_AGENTS_PER_ITERATION:
            return [
                f"Error: Cannot spawn more than {MAX_AGENTS_PER_ITERATION} "
                f"agents per iteration (requested {len(tasks)})."
            ]

        # Enforce per-loop lifetime limit
        current_count = self.get_loop_agent_count(loop_id)
        if current_count + len(tasks) > MAX_AGENTS_PER_LOOP:
            remaining = MAX_AGENTS_PER_LOOP - current_count
            return [
                f"Error: Loop '{loop_id}' has spawned {current_count} agents "
                f"(limit {MAX_AGENTS_PER_LOOP}). Can spawn {remaining} more."
            ]

        if loop_id not in self._loop_agents:
            self._loop_agents[loop_id] = []

        results: list[str] = []
        for task in tasks:
            label = task.get("label", "")
            goal = task.get("goal", "")

            # Inject loop context into the agent goal
            enriched_goal = (
                f"[Spawned by loop '{loop_id}', iteration {iteration}]\n"
                f"Loop goal: {loop_goal}\n"
                f"Agent task: {goal}"
            )

            agent_id = self._agent_manager.spawn(
                label=label,
                goal=enriched_goal,
                channel_id=channel_id,
                requester_id=requester_id,
                requester_name=requester_name,
                iteration_callback=iteration_callback,
                tool_executor_callback=tool_executor_callback,
                announce_callback=announce_callback,
                tools=tools,
                system_prompt=system_prompt,
            )

            if not agent_id.startswith("Error"):
                self._loop_agents[loop_id].append(
                    LoopAgentRecord(
                        agent_id=agent_id,
                        loop_id=loop_id,
                        iteration=iteration,
                        label=label,
                    )
                )
                log.info(
                    "Loop %s iter %d spawned agent %s (%s)",
                    loop_id, iteration, agent_id, label,
                )
            results.append(agent_id)

        return results

    async def wait_and_collect(
        self,
        loop_id: str,
        agent_ids: list[str] | None = None,
        timeout: float = LOOP_AGENT_WAIT_TIMEOUT,
    ) -> dict[str, dict]:
        """Wait for agents spawned by a loop and collect their results.

        If agent_ids is None, waits for all uncollected agents from this loop.
        Returns {agent_id: results_dict}.
        """
        if agent_ids is None:
            records = self._loop_agents.get(loop_id, [])
            agent_ids = [r.agent_id for r in records if not r.collected]

        if not agent_ids:
            return {}

        results = await self._agent_manager.wait_for_agents(
            agent_ids, timeout=timeout,
        )

        # Mark collected
        for record in self._loop_agents.get(loop_id, []):
            if record.agent_id in agent_ids:
                record.collected = True

        return results

    def format_agent_results_for_context(self, results: dict[str, dict]) -> str:
        """Format agent results into a string for loop iteration context."""
        if not results:
            return ""

        lines = ["Agent results:"]
        for aid, r in results.items():
            status = r.get("status", "unknown")
            label = r.get("label", aid)
            result_text = r.get("result", "") or r.get("error", "(no output)")
            # Truncate individual result
            if len(result_text) > 500:
                result_text = result_text[:500] + "..."
            lines.append(f"- [{label}] ({status}): {result_text}")

        return "\n".join(lines)

    def cleanup_loop(self, loop_id: str) -> int:
        """Remove agent records for a finished loop. Returns count removed."""
        records = self._loop_agents.pop(loop_id, [])
        return len(records)

    def get_active_loop_agents(self, loop_id: str) -> list[dict]:
        """Return info about active (uncollected) agents for a loop."""
        records = self._loop_agents.get(loop_id, [])
        active = []
        for r in records:
            if r.collected:
                continue
            agent_results = self._agent_manager.get_results(r.agent_id)
            if agent_results:
                active.append({
                    "agent_id": r.agent_id,
                    "label": r.label,
                    "iteration": r.iteration,
                    "status": agent_results.get("status", "unknown"),
                })
        return active

    @property
    def tracked_loop_count(self) -> int:
        """Number of loops with tracked agent records."""
        return len(self._loop_agents)
