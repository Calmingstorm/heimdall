"""Multi-agent orchestration — spawn autonomous agents for parallel sub-tasks."""
from __future__ import annotations

from .loop_bridge import LoopAgentBridge
from .manager import (
    AGENT_BLOCKED_TOOLS,
    ITERATION_CB_TIMEOUT,
    TOOL_EXEC_TIMEOUT,
    AgentInfo,
    AgentManager,
    filter_agent_tools,
)

__all__ = [
    "AGENT_BLOCKED_TOOLS",
    "ITERATION_CB_TIMEOUT",
    "TOOL_EXEC_TIMEOUT",
    "AgentInfo",
    "AgentManager",
    "LoopAgentBridge",
    "filter_agent_tools",
]
