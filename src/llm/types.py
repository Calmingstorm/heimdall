"""Backend-agnostic types for LLM responses with tool calling."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ToolCall:
    """A single tool call extracted from an LLM response.

    Works with OpenAI (function_call items) and internal tool_use blocks.
    """

    id: str  # call_id (OpenAI) or internal tool_use_id
    name: str  # tool name
    input: dict  # parsed tool arguments


@dataclass(slots=True)
class LLMResponse:
    """Normalized response from any LLM backend.

    Unifies LLM backend responses into a single structure that
    the tool loop can consume.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn" or "tool_use"

    @property
    def is_tool_use(self) -> bool:
        return self.stop_reason == "tool_use" or len(self.tool_calls) > 0
