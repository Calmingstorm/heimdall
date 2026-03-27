"""Background task delegation — run multi-step tool sequences without blocking conversation.

The LLM constructs a list of steps upfront, the user approves once, and the task
runs in the background with progress updates via an editable Discord message.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import discord

from ..llm.secret_scrubber import scrub_output_secrets
from ..logging import get_logger

if TYPE_CHECKING:
    from ..audit.logger import AuditLogger
    from ..tools.executor import ToolExecutor
    from ..tools.skill_manager import SkillManager
    from ..knowledge.store import KnowledgeStore
    from ..search.embedder import LocalEmbedder

# Type for Codex chat callback: takes (messages, system, max_tokens) -> response text
CodexCallback = Callable[[list[dict], str, int], Awaitable[str]]

log = get_logger("background_task")

# Tools that cannot run in background tasks (need Discord/interactive context)
BLOCKED_TOOLS = {
    "purge_messages", "browser_screenshot", "generate_file", "post_file",
    "browser_click", "browser_fill", "browser_evaluate",
    "delegate_task",  # no nesting
    "schedule_task", "delete_schedule", "create_digest",
    "create_skill", "edit_skill", "delete_skill",
    "start_loop", "stop_loop",  # need LoopManager from client
    "spawn_agent", "send_to_agent", "kill_agent",  # no agent nesting
}

MAX_STEPS = 200
PROGRESS_UPDATE_INTERVAL = 2.0  # seconds between Discord message edits


@dataclass
class StepResult:
    index: int
    tool_name: str
    description: str
    status: str  # "ok", "error", "skipped", "cancelled"
    output: str = ""
    elapsed_ms: int = 0


@dataclass
class BackgroundTask:
    task_id: str
    description: str
    steps: list[dict]
    channel: discord.abc.Messageable
    requester: str
    requester_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "running"  # running, completed, failed, cancelled
    results: list[StepResult] = field(default_factory=list)
    current_step: int = 0
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    _asyncio_task: asyncio.Task | None = field(default=None, repr=False)

    def cancel(self) -> None:
        self._cancel_event.set()


async def run_background_task(
    task: BackgroundTask,
    executor: ToolExecutor,
    skill_manager: SkillManager,
    knowledge_store: KnowledgeStore | None = None,
    embedder: LocalEmbedder | None = None,
    audit_logger: AuditLogger | None = None,
    codex_callback: CodexCallback | None = None,
) -> None:
    """Execute a background task's steps sequentially with progress updates."""

    # Post initial progress message
    progress_msg = await _send_progress(task, None)

    variables: dict[str, str] = {}
    prev_output = ""
    last_update = time.monotonic()

    for i, step in enumerate(task.steps):
        # Check cancellation
        if task._cancel_event.is_set():
            task.status = "cancelled"
            task.results.append(StepResult(
                index=i, tool_name=step.get("tool_name", ""),
                description=step.get("description", ""), status="cancelled",
            ))
            break

        task.current_step = i
        tool_name = step["tool_name"]
        tool_input = step.get("tool_input", {})
        condition = step.get("condition")
        on_failure = step.get("on_failure", "abort")
        step_desc = step.get("description", tool_name)
        store_as = step.get("store_as")

        # Variable substitution in tool_input string values
        tool_input = _substitute_vars(tool_input, variables, prev_output)

        # Evaluate condition
        if condition and prev_output:
            if not _check_condition(condition, prev_output):
                task.results.append(StepResult(
                    index=i, tool_name=tool_name,
                    description=step_desc, status="skipped",
                    output=f"Condition not met: {condition}",
                ))
                # Update progress periodically
                now = time.monotonic()
                if now - last_update >= PROGRESS_UPDATE_INTERVAL:
                    progress_msg = await _send_progress(task, progress_msg)
                    last_update = now
                continue

        # Check blocked tools
        if tool_name in BLOCKED_TOOLS:
            task.results.append(StepResult(
                index=i, tool_name=tool_name,
                description=step_desc, status="error",
                output=f"Tool '{tool_name}' cannot run in background tasks.",
            ))
            if on_failure == "abort":
                task.status = "failed"
                break
            continue

        # Execute the tool
        t0 = time.monotonic()
        try:
            output = await _execute_tool(
                tool_name, tool_input, executor, skill_manager,
                knowledge_store, embedder, task.requester,
                step_desc=step_desc,
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            output = scrub_output_secrets(output)
            prev_output = output

            if store_as:
                variables[store_as] = output

            # Detect error strings returned by executor (not raised as exceptions)
            is_error = _is_error_output(output)
            task.results.append(StepResult(
                index=i, tool_name=tool_name,
                description=step_desc, status="error" if is_error else "ok",
                output=output[:500], elapsed_ms=elapsed_ms,
            ))

            if audit_logger:
                try:
                    log_kwargs: dict = dict(
                        user_id=task.requester_id, user_name=task.requester,
                        channel_id=str(getattr(task.channel, "id", "")),
                        tool_name=tool_name, tool_input=tool_input, approved=True,
                        result_summary=output[:500],
                        execution_time_ms=elapsed_ms,
                    )
                    if is_error:
                        log_kwargs["error"] = output[:500]
                    await audit_logger.log_execution(**log_kwargs)
                except Exception:
                    log.warning("Failed to audit log step %d of task %s", i, task.task_id)

            # Abort on detected error if on_failure policy requires it
            if is_error and on_failure == "abort":
                task.status = "failed"
                break

        except Exception as e:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            error_msg = str(e)
            task.results.append(StepResult(
                index=i, tool_name=tool_name,
                description=step_desc, status="error",
                output=error_msg[:500], elapsed_ms=elapsed_ms,
            ))

            if audit_logger:
                try:
                    await audit_logger.log_execution(
                        user_id=task.requester_id, user_name=task.requester,
                        channel_id=str(getattr(task.channel, "id", "")),
                        tool_name=tool_name, tool_input=tool_input, approved=True,
                        result_summary=error_msg[:500],
                        execution_time_ms=elapsed_ms, error=error_msg[:500],
                    )
                except Exception:
                    log.warning("Failed to audit log step %d of task %s", i, task.task_id)
            if on_failure == "abort":
                task.status = "failed"
                break

        # Update progress
        now = time.monotonic()
        if now - last_update >= PROGRESS_UPDATE_INTERVAL:
            progress_msg = await _send_progress(task, progress_msg)
            last_update = now

    # Final status
    if task.status == "running":
        task.status = "completed"

    # Final progress update
    await _send_progress(task, progress_msg)
    await _send_summary(task)

    # Generate conversational follow-up via LLM
    if codex_callback:
        await _send_conversational_followup(task, codex_callback)

    log.info(
        "Background task %s finished: %s (%d/%d steps)",
        task.task_id, task.status, len(task.results), len(task.steps),
    )


def _get_default_host(executor: ToolExecutor) -> str:
    """Get the first configured host alias, falling back to 'localhost'."""
    try:
        hosts = executor.config.hosts
        if hosts and isinstance(hosts, dict):
            return next(iter(hosts))
    except (AttributeError, StopIteration):
        pass
    return "localhost"


def _is_error_output(output: str) -> bool:
    """Detect error strings returned as successful results by executor/handlers."""
    if not output:
        return False
    # executor.execute() returns "Error executing <tool>: <msg>" on exception
    if output.startswith("Error executing "):
        return True
    # executor.execute() returns "Unknown tool: <name>" for missing handlers
    if output.startswith("Unknown tool: "):
        return True
    return False


async def _execute_tool(
    tool_name: str,
    tool_input: dict,
    executor: ToolExecutor,
    skill_manager: SkillManager,
    knowledge_store: KnowledgeStore | None,
    embedder: LocalEmbedder | None,
    requester: str,
    step_desc: str = "",
) -> str:
    """Execute a single tool, routing to the right handler."""
    # Knowledge base tools need special handling (not in executor)
    if tool_name == "ingest_document" and knowledge_store and embedder:
        source = tool_input.get("source", "")
        content = tool_input.get("content", "")
        if not source or not content:
            return "Both 'source' and 'content' are required."
        count = await knowledge_store.ingest(
            content=content, source=source, embedder=embedder,
            uploader=requester,
        )
        return f"Ingested '{source}' ({count} chunks)."

    if tool_name == "search_knowledge" and knowledge_store and embedder:
        query = tool_input.get("query", "")
        limit = min(tool_input.get("limit", 5), 10)
        results = await knowledge_store.search(query, embedder, limit=limit)
        if not results:
            return f"No results for '{query}'."
        lines = [f"[{r['source']}] (score: {r['score']}): {r['content'][:200]}" for r in results]
        return "\n".join(lines)

    if tool_name == "list_knowledge" and knowledge_store:
        sources = knowledge_store.list_sources()
        if not sources:
            return "Knowledge base is empty."
        return "\n".join(f"- {s['source']} ({s['chunks']} chunks)" for s in sources)

    # Skills
    if skill_manager.has_skill(tool_name):
        return await skill_manager.execute(tool_name, tool_input)

    # Built-in tools via executor — default missing required fields
    if "host" not in tool_input:
        tool_input = {**tool_input, "host": _get_default_host(executor)}
    # run_command/run_script: if 'command'/'script' missing, let executor handle it
    # (it will return an error that _is_error_output catches)
    return await executor.execute(tool_name, tool_input)


def _substitute_vars(
    tool_input: dict, variables: dict[str, str], prev_output: str,
) -> dict:
    """Replace {prev_output} and {var.name} in string values."""
    result = {}
    for key, value in tool_input.items():
        if isinstance(value, str):
            value = value.replace("{prev_output}", prev_output)
            for var_name, var_value in variables.items():
                value = value.replace(f"{{var.{var_name}}}", var_value)
            result[key] = value
        else:
            result[key] = value
    return result


def _check_condition(condition: str, prev_output: str) -> bool:
    """Check if a condition is met against previous output."""
    prev_lower = prev_output.lower()
    if condition.startswith("!"):
        # Negated: true if substring is NOT present
        return condition[1:].lower() not in prev_lower
    else:
        # Normal: true if substring IS present
        return condition.lower() in prev_lower


async def _send_progress(
    task: BackgroundTask,
    existing_msg: discord.Message | None,
) -> discord.Message | None:
    """Post or edit a progress message in the channel."""
    total = len(task.steps)
    done = len(task.results)
    ok = sum(1 for r in task.results if r.status == "ok")
    errors = sum(1 for r in task.results if r.status == "error")
    skipped = sum(1 for r in task.results if r.status == "skipped")

    # Status emoji
    if task.status == "completed":
        status_icon = "DONE"
    elif task.status == "failed":
        status_icon = "FAILED"
    elif task.status == "cancelled":
        status_icon = "CANCELLED"
    else:
        status_icon = f"Step {task.current_step + 1}/{total}"

    # Build progress bar
    if total > 0:
        pct = done / total
        filled = int(pct * 20)
        bar = "\u2588" * filled + "\u2591" * (20 - filled)
        progress_line = f"`[{bar}]` {done}/{total}"
    else:
        progress_line = "No steps"

    lines = [
        f"**Background Task: {task.description}** ({status_icon})",
        f"ID: `{task.task_id}` | {progress_line}",
    ]

    if ok or errors or skipped:
        lines.append(f"OK: {ok} | Errors: {errors} | Skipped: {skipped}")

    # When finished, show ALL steps; while running, show last 3
    is_finished = task.status in ("completed", "failed", "cancelled")
    show_results = task.results if is_finished else task.results[-3:]
    if show_results:
        lines.append("")
        for r in show_results:
            icon = {"ok": "+", "error": "!", "skipped": "-", "cancelled": "x"}.get(r.status, "?")
            output_preview = r.output.split("\n")[0][:120]  # first line, truncated
            lines.append(f"`[{icon}]` Step {r.index + 1} ({r.description}): {output_preview}")

    text = "\n".join(lines)

    try:
        if len(text) > 1900 and is_finished:
            # Too long for Discord — post a short summary in the message,
            # attach the full report as a file
            import io
            short = "\n".join(lines[:3])  # header + progress bar + counts
            short += f"\n\nFull report attached ({len(task.results)} steps)."
            file_bytes = text.encode("utf-8")
            discord_file = discord.File(
                io.BytesIO(file_bytes),
                filename=f"task_{task.task_id}_report.txt",
            )
            if existing_msg:
                await existing_msg.edit(content=short)
                await task.channel.send(file=discord_file)
            else:
                await task.channel.send(content=short, file=discord_file)
            return existing_msg
        elif len(text) > 1900:
            text = text[:1900] + "\n..."

        if existing_msg:
            await existing_msg.edit(content=text)
            return existing_msg
        else:
            return await task.channel.send(text)
    except Exception as e:
        log.warning("Failed to update progress message: %s", e)
        return existing_msg


async def _send_summary(task: BackgroundTask) -> None:
    """Post a natural language summary of the completed task."""
    ok = [r for r in task.results if r.status == "ok"]
    errors = [r for r in task.results if r.status == "error"]
    skipped = [r for r in task.results if r.status == "skipped"]

    lines = [f"**Task complete: {task.description}**"]

    if task.status == "completed" and not errors:
        lines.append(f"All {len(ok)} steps succeeded.")
    elif task.status == "completed" and errors:
        lines.append(f"{len(ok)} succeeded, {len(errors)} failed.")
    elif task.status == "failed":
        lines.append(f"Task aborted after {len(task.results)} of {len(task.steps)} steps ({len(errors)} error(s)).")
    elif task.status == "cancelled":
        lines.append(f"Task was cancelled after {len(task.results)} of {len(task.steps)} steps.")

    # Include all results with their output
    if ok or errors:
        lines.append("")
        for r in task.results:
            if r.status == "skipped":
                continue
            icon = "+" if r.status == "ok" else "x"
            # Show meaningful output, not just truncated first line
            output = r.output.strip()
            if len(output) > 200:
                output = output[:200] + "..."
            lines.append(f"**{r.description}**: {output}")

    text = "\n".join(lines)

    try:
        if len(text) > 1900:
            import io
            short_lines = lines[:3]  # header + status line
            short = "\n".join(short_lines) + "\n\nFull summary attached."
            file_bytes = text.encode("utf-8")
            discord_file = discord.File(
                io.BytesIO(file_bytes),
                filename=f"task_{task.task_id}_summary.txt",
            )
            await task.channel.send(content=short, file=discord_file)
        else:
            await task.channel.send(text)
    except Exception as e:
        log.warning("Failed to send task summary: %s", e)


async def _send_conversational_followup(
    task: BackgroundTask,
    codex_callback: CodexCallback,
) -> None:
    """Generate and post an LLM-written conversational summary of the task results."""
    # Build a concise context of what happened
    result_lines = []
    for r in task.results:
        icon = {"ok": "OK", "error": "ERROR", "skipped": "SKIPPED"}.get(r.status, r.status)
        output_preview = r.output.strip()[:150] if r.output else ""
        result_lines.append(f"[{icon}] {r.description}: {output_preview}")

    results_text = "\n".join(result_lines) if result_lines else "No step results."

    messages = [
        {
            "role": "user",
            "content": (
                f"A background task just finished. Summarize the results conversationally.\n\n"
                f"Task: {task.description}\n"
                f"Status: {task.status}\n"
                f"Requested by: {task.requester}\n\n"
                f"Step results:\n{results_text}\n\n"
                f"Write a concise, personality-infused summary (2-4 sentences). "
                f"Highlight any failures. Do NOT repeat every step — focus on the outcome."
            ),
        }
    ]
    system = (
        "You are Heimdall, the All-Seeing — a capable but eternally vigilant infrastructure guardian. "
        "Summarize this background task result conversationally. Be concise and direct."
    )

    try:
        response = await codex_callback(messages, system, 200)
        response = scrub_output_secrets(response.strip())
        if response:
            await task.channel.send(response)
    except Exception as e:
        log.warning("Failed to generate conversational follow-up for task %s: %s", task.task_id, e)


def create_task_id() -> str:
    return uuid.uuid4().hex[:8]
