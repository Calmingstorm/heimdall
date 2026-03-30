from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any

from croniter import croniter

from ..logging import get_logger

log = get_logger("scheduler")

# Tools that can be scheduled for "check" actions
ALLOWED_CHECK_TOOLS = {
    "run_command", "run_command_multi", "run_script",
}


class Scheduler:
    """Manages scheduled tasks — recurring (cron), one-time, and webhook-triggered."""

    def __init__(self, data_path: str) -> None:
        self.data_path = Path(data_path)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self._schedules: list[dict] = []
        self._task: asyncio.Task | None = None
        self._callback: Callable[[dict], Awaitable[None]] | None = None
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        if self.data_path.exists():
            try:
                self._schedules = json.loads(self.data_path.read_text())
                log.info("Loaded %d schedule(s)", len(self._schedules))
            except Exception as e:
                log.error("Failed to load schedules: %s", e)
                self._schedules = []

    def _save(self) -> None:
        self.data_path.write_text(json.dumps(self._schedules, indent=2))

    def add(
        self,
        description: str,
        action: str,
        channel_id: str,
        cron: str | None = None,
        run_at: str | None = None,
        message: str | None = None,
        tool_name: str | None = None,
        tool_input: dict | None = None,
        steps: list[dict] | None = None,
        trigger: dict | None = None,
    ) -> dict:
        if action == "digest":
            # Digest is a predefined action, no tool validation needed
            pass
        elif action == "check":
            if not tool_name:
                raise ValueError("tool_name is required for 'check' actions")
            if tool_name not in ALLOWED_CHECK_TOOLS:
                raise ValueError(
                    f"Tool '{tool_name}' is not allowed for scheduled checks. "
                    f"Allowed: {', '.join(sorted(ALLOWED_CHECK_TOOLS))}"
                )
        elif action == "workflow":
            if not steps or not isinstance(steps, list):
                raise ValueError("'steps' (list) is required for 'workflow' actions")
            for i, step in enumerate(steps):
                if not isinstance(step, dict) or "tool_name" not in step:
                    raise ValueError(f"Step {i}: must be a dict with 'tool_name'")

        if trigger is not None:
            self._validate_trigger(trigger)
        elif not cron and not run_at:
            raise ValueError("Either 'cron', 'run_at', or 'trigger' is required")

        schedule: dict[str, Any] = {
            "id": uuid.uuid4().hex[:8],
            "description": description,
            "action": action,
            "channel_id": channel_id,
            "created_at": datetime.now().isoformat(),
            "last_run": None,
        }

        if trigger is not None:
            schedule["trigger"] = trigger
            schedule["one_time"] = False
        elif cron:
            # Validate cron expression
            if not croniter.is_valid(cron):
                raise ValueError(f"Invalid cron expression: {cron}")
            schedule["cron"] = cron
            schedule["one_time"] = False
            cr = croniter(cron, datetime.now())
            schedule["next_run"] = cr.get_next(datetime).isoformat()
        else:
            if run_at:
                try:
                    datetime.fromisoformat(run_at)
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid ISO datetime for run_at: {run_at!r}")
            schedule["run_at"] = run_at
            schedule["next_run"] = run_at
            schedule["one_time"] = True

        if action == "reminder":
            schedule["message"] = message or description
        elif action == "check":
            schedule["tool_name"] = tool_name
            schedule["tool_input"] = tool_input or {}
        elif action == "workflow":
            schedule["steps"] = steps

        self._schedules.append(schedule)
        self._save()
        log_next = schedule.get("next_run", "on trigger")
        log.info("Added schedule %s: %s (next: %s)", schedule["id"], description, log_next)
        return schedule

    @staticmethod
    def _validate_trigger(trigger: dict) -> None:
        """Validate a webhook trigger definition."""
        if not isinstance(trigger, dict):
            raise ValueError("'trigger' must be a dict")
        valid_keys = {"source", "event", "repo", "alert_name"}
        unknown = set(trigger.keys()) - valid_keys
        if unknown:
            raise ValueError(f"Unknown trigger keys: {', '.join(sorted(unknown))}")
        valid_sources = {"gitea", "grafana", "generic"}
        source = trigger.get("source")
        if source and source not in valid_sources:
            raise ValueError(
                f"Invalid trigger source '{source}'. "
                f"Valid: {', '.join(sorted(valid_sources))}"
            )
        if not trigger:
            raise ValueError("Trigger must have at least one condition")

    @staticmethod
    def _trigger_matches(trigger: dict, source: str, event_data: dict) -> bool:
        """Check if webhook event data matches a trigger definition.

        Matching rules:
        - source: exact match (required if specified)
        - event: exact match against event_data["event"]
        - repo: case-insensitive substring match against event_data["repo"]
        - alert_name: case-insensitive substring match against event_data["alert_name"]

        All specified fields must match (AND logic).
        """
        if trigger.get("source") and trigger["source"] != source:
            return False
        if trigger.get("event"):
            if trigger["event"] != event_data.get("event"):
                return False
        if trigger.get("repo"):
            repo = event_data.get("repo", "")
            if trigger["repo"].lower() not in repo.lower():
                return False
        if trigger.get("alert_name"):
            alert = event_data.get("alert_name", "")
            if trigger["alert_name"].lower() not in alert.lower():
                return False
        return True

    async def fire_triggers(self, source: str, event_data: dict) -> int:
        """Check all trigger-based schedules against an incoming webhook event.

        Returns the number of triggers that fired.
        Holds _lock to prevent concurrent mutation with _tick().
        """
        if not self._callback:
            return 0

        async with self._lock:
            fired = 0
            now = datetime.now()
            for schedule in self._schedules:
                trigger = schedule.get("trigger")
                if not trigger:
                    continue
                if not self._trigger_matches(trigger, source, event_data):
                    continue

                log.info(
                    "Webhook trigger fired: schedule %s (%s) on %s event",
                    schedule["id"], schedule["description"], source,
                )
                schedule["last_run"] = now.isoformat()
                fired += 1

                try:
                    await self._callback(schedule)
                except Exception as e:
                    log.error("Trigger schedule %s callback failed: %s", schedule["id"], e)

            if fired:
                await asyncio.to_thread(self._save)
            return fired

    def list_all(self) -> list[dict]:
        return list(self._schedules)

    def delete(self, schedule_id: str) -> bool:
        before = len(self._schedules)
        self._schedules = [s for s in self._schedules if s["id"] != schedule_id]
        if len(self._schedules) < before:
            self._save()
            log.info("Deleted schedule %s", schedule_id)
            return True
        return False

    def start(self, callback: Callable[[dict], Awaitable[None]]) -> None:
        self._callback = callback
        self._task = asyncio.create_task(self._loop())
        log.info("Scheduler started with %d schedule(s)", len(self._schedules))

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            log.info("Scheduler stopped")

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(60)
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Scheduler tick error: %s", e, exc_info=True)

    async def _tick(self) -> None:
        async with self._lock:
            now = datetime.now()
            fired = False
            to_remove: list[str] = []

            for schedule in self._schedules:
                next_run_str = schedule.get("next_run")
                if not next_run_str:
                    continue

                next_run = datetime.fromisoformat(next_run_str)
                # Strip timezone info so comparison with naive now() always works
                if next_run.tzinfo is not None:
                    next_run = next_run.replace(tzinfo=None)
                if now < next_run:
                    continue

                log.info("Firing schedule %s: %s", schedule["id"], schedule["description"])
                schedule["last_run"] = now.isoformat()
                fired = True

                if self._callback:
                    try:
                        await self._callback(schedule)
                    except Exception as e:
                        log.error("Schedule %s callback failed: %s", schedule["id"], e)

                if schedule.get("one_time"):
                    to_remove.append(schedule["id"])
                elif schedule.get("cron"):
                    cr = croniter(schedule["cron"], now)
                    schedule["next_run"] = cr.get_next(datetime).isoformat()

            for sid in to_remove:
                self._schedules = [s for s in self._schedules if s["id"] != sid]

            if fired or to_remove:
                await asyncio.to_thread(self._save)
