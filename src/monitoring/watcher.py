"""Proactive infrastructure monitor.

Runs deterministic checks on intervals and alerts when thresholds are crossed.
No LLM in the loop — checks are config-driven and parsed programmatically.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from collections.abc import Awaitable, Callable

from ..config.schema import MonitoringConfig, MonitorCheck
from ..logging import get_logger
from ..tools.executor import ToolExecutor

log = get_logger("monitoring")


@dataclass
class AlertState:
    """Tracks cooldown per (check_name, host) to prevent alert storms."""
    last_alerted: dict[str, float] = field(default_factory=dict)

    def should_alert(self, key: str, cooldown_seconds: int) -> bool:
        last = self.last_alerted.get(key, 0)
        if time.time() - last < cooldown_seconds:
            return False
        return True

    def mark_alerted(self, key: str) -> None:
        self.last_alerted[key] = time.time()


class InfraWatcher:
    """Periodically runs infrastructure checks and alerts on threshold breaches."""

    def __init__(
        self,
        config: MonitoringConfig,
        executor: ToolExecutor,
        alert_callback: Callable[[str], Awaitable[None]],
    ) -> None:
        self.config = config
        self._executor = executor
        self._alert_callback = alert_callback
        self._alert_state = AlertState()
        self._task: asyncio.Task | None = None
        self._check_tasks: dict[str, asyncio.Task] = {}

    # Seconds between each check's startup — spreads SSH connections over time
    STAGGER_INTERVAL = 5

    def start(self) -> None:
        if not self.config.enabled:
            log.info("Monitoring disabled in config")
            return
        if not self.config.checks:
            log.info("No monitoring checks configured")
            return
        for i, check in enumerate(self.config.checks):
            stagger = self.STAGGER_INTERVAL * (i + 1)
            task = asyncio.create_task(self._check_loop(check, stagger))
            self._check_tasks[check.name] = task
        log.info("Started %d monitoring check(s)", len(self._check_tasks))

    async def stop(self) -> None:
        for name, task in self._check_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._check_tasks.clear()
        log.info("Monitoring stopped")

    async def _check_loop(self, check: MonitorCheck, stagger_seconds: int = 5) -> None:
        """Run a single check on its configured interval."""
        await asyncio.sleep(stagger_seconds)
        while True:
            try:
                await self._run_check(check)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Monitor check '%s' failed: %s", check.name, e, exc_info=True)
            await asyncio.sleep(check.interval_minutes * 60)

    async def _run_check(self, check: MonitorCheck) -> None:
        """Execute a single check and alert if threshold is crossed."""
        if check.type == "disk":
            await self._check_disk(check)
        elif check.type == "memory":
            await self._check_memory(check)
        elif check.type == "service":
            await self._check_service(check)
        elif check.type == "promql":
            await self._check_promql(check)
        else:
            log.warning("Unknown check type: %s", check.type)

    async def _check_disk(self, check: MonitorCheck) -> None:
        for host in check.hosts:
            output = await self._executor.execute("run_command", {
                "host": host,
                "command": "df -h --exclude-type=tmpfs --exclude-type=devtmpfs",
            })
            # Parse df -h output for usage percentages
            for line in output.strip().split("\n"):
                match = re.search(r"(\d+)%\s+(/\S*)", line)
                if match:
                    usage = int(match.group(1))
                    mount = match.group(2)
                    if usage >= check.threshold:
                        key = f"{check.name}:{host}:{mount}"
                        if self._alert_state.should_alert(key, self.config.cooldown_minutes * 60):
                            await self._alert_callback(
                                f"**Disk Alert** on `{host}`: `{mount}` is at **{usage}%** "
                                f"(threshold: {check.threshold}%)"
                            )
                            self._alert_state.mark_alerted(key)

    async def _check_memory(self, check: MonitorCheck) -> None:
        for host in check.hosts:
            output = await self._executor.execute("run_command", {
                "host": host,
                "command": "free -h",
            })
            # Parse "free -h" output — look for Mem: line
            for line in output.strip().split("\n"):
                if line.startswith("Mem:"):
                    parts = line.split()
                    if len(parts) >= 3:
                        total = self._parse_mem_value(parts[1])
                        used = self._parse_mem_value(parts[2])
                        if total > 0:
                            pct = int(used / total * 100)
                            if pct >= check.threshold:
                                key = f"{check.name}:{host}"
                                if self._alert_state.should_alert(key, self.config.cooldown_minutes * 60):
                                    await self._alert_callback(
                                        f"**Memory Alert** on `{host}`: **{pct}%** used "
                                        f"(threshold: {check.threshold}%)"
                                    )
                                    self._alert_state.mark_alerted(key)

    async def _check_service(self, check: MonitorCheck) -> None:
        for host in check.hosts:
            for service in check.services:
                output = await self._executor.execute("run_command", {
                    "host": host,
                    "command": f"systemctl status {service} --no-pager -l",
                })
                # Check if the service is not active
                is_down = (
                    "could not be found" in output.lower()
                    or "inactive" in output.lower()
                    or "failed" in output.lower()
                )
                if is_down:
                    key = f"{check.name}:{host}:{service}"
                    if self._alert_state.should_alert(key, self.config.cooldown_minutes * 60):
                        # Extract just the first relevant line
                        status_line = output.strip().split("\n")[0][:200]
                        await self._alert_callback(
                            f"**Service Alert**: `{service}` on `{host}` appears down\n"
                            f"```{status_line}```"
                        )
                        self._alert_state.mark_alerted(key)

    async def _check_promql(self, check: MonitorCheck) -> None:
        # PromQL checks require a prometheus host — use run_command with curl
        # The user configures which host runs the prometheus query via check.hosts
        host = check.hosts[0] if check.hosts else None
        if not host:
            log.warning("PromQL check '%s' has no hosts configured", check.name)
            return
        from urllib.parse import quote as url_quote
        safe_query = url_quote(check.query)
        output = await self._executor.execute("run_command", {
            "host": host,
            "command": f"curl -s 'http://127.0.0.1:9090/api/v1/query?query={safe_query}'",
        })
        # If the query returns results (non-empty result array), alert.
        if output.strip() != "No results." and '"result":[]' not in output and '"result": []' not in output:
            key = f"{check.name}:promql"
            if self._alert_state.should_alert(key, self.config.cooldown_minutes * 60):
                await self._alert_callback(
                    f"**Prometheus Alert** (`{check.name}`): query `{check.query}` returned results\n"
                    f"```{output[:500]}```"
                )
                self._alert_state.mark_alerted(key)

    @staticmethod
    def _parse_mem_value(s: str) -> float:
        """Parse memory values like '62Gi', '3.2G', '512Mi' to bytes."""
        s = s.strip()
        multipliers = {"K": 1e3, "Ki": 1024, "M": 1e6, "Mi": 1024**2,
                        "G": 1e9, "Gi": 1024**3, "T": 1e12, "Ti": 1024**4}
        for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
            if s.endswith(suffix):
                try:
                    return float(s[:-len(suffix)]) * mult
                except ValueError:
                    return 0
        try:
            return float(s)
        except ValueError:
            return 0

    def get_status(self) -> dict:
        """Return current monitoring status for system prompt injection."""
        return {
            "enabled": self.config.enabled,
            "checks": len(self.config.checks),
            "running": len(self._check_tasks),
            "active_alerts": len(self._alert_state.last_alerted),
        }
