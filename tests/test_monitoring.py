"""Tests for monitoring/watcher.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.config.schema import MonitoringConfig, MonitorCheck, ToolsConfig, ToolHost
from src.monitoring.watcher import InfraWatcher, AlertState
from src.tools.executor import ToolExecutor


@pytest.fixture
def executor(tools_config: ToolsConfig) -> ToolExecutor:
    return ToolExecutor(tools_config)


@pytest.fixture
def alert_callback() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def monitoring_config() -> MonitoringConfig:
    return MonitoringConfig(
        enabled=True,
        alert_channel_id="12345",
        cooldown_minutes=60,
        checks=[
            MonitorCheck(
                name="disk_critical",
                type="disk",
                hosts=["server"],
                threshold=90,
                interval_minutes=30,
            ),
            MonitorCheck(
                name="memory_high",
                type="memory",
                hosts=["server"],
                threshold=90,
                interval_minutes=15,
            ),
            MonitorCheck(
                name="services_up",
                type="service",
                hosts=["server"],
                services=["apache2", "prometheus"],
                interval_minutes=5,
            ),
            MonitorCheck(
                name="prom_alerts",
                type="promql",
                hosts=["server"],
                query='ALERTS{alertstate="firing"}',
                interval_minutes=5,
            ),
        ],
    )


class TestAlertState:
    def test_should_alert_first_time(self):
        state = AlertState()
        assert state.should_alert("test", 3600) is True

    def test_should_not_alert_within_cooldown(self):
        state = AlertState()
        state.mark_alerted("test")
        assert state.should_alert("test", 3600) is False

    def test_different_keys_independent(self):
        state = AlertState()
        state.mark_alerted("key1")
        assert state.should_alert("key2", 3600) is True


class TestDiskCheck:
    @pytest.mark.asyncio
    async def test_alerts_on_high_usage(self, executor, monitoring_config, alert_callback):
        watcher = InfraWatcher(monitoring_config, executor, alert_callback)
        check = monitoring_config.checks[0]  # disk_critical

        df_output = (
            "Filesystem      Size  Used Avail Use% Mounted on\n"
            "/dev/nvme0n1p2  916G  870G   46G  95% /\n"
            "/dev/sda1       7.3T  6.5T  800G  89% /mnt/media"
        )

        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, df_output)
            await watcher._run_check(check)

        # Should alert for / at 95% (above 90 threshold) but not /mnt/media at 89%
        alert_callback.assert_called_once()
        msg = alert_callback.call_args[0][0]
        assert "95%" in msg
        assert "/" in msg

    @pytest.mark.asyncio
    async def test_no_alert_below_threshold(self, executor, monitoring_config, alert_callback):
        watcher = InfraWatcher(monitoring_config, executor, alert_callback)
        check = monitoring_config.checks[0]

        df_output = (
            "Filesystem      Size  Used Avail Use% Mounted on\n"
            "/dev/nvme0n1p2  916G  400G  516G  44% /"
        )

        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, df_output)
            await watcher._run_check(check)

        alert_callback.assert_not_called()


class TestMemoryCheck:
    @pytest.mark.asyncio
    async def test_alerts_on_high_memory(self, executor, monitoring_config, alert_callback):
        watcher = InfraWatcher(monitoring_config, executor, alert_callback)
        check = monitoring_config.checks[1]  # memory_high

        free_output = (
            "               total        used        free      shared  buff/cache   available\n"
            "Mem:            62Gi        58Gi       1.2Gi       0.5Gi       2.8Gi       3.0Gi\n"
            "Swap:          8.0Gi       0.0Gi       8.0Gi"
        )

        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, free_output)
            await watcher._run_check(check)

        alert_callback.assert_called_once()
        msg = alert_callback.call_args[0][0]
        assert "Memory Alert" in msg

    @pytest.mark.asyncio
    async def test_no_alert_low_memory(self, executor, monitoring_config, alert_callback):
        watcher = InfraWatcher(monitoring_config, executor, alert_callback)
        check = monitoring_config.checks[1]

        free_output = (
            "               total        used        free\n"
            "Mem:            62Gi        20Gi        42Gi\n"
        )

        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, free_output)
            await watcher._run_check(check)

        alert_callback.assert_not_called()


class TestServiceCheck:
    @pytest.mark.asyncio
    async def test_alerts_on_down_service(self, executor, monitoring_config, alert_callback):
        watcher = InfraWatcher(monitoring_config, executor, alert_callback)
        check = monitoring_config.checks[2]  # services_up

        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            # apache2 is down, prometheus is up
            def side_effect(**kwargs):
                cmd = kwargs.get("command", "")
                if "apache2" in cmd:
                    return (3, "apache2.service - inactive (dead)")
                return (0, "active (running)")
            mock_ssh.side_effect = side_effect
            await watcher._run_check(check)

        alert_callback.assert_called_once()
        msg = alert_callback.call_args[0][0]
        assert "apache2" in msg

    @pytest.mark.asyncio
    async def test_no_alert_all_up(self, executor, monitoring_config, alert_callback):
        watcher = InfraWatcher(monitoring_config, executor, alert_callback)
        check = monitoring_config.checks[2]

        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "active (running)")
            await watcher._run_check(check)

        alert_callback.assert_not_called()


class TestPromqlCheck:
    @pytest.mark.asyncio
    async def test_alerts_on_firing(self, executor, monitoring_config, alert_callback):
        watcher = InfraWatcher(monitoring_config, executor, alert_callback)
        check = monitoring_config.checks[3]  # prom_alerts

        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, '{"status":"success","data":{"resultType":"vector","result":[{"metric":{},"value":[1,"1"]}]}}')
            await watcher._run_check(check)

        alert_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_alert_when_empty(self, executor, monitoring_config, alert_callback):
        watcher = InfraWatcher(monitoring_config, executor, alert_callback)
        check = monitoring_config.checks[3]

        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, '{"status":"success","data":{"resultType":"vector","result":[]}}')
            await watcher._run_check(check)

        alert_callback.assert_not_called()


class TestCooldown:
    @pytest.mark.asyncio
    async def test_respects_cooldown(self, executor, monitoring_config, alert_callback):
        watcher = InfraWatcher(monitoring_config, executor, alert_callback)
        check = monitoring_config.checks[0]  # disk_critical

        df_output = (
            "Filesystem      Size  Used Avail Use% Mounted on\n"
            "/dev/nvme0n1p2  916G  870G   46G  95% /"
        )

        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, df_output)
            # First check — should alert
            await watcher._run_check(check)
            # Second check — should be suppressed by cooldown
            await watcher._run_check(check)

        # Only one alert despite two checks
        assert alert_callback.call_count == 1


class TestMemParsing:
    def test_parse_gi(self):
        assert InfraWatcher._parse_mem_value("62Gi") == 62 * 1024**3

    def test_parse_mi(self):
        assert InfraWatcher._parse_mem_value("512Mi") == 512 * 1024**2

    def test_parse_g(self):
        assert InfraWatcher._parse_mem_value("3.2G") == 3.2 * 1e9

    def test_parse_plain(self):
        assert InfraWatcher._parse_mem_value("1024") == 1024

    def test_parse_invalid(self):
        assert InfraWatcher._parse_mem_value("abc") == 0


class TestGetStatus:
    def test_status(self, executor, monitoring_config, alert_callback):
        watcher = InfraWatcher(monitoring_config, executor, alert_callback)
        status = watcher.get_status()
        assert status["enabled"] is True
        assert status["checks"] == 4
        assert status["running"] == 0  # not started yet
