"""Tests for webhook-triggered scheduled actions.

Tests cover:
- Scheduler: trigger validation, matching, fire_triggers
- HealthServer: trigger callback wiring and invocation from each webhook endpoint
- Registry: schedule_task tool schema includes trigger property
- Client: schedule_task handler passes trigger and formats responses
"""
from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import AioHTTPTestCase, TestClient

from src.scheduler.scheduler import Scheduler
from src.health.server import HealthServer
from src.config.schema import WebhookConfig
from src.tools.registry import TOOLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scheduler(tmp_dir: Path) -> Scheduler:
    return Scheduler(data_path=str(tmp_dir / "schedules.json"))


TRIGGER_TEST_SECRET = "trigger-test-secret"


@pytest.fixture
def webhook_config() -> WebhookConfig:
    return WebhookConfig(
        enabled=True,
        secret=TRIGGER_TEST_SECRET,
        channel_id="12345",
    )


# ---------------------------------------------------------------------------
# Scheduler: trigger validation
# ---------------------------------------------------------------------------

class TestTriggerValidation:
    def test_add_with_trigger(self, scheduler: Scheduler):
        schedule = scheduler.add(
            description="On push, run tests",
            action="check",
            channel_id="ch1",
            tool_name="check_service",
            tool_input={"host": "server", "service": "docker"},
            trigger={"source": "gitea", "event": "push"},
        )
        assert schedule["trigger"] == {"source": "gitea", "event": "push"}
        assert schedule["one_time"] is False
        assert "next_run" not in schedule

    def test_add_trigger_no_cron_no_run_at_ok(self, scheduler: Scheduler):
        """trigger is a valid alternative to cron/run_at."""
        schedule = scheduler.add(
            description="On alert, check hosts",
            action="reminder",
            channel_id="ch1",
            message="Alert fired!",
            trigger={"source": "grafana"},
        )
        assert "trigger" in schedule
        assert "cron" not in schedule
        assert "run_at" not in schedule

    def test_add_no_trigger_no_cron_no_run_at_fails(self, scheduler: Scheduler):
        with pytest.raises(ValueError, match="Either 'cron', 'run_at', or 'trigger'"):
            scheduler.add(
                description="No timing",
                action="reminder",
                channel_id="ch1",
                message="hi",
            )

    def test_trigger_invalid_source(self, scheduler: Scheduler):
        with pytest.raises(ValueError, match="Invalid trigger source"):
            scheduler.add(
                description="Bad source",
                action="reminder",
                channel_id="ch1",
                message="hi",
                trigger={"source": "slack"},
            )

    def test_trigger_unknown_keys(self, scheduler: Scheduler):
        with pytest.raises(ValueError, match="Unknown trigger keys"):
            scheduler.add(
                description="Bad key",
                action="reminder",
                channel_id="ch1",
                message="hi",
                trigger={"source": "gitea", "branch": "main"},
            )

    def test_trigger_empty_dict(self, scheduler: Scheduler):
        with pytest.raises(ValueError, match="at least one condition"):
            scheduler.add(
                description="Empty trigger",
                action="reminder",
                channel_id="ch1",
                message="hi",
                trigger={},
            )

    def test_trigger_not_dict(self, scheduler: Scheduler):
        with pytest.raises(ValueError, match="must be a dict"):
            scheduler.add(
                description="Bad trigger",
                action="reminder",
                channel_id="ch1",
                message="hi",
                trigger="gitea:push",
            )

    def test_trigger_persists(self, tmp_dir: Path):
        path = str(tmp_dir / "schedules.json")
        s1 = Scheduler(data_path=path)
        s1.add(
            description="Persist test",
            action="reminder",
            channel_id="ch1",
            message="hi",
            trigger={"source": "gitea", "event": "push"},
        )
        s2 = Scheduler(data_path=path)
        assert len(s2.list_all()) == 1
        assert s2.list_all()[0]["trigger"] == {"source": "gitea", "event": "push"}

    def test_trigger_valid_sources(self, scheduler: Scheduler):
        """All three valid sources are accepted."""
        for source in ("gitea", "grafana", "generic"):
            schedule = scheduler.add(
                description=f"{source} trigger",
                action="reminder",
                channel_id="ch1",
                message="hi",
                trigger={"source": source},
            )
            assert schedule["trigger"]["source"] == source


# ---------------------------------------------------------------------------
# Scheduler: trigger matching
# ---------------------------------------------------------------------------

class TestTriggerMatching:
    def test_source_match(self):
        assert Scheduler._trigger_matches(
            {"source": "gitea"}, "gitea", {"event": "push"}
        ) is True

    def test_source_mismatch(self):
        assert Scheduler._trigger_matches(
            {"source": "gitea"}, "grafana", {"event": "alert"}
        ) is False

    def test_event_match(self):
        assert Scheduler._trigger_matches(
            {"event": "push"}, "gitea", {"event": "push"}
        ) is True

    def test_event_mismatch(self):
        assert Scheduler._trigger_matches(
            {"event": "push"}, "gitea", {"event": "pull_request"}
        ) is False

    def test_repo_substring_match(self):
        assert Scheduler._trigger_matches(
            {"repo": "myproject"},
            "gitea",
            {"event": "push", "repo": "user/myproject"},
        ) is True

    def test_repo_case_insensitive(self):
        assert Scheduler._trigger_matches(
            {"repo": "MyProject"},
            "gitea",
            {"event": "push", "repo": "user/myproject"},
        ) is True

    def test_repo_mismatch(self):
        assert Scheduler._trigger_matches(
            {"repo": "myproject"},
            "gitea",
            {"event": "push", "repo": "calmingstorm/other-project"},
        ) is False

    def test_alert_name_match(self):
        assert Scheduler._trigger_matches(
            {"alert_name": "HighCPU"},
            "grafana",
            {"event": "alert", "alert_name": "HighCPU on server"},
        ) is True

    def test_alert_name_case_insensitive(self):
        assert Scheduler._trigger_matches(
            {"alert_name": "highcpu"},
            "grafana",
            {"event": "alert", "alert_name": "HighCPU"},
        ) is True

    def test_alert_name_mismatch(self):
        assert Scheduler._trigger_matches(
            {"alert_name": "DiskFull"},
            "grafana",
            {"event": "alert", "alert_name": "HighCPU"},
        ) is False

    def test_multiple_conditions_all_match(self):
        assert Scheduler._trigger_matches(
            {"source": "gitea", "event": "push", "repo": "myproject"},
            "gitea",
            {"event": "push", "repo": "user/myproject"},
        ) is True

    def test_multiple_conditions_one_fails(self):
        assert Scheduler._trigger_matches(
            {"source": "gitea", "event": "push", "repo": "myproject"},
            "gitea",
            {"event": "pull_request", "repo": "user/myproject"},
        ) is False

    def test_empty_event_data_fields(self):
        """Missing event_data fields don't match non-empty trigger conditions."""
        assert Scheduler._trigger_matches(
            {"repo": "myproject"}, "gitea", {"event": "push"}
        ) is False

    def test_no_conditions_matches_everything(self):
        """Trigger with no source/event/repo/alert matches any event from any source."""
        # This shouldn't happen in practice (empty triggers rejected by validation)
        # but _trigger_matches alone doesn't enforce this.
        assert Scheduler._trigger_matches(
            {}, "gitea", {"event": "push", "repo": "foo"}
        ) is True


# ---------------------------------------------------------------------------
# Scheduler: fire_triggers
# ---------------------------------------------------------------------------

class TestFireTriggers:
    @pytest.mark.asyncio
    async def test_fires_matching_trigger(self, scheduler: Scheduler):
        callback = AsyncMock()
        scheduler.start(callback)

        scheduler.add(
            description="On myproject push",
            action="reminder",
            channel_id="ch1",
            message="Push detected!",
            trigger={"source": "gitea", "event": "push", "repo": "myproject"},
        )

        fired = await scheduler.fire_triggers(
            "gitea", {"event": "push", "repo": "user/myproject"}
        )
        assert fired == 1
        callback.assert_called_once()
        # The schedule dict was passed to the callback
        schedule_arg = callback.call_args[0][0]
        assert schedule_arg["description"] == "On myproject push"
        assert schedule_arg["last_run"] is not None

    @pytest.mark.asyncio
    async def test_does_not_fire_non_matching(self, scheduler: Scheduler):
        callback = AsyncMock()
        scheduler.start(callback)

        scheduler.add(
            description="Only grafana",
            action="reminder",
            channel_id="ch1",
            message="alert!",
            trigger={"source": "grafana"},
        )

        fired = await scheduler.fire_triggers(
            "gitea", {"event": "push", "repo": "foo"}
        )
        assert fired == 0
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_fires_multiple_matching_triggers(self, scheduler: Scheduler):
        callback = AsyncMock()
        scheduler.start(callback)

        scheduler.add(
            description="Trigger A",
            action="reminder",
            channel_id="ch1",
            message="A",
            trigger={"source": "gitea"},
        )
        scheduler.add(
            description="Trigger B",
            action="reminder",
            channel_id="ch1",
            message="B",
            trigger={"source": "gitea", "event": "push"},
        )
        scheduler.add(
            description="Trigger C (no match)",
            action="reminder",
            channel_id="ch1",
            message="C",
            trigger={"source": "grafana"},
        )

        fired = await scheduler.fire_triggers(
            "gitea", {"event": "push", "repo": "foo"}
        )
        assert fired == 2
        assert callback.call_count == 2

    @pytest.mark.asyncio
    async def test_skips_time_based_schedules(self, scheduler: Scheduler):
        callback = AsyncMock()
        scheduler.start(callback)

        # Add a cron schedule — should be ignored by fire_triggers
        scheduler.add(
            description="Cron task",
            action="reminder",
            channel_id="ch1",
            message="cron",
            cron="0 9 * * *",
        )

        fired = await scheduler.fire_triggers("gitea", {"event": "push"})
        assert fired == 0
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_callback_returns_zero(self, scheduler: Scheduler):
        # Don't call start() — no callback set
        scheduler.add(
            description="No callback",
            action="reminder",
            channel_id="ch1",
            message="hi",
            trigger={"source": "gitea"},
        )
        fired = await scheduler.fire_triggers("gitea", {"event": "push"})
        assert fired == 0

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_stop_others(self, scheduler: Scheduler):
        callback = AsyncMock(side_effect=[Exception("boom"), None])
        scheduler.start(callback)

        scheduler.add(
            description="Trigger 1",
            action="reminder",
            channel_id="ch1",
            message="1",
            trigger={"source": "gitea"},
        )
        scheduler.add(
            description="Trigger 2",
            action="reminder",
            channel_id="ch1",
            message="2",
            trigger={"source": "gitea"},
        )

        fired = await scheduler.fire_triggers("gitea", {"event": "push"})
        assert fired == 2
        assert callback.call_count == 2

    @pytest.mark.asyncio
    async def test_fire_triggers_saves(self, scheduler: Scheduler):
        callback = AsyncMock()
        scheduler.start(callback)

        scheduler.add(
            description="Save test",
            action="reminder",
            channel_id="ch1",
            message="hi",
            trigger={"source": "gitea"},
        )

        with patch.object(scheduler, "_save") as mock_save:
            await scheduler.fire_triggers("gitea", {"event": "push"})
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_not_removed_after_fire(self, scheduler: Scheduler):
        """Trigger schedules are persistent — they stay after firing."""
        callback = AsyncMock()
        scheduler.start(callback)

        scheduler.add(
            description="Persistent trigger",
            action="reminder",
            channel_id="ch1",
            message="hi",
            trigger={"source": "gitea"},
        )

        await scheduler.fire_triggers("gitea", {"event": "push"})
        assert len(scheduler.list_all()) == 1  # Still there


# ---------------------------------------------------------------------------
# HealthServer: trigger callback
# ---------------------------------------------------------------------------

class TestHealthServerTriggers:
    @pytest.fixture
    def health(self, webhook_config: WebhookConfig) -> HealthServer:
        return HealthServer(port=0, webhook_config=webhook_config)

    def test_set_trigger_callback(self, health: HealthServer):
        cb = AsyncMock()
        health.set_trigger_callback(cb)
        assert health._trigger_callback is cb

    @pytest.mark.asyncio
    async def test_notify_triggers_calls_callback(self, health: HealthServer):
        cb = AsyncMock(return_value=2)
        health.set_trigger_callback(cb)
        await health._notify_triggers("gitea", {"event": "push"})
        cb.assert_called_once_with("gitea", {"event": "push"})

    @pytest.mark.asyncio
    async def test_notify_triggers_no_callback(self, health: HealthServer):
        # No callback set — should not raise
        await health._notify_triggers("gitea", {"event": "push"})

    @pytest.mark.asyncio
    async def test_notify_triggers_callback_exception(self, health: HealthServer):
        cb = AsyncMock(side_effect=Exception("boom"))
        health.set_trigger_callback(cb)
        # Should not raise — logs the error
        await health._notify_triggers("gitea", {"event": "push"})

    @pytest.mark.asyncio
    async def test_gitea_webhook_calls_triggers(self, health: HealthServer):
        trigger_cb = AsyncMock(return_value=0)
        health.set_trigger_callback(trigger_cb)
        send_cb = AsyncMock()
        health.set_send_message(send_cb)

        from aiohttp.test_utils import make_mocked_request
        payload = json.dumps({
            "repository": {"full_name": "user/myproject"},
            "pusher": {"login": "aaron"},
            "commits": [{"id": "abc1234", "message": "test commit"}],
            "ref": "refs/heads/main",
        }).encode()
        signature = hmac_mod.new(
            TRIGGER_TEST_SECRET.encode(), payload, hashlib.sha256
        ).hexdigest()

        request = make_mocked_request(
            "POST", "/webhook/gitea",
            headers={
                "X-Gitea-Event": "push",
                "X-Gitea-Signature": signature,
            },
            payload=payload,
        )
        # Mock request.read() to return our payload
        request.read = AsyncMock(return_value=payload)

        await health._webhook_gitea(request)

        trigger_cb.assert_called_once_with(
            "gitea",
            {"event": "push", "repo": "user/myproject"},
        )

    @pytest.mark.asyncio
    async def test_grafana_webhook_calls_triggers(self, health: HealthServer):
        trigger_cb = AsyncMock(return_value=0)
        health.set_trigger_callback(trigger_cb)
        send_cb = AsyncMock()
        health.set_send_message(send_cb)

        payload = json.dumps({
            "alerts": [{
                "status": "firing",
                "labels": {"alertname": "HighCPU", "instance": "server:9090"},
                "annotations": {"summary": "CPU is high"},
            }],
        }).encode()

        from aiohttp.test_utils import make_mocked_request
        request = make_mocked_request(
            "POST", "/webhook/grafana",
            headers={"X-Webhook-Secret": TRIGGER_TEST_SECRET},
            payload=payload,
        )
        request.read = AsyncMock(return_value=payload)

        await health._webhook_grafana(request)

        trigger_cb.assert_called_once_with(
            "grafana",
            {"event": "alert", "alert_name": "HighCPU"},
        )

    @pytest.mark.asyncio
    async def test_grafana_no_alerts_uses_rulename(self, health: HealthServer):
        trigger_cb = AsyncMock(return_value=0)
        health.set_trigger_callback(trigger_cb)
        send_cb = AsyncMock()
        health.set_send_message(send_cb)

        payload = json.dumps({
            "ruleName": "DiskSpaceAlert",
            "message": "disk is full",
        }).encode()

        from aiohttp.test_utils import make_mocked_request
        request = make_mocked_request(
            "POST", "/webhook/grafana",
            headers={"X-Webhook-Secret": TRIGGER_TEST_SECRET},
            payload=payload,
        )
        request.read = AsyncMock(return_value=payload)

        await health._webhook_grafana(request)

        trigger_cb.assert_called_once_with(
            "grafana",
            {"event": "alert", "alert_name": "DiskSpaceAlert"},
        )

    @pytest.mark.asyncio
    async def test_generic_webhook_calls_triggers(self, health: HealthServer):
        trigger_cb = AsyncMock(return_value=0)
        health.set_trigger_callback(trigger_cb)
        send_cb = AsyncMock()
        health.set_send_message(send_cb)

        payload = json.dumps({
            "title": "Deploy Complete",
            "message": "v2.1 deployed",
            "event": "deploy",
        }).encode()

        from aiohttp.test_utils import make_mocked_request
        request = make_mocked_request(
            "POST", "/webhook/generic",
            headers={"X-Webhook-Secret": TRIGGER_TEST_SECRET},
            payload=payload,
        )
        request.read = AsyncMock(return_value=payload)

        await health._webhook_generic(request)

        trigger_cb.assert_called_once_with(
            "generic",
            {"event": "deploy", "title": "Deploy Complete"},
        )


# ---------------------------------------------------------------------------
# Registry: schedule_task tool schema
# ---------------------------------------------------------------------------

class TestRegistryTriggerSchema:
    def _get_schedule_tool(self):
        return next(t for t in TOOLS if t["name"] == "schedule_task")

    def test_trigger_property_exists(self):
        tool = self._get_schedule_tool()
        props = tool["input_schema"]["properties"]
        assert "trigger" in props

    def test_trigger_is_object_type(self):
        tool = self._get_schedule_tool()
        trigger_schema = tool["input_schema"]["properties"]["trigger"]
        assert trigger_schema["type"] == "object"

    def test_trigger_has_source_property(self):
        tool = self._get_schedule_tool()
        trigger_props = tool["input_schema"]["properties"]["trigger"]["properties"]
        assert "source" in trigger_props
        assert trigger_props["source"]["enum"] == ["gitea", "grafana", "generic"]

    def test_trigger_has_event_property(self):
        tool = self._get_schedule_tool()
        trigger_props = tool["input_schema"]["properties"]["trigger"]["properties"]
        assert "event" in trigger_props

    def test_trigger_has_repo_property(self):
        tool = self._get_schedule_tool()
        trigger_props = tool["input_schema"]["properties"]["trigger"]["properties"]
        assert "repo" in trigger_props

    def test_trigger_has_alert_name_property(self):
        tool = self._get_schedule_tool()
        trigger_props = tool["input_schema"]["properties"]["trigger"]["properties"]
        assert "alert_name" in trigger_props

    def test_description_mentions_webhook(self):
        tool = self._get_schedule_tool()
        assert "webhook" in tool["description"].lower()


# ---------------------------------------------------------------------------
# Integration: end-to-end trigger flow
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_gitea_push_fires_scheduled_check(self, tmp_dir: Path):
        """Full flow: register trigger → receive webhook → callback fires."""
        scheduler = Scheduler(data_path=str(tmp_dir / "schedules.json"))
        callback = AsyncMock()
        scheduler.start(callback)

        scheduler.add(
            description="Run tests on push to myproject",
            action="check",
            channel_id="ch1",
            tool_name="check_service",
            tool_input={"host": "server", "service": "docker"},
            trigger={"source": "gitea", "event": "push", "repo": "myproject"},
        )

        # Simulate what HealthServer does: call fire_triggers
        fired = await scheduler.fire_triggers(
            "gitea",
            {"event": "push", "repo": "user/myproject"},
        )

        assert fired == 1
        schedule_arg = callback.call_args[0][0]
        assert schedule_arg["action"] == "check"
        assert schedule_arg["tool_name"] == "check_service"

    @pytest.mark.asyncio
    async def test_grafana_alert_fires_workflow(self, tmp_dir: Path):
        """Grafana alert triggers a multi-step workflow."""
        scheduler = Scheduler(data_path=str(tmp_dir / "schedules.json"))
        callback = AsyncMock()
        scheduler.start(callback)

        scheduler.add(
            description="On CPU alert, check all hosts",
            action="workflow",
            channel_id="ch1",
            steps=[
                {"tool_name": "check_memory", "tool_input": {"host": "server"}},
                {"tool_name": "check_disk", "tool_input": {"host": "server"}},
            ],
            trigger={"source": "grafana", "alert_name": "HighCPU"},
        )

        fired = await scheduler.fire_triggers(
            "grafana",
            {"event": "alert", "alert_name": "HighCPU on server:9090"},
        )

        assert fired == 1
        schedule_arg = callback.call_args[0][0]
        assert schedule_arg["action"] == "workflow"
        assert len(schedule_arg["steps"]) == 2

    @pytest.mark.asyncio
    async def test_unrelated_webhook_does_not_fire(self, tmp_dir: Path):
        """Webhook that doesn't match any trigger fires nothing."""
        scheduler = Scheduler(data_path=str(tmp_dir / "schedules.json"))
        callback = AsyncMock()
        scheduler.start(callback)

        scheduler.add(
            description="Only myproject pushes",
            action="reminder",
            channel_id="ch1",
            message="hi",
            trigger={"source": "gitea", "event": "push", "repo": "myproject"},
        )

        fired = await scheduler.fire_triggers(
            "gitea",
            {"event": "push", "repo": "calmingstorm/other-project"},
        )
        assert fired == 0
        callback.assert_not_called()
