"""Tests targeting coverage gaps in src/health/server.py.

Covers:
- set_ready, set_send_message, set_trigger_callback
- _health endpoint (ready and not ready states)
- _verify_hmac_sha256 (with and without secret)
- _get_channel_id (gitea, grafana, default)
- _send (no channel, no callback, delivery exception, success)
- _webhook_gitea: push, PR, issue, unknown event, invalid signature, invalid JSON
- _webhook_grafana: with alerts, without alerts, invalid JSON
- _webhook_generic: valid, invalid secret, invalid JSON
- start / stop lifecycle
"""
from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient

from src.config.schema import WebhookConfig
from src.health.server import HealthServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(body: bytes | dict, headers: dict | None = None):
    """Create a mock aiohttp request."""
    if isinstance(body, dict):
        body = json.dumps(body).encode()
    req = MagicMock()
    req.read = AsyncMock(return_value=body)
    req.headers = headers or {}
    return req


# ---------------------------------------------------------------------------
# Unit tests (no HTTP server needed)
# ---------------------------------------------------------------------------

class TestSetters:
    """Tests for setter methods."""

    def test_set_ready(self):
        """set_ready should update the _ready flag."""
        server = HealthServer(webhook_config=WebhookConfig())
        assert server._ready is False
        server.set_ready(True)
        assert server._ready is True
        server.set_ready(False)
        assert server._ready is False

    def test_set_send_message(self):
        """set_send_message should store the callback."""
        server = HealthServer(webhook_config=WebhookConfig())
        assert server._send_message is None
        cb = AsyncMock()
        server.set_send_message(cb)
        assert server._send_message is cb

    def test_set_trigger_callback(self):
        """set_trigger_callback should store the callback."""
        server = HealthServer(webhook_config=WebhookConfig())
        assert server._trigger_callback is None
        cb = AsyncMock()
        server.set_trigger_callback(cb)
        assert server._trigger_callback is cb


class TestHealth:
    """Tests for the _health endpoint handler."""

    async def test_health_ready(self):
        """Should return 200 with status ok when ready."""
        server = HealthServer(webhook_config=WebhookConfig())
        server.set_ready(True)
        response = await server._health(MagicMock())
        assert response.status == 200
        body = json.loads(response.body)
        assert body["status"] == "ok"

    async def test_health_not_ready(self):
        """Should return 503 with status starting when not ready."""
        server = HealthServer(webhook_config=WebhookConfig())
        response = await server._health(MagicMock())
        assert response.status == 503
        body = json.loads(response.body)
        assert body["status"] == "starting"


class TestVerifyHmac:
    """Tests for _verify_hmac_sha256."""

    def test_no_secret_configured_rejects(self):
        """Should return False (reject) when no secret is configured."""
        server = HealthServer(webhook_config=WebhookConfig(secret=""))
        assert server._verify_hmac_sha256(b"body", "any_sig") is False

    def test_valid_signature(self):
        """Should return True for a valid HMAC-SHA256 signature."""
        secret = "my-webhook-secret"
        body = b'{"event":"push"}'
        expected = hmac_mod.new(secret.encode(), body, hashlib.sha256).hexdigest()

        server = HealthServer(webhook_config=WebhookConfig(secret=secret))
        assert server._verify_hmac_sha256(body, expected) is True

    def test_invalid_signature(self):
        """Should return False for an invalid HMAC-SHA256 signature."""
        server = HealthServer(webhook_config=WebhookConfig(secret="my-secret"))
        assert server._verify_hmac_sha256(b"body", "wrong_signature") is False


class TestGetChannelId:
    """Tests for _get_channel_id."""

    def test_gitea_specific_channel(self):
        """Should return gitea-specific channel when configured."""
        server = HealthServer(webhook_config=WebhookConfig(
            gitea_channel_id="111", channel_id="999",
        ))
        assert server._get_channel_id("gitea") == "111"

    def test_grafana_specific_channel(self):
        """Should return grafana-specific channel when configured."""
        server = HealthServer(webhook_config=WebhookConfig(
            grafana_channel_id="222", channel_id="999",
        ))
        assert server._get_channel_id("grafana") == "222"

    def test_default_channel(self):
        """Should return default channel_id for unknown sources."""
        server = HealthServer(webhook_config=WebhookConfig(channel_id="999"))
        assert server._get_channel_id("generic") == "999"

    def test_no_channel_configured(self):
        """Should return None when no channel is configured."""
        server = HealthServer(webhook_config=WebhookConfig())
        assert server._get_channel_id("generic") is None


class TestSend:
    """Tests for the _send helper."""

    async def test_send_no_channel(self):
        """Should return 500 when no channel is configured."""
        server = HealthServer(webhook_config=WebhookConfig())
        response = await server._send("generic", "test message")
        assert response.status == 500
        body = json.loads(response.body)
        assert "no channel" in body["error"]

    async def test_send_no_callback(self):
        """Should return 503 when no send_message callback is set."""
        server = HealthServer(webhook_config=WebhookConfig(channel_id="123"))
        # No callback set
        response = await server._send("generic", "test message")
        assert response.status == 503
        body = json.loads(response.body)
        assert "not ready" in body["error"]

    async def test_send_success(self):
        """Should return 200 when message is delivered successfully."""
        server = HealthServer(webhook_config=WebhookConfig(channel_id="123"))
        server.set_send_message(AsyncMock())
        response = await server._send("generic", "test message")
        assert response.status == 200
        body = json.loads(response.body)
        assert body["status"] == "delivered"

    async def test_send_delivery_failure(self):
        """Should return 500 when send_message callback raises."""
        server = HealthServer(webhook_config=WebhookConfig(channel_id="123"))
        server.set_send_message(AsyncMock(side_effect=Exception("Discord down")))
        response = await server._send("generic", "test message")
        assert response.status == 500
        body = json.loads(response.body)
        assert "Discord down" in body["error"]


def _gitea_headers(body: bytes, secret: str, event: str = "push") -> dict:
    """Build Gitea webhook headers with valid HMAC signature."""
    sig = hmac_mod.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return {"X-Gitea-Event": event, "X-Gitea-Signature": sig}


GITEA_SECRET = "gitea-test-secret"


class TestWebhookGitea:
    """Tests for _webhook_gitea handler."""

    async def test_push_event(self):
        """Should format push event with commits."""
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret=GITEA_SECRET,
        ))
        server.set_send_message(AsyncMock())

        body = {
            "repository": {"full_name": "user/repo"},
            "pusher": {"login": "testuser"},
            "ref": "refs/heads/main",
            "commits": [
                {"id": "abc1234567890", "message": "Fix bug"},
                {"id": "def5678901234", "message": "Add feature"},
            ],
        }
        body_bytes = json.dumps(body).encode()
        req = _make_request(body_bytes, headers=_gitea_headers(body_bytes, GITEA_SECRET, "push"))
        response = await server._webhook_gitea(req)
        assert response.status == 200

    async def test_pull_request_event(self):
        """Should format pull request event."""
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret=GITEA_SECRET,
        ))
        server.set_send_message(AsyncMock())

        body = {
            "repository": {"full_name": "user/repo"},
            "action": "opened",
            "pull_request": {
                "title": "Add new feature",
                "user": {"login": "testuser"},
            },
        }
        body_bytes = json.dumps(body).encode()
        req = _make_request(body_bytes, headers=_gitea_headers(body_bytes, GITEA_SECRET, "pull_request"))
        response = await server._webhook_gitea(req)
        assert response.status == 200

    async def test_issue_event(self):
        """Should format issue event."""
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret=GITEA_SECRET,
        ))
        server.set_send_message(AsyncMock())

        body = {
            "repository": {"full_name": "user/repo"},
            "action": "opened",
            "issue": {"title": "Bug report"},
            "sender": {"login": "testuser"},
        }
        body_bytes = json.dumps(body).encode()
        req = _make_request(body_bytes, headers=_gitea_headers(body_bytes, GITEA_SECRET, "issues"))
        response = await server._webhook_gitea(req)
        assert response.status == 200

    async def test_unknown_event(self):
        """Should handle unknown events with generic message."""
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret=GITEA_SECRET,
        ))
        server.set_send_message(AsyncMock())

        body = {"repository": {"full_name": "user/repo"}}
        body_bytes = json.dumps(body).encode()
        req = _make_request(body_bytes, headers=_gitea_headers(body_bytes, GITEA_SECRET, "release"))
        response = await server._webhook_gitea(req)
        assert response.status == 200

    async def test_invalid_signature(self):
        """Should return 403 for invalid HMAC signature."""
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret="my-secret",
        ))
        req = _make_request(b'{"test": true}', headers={
            "X-Gitea-Event": "push",
            "X-Gitea-Signature": "wrong_signature",
        })
        response = await server._webhook_gitea(req)
        assert response.status == 403

    async def test_no_secret_rejects(self):
        """Should return 403 when no secret is configured (HMAC rejects)."""
        server = HealthServer(webhook_config=WebhookConfig(channel_id="123"))
        body_bytes = b'{"repository": {"full_name": "user/repo"}}'
        req = _make_request(body_bytes, headers={
            "X-Gitea-Event": "push",
            "X-Gitea-Signature": "",
        })
        response = await server._webhook_gitea(req)
        assert response.status == 403

    async def test_invalid_json(self):
        """Should return 400 for invalid JSON body."""
        secret = "json-test-secret"
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret=secret,
        ))
        body_bytes = b"not json"
        req = _make_request(body_bytes, headers=_gitea_headers(body_bytes, secret, "push"))
        response = await server._webhook_gitea(req)
        assert response.status == 400


GRAFANA_SECRET = "grafana-test-secret"


class TestWebhookGrafana:
    """Tests for _webhook_grafana handler."""

    async def test_grafana_with_alerts(self):
        """Should format alerts with status, labels, and annotations."""
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret=GRAFANA_SECRET,
        ))
        server.set_send_message(AsyncMock())

        body = {
            "alerts": [
                {
                    "status": "firing",
                    "labels": {"alertname": "HighCPU", "instance": "server:9090"},
                    "annotations": {"summary": "CPU above 90% for 5 minutes"},
                },
                {
                    "status": "resolved",
                    "labels": {"alertname": "DiskSpace"},
                    "annotations": {},
                },
            ],
        }
        req = _make_request(body, headers={"X-Webhook-Secret": GRAFANA_SECRET})
        response = await server._webhook_grafana(req)
        assert response.status == 200

    async def test_grafana_no_alerts(self):
        """Should handle legacy format without alerts array."""
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret=GRAFANA_SECRET,
        ))
        server.set_send_message(AsyncMock())

        body = {
            "title": "High Memory Usage",
            "message": "Memory exceeded threshold",
        }
        req = _make_request(body, headers={"X-Webhook-Secret": GRAFANA_SECRET})
        response = await server._webhook_grafana(req)
        assert response.status == 200

    async def test_grafana_invalid_json(self):
        """Should return 400 for invalid JSON."""
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret=GRAFANA_SECRET,
        ))
        req = _make_request(b"not json", headers={"X-Webhook-Secret": GRAFANA_SECRET})
        response = await server._webhook_grafana(req)
        assert response.status == 400

    async def test_grafana_invalid_secret(self):
        """Should return 403 when secret doesn't match."""
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret=GRAFANA_SECRET,
        ))
        body = {"alerts": []}
        req = _make_request(body, headers={"X-Webhook-Secret": "wrong-secret"})
        response = await server._webhook_grafana(req)
        assert response.status == 403

    async def test_grafana_missing_secret_header(self):
        """Should return 403 when secret header is not provided but secret is configured."""
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret=GRAFANA_SECRET,
        ))
        body = {"alerts": []}
        req = _make_request(body)
        response = await server._webhook_grafana(req)
        assert response.status == 403

    async def test_grafana_no_secret_configured_allows(self):
        """Should accept Grafana webhooks when no secret is configured."""
        server = HealthServer(webhook_config=WebhookConfig(channel_id="123"))
        server.set_send_message(AsyncMock())

        body = {"alerts": []}
        req = _make_request(body)
        response = await server._webhook_grafana(req)
        assert response.status == 200


GENERIC_SECRET = "generic-test-secret"


class TestWebhookGeneric:
    """Tests for _webhook_generic handler."""

    async def test_generic_valid(self):
        """Should accept valid generic webhook with correct secret."""
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret=GENERIC_SECRET,
        ))
        server.set_send_message(AsyncMock())

        body = {"title": "Deploy Complete", "message": "v1.2.3 deployed to prod"}
        req = _make_request(body, headers={"X-Webhook-Secret": GENERIC_SECRET})
        response = await server._webhook_generic(req)
        assert response.status == 200

    async def test_generic_no_secret_allows(self):
        """Should accept generic webhooks when no secret is configured."""
        server = HealthServer(webhook_config=WebhookConfig(channel_id="123"))
        server.set_send_message(AsyncMock())

        body = {"title": "Deploy Complete", "message": "v1.2.3 deployed to prod"}
        req = _make_request(body, headers={"X-Webhook-Secret": ""})
        response = await server._webhook_generic(req)
        assert response.status == 200

    async def test_generic_invalid_secret(self):
        """Should return 403 when secret doesn't match."""
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret="correct-secret",
        ))
        body = {"title": "Test"}
        req = _make_request(body, headers={"X-Webhook-Secret": "wrong-secret"})
        response = await server._webhook_generic(req)
        assert response.status == 403

    async def test_generic_uses_constant_time_comparison(self):
        """Should use hmac.compare_digest for secret comparison (not plain !=)."""
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret=GENERIC_SECRET,
        ))
        server.set_send_message(AsyncMock())

        body = {"title": "Test"}
        req = _make_request(body, headers={"X-Webhook-Secret": GENERIC_SECRET})

        with patch("src.health.server.hmac.compare_digest", return_value=True) as mock_cd:
            response = await server._webhook_generic(req)
            assert response.status == 200
            mock_cd.assert_called_once_with(GENERIC_SECRET, GENERIC_SECRET)

    async def test_generic_invalid_json(self):
        """Should return 400 for invalid JSON."""
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret=GENERIC_SECRET,
        ))
        req = _make_request(b"not json", headers={"X-Webhook-Secret": GENERIC_SECRET})
        response = await server._webhook_generic(req)
        assert response.status == 400

    async def test_generic_no_message(self):
        """Should handle webhook with title only (no message)."""
        server = HealthServer(webhook_config=WebhookConfig(
            channel_id="123", secret=GENERIC_SECRET,
        ))
        server.set_send_message(AsyncMock())

        body = {"title": "Status Check"}
        req = _make_request(body, headers={"X-Webhook-Secret": GENERIC_SECRET})
        response = await server._webhook_generic(req)
        assert response.status == 200


class TestStartStop:
    """Tests for server start/stop lifecycle."""

    async def test_start_and_stop(self):
        """Should start and stop without errors."""
        server = HealthServer(port=0, webhook_config=WebhookConfig())

        mock_runner = AsyncMock()
        mock_runner.setup = AsyncMock()
        mock_runner.cleanup = AsyncMock()
        mock_runner.server = MagicMock()  # Prevent TCPSite RuntimeError

        with patch("src.health.server.web.AppRunner", return_value=mock_runner), \
             patch("src.health.server.web.TCPSite") as mock_site_cls:
            mock_site_cls.return_value.start = AsyncMock()
            await server.start()
            assert server._runner is not None
            mock_runner.setup.assert_called_once()
            await server.stop()
            mock_runner.cleanup.assert_called_once()

    async def test_stop_without_start(self):
        """Should handle stop being called without start."""
        server = HealthServer(webhook_config=WebhookConfig())
        # Should not raise
        await server.stop()
