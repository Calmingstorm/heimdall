"""Shared fixtures for Loki test suite."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.schema import (
    Config,
    ContextConfig,
    DiscordConfig,
    LoggingConfig,
    OpenAICodexConfig,
    SessionsConfig,
    ToolHost,
    ToolsConfig,
    UsageConfig,
    WebhookConfig,
)


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test data."""
    return tmp_path


@pytest.fixture
def tools_config(tmp_dir: Path) -> ToolsConfig:
    return ToolsConfig(
        ssh_key_path=str(tmp_dir / "id_ed25519"),
        ssh_known_hosts_path=str(tmp_dir / "known_hosts"),
        hosts={
            "server": ToolHost(address="10.0.0.1", ssh_user="root", os="linux"),
            "desktop": ToolHost(address="10.0.0.2", ssh_user="root", os="linux"),
            "macbook": ToolHost(address="10.0.0.3", ssh_user="deploy", os="macos"),
        },
        allowed_services=["apache2", "prometheus", "grafana-server"],
        allowed_playbooks=["check-services.yml", "update-all.yml"],
        ansible_directory="/ansible",
        command_timeout_seconds=5,
        prometheus_host="server",
        ansible_host="desktop",
        claude_code_host="desktop",
        claude_code_user="deploy",
        claude_code_dir="/opt/project",
        incus_host="desktop",
    )


@pytest.fixture
def config(tmp_dir: Path, tools_config: ToolsConfig) -> Config:
    return Config(
        discord=DiscordConfig(
            token="test-token-not-real",
            allowed_users=["12345"],
            channels=["67890"],
        ),
        openai_codex=OpenAICodexConfig(enabled=False),
        context=ContextConfig(directory=str(tmp_dir / "context")),
        sessions=SessionsConfig(
            max_history=50,
            max_age_hours=24,
            persist_directory=str(tmp_dir / "sessions"),
        ),
        tools=tools_config,
        logging=LoggingConfig(directory=str(tmp_dir / "logs")),
        usage=UsageConfig(directory=str(tmp_dir / "usage")),
        webhook=WebhookConfig(
            enabled=True,
            secret="test-webhook-secret",
            channel_id="67890",
        ),
    )


@pytest.fixture
def mock_ssh():
    """Return a mock for run_ssh_command that returns (0, 'ok')."""
    mock = AsyncMock(return_value=(0, "ok"))
    return mock


