from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DiscordConfig(BaseModel):
    token: str
    allowed_users: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    respond_to_bots: bool = False
    require_mention: bool = False


class AnthropicConfig(BaseModel):
    """Anthropic API config — used for Haiku classification only."""
    api_key: str = ""
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 16384
    daily_budget_tokens: int = 0


class ContextConfig(BaseModel):
    directory: str = "./data/context"
    max_system_prompt_tokens: int = 32000


class SessionsConfig(BaseModel):
    max_history: int = 50
    max_age_hours: int = 24
    persist_directory: str = "./data/sessions"


class ToolHost(BaseModel):
    address: str
    ssh_user: str = "root"
    os: str = "linux"  # "linux" or "macos"


class ToolsConfig(BaseModel):
    enabled: bool = True
    ssh_key_path: str = "/app/.ssh/id_ed25519"
    ssh_known_hosts_path: str = "/app/.ssh/known_hosts"
    hosts: dict[str, ToolHost] = Field(default_factory=dict)
    allowed_services: list[str] = Field(default_factory=list)
    allowed_playbooks: list[str] = Field(default_factory=list)
    ansible_directory: str = "/ansible"
    approval_timeout_seconds: int = 60
    command_timeout_seconds: int = 30
    tool_timeout_seconds: int = 300
    prometheus_host: str = ""
    ansible_host: str = ""
    claude_code_host: str = ""
    claude_code_user: str = ""
    claude_code_dir: str = "/opt/project"
    incus_host: str = ""


class LoggingConfig(BaseModel):
    level: str = "INFO"
    directory: str = "./data/logs"


class UsageConfig(BaseModel):
    directory: str = "./data/usage"


class OpenAICodexConfig(BaseModel):
    enabled: bool = False
    model: str = "gpt-4o"
    max_tokens: int = 4096
    credentials_path: str = "./data/codex_auth.json"


class WebhookConfig(BaseModel):
    enabled: bool = False
    secret: str = ""
    channel_id: str = ""
    gitea_channel_id: str = ""
    grafana_channel_id: str = ""


class LearningConfig(BaseModel):
    enabled: bool = True
    max_entries: int = 30
    consolidation_target: int = 20


class SearchConfig(BaseModel):
    enabled: bool = True
    ollama_url: str = "http://host.docker.internal:11434"
    embed_model: str = "nomic-embed-text"
    chromadb_path: str = "./data/chromadb"


class VoiceConfig(BaseModel):
    enabled: bool = False
    voice_service_url: str = "ws://loki-voice:3940/ws"
    auto_join: bool = False
    transcript_channel_id: str = ""
    default_voice: str = "en_US-lessac-medium"
    wake_word: str = "loki"


class BrowserConfig(BaseModel):
    enabled: bool = False
    cdp_url: str = "ws://loki-browser:3000?token=loki-internal"
    default_timeout_ms: int = 30000
    viewport_width: int = 1280
    viewport_height: int = 720


class PermissionsConfig(BaseModel):
    tiers: dict[str, str] = Field(default_factory=dict)
    default_tier: str = "user"
    overrides_path: str = "./data/permissions.json"


class MonitorCheck(BaseModel):
    name: str
    type: str  # "disk", "memory", "service", "promql"
    hosts: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)  # for type "service"
    threshold: int = 90  # percent, for disk/memory
    query: str = ""  # for type "promql"
    interval_minutes: int = 30


class MonitoringConfig(BaseModel):
    enabled: bool = False
    checks: list[MonitorCheck] = Field(default_factory=list)
    alert_channel_id: str = ""
    cooldown_minutes: int = 60


class Config(BaseModel):
    timezone: str = "UTC"
    discord: DiscordConfig
    anthropic: AnthropicConfig = AnthropicConfig()
    openai_codex: OpenAICodexConfig = OpenAICodexConfig()
    context: ContextConfig = ContextConfig()
    sessions: SessionsConfig = SessionsConfig()
    tools: ToolsConfig = ToolsConfig()
    logging: LoggingConfig = LoggingConfig()
    usage: UsageConfig = UsageConfig()
    webhook: WebhookConfig = WebhookConfig()
    learning: LearningConfig = LearningConfig()
    search: SearchConfig = SearchConfig()
    voice: VoiceConfig = VoiceConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    browser: BrowserConfig = BrowserConfig()
    permissions: PermissionsConfig = PermissionsConfig()


def _substitute_env_vars(text: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(f"Environment variable {var_name} is not set")
        return value
    return re.sub(r"\$\{(\w+)}", replacer, text)


def load_config(path: str | Path = "config.yml") -> Config:
    path = Path(path)
    raw = path.read_text()
    raw = _substitute_env_vars(raw)
    data = yaml.safe_load(raw)
    return Config(**data)
