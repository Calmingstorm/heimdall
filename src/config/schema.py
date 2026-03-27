from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DiscordConfig(BaseModel):
    token: str
    allowed_users: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    respond_to_bots: bool = False
    require_mention: bool = False
    ignore_bot_ids: list[str] = Field(default_factory=list)  # Bot user IDs to never auto-respond to


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
    command_timeout_seconds: int = 300
    tool_timeout_seconds: int = 300
    prometheus_host: str = ""
    ansible_host: str = ""
    claude_code_host: str = ""
    claude_code_user: str = ""
    claude_code_dir: str = "/opt/project"
    incus_host: str = ""
    # Empty = all tools. Options: docker, systemd, incus, ansible, prometheus, git, comfyui
    tool_packs: list[str] = Field(default_factory=list)


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
    model_config = ConfigDict(populate_by_name=True)
    enabled: bool = True
    # Accepts "chromadb_path" from old configs for backward compat
    search_db_path: str = Field(default="./data/chromadb", validation_alias="chromadb_path")


class VoiceConfig(BaseModel):
    enabled: bool = False
    voice_service_url: str = "ws://heimdall-voice:3940/ws"
    auto_join: bool = False
    transcript_channel_id: str = ""
    default_voice: str = "en_US-lessac-medium"
    wake_word: str = "heimdall"


class BrowserConfig(BaseModel):
    enabled: bool = False
    cdp_url: str = "ws://heimdall-browser:3000?token=heimdall-internal"  # Override token via BROWSER_TOKEN env var
    default_timeout_ms: int = 30000
    viewport_width: int = 1280
    viewport_height: int = 720


class PermissionsConfig(BaseModel):
    tiers: dict[str, str] = Field(default_factory=dict)
    default_tier: str = "user"
    overrides_path: str = "./data/permissions.json"


class WebConfig(BaseModel):
    enabled: bool = True
    api_token: str = ""  # Empty = no auth required (dev mode)
    session_timeout_minutes: int = 0  # 0 = no timeout (sessions persist until logout)
    port: int = 3000  # HTTP server port for health checks + web UI


class ComfyUIConfig(BaseModel):
    enabled: bool = False
    url: str = "http://localhost:8188"


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
    comfyui: ComfyUIConfig = ComfyUIConfig()
    web: WebConfig = WebConfig()


def _substitute_env_vars(text: str) -> str:
    """Replace ${VAR} and ${VAR:-default} patterns with environment variable values.

    ${VAR} — required, raises ValueError if not set.
    ${VAR:-default} — optional, uses *default* when VAR is unset.
    """
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)  # None when no :- syntax used
        value = os.environ.get(var_name)
        if value is None:
            if default is not None:
                return default
            raise ValueError(f"Environment variable {var_name} is not set")
        return value
    return re.sub(r"\$\{(\w+)(?::-([^}]*))?\}", replacer, text)


def load_config(path: str | Path = "config.yml") -> Config:
    path = Path(path)
    raw = path.read_text()
    try:
        raw = _substitute_env_vars(raw)
    except ValueError as exc:
        raise SystemExit(
            f"Configuration error: {exc}\n"
            "Set the variable in your .env file or shell environment.\n"
            "See .env.example for required variables."
        ) from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise SystemExit(
            f"Failed to parse {path}: {exc}\n"
            "Check your YAML syntax (indentation, colons, quotes)."
        ) from exc
    if not isinstance(data, dict):
        raise SystemExit(
            f"Config file {path} is empty or invalid.\n"
            "It must contain a YAML mapping with at least a 'discord' section.\n"
            "See config.yml comments for examples."
        )
    return Config(**data)
