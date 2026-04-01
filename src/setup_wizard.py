"""
Interactive CLI setup wizard for Heimdall.

Usage:
    python -m src.setup wizard              # Full interactive setup
    python -m src.setup wizard --headless   # Headless mode (no browser)
    python -m src.setup wizard --check      # Check if setup is needed

Walks the user through configuring Discord, Codex auth, hosts, features,
and web UI.  Generates config.yml and .env from answers.  All I/O goes
through injectable callables so tests can mock input()/print().
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import string
import subprocess
import sys
from pathlib import Path
from collections.abc import Callable
from typing import Any

import aiohttp
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("config.yml")
DEFAULT_ENV_PATH = Path(".env")
ENV_EXAMPLE_PATH = Path(".env.example")

DISCORD_API_BASE = "https://discord.com/api/v10"

# Placeholder token value that indicates setup hasn't been done
PLACEHOLDER_TOKEN = "your-discord-bot-token-here"

# Default config template — matches the structure in config.yml
_DEFAULT_CONFIG: dict[str, Any] = {
    "timezone": "UTC",
    "discord": {
        "token": "${DISCORD_TOKEN}",
        "allowed_users": [],
        "channels": [],
        "respond_to_bots": False,
        "require_mention": False,
    },
    "openai_codex": {
        "enabled": True,
        "model": "gpt-5.4",
        "max_tokens": 4096,
        "credentials_path": "./data/codex_auth.json",
    },
    "context": {
        "directory": "./data/context",
        "max_system_prompt_tokens": 32000,
    },
    "sessions": {
        "max_history": 50,
        "max_age_hours": 24,
        "persist_directory": "./data/sessions",
    },
    "tools": {
        "enabled": True,
        "ssh_key_path": "~/.ssh/id_ed25519",
        "ssh_known_hosts_path": "~/.ssh/known_hosts",
        "hosts": {},
        "command_timeout_seconds": 300,
    },
    "webhook": {
        "enabled": False,
        "secret": "${WEBHOOK_SECRET:-}",
        "channel_id": "",
        "gitea_channel_id": "",
        "grafana_channel_id": "",
    },
    "learning": {
        "enabled": True,
        "max_entries": 30,
        "consolidation_target": 20,
    },
    "search": {
        "enabled": True,
        "search_db_path": "./data/search",
    },
    "logging": {
        "level": "INFO",
        "directory": "./data/logs",
    },
    "usage": {
        "directory": "./data/usage",
    },
    "voice": {
        "enabled": False,
    },
    "browser": {
        "enabled": False,
    },
    "monitoring": {
        "enabled": False,
        "checks": [],
        "alert_channel_id": "",
        "cooldown_minutes": 60,
    },
    "permissions": {
        "tiers": {},
        "default_tier": "user",
        "overrides_path": "./data/permissions.json",
    },
    "comfyui": {
        "enabled": False,
    },
    "web": {
        "enabled": True,
        "port": 3000,
        "api_token": "",
    },
}


# ---------------------------------------------------------------------------
# Validation helpers (pure functions, no I/O)
# ---------------------------------------------------------------------------

def validate_token_format(token: str) -> bool:
    """Check if a string looks like a valid Discord bot token.

    Discord tokens are base64-encoded and have three dot-separated parts.
    We don't validate cryptographically — just format.
    """
    if not token or not token.strip():
        return False
    parts = token.strip().split(".")
    return len(parts) == 3 and all(len(p) > 0 for p in parts)


async def validate_discord_token(token: str) -> tuple[bool, dict[str, str]]:
    """Validate a Discord bot token by calling GET /users/@me.

    Returns (valid, info_dict).  info_dict has 'username' and 'id' on success.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{DISCORD_API_BASE}/users/@me",
                headers={"Authorization": f"Bot {token.strip()}"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return True, {
                        "username": data.get("username", "unknown"),
                        "id": data.get("id", ""),
                        "discriminator": data.get("discriminator", "0"),
                    }
                return False, {"error": f"HTTP {resp.status}"}
    except Exception as exc:
        return False, {"error": str(exc)}


def validate_host_address(address: str) -> bool:
    """Check if a string is a plausible host address (IP or hostname)."""
    if not address or not address.strip():
        return False
    addr = address.strip()
    # IPv4
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", addr):
        return all(0 <= int(p) <= 255 for p in addr.split("."))
    # Hostname (simple check)
    if re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$", addr):
        return True
    return False


def validate_ssh_user(user: str) -> bool:
    """Check if a string is a plausible SSH username."""
    if not user or not user.strip():
        return False
    return bool(re.match(r"^[a-z_][a-z0-9_-]*$", user.strip()))


def generate_api_token(length: int = 32) -> str:
    """Generate a cryptographically random API token."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


# ---------------------------------------------------------------------------
# Config / env generation (pure functions)
# ---------------------------------------------------------------------------

def build_config(
    *,
    discord_token_env_ref: str = "${DISCORD_TOKEN}",
    timezone: str = "UTC",
    hosts: dict[str, dict[str, str]] | None = None,
    features: dict[str, bool] | None = None,
    web_api_token: str = "",
    claude_code_host: str = "",
) -> dict[str, Any]:
    """Build a config dict from wizard answers.

    Returns a dict ready to be written as YAML.
    """
    import copy
    cfg = copy.deepcopy(_DEFAULT_CONFIG)

    cfg["timezone"] = timezone
    cfg["discord"]["token"] = discord_token_env_ref

    if hosts:
        for name, host_info in hosts.items():
            cfg["tools"]["hosts"][name] = {
                "address": host_info["address"],
                "ssh_user": host_info.get("ssh_user", "root"),
            }

    features = features or {}
    cfg["browser"]["enabled"] = features.get("browser", False)
    cfg["voice"]["enabled"] = features.get("voice", False)
    cfg["comfyui"]["enabled"] = features.get("comfyui", False)

    if claude_code_host and claude_code_host in (hosts or {}):
        cfg["tools"]["claude_code_host"] = claude_code_host

    cfg["web"]["api_token"] = web_api_token

    return cfg


def build_env(discord_token: str, extra: dict[str, str] | None = None) -> str:
    """Build .env file content from wizard answers.

    Reads .env.example as a base if it exists, otherwise generates minimal content.
    """
    lines = [
        "# Heimdall environment configuration",
        "# Generated by setup wizard",
        "",
        f"DISCORD_TOKEN={discord_token}",
    ]
    if extra:
        lines.append("")
        for key, value in extra.items():
            lines.append(f"{key}={value}")
    lines.append("")
    return "\n".join(lines)


def is_setup_needed(
    config_path: Path = DEFAULT_CONFIG_PATH,
    env_path: Path = DEFAULT_ENV_PATH,
) -> bool:
    """Check if initial setup is needed.

    Returns True if:
    - .env doesn't exist, OR
    - DISCORD_TOKEN is the placeholder value, OR
    - config.yml doesn't exist
    """
    if not config_path.exists():
        return True
    if not env_path.exists():
        return True

    # Check if token is still the placeholder
    try:
        env_content = env_path.read_text()
        for line in env_content.splitlines():
            stripped = line.strip()
            if stripped.startswith("DISCORD_TOKEN="):
                value = stripped.split("=", 1)[1].strip()
                if value == PLACEHOLDER_TOKEN or not value:
                    return True
                return False
    except Exception:
        return True

    return True  # No DISCORD_TOKEN line found


def detect_systemd() -> bool:
    """Check if systemd is available on this system."""
    return shutil.which("systemctl") is not None


# ---------------------------------------------------------------------------
# Wizard runner
# ---------------------------------------------------------------------------

class SetupWizard:
    """Interactive CLI setup wizard.

    All I/O goes through ``input_fn`` and ``print_fn`` so tests can
    inject mocks.  ``run()`` is a sync entry point that drives the
    async validation calls internally.
    """

    def __init__(
        self,
        *,
        config_path: Path = DEFAULT_CONFIG_PATH,
        env_path: Path = DEFAULT_ENV_PATH,
        credentials_path: Path = Path("data/codex_auth.json"),
        headless: bool = False,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[..., None] = print,
    ):
        self.config_path = config_path
        self.env_path = env_path
        self.credentials_path = credentials_path
        self.headless = headless
        self._input = input_fn
        self._print = print_fn

        # Collected answers
        self.discord_token: str = ""
        self.bot_info: dict[str, str] = {}
        self.hosts: dict[str, dict[str, str]] = {}
        self.features: dict[str, bool] = {}
        self.web_api_token: str = ""
        self.claude_code_host: str = ""
        self.timezone: str = "UTC"
        self.codex_configured: bool = False

    # -- Prompt helpers -----------------------------------------------------

    def _prompt(self, message: str, default: str = "") -> str:
        """Prompt for input with optional default."""
        if default:
            display = f"{message} [{default}]: "
        else:
            display = f"{message}: "
        value = self._input(display).strip()
        return value if value else default

    def _prompt_yes_no(self, message: str, default: bool = False) -> bool:
        """Prompt for a yes/no answer."""
        suffix = " [Y/n]" if default else " [y/N]"
        answer = self._input(f"{message}{suffix}: ").strip().lower()
        if not answer:
            return default
        return answer in ("y", "yes")

    def _prompt_secret(self, message: str) -> str:
        """Prompt for a secret value (no default shown)."""
        return self._input(f"{message}: ").strip()

    # -- Wizard steps -------------------------------------------------------

    def step_discord_token(self) -> bool:
        """Step 1: Prompt for Discord bot token and validate it.

        Returns True if token is valid, False to abort.
        """
        self._print("\n=== Step 1: Discord Bot Token ===\n")
        self._print("Create a bot at https://discord.com/developers/applications")
        self._print("Required intents: MESSAGE CONTENT, SERVER MEMBERS\n")

        for attempt in range(3):
            token = self._prompt_secret("Paste your bot token")
            if not token:
                self._print("No token provided.")
                continue

            if not validate_token_format(token):
                self._print("Token format looks invalid (expected 3 dot-separated parts).")
                continue

            self._print("Validating token...")
            valid, info = asyncio.run(validate_discord_token(token))
            if valid:
                self.discord_token = token.strip()
                self.bot_info = info
                self._print(f"  Bot: {info.get('username', 'unknown')} (ID: {info.get('id', '?')})")
                return True
            else:
                self._print(f"  Token validation failed: {info.get('error', 'unknown error')}")

        self._print("\nFailed to validate token after 3 attempts.")
        return False

    def step_codex_auth(self) -> None:
        """Step 2: Set up Codex (ChatGPT) authentication."""
        self._print("\n=== Step 2: Codex Authentication ===\n")
        self._print("Codex (ChatGPT) provides the AI backend.  You need a free")
        self._print("ChatGPT account.  This step opens a browser for OAuth login.\n")

        if not self._prompt_yes_no("Set up Codex authentication now?"):
            self._print("Skipped.  Run 'python -m src.setup add' later.")
            return

        # Delegate to existing setup flow
        try:
            from .setup import _get_auth_code_headless, _get_auth_code_browser
            from .setup import _save_accounts, _load_accounts
            from .llm.codex_auth import CodexAuth

            if self.headless:
                auth_code, code_verifier = _get_auth_code_headless()
            else:
                auth_code, code_verifier = _get_auth_code_browser()

            self._print("Exchanging code for tokens...")
            creds = asyncio.get_event_loop().run_until_complete(
                CodexAuth.exchange_code(auth_code, code_verifier)
            )
            accounts = _load_accounts(self.credentials_path)
            accounts.append(creds)
            _save_accounts(self.credentials_path, accounts)
            self.codex_configured = True
            self._print(f"  Codex account added: {creds.get('email', 'unknown')}")
        except KeyboardInterrupt:
            self._print("\nCodex setup cancelled.")
        except Exception as exc:
            self._print(f"  Codex setup failed: {exc}")
            self._print("  Run 'python -m src.setup add' later to retry.")

    def step_hosts(self) -> None:
        """Step 3: Configure remote hosts."""
        self._print("\n=== Step 3: Remote Hosts ===\n")
        self._print("Heimdall can manage remote servers via SSH.")
        self._print("You can also add hosts later in config.yml.\n")

        while self._prompt_yes_no("Add a remote host?"):
            name = self._prompt("Host name (e.g. 'myserver')")
            if not name:
                continue

            address = self._prompt("Host address (IP or hostname)")
            if not validate_host_address(address):
                self._print(f"  Invalid address: {address}")
                continue

            ssh_user = self._prompt("SSH user", default="root")
            if not validate_ssh_user(ssh_user):
                self._print(f"  Invalid SSH user: {ssh_user}")
                continue

            self.hosts[name] = {"address": address, "ssh_user": ssh_user}
            self._print(f"  Added host '{name}' ({ssh_user}@{address})")

        if self.hosts:
            self._print(f"\n  Configured {len(self.hosts)} host(s)")

    def step_features(self) -> None:
        """Step 4: Toggle optional features."""
        self._print("\n=== Step 4: Optional Features ===\n")
        self._print("Enable extra capabilities (all can be changed later).\n")

        self.features["browser"] = self._prompt_yes_no("Enable browser automation?")
        self.features["voice"] = self._prompt_yes_no("Enable voice channel support?")
        self.features["comfyui"] = self._prompt_yes_no("Enable ComfyUI image generation?")

        if self.hosts:
            if self._prompt_yes_no("Enable Claude Code (AI coding assistant)?"):
                self.features["claude_code"] = True
                host_names = list(self.hosts.keys())
                if len(host_names) == 1:
                    self.claude_code_host = host_names[0]
                else:
                    host_list = ", ".join(host_names)
                    choice = self._prompt(f"Which host for Claude Code? ({host_list})")
                    if choice in self.hosts:
                        self.claude_code_host = choice
            else:
                self.features["claude_code"] = False

        enabled = [k for k, v in self.features.items() if v]
        if enabled:
            self._print(f"\n  Enabled: {', '.join(enabled)}")
        else:
            self._print("\n  No optional features enabled")

    def step_web_token(self) -> None:
        """Step 5: Set up web UI authentication."""
        self._print("\n=== Step 5: Web UI Authentication ===\n")
        self._print("The web management UI can be protected with an API token.\n")

        if self._prompt_yes_no("Generate a random API token for the web UI?", default=True):
            self.web_api_token = generate_api_token()
            self._print(f"  Token: {self.web_api_token}")
            self._print("  Save this token — you'll need it to log in to the web UI.")
        else:
            custom = self._prompt("Enter a custom API token (empty = no auth)")
            self.web_api_token = custom

    def step_write_config(self) -> tuple[bool, bool]:
        """Step 6: Write config.yml and .env files.

        Returns (config_written, env_written).
        """
        self._print("\n=== Step 6: Write Configuration ===\n")

        cfg = build_config(
            timezone=self.timezone,
            hosts=self.hosts,
            features=self.features,
            web_api_token=self.web_api_token,
            claude_code_host=self.claude_code_host,
        )
        env_content = build_env(self.discord_token)

        config_written = False
        env_written = False

        # Write config.yml
        if self.config_path.exists():
            if not self._prompt_yes_no(f"Overwrite {self.config_path}?"):
                self._print(f"  Skipped {self.config_path}")
            else:
                self._write_config_file(cfg)
                config_written = True
        else:
            self._write_config_file(cfg)
            config_written = True

        # Write .env
        if self.env_path.exists():
            if not self._prompt_yes_no(f"Overwrite {self.env_path}?"):
                self._print(f"  Skipped {self.env_path}")
            else:
                self._write_env_file(env_content)
                env_written = True
        else:
            self._write_env_file(env_content)
            env_written = True

        return config_written, env_written

    def _write_config_file(self, cfg: dict) -> None:
        """Write config dict as YAML."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        self._print(f"  Wrote {self.config_path}")

    def _write_env_file(self, content: str) -> None:
        """Write .env file with restricted permissions."""
        self.env_path.parent.mkdir(parents=True, exist_ok=True)
        self.env_path.write_text(content)
        try:
            self.env_path.chmod(0o600)
        except OSError:
            pass  # Windows or permission issue
        self._print(f"  Wrote {self.env_path}")

    def step_start_service(self) -> None:
        """Step 7: Offer to start the systemd service."""
        if not detect_systemd():
            self._print("\n  systemd not detected — start Heimdall manually.")
            return

        self._print("\n=== Step 7: Start Heimdall ===\n")
        if self._prompt_yes_no("Start Heimdall service now?"):
            try:
                subprocess.run(
                    ["sudo", "systemctl", "start", "heimdall"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self._print("  Heimdall service started!")
            except subprocess.CalledProcessError as exc:
                self._print(f"  Failed to start service: {exc.stderr}")
            except FileNotFoundError:
                self._print("  'sudo' not found — start manually: systemctl start heimdall")
            except subprocess.TimeoutExpired:
                self._print("  Start command timed out.")

    # -- Main entry point ---------------------------------------------------

    def run(self) -> bool:
        """Run the full wizard flow.

        Returns True if setup completed successfully, False otherwise.
        """
        self._print("=" * 50)
        self._print("  Heimdall Setup Wizard")
        self._print("=" * 50)

        # Step 1: Discord token (required)
        if not self.step_discord_token():
            self._print("\nSetup cancelled — Discord token is required.")
            return False

        # Step 2: Codex auth (optional but recommended)
        self.step_codex_auth()

        # Step 3: Remote hosts (optional)
        self.step_hosts()

        # Step 4: Feature toggles (optional)
        self.step_features()

        # Step 5: Web UI token (optional)
        self.step_web_token()

        # Step 6: Write config files
        config_written, env_written = self.step_write_config()

        # Step 7: Start service (optional)
        self.step_start_service()

        # Summary
        self._print("\n" + "=" * 50)
        self._print("  Setup complete!")
        self._print("=" * 50)
        if config_written:
            self._print(f"  Config:  {self.config_path}")
        if env_written:
            self._print(f"  Env:     {self.env_path}")
        if self.codex_configured:
            self._print(f"  Codex:   {self.credentials_path}")
        self._print("")

        return True


# ---------------------------------------------------------------------------
# CLI entry point (called from src/setup.py)
# ---------------------------------------------------------------------------

def run_wizard(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    env_path: Path = DEFAULT_ENV_PATH,
    credentials_path: Path = Path("data/codex_auth.json"),
    headless: bool = False,
    check_only: bool = False,
) -> None:
    """Entry point for ``python -m src.setup wizard``."""
    if check_only:
        needed = is_setup_needed(config_path, env_path)
        print(f"Setup needed: {needed}")
        sys.exit(0 if not needed else 1)

    wizard = SetupWizard(
        config_path=config_path,
        env_path=env_path,
        credentials_path=credentials_path,
        headless=headless,
    )
    try:
        success = wizard.run()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(130)

    sys.exit(0 if success else 1)
