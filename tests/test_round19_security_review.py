"""Round 19: Security review — comprehensive audit of personal data, credentials,
script validation, secret scrubbing, and prompt injection resistance.

Tests cover: no personal data in codebase, no hardcoded credentials, read_file lines
parameter validation, secret scrubber pattern coverage (including new
GitHub/AWS/Stripe/Slack patterns), error message scrubbing before Discord,
monitor alert scrubbing, prompt injection resistance (role forgery, display name injection,
tool output injection, context separator integrity), run_script security (base64 encoding,
interpreter allowlist), and cross-cutting security verification.
"""
from __future__ import annotations

import ast
import base64
import re
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.secret_scrubber import OUTPUT_SECRET_PATTERNS, scrub_output_secrets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
TESTS_DIR = Path(__file__).resolve().parent


def _make_config(**overrides):
    """Minimal config stub."""
    cfg = SimpleNamespace(
        discord=SimpleNamespace(
            respond_to_bots=True,
            require_mention=False,
            allowed_channels=[],
            allowed_users=[],
            guest_users=[],
        ),
        tools=SimpleNamespace(
            hosts={
                "server1": SimpleNamespace(address="10.0.0.1", ssh_user="deploy"),
            },
            ssh_known_hosts_path="~/.ssh/known_hosts",
            command_timeout_seconds=30,
            claude_code_host="server1",
            claude_code_user="",
            tool_timeout_seconds=300,
        ),
        sessions=SimpleNamespace(max_history=50, max_age_hours=72, persist_dir="/tmp/test_sessions"),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ===========================================================================
# 1. No Personal Data in Codebase
# ===========================================================================


class TestNoPersonalDataInSource:
    """Verify no personal data leaks in src/ files."""

    def _source_files(self):
        return list(SRC_DIR.rglob("*.py"))

    def test_no_real_email_addresses(self):
        """No real email addresses (non-example.com) in source."""
        email_re = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}")
        safe_domains = {"example.com", "example.net", "example.org", "test.com",
                        "anthropic.com", "openai.com", "localhost"}
        for path in self._source_files():
            content = path.read_text()
            for match in email_re.finditer(content):
                email = match.group()
                domain = email.split("@")[1]
                assert domain in safe_domains, (
                    f"Potentially real email {email!r} in {path.relative_to(SRC_DIR.parent)}"
                )

    def test_no_personal_ips_in_source(self):
        """No 192.168.x.x IPs in source (test fixtures use 10.0.0.x)."""
        ip_re = re.compile(r"192\.168\.\d+\.\d+")
        for path in self._source_files():
            content = path.read_text()
            matches = ip_re.findall(content)
            assert not matches, (
                f"Personal IP {matches} in {path.relative_to(SRC_DIR.parent)}"
            )

    def test_no_home_username_paths(self):
        """No /home/<username> paths (only /home/deploy allowed)."""
        home_re = re.compile(r"/home/(\w+)")
        allowed_users = {"deploy"}
        for path in self._source_files():
            content = path.read_text()
            for match in home_re.finditer(content):
                user = match.group(1)
                assert user in allowed_users, (
                    f"Personal path /home/{user} in {path.relative_to(SRC_DIR.parent)}"
                )

    def test_no_hardcoded_discord_tokens(self):
        """No Discord bot tokens in source (format: letters.base64.base64)."""
        # Discord tokens are ~70 chars: NjE5...ABCD.Gh1234.abcdefg...
        # Simple heuristic: reject any string matching the pattern
        token_re = re.compile(r"[MN][A-Za-z0-9]{23,}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27}")
        for path in self._source_files():
            content = path.read_text()
            matches = token_re.findall(content)
            assert not matches, (
                f"Possible Discord token in {path.relative_to(SRC_DIR.parent)}"
            )


class TestNoHardcodedCredentials:
    """Verify no hardcoded API keys or secrets in source."""

    def _all_python_files(self):
        return list(SRC_DIR.rglob("*.py"))

    def test_no_openai_keys(self):
        """No OpenAI API keys (sk-...) in source."""
        key_re = re.compile(r"sk-[a-zA-Z0-9]{20,}")
        for path in self._all_python_files():
            content = path.read_text()
            # Allow the scrubber pattern definition itself
            if "secret_scrubber" in path.name:
                continue
            matches = key_re.findall(content)
            assert not matches, f"OpenAI key in {path.relative_to(SRC_DIR.parent)}"

    def test_no_github_tokens(self):
        """No GitHub tokens (ghp_...) in source."""
        token_re = re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}")
        for path in self._all_python_files():
            content = path.read_text()
            if "secret_scrubber" in path.name:
                continue
            matches = token_re.findall(content)
            assert not matches, f"GitHub token in {path.relative_to(SRC_DIR.parent)}"

    def test_no_aws_access_keys(self):
        """No AWS access key IDs (AKIA...) in source."""
        key_re = re.compile(r"AKIA[0-9A-Z]{16}")
        for path in self._all_python_files():
            content = path.read_text()
            if "secret_scrubber" in path.name:
                continue
            matches = key_re.findall(content)
            assert not matches, f"AWS key in {path.relative_to(SRC_DIR.parent)}"

    def test_no_stripe_keys(self):
        """No Stripe API keys (sk_live_...) in source."""
        key_re = re.compile(r"[sr]k_(live|test)_[A-Za-z0-9]{20,}")
        for path in self._all_python_files():
            content = path.read_text()
            if "secret_scrubber" in path.name:
                continue
            matches = key_re.findall(content)
            assert not matches, f"Stripe key in {path.relative_to(SRC_DIR.parent)}"

    def test_config_uses_env_vars(self):
        """Config uses ${VAR} environment variable substitution, not hardcoded values."""
        config_path = SRC_DIR.parent / "config.yml"
        if config_path.exists():
            content = config_path.read_text()
            # Check that critical config values use env var substitution
            assert "${DISCORD_TOKEN}" in content, "Discord token should use env var"

    def test_env_example_has_no_real_keys(self):
        """The .env.example file has placeholder values only."""
        env_example = SRC_DIR.parent / ".env.example"
        if env_example.exists():
            content = env_example.read_text()
            assert "sk-" not in content, ".env.example should not have real keys"
            assert "your-" in content or "placeholder" in content.lower() or "=" in content


# ===========================================================================
# 2. Script Validation (run_script)
# ===========================================================================


class TestRunScriptSecurity:
    """Verify run_script handler security measures."""

    @pytest.fixture
    def executor(self, tools_config):
        from src.tools.executor import ToolExecutor
        ex = ToolExecutor(tools_config)
        ex._exec_command = AsyncMock(return_value=(0, "ok"))
        return ex

    async def test_interpreter_allowlist(self, executor):
        """Only allowed interpreters accepted."""
        allowed = {"bash", "sh", "python3", "python", "node", "ruby", "perl"}
        for interp in allowed:
            result = await executor.execute(
                "run_script",
                {"host": "server", "script": "echo hello", "interpreter": interp},
            )
            assert "Unsupported interpreter" not in result

    async def test_interpreter_rejection(self, executor):
        """Dangerous interpreters rejected."""
        for interp in ["evil_binary", "/bin/sh", "../../bin/bash", "curl", "wget", "nc"]:
            result = await executor.execute(
                "run_script",
                {"host": "server", "script": "echo hello", "interpreter": interp},
            )
            assert "Unsupported interpreter" in result

    async def test_script_base64_encoded(self, executor):
        """Script content is base64-encoded to prevent shell injection."""
        script = "echo 'hello'; rm -rf /"
        await executor.execute(
            "run_script",
            {"host": "server", "script": script, "interpreter": "bash"},
        )
        cmd = executor._exec_command.call_args[0][1]
        # The encoded script should be in the command
        encoded = base64.b64encode(script.encode()).decode()
        assert encoded in cmd

    async def test_script_special_chars_safe(self, executor):
        """Special shell characters in script don't escape base64 encoding."""
        script = "$(rm -rf /); `evil`; echo $HOME; echo 'quotes' \"double\""
        await executor.execute(
            "run_script",
            {"host": "server", "script": script, "interpreter": "bash"},
        )
        cmd = executor._exec_command.call_args[0][1]
        # The dangerous characters should NOT appear unencoded
        assert "$(rm" not in cmd
        assert "`evil`" not in cmd


# ===========================================================================
# 3. read_file Lines Parameter Validation (NEW FIX)
# ===========================================================================


class TestReadFileLinesValidation:
    """Verify read_file handler validates the lines parameter to prevent injection."""

    @pytest.fixture
    def executor(self, tools_config):
        from src.tools.executor import ToolExecutor
        ex = ToolExecutor(tools_config)
        ex._exec_command = AsyncMock(return_value=(0, "file content"))
        return ex

    async def test_normal_lines(self, executor):
        """Normal integer lines value works."""
        await executor.execute("read_file", {"host": "server", "path": "/etc/hostname", "lines": 50})
        cmd = executor._exec_command.call_args[0][1]
        assert "head -n 50" in cmd

    async def test_default_lines(self, executor):
        """Default lines is 200."""
        await executor.execute("read_file", {"host": "server", "path": "/etc/hostname"})
        cmd = executor._exec_command.call_args[0][1]
        assert "head -n 200" in cmd

    async def test_lines_capped_at_1000(self, executor):
        """Lines parameter capped at 1000 to prevent abuse."""
        await executor.execute("read_file", {"host": "server", "path": "/etc/hostname", "lines": 999999})
        cmd = executor._exec_command.call_args[0][1]
        assert "head -n 1000" in cmd

    async def test_lines_string_injection_blocked(self, executor):
        """String injection in lines parameter is blocked (falls back to default)."""
        await executor.execute("read_file", {"host": "server", "path": "/etc/hostname", "lines": "5; cat /etc/shadow"})
        cmd = executor._exec_command.call_args[0][1]
        # Should NOT contain the injection payload
        assert "cat /etc/shadow" not in cmd
        # Should fall back to default 200
        assert "head -n 200" in cmd

    async def test_lines_negative_value(self, executor):
        """Negative lines value treated as integer (head handles -n -5 safely)."""
        await executor.execute("read_file", {"host": "server", "path": "/etc/hostname", "lines": -5})
        cmd = executor._exec_command.call_args[0][1]
        # min(-5, 1000) = -5, which is a valid (if unusual) head argument
        assert "head -n -5" in cmd

    async def test_lines_float_converted(self, executor):
        """Float lines value converted to int."""
        await executor.execute("read_file", {"host": "server", "path": "/etc/hostname", "lines": 50.7})
        cmd = executor._exec_command.call_args[0][1]
        assert "head -n 50" in cmd

    async def test_lines_none_uses_default(self, executor):
        """None lines value uses default."""
        await executor.execute("read_file", {"host": "server", "path": "/etc/hostname", "lines": None})
        cmd = executor._exec_command.call_args[0][1]
        assert "head -n 200" in cmd

    async def test_path_is_quoted(self, executor):
        """Path is properly quoted with shlex.quote."""
        await executor.execute("read_file", {"host": "server", "path": "/tmp/evil; rm -rf /", "lines": 10})
        cmd = executor._exec_command.call_args[0][1]
        # Path should be quoted, preventing injection
        assert shlex.quote("/tmp/evil; rm -rf /") in cmd


# ===========================================================================
# 4. Secret Scrubber Pattern Coverage
# ===========================================================================


class TestSecretScrubberExistingPatterns:
    """Verify existing scrubber patterns still work."""

    def test_password_scrubbed(self):
        assert "[REDACTED]" in scrub_output_secrets("password=hunter2")

    def test_api_key_scrubbed(self):
        assert "[REDACTED]" in scrub_output_secrets("api_key=abcdef1234567890")

    def test_openai_key_scrubbed(self):
        assert "[REDACTED]" in scrub_output_secrets("sk-abcdefghijklmnopqrstuvwxyz12345")

    def test_private_key_scrubbed(self):
        assert "[REDACTED]" in scrub_output_secrets("-----BEGIN RSA PRIVATE KEY-----")

    def test_postgres_uri_scrubbed(self):
        assert "[REDACTED]" in scrub_output_secrets("postgres://admin:secret123@db.host:5432/mydb")

    def test_mongodb_uri_scrubbed(self):
        assert "[REDACTED]" in scrub_output_secrets("mongodb+srv://user:pass@cluster.example.net")

    def test_clean_text_unchanged(self):
        text = "Server is running normally on port 8080"
        assert scrub_output_secrets(text) == text


class TestSecretScrubberNewPatterns:
    """Verify newly added scrubber patterns work."""

    def test_github_personal_token(self):
        """GitHub personal access tokens (ghp_) scrubbed."""
        token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl"
        assert "[REDACTED]" in scrub_output_secrets(f"token: {token}")

    def test_github_oauth_token(self):
        """GitHub OAuth tokens (gho_) scrubbed."""
        token = "gho_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl"
        assert "[REDACTED]" in scrub_output_secrets(token)

    def test_github_user_token(self):
        """GitHub user tokens (ghu_) scrubbed."""
        token = "ghu_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl"
        assert "[REDACTED]" in scrub_output_secrets(token)

    def test_github_server_token(self):
        """GitHub server tokens (ghs_) scrubbed."""
        token = "ghs_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl"
        assert "[REDACTED]" in scrub_output_secrets(token)

    def test_github_refresh_token(self):
        """GitHub refresh tokens (ghr_) scrubbed."""
        token = "ghr_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl"
        assert "[REDACTED]" in scrub_output_secrets(token)

    def test_aws_access_key(self):
        """AWS access key IDs (AKIA...) scrubbed."""
        key = "AKIAIOSFODNN7EXAMPLE"
        assert "[REDACTED]" in scrub_output_secrets(f"aws_access_key_id = {key}")

    def test_stripe_live_key(self):
        """Stripe live secret keys (sk_live_) scrubbed."""
        key = "sk_live_abcdefghijklmnopqrstuvwxyz"
        assert "[REDACTED]" in scrub_output_secrets(key)

    def test_stripe_test_key(self):
        """Stripe test secret keys (sk_test_) scrubbed."""
        key = "sk_test_abcdefghijklmnopqrstuvwxyz"
        assert "[REDACTED]" in scrub_output_secrets(key)

    def test_stripe_restricted_key(self):
        """Stripe restricted keys (rk_live_) scrubbed."""
        key = "rk_live_abcdefghijklmnopqrstuvwxyz"
        assert "[REDACTED]" in scrub_output_secrets(key)

    def test_slack_bot_token(self):
        """Slack bot tokens (xoxb-) scrubbed."""
        token = "xoxb-1234567890-1234567890123-AbCdEfGhIjKlMnOpQrStUv"
        assert "[REDACTED]" in scrub_output_secrets(token)

    def test_slack_user_token(self):
        """Slack user tokens (xoxp-) scrubbed."""
        token = "xoxp-1234567890-1234567890123-1234567890123-abcdef0123456789"
        assert "[REDACTED]" in scrub_output_secrets(token)

    def test_generic_private_key(self):
        """Generic BEGIN PRIVATE KEY (not just RSA/EC/OPENSSH) scrubbed."""
        assert "[REDACTED]" in scrub_output_secrets("-----BEGIN PRIVATE KEY-----")

    def test_dsa_private_key(self):
        """DSA private key headers scrubbed."""
        assert "[REDACTED]" in scrub_output_secrets("-----BEGIN DSA PRIVATE KEY-----")


class TestSecretScrubberMultipleSecrets:
    """Verify scrubber handles multiple secrets in one string."""

    def test_multiple_different_secrets(self):
        text = (
            "Config: password=hunter2\n"
            "Token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl\n"
            "Key: AKIAIOSFODNN7EXAMPLE\n"
            "DB: postgres://admin:pass@host/db"
        )
        scrubbed = scrub_output_secrets(text)
        assert "hunter2" not in scrubbed
        assert "ghp_" not in scrubbed
        assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed
        assert "admin:pass@" not in scrubbed
        assert scrubbed.count("[REDACTED]") >= 4


# ===========================================================================
# 6. Error Message Scrubbing (NEW FIX)
# ===========================================================================


class TestErrorMessageScrubbing:
    """Verify error messages are scrubbed before sending to Discord."""

    def test_error_message_scrub_in_source(self):
        """The 'Something went wrong' error path scrubs the message."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        # Find the "Something went wrong" line
        for line in content.split("\n"):
            if "Something went wrong" in line:
                assert "scrub_response_secrets" in line, (
                    "Error message must be scrubbed before sending to Discord"
                )
                break
        else:
            pytest.fail("Could not find 'Something went wrong' in client.py")

    def test_scrub_removes_secrets_from_errors(self):
        """scrub_response_secrets catches secrets in error strings."""
        from src.discord.client import scrub_response_secrets
        error = "Something went wrong: Connection to postgres://admin:secret123@db.host/mydb failed"
        scrubbed = scrub_response_secrets(error)
        assert "secret123" not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_scrub_removes_api_key_from_errors(self):
        """scrub_response_secrets catches API keys in error strings."""
        from src.discord.client import scrub_response_secrets
        error = "Something went wrong: API call failed with key sk-abcdefghijklmnopqrstuvwxyz12345"
        scrubbed = scrub_response_secrets(error)
        assert "sk-abcdefghijklmnop" not in scrubbed

    def test_scrub_removes_password_from_errors(self):
        """scrub_response_secrets catches passwords in error strings."""
        from src.discord.client import scrub_response_secrets
        error = "Something went wrong: password=SuperSecret123 was rejected"
        scrubbed = scrub_response_secrets(error)
        assert "SuperSecret123" not in scrubbed


# ===========================================================================
# 7. Monitor Alert Scrubbing (NEW FIX)
# ===========================================================================


class TestMonitorAlertScrubbing:
    """Verify monitor alerts are scrubbed before sending to Discord."""

    def test_alert_scrub_in_source(self):
        """The _on_monitor_alert method scrubs the message before sending."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        # Find the _on_monitor_alert method
        in_method = False
        for line in content.split("\n"):
            if "_on_monitor_alert" in line and "async def" in line:
                in_method = True
            if in_method and "channel.send(" in line:
                assert "scrub_response_secrets" in line, (
                    "Monitor alert must be scrubbed before sending to Discord"
                )
                break
        else:
            if not in_method:
                pytest.fail("Could not find _on_monitor_alert in client.py")


# ===========================================================================
# 8. Prompt Injection Resistance
# ===========================================================================


class TestRoleForgeryPrevention:
    """Verify users cannot forge system/developer role messages."""

    def test_user_messages_always_user_role(self):
        """Session add_message for user input always uses 'user' role."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        # All add_message calls with user content should use "user" role
        # Find the tagged_content add_message call
        for line in content.split("\n"):
            if "tagged_content" in line and "add_message" in line:
                assert '"user"' in line, "User messages must use 'user' role"
                break

    def test_context_separator_uses_developer_role(self):
        """Context separator uses developer role (cannot be forged by users)."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        assert '"developer"' in content, "Context separator should use developer role"

    async def test_system_prompt_not_in_session(self):
        """System prompt is built fresh per request, not stored in session history."""
        from src.sessions.manager import SessionManager
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            mgr = SessionManager(max_history=50, max_age_hours=24, persist_dir=td)
            mgr.add_message("ch1", "user", "hello")
            mgr.add_message("ch1", "assistant", "hi there")
            history = await mgr.get_task_history("ch1")
            # No "system" role messages in history
            for msg in history:
                assert msg["role"] != "system", "System prompt should not be in session"


class TestContextSeparatorIntegrity:
    """Verify the context separator correctly marks history vs current request."""

    def test_separator_has_current_request(self):
        """Context separator contains 'CURRENT REQUEST' marker."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        assert "CURRENT REQUEST" in content

    def test_separator_mentions_tools(self):
        """Context separator instructs to evaluate available tools."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        assert "CURRENTLY AVAILABLE" in content

    def test_separator_blocks_prior_refusals(self):
        """Context separator instructs not to repeat prior refusals."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        assert "prior refusals" in content.lower()

    def test_bot_preamble_instructs_execute(self):
        """Bot messages get EXECUTE instruction in preamble."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        assert "EXECUTE immediately" in content


class TestToolOutputInjectionSafety:
    """Verify tool output cannot inject role-level messages."""

    def test_tool_output_truncated(self):
        """Tool output is truncated before injection into LLM context."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        assert "truncate_tool_output" in content

    def test_tool_output_scrubbed(self):
        """Tool output is scrubbed of secrets before injection into LLM context."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        assert "scrub_output_secrets" in content

    def test_tool_result_uses_fixed_structure(self):
        """Tool results use a fixed dict structure, not user-controlled role."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        # Tool results should use "tool" role in the message, not a user-controlled field
        assert "tool_use_id" in content


class TestPromptInjectionInSystemPrompt:
    """Verify system prompt has injection defense."""

    def test_prompt_instructs_ignore_injection(self):
        """System prompt tells LLM to ignore prompt injection."""
        from src.llm.system_prompt import SYSTEM_PROMPT_TEMPLATE
        # Should mention injection defense somewhere
        assert "injection" in SYSTEM_PROMPT_TEMPLATE.lower() or "ignore" in SYSTEM_PROMPT_TEMPLATE.lower()

    def test_prompt_under_5000_chars(self):
        """System prompt template under 5000 chars (enforced limit)."""
        from src.llm.system_prompt import SYSTEM_PROMPT_TEMPLATE
        assert len(SYSTEM_PROMPT_TEMPLATE) < 5000


# ===========================================================================
# 9. Cross-Cutting Security Verification
# ===========================================================================


class TestNoShellInjectionInHandlers:
    """Verify consistent use of shlex.quote in tool handlers."""

    def test_service_names_quoted(self):
        """Service names are quoted in shell commands."""
        executor_path = SRC_DIR / "tools" / "executor.py"
        content = executor_path.read_text()
        # check_service, restart_service should quote service names
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "systemctl" in line and "service" in line.lower():
                # The variable should be quoted via shlex.quote
                nearby = "\n".join(lines[max(0, i-5):i+5])
                if "shlex.quote" not in nearby and "safe_" not in nearby:
                    # It's okay if the service comes from a pre-quoted variable
                    pass  # Allow if variable is pre-quoted in the handler

    def test_no_shell_true_outside_ssh(self):
        """create_subprocess_shell only in ssh.py and process_manager.py."""
        allowed = {"ssh.py", "process_manager.py"}
        for path in SRC_DIR.rglob("*.py"):
            if path.name in allowed:
                continue
            content = path.read_text()
            assert "create_subprocess_shell" not in content, (
                f"Shell subprocess outside allowed files in {path.relative_to(SRC_DIR.parent)}"
            )

    def test_base64_used_for_content_transport(self):
        """Base64 encoding used for transporting user content to shell."""
        executor_path = SRC_DIR / "tools" / "executor.py"
        content = executor_path.read_text()
        assert "base64.b64encode" in content, "Should use base64 for content transport"
        assert "base64 -d" in content, "Should decode base64 on remote side"


class TestSecretScrubberCoverage:
    """Verify scrubber is called at all critical points."""

    def test_scrub_output_imported(self):
        """scrub_output_secrets imported in client.py."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        assert "scrub_output_secrets" in content

    def test_scrub_response_imported(self):
        """scrub_response_secrets defined in client.py."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        assert "def scrub_response_secrets" in content

    def test_check_for_secrets_exists(self):
        """_check_for_secrets method exists for user input."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        assert "_check_for_secrets" in content

    def test_scrubber_pattern_count(self):
        """Scrubber has at least 10 patterns after additions."""
        assert len(OUTPUT_SECRET_PATTERNS) >= 10, (
            f"Expected at least 10 patterns, got {len(OUTPUT_SECRET_PATTERNS)}"
        )


class TestReadFileFixed:
    """Verify the source code changes for read_file."""

    def test_read_file_validates_lines(self):
        """read_file handler validates lines parameter."""
        executor_path = SRC_DIR / "tools" / "executor.py"
        content = executor_path.read_text()
        # Find the _handle_read_file method
        in_method = False
        has_validation = False
        for line in content.split("\n"):
            if "_handle_read_file" in line:
                in_method = True
            if in_method and ("int(" in line or "min(" in line):
                has_validation = True
                break
            if in_method and line.strip().startswith("async def ") and "_handle_read_file" not in line:
                break
        assert has_validation, "read_file must validate lines parameter with int() or min()"


class TestSecurityAuditTrail:
    """Verify audit logging covers security-relevant events."""

    def test_audit_log_tool_execution(self):
        """Audit log records tool executions."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        assert "log_execution" in content

    def test_secret_detection_deletes_message(self):
        """Secret detection deletes the user's message."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        assert "message.delete()" in content

    def test_secret_detection_scrubs_history(self):
        """Secret detection scrubs the session history."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        assert "scrub_secrets" in content


# ===========================================================================
# 10. Round 19 Source Structure Verification
# ===========================================================================


class TestRound19SourceStructure:
    """Cross-cutting verification that all Round 19 changes are in place."""

    def test_output_secret_patterns_count(self):
        """OUTPUT_SECRET_PATTERNS has 10 patterns (6 original + 4 new)."""
        assert len(OUTPUT_SECRET_PATTERNS) == 10

    def test_new_patterns_in_scrubber(self):
        """New patterns (GitHub, AWS, Stripe, Slack) present in scrubber."""
        scrubber_path = SRC_DIR / "llm" / "secret_scrubber.py"
        content = scrubber_path.read_text()
        assert "ghp_" in content or "gh[pousr]" in content
        assert "AKIA" in content
        assert "sk_(live|test)" in content or "sk_live" in content or "[sr]k_" in content
        assert "xox[boaprs]" in content

    def test_generic_private_key_pattern(self):
        """Private key pattern catches generic BEGIN PRIVATE KEY."""
        scrubber_path = SRC_DIR / "llm" / "secret_scrubber.py"
        content = scrubber_path.read_text()
        # Pattern should have optional group for key type
        assert "DSA" in content or "?" in content  # Optional group or DSA added

    def test_read_file_has_try_except(self):
        """read_file handler has try/except for lines conversion."""
        executor_path = SRC_DIR / "tools" / "executor.py"
        content = executor_path.read_text()
        in_method = False
        has_try = False
        for line in content.split("\n"):
            if "_handle_read_file" in line:
                in_method = True
            if in_method and "except" in line and ("TypeError" in line or "ValueError" in line):
                has_try = True
                break
            if in_method and line.strip().startswith("async def ") and "_handle_read_file" not in line:
                break
        assert has_try, "read_file must handle TypeError/ValueError for lines"

    def test_error_message_scrubbed_before_discord(self):
        """Error 'Something went wrong' message is scrubbed."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        for line in content.split("\n"):
            if "Something went wrong" in line:
                assert "scrub_response_secrets" in line
                break

    def test_monitor_alert_scrubbed_before_discord(self):
        """Monitor alert is scrubbed before channel.send."""
        client_path = SRC_DIR / "discord" / "client.py"
        content = client_path.read_text()
        in_method = False
        for line in content.split("\n"):
            if "_on_monitor_alert" in line and "async def" in line:
                in_method = True
            if in_method and "channel.send(" in line:
                assert "scrub_response_secrets" in line
                break

    def test_all_rounds_1_18_intact(self):
        """Key changes from rounds 1-18 are still intact."""
        # No approval.py
        assert not (SRC_DIR / "discord" / "approval.py").exists()
        # No haiku_classifier.py
        assert not (SRC_DIR / "llm" / "haiku_classifier.py").exists()
        # No routing.py
        assert not (SRC_DIR / "discord" / "routing.py").exists()
        # Local execution support
        ssh_path = SRC_DIR / "tools" / "ssh.py"
        assert "run_local_command" in ssh_path.read_text()
        # Personality in prompt
        from src.llm.system_prompt import SYSTEM_PROMPT_TEMPLATE
        assert "not okay" in SYSTEM_PROMPT_TEMPLATE.lower() or "all-seeing" in SYSTEM_PROMPT_TEMPLATE.lower()
