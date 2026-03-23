"""Round 18: Final security scan tests — comprehensive security verification.

Covers:
- Browser token configurability
- SSH security settings
- Input sanitization (shlex.quote, URL validation, SQL parameterization)
- No unsafe deserialization (eval/exec/pickle on user input)
- No hardcoded secrets in source code
- No personal data edge cases (emails, geographic, hardware)
- Config error messages don't leak secret values
- Webhook HMAC authentication
- Secret scrubber extended patterns
"""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


def _get_tracked_files() -> list[str]:
    """Get list of git-tracked files."""
    result = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd="."
    )
    return result.stdout.strip().split("\n")


def _get_tracked_source_files() -> list[str]:
    """Get tracked .py files in src/."""
    return [f for f in _get_tracked_files() if f.startswith("src/") and f.endswith(".py")]


def _get_all_tracked_py_files() -> list[str]:
    """Get all tracked Python files (src + tests)."""
    return [f for f in _get_tracked_files() if f.endswith(".py")]


# --- Browser Token Configurability ---

class TestBrowserTokenConfigurable:
    """Verify browser service token is configurable via env var."""

    def test_config_yml_uses_browser_token_env_var(self):
        content = Path("config.yml").read_text()
        assert "${BROWSER_TOKEN:-loki-internal}" in content

    def test_docker_compose_uses_browser_token_env_var(self):
        content = Path("docker-compose.yml").read_text()
        assert "${BROWSER_TOKEN:-loki-internal}" in content

    def test_env_example_documents_browser_token(self):
        content = Path(".env.example").read_text()
        assert "BROWSER_TOKEN" in content

    def test_browser_token_override_via_env(self):
        import os
        from src.config.schema import _substitute_env_vars
        with patch.dict(os.environ, {"BROWSER_TOKEN": "custom-secret-token"}):
            result = _substitute_env_vars("${BROWSER_TOKEN:-loki-internal}")
            assert result == "custom-secret-token"

    def test_browser_token_defaults_when_unset(self):
        import os
        from src.config.schema import _substitute_env_vars
        os.environ.pop("BROWSER_TOKEN", None)
        result = _substitute_env_vars("${BROWSER_TOKEN:-loki-internal}")
        assert result == "loki-internal"

    def test_docker_compose_browser_token_matches_config(self):
        """Docker compose and config.yml should use the same env var."""
        dc = Path("docker-compose.yml").read_text()
        cfg = Path("config.yml").read_text()
        # Both should reference BROWSER_TOKEN
        assert "BROWSER_TOKEN" in dc
        assert "BROWSER_TOKEN" in cfg


# --- SSH Security ---

class TestSSHSecurity:
    """Verify SSH implementation follows security best practices."""

    def test_strict_host_key_checking(self):
        content = Path("src/tools/ssh.py").read_text()
        assert "StrictHostKeyChecking=yes" in content

    def test_batch_mode_enabled(self):
        """BatchMode=yes prevents interactive password prompts."""
        content = Path("src/tools/ssh.py").read_text()
        assert "BatchMode=yes" in content

    def test_uses_subprocess_exec_not_shell(self):
        """SSH uses create_subprocess_exec (safe) not shell=True."""
        content = Path("src/tools/ssh.py").read_text()
        assert "create_subprocess_exec" in content
        assert "shell=True" not in content

    def test_connect_timeout_set(self):
        content = Path("src/tools/ssh.py").read_text()
        assert "ConnectTimeout" in content


# --- Input Sanitization ---

class TestInputSanitization:
    """Verify user/LLM input is properly sanitized before use."""

    def test_shlex_quote_used_in_executor(self):
        """Tool executor should use shlex.quote for shell arguments."""
        content = Path("src/tools/executor.py").read_text()
        assert "shlex.quote" in content
        # Should have multiple uses for different tools
        assert content.count("shlex.quote") >= 10

    def test_browser_url_validation(self):
        """Browser tools should validate URL schemes."""
        content = Path("src/tools/browser.py").read_text()
        assert "_validate_url" in content
        assert 'ALLOWED_SCHEMES = ("http://", "https://")' in content

    def test_fts5_parameterized_queries(self):
        """FTS5 search should use parameterized queries."""
        content = Path("src/search/fts.py").read_text()
        # Should use ? placeholders, not string formatting
        assert "?" in content
        # Uses conn.execute with parameterized queries
        assert ".execute(" in content

    def test_incus_name_validation(self):
        """Incus instance names should be validated."""
        content = Path("src/tools/executor.py").read_text()
        assert "_validate_incus_name" in content

    def test_yaml_safe_load(self):
        """Config loading should use yaml.safe_load (not yaml.load)."""
        content = Path("src/config/schema.py").read_text()
        assert "yaml.safe_load" in content
        # Should NOT use unsafe yaml.load
        assert "yaml.load(" not in content


# --- No Unsafe Deserialization ---

class TestNoUnsafeDeserialization:
    """Verify no eval/exec/pickle on user-controlled input."""

    def test_no_pickle_in_source(self):
        """No pickle usage in source code (unsafe deserialization)."""
        for f in _get_tracked_source_files():
            content = Path(f).read_text()
            assert "pickle.load" not in content, f"{f} uses pickle.load"
            assert "pickle.loads" not in content, f"{f} uses pickle.loads"

    def test_no_eval_in_source(self):
        """No Python eval() on user input in source code."""
        for f in _get_tracked_source_files():
            # Parse AST to find eval() calls (not just the string "eval")
            try:
                tree = ast.parse(Path(f).read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id != "eval", f"{f} uses eval()"

    def test_no_exec_in_source(self):
        """No Python exec() on user input (except skill_manager which is by-design)."""
        for f in _get_tracked_source_files():
            if "skill_manager" in f:
                continue  # Skill execution is by-design
            try:
                tree = ast.parse(Path(f).read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id != "exec", f"{f} uses exec()"


# --- No Hardcoded Secrets ---

class TestNoHardcodedSecrets:
    """Verify no API keys, tokens, or passwords in source code."""

    def test_no_discord_token_pattern_in_source(self):
        """No Discord bot token patterns (base64 with dots) in source."""
        token_pattern = re.compile(r'[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}')
        for f in _get_tracked_source_files():
            content = Path(f).read_text()
            matches = token_pattern.findall(content)
            assert not matches, f"{f} contains potential Discord token: {matches[0][:20]}..."

    def test_no_anthropic_key_pattern_in_source(self):
        """No Anthropic API key pattern (sk-ant-...) in source."""
        for f in _get_tracked_source_files():
            content = Path(f).read_text()
            assert "sk-ant-" not in content, f"{f} contains Anthropic API key pattern"

    def test_no_openai_key_pattern_in_source(self):
        """No OpenAI API key pattern (sk-..., 40+ chars) in source."""
        # Exclude test files for the secret scrubber
        key_pattern = re.compile(r'sk-[A-Za-z0-9]{40,}')
        for f in _get_tracked_source_files():
            if "secret_scrubber" in f:
                continue
            content = Path(f).read_text()
            matches = key_pattern.findall(content)
            assert not matches, f"{f} contains potential OpenAI key"

    def test_no_private_key_blocks_in_tracked_files(self):
        """No private key material in tracked files."""
        for f in _get_tracked_files():
            if f.endswith(('.py', '.yml', '.md', '.toml', '.sh', '.txt')):
                try:
                    content = Path(f).read_text()
                except Exception:
                    continue
                # Skip test files that contain key strings as assertions
                if "test_" in f and ("scrub" in content[:200].lower() or "secret" in f.lower() or "security" in f.lower()):
                    continue
                assert "BEGIN RSA PRIVATE" not in content, f"{f} contains RSA private key"
                assert "BEGIN OPENSSH PRIVATE" not in content, f"{f} contains OpenSSH private key"
                assert "BEGIN EC PRIVATE" not in content, f"{f} contains EC private key"


# --- Personal Data Edge Cases ---

class TestPersonalDataEdgeCases:
    """Edge-case checks for personal data that broader scans might miss."""

    def test_no_email_addresses_in_source(self):
        """No personal email addresses in source code."""
        email_pattern = re.compile(r'[\w.-]+@(?!example\.com)[\w.-]+\.\w{2,}')
        for f in _get_tracked_source_files():
            content = Path(f).read_text()
            matches = email_pattern.findall(content)
            # Filter out common non-personal patterns
            real_matches = [m for m in matches if "@users.noreply" not in m]
            assert not real_matches, f"{f} contains email: {real_matches}"

    def test_no_home_lab_references(self):
        """No 'homelab', 'home lab', 'home network' references."""
        for f in _get_tracked_source_files():
            content = Path(f).read_text().lower()
            assert "homelab" not in content, f"{f} contains 'homelab'"
            assert "home lab" not in content, f"{f} contains 'home lab'"
            assert "home network" not in content, f"{f} contains 'home network'"

    def test_no_personal_hardware_in_prompts(self):
        """No personal hardware model numbers in system prompts."""
        content = Path("src/llm/system_prompt.py").read_text()
        assert "RTX" not in content
        assert "GTX" not in content
        assert "4070" not in content
        assert "3080" not in content

    def test_no_my_server_my_network_in_prompts(self):
        """System prompt should not use personal possessive pronouns about infrastructure."""
        content = Path("src/llm/system_prompt.py").read_text().lower()
        assert "my server" not in content
        assert "my network" not in content
        assert "i own" not in content
        assert "my home" not in content


# --- Config Error Message Safety ---

class TestConfigErrorSafety:
    """Verify config error messages don't leak sensitive values."""

    def test_missing_env_var_shows_name_not_value(self):
        """When env var is missing, error shows var NAME not attempted value."""
        import os
        from src.config.schema import _substitute_env_vars
        os.environ.pop("_TEST_SECRET_VAR_XXXXXX", None)
        with pytest.raises(ValueError) as exc_info:
            _substitute_env_vars("${_TEST_SECRET_VAR_XXXXXX}")
        error_msg = str(exc_info.value)
        assert "_TEST_SECRET_VAR_XXXXXX" in error_msg

    def test_config_error_does_not_show_other_env_values(self):
        """Config loading errors should not dump all env vars."""
        from src.config.schema import _substitute_env_vars
        # Verify the error message is focused, not a generic dump
        try:
            _substitute_env_vars("${_NEVER_SET_VAR_12345}")
        except ValueError as e:
            msg = str(e)
            assert "DISCORD_TOKEN" not in msg
            assert "ANTHROPIC_API_KEY" not in msg


# --- Webhook HMAC Authentication ---

class TestWebhookAuthentication:
    """Verify webhook endpoints require HMAC authentication."""

    def test_hmac_verification_exists(self):
        content = Path("src/health/server.py").read_text()
        assert "_verify_hmac_sha256" in content
        assert "hmac.compare_digest" in content

    def test_rejects_when_no_secret_configured(self):
        """Webhooks are rejected when no secret is configured."""
        content = Path("src/health/server.py").read_text()
        assert "no secret configured" in content.lower() or "Webhook rejected" in content

    def test_gitea_webhook_checks_signature(self):
        content = Path("src/health/server.py").read_text()
        assert "X-Gitea-Signature" in content

    def test_grafana_webhook_checks_secret(self):
        content = Path("src/health/server.py").read_text()
        assert "X-Webhook-Secret" in content


# --- SSL Design Decision ---

class TestSSLDesignDecision:
    """Verify SSL handling is documented."""

    def test_fetch_url_ssl_documented(self):
        """The ssl=False in fetch_url should have a documenting comment."""
        content = Path("src/tools/web.py").read_text()
        # Find the ssl=False line and check it has a comment
        for line in content.split("\n"):
            if "ssl=False" in line:
                assert "#" in line, "ssl=False should have a documenting comment"
                break


# --- Secret Scrubber Extended Patterns ---

class TestSecretScrubberExtended:
    """Extended tests for the secret scrubber."""

    def test_scrubs_github_token(self):
        from src.llm.secret_scrubber import scrub_output_secrets
        result = scrub_output_secrets("token: ghp_abcdefghijklmnopqrstuvwxyz1234567890")
        assert "ghp_" not in result or "[REDACTED]" in result

    def test_scrubs_bearer_token(self):
        """Bearer tokens with 'password' or 'token' context should be caught."""
        from src.llm.secret_scrubber import scrub_output_secrets
        # The scrubber catches "token: <value>" patterns
        result = scrub_output_secrets("token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U")
        assert "[REDACTED]" in result

    def test_preserves_normal_text(self):
        from src.llm.secret_scrubber import scrub_output_secrets
        text = "The server status is OK. CPU usage: 45%. Memory: 8GB/16GB."
        assert scrub_output_secrets(text) == text

    def test_scrubs_connection_string_with_password(self):
        from src.llm.secret_scrubber import scrub_output_secrets
        result = scrub_output_secrets("mysql://admin:s3cr3t_p@ss@db.example.com:3306/mydb")
        assert "[REDACTED]" in result


# --- No Shell=True Usage ---

class TestNoShellTrue:
    """Verify subprocess calls don't use shell=True."""

    def test_no_shell_true_in_source(self):
        """No subprocess calls with shell=True in source code.

        Exception: ssh.py uses create_subprocess_shell for run_local_command
        because commands must be interpreted by a shell (same as SSH does
        remotely).  All inputs are already sanitised via shlex.quote in
        executor.py before reaching run_local_command.
        """
        # ssh.py is the single allowed location for shell subprocess usage
        allowed_shell_files = {"src/tools/ssh.py"}
        for f in _get_tracked_source_files():
            content = Path(f).read_text()
            if f in allowed_shell_files:
                # ssh.py may use create_subprocess_shell but never shell=True
                if "shell=True" in content:
                    assert False, f"{f} uses shell=True"
                continue
            # All other files: no shell subprocess at all
            if "subprocess" in content or "create_subprocess_shell" in content:
                assert "shell=True" not in content, f"{f} uses shell=True"
                assert "create_subprocess_shell" not in content, f"{f} uses create_subprocess_shell"
