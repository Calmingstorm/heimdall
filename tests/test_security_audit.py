"""Round 9: Security audit tests — verify no personal data or secrets remain.

These tests scan tracked source, config, and test files to ensure:
- No personal IPs, user IDs, bot IDs, webhook IDs, or channel IDs
- No personal names, paths, or hostnames in source/config
- No hardcoded secrets or credentials
- .gitignore covers all sensitive file patterns
- .env.example is safe (no real credentials)
- Secret scrubber catches common credential patterns
- No niche personal tool references remain
"""
from __future__ import annotations

import subprocess
from pathlib import Path

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


def _get_tracked_test_files() -> list[str]:
    """Get tracked .py files in tests/ (excluding audit/verification tests that
    contain personal data strings as negative assertions)."""
    # These files contain strings like assert "192.168.1" not in ...
    # or assert "AnsiblexBot" not in ... — verification tests, not leaks.
    excluded = {
        "tests/test_config.py",
        "tests/test_security_audit.py",
        "tests/test_full_verification.py",
    }
    return [
        f for f in _get_tracked_files()
        if f.startswith("tests/") and f.endswith(".py") and f not in excluded
    ]


class TestNoPersonalIPs:
    """Verify no personal 192.168.1.x IPs in any tracked file."""

    def test_no_personal_ips_in_source(self):
        for f in _get_tracked_source_files():
            content = Path(f).read_text()
            assert "192.168.1" not in content, f"{f} contains personal IP"

    def test_no_personal_ips_in_tests(self):
        for f in _get_tracked_test_files():
            content = Path(f).read_text()
            assert "192.168.1" not in content, f"{f} contains personal IP"

    def test_no_personal_ips_in_config(self):
        for name in ["config.yml", ".env.example", "docker-compose.yml", "Dockerfile"]:
            p = Path(name)
            if p.exists():
                content = p.read_text()
                assert "192.168.1" not in content, f"{name} contains personal IP"


class TestNoPersonalDiscordIDs:
    """Verify no personal Discord snowflake IDs in tracked files."""

    PERSONAL_IDS = [
        "441602773310767105",   # personal user ID
        "757383353141035140",   # personal user ID
        "1469121766910726210",  # bot ID
        "1485046995650482406",  # webhook ID
        "1469135502115606792",  # channel ID
        "1483270529061228586",  # channel ID
        "1485046396775043142",  # channel ID
        "1481679552181698610",  # channel ID
        "1482174235743879249",  # channel ID
    ]

    def test_no_personal_ids_in_source(self):
        for f in _get_tracked_source_files():
            content = Path(f).read_text()
            for pid in self.PERSONAL_IDS:
                assert pid not in content, f"{f} contains personal ID {pid}"

    def test_no_personal_ids_in_config(self):
        for name in ["config.yml", ".env.example"]:
            p = Path(name)
            if p.exists():
                content = p.read_text()
                for pid in self.PERSONAL_IDS:
                    assert pid not in content, f"{name} contains personal ID {pid}"


class TestNoPersonalNames:
    """Verify no personal names in source or config files."""

    def test_no_calmingstorm_in_source(self):
        for f in _get_tracked_source_files():
            content = Path(f).read_text().lower()
            assert "calmingstorm" not in content, f"{f} contains 'calmingstorm'"

    def test_no_calmingstorm_in_tests(self):
        for f in _get_tracked_test_files():
            content = Path(f).read_text().lower()
            assert "calmingstorm" not in content, f"{f} contains 'calmingstorm'"

    def test_no_audrastaia_in_tracked_files(self):
        for f in _get_tracked_source_files() + _get_tracked_test_files():
            content = Path(f).read_text().lower()
            assert "audrastaia" not in content, f"{f} contains 'audrastaia'"

    def test_no_personal_names_in_source(self):
        """Check source files for personal names used as identifiers (not generic words)."""
        import re
        for f in _get_tracked_source_files():
            content = Path(f).read_text()
            # "Aaron" as a standalone word (not part of another word)
            matches = re.findall(r'\baaron\b', content, re.IGNORECASE)
            assert not matches, f"{f} contains personal name 'Aaron'"

    def test_no_personal_names_in_tests(self):
        """Check test files for personal names used as identifiers."""
        import re
        for f in _get_tracked_test_files():
            content = Path(f).read_text()
            matches = re.findall(r'\baaron\b', content, re.IGNORECASE)
            assert not matches, f"{f} contains personal name 'Aaron'"


class TestNoAnsiblexReferences:
    """Verify 'ansiblex' is fully renamed to 'loki' everywhere."""

    def test_no_ansiblex_in_source(self):
        for f in _get_tracked_source_files():
            content = Path(f).read_text().lower()
            assert "ansiblex" not in content, f"{f} still contains 'ansiblex'"

    def test_no_ansiblex_in_tests(self):
        for f in _get_tracked_test_files():
            content = Path(f).read_text().lower()
            assert "ansiblex" not in content, f"{f} still contains 'ansiblex'"

    def test_no_ansiblex_in_config(self):
        for name in ["config.yml", ".env.example", "docker-compose.yml", "Dockerfile"]:
            p = Path(name)
            if p.exists():
                content = p.read_text().lower()
                assert "ansiblex" not in content, f"{name} still contains 'ansiblex'"

    def test_no_ansiblex_in_pyproject(self):
        content = Path("pyproject.toml").read_text().lower()
        assert "ansiblex" not in content


class TestNoPersonalPaths:
    """Verify no personal paths remain in source or config."""

    def test_no_personal_paths_in_source(self):
        for f in _get_tracked_source_files():
            content = Path(f).read_text()
            assert "/root/ansiblex" not in content, f"{f} contains /root/ansiblex"
            assert "/opt/ansiblex" not in content, f"{f} contains /opt/ansiblex"

    def test_no_personal_paths_in_config(self):
        for name in ["config.yml", "docker-compose.yml"]:
            p = Path(name)
            if p.exists():
                content = p.read_text()
                assert "/root/ansiblex" not in content, f"{name} contains /root/ansiblex"
                assert "/opt/ansiblex" not in content, f"{name} contains /opt/ansiblex"


class TestNoGiteaURLs:
    """Verify no personal Gitea URLs remain."""

    def test_no_gitea_lan_in_source(self):
        for f in _get_tracked_source_files():
            content = Path(f).read_text()
            assert "gitea.lan" not in content, f"{f} contains gitea.lan"

    def test_no_gitea_port_in_source(self):
        for f in _get_tracked_source_files():
            content = Path(f).read_text()
            assert "192.168.1.13:3300" not in content, f"{f} contains personal Gitea URL"
            assert ":3300" not in content, f"{f} contains personal Gitea port 3300"


class TestNoNicheToolReferences:
    """Verify no niche/personal tool references remain in routing."""

    def test_no_siglos_in_source(self):
        """Siglos was a personal project management tool — should be removed."""
        for f in _get_tracked_source_files():
            content = Path(f).read_text().lower()
            assert "siglos" not in content, f"{f} contains niche 'siglos' reference"

    def test_no_siglos_in_tests(self):
        for f in _get_tracked_test_files():
            content = Path(f).read_text().lower()
            assert "siglos" not in content, f"{f} contains niche 'siglos' reference"


class TestGitignoreCoverage:
    """Verify .gitignore covers all sensitive file patterns."""

    def test_env_file_ignored(self):
        content = Path(".gitignore").read_text()
        assert ".env" in content

    def test_ssh_directory_ignored(self):
        content = Path(".gitignore").read_text()
        assert "ssh/" in content

    def test_runtime_data_ignored(self):
        content = Path(".gitignore").read_text()
        assert "data/sessions/" in content
        assert "data/logs/" in content
        assert "data/audit.jsonl" in content

    def test_user_context_files_ignored(self):
        content = Path(".gitignore").read_text()
        assert "data/context/*.md" in content

    def test_user_skills_ignored(self):
        content = Path(".gitignore").read_text()
        assert "data/skills/*.py" in content

    def test_credentials_file_ignored(self):
        content = Path(".gitignore").read_text()
        assert "data/codex_auth.json" in content


class TestDockerignoreCoverage:
    """Verify .dockerignore prevents sensitive files from entering images."""

    def test_env_excluded(self):
        content = Path(".dockerignore").read_text()
        assert ".env" in content

    def test_ssh_excluded(self):
        content = Path(".dockerignore").read_text()
        assert "ssh/" in content

    def test_git_excluded(self):
        content = Path(".dockerignore").read_text()
        assert ".git/" in content


class TestEnvExampleSafety:
    """Verify .env.example contains only placeholder values."""

    def test_no_real_tokens(self):
        content = Path(".env.example").read_text()
        # Should contain placeholder text, not real tokens
        assert "your-discord-bot-token" in content
        assert "your-anthropic-api-key" in content

    def test_no_real_discord_token_format(self):
        """Real Discord tokens are base64-encoded, not plaintext."""
        import re
        content = Path(".env.example").read_text()
        # Discord tokens are typically 59+ chars of base64
        tokens = re.findall(r'DISCORD_TOKEN=(\S+)', content)
        for token in tokens:
            assert len(token) < 50, "DISCORD_TOKEN looks like a real token"
            assert "." not in token, "DISCORD_TOKEN looks like a real JWT token"


class TestSecretScrubber:
    """Verify the secret scrubber catches common credential patterns."""

    def test_scrubs_password(self):
        from src.llm.secret_scrubber import scrub_output_secrets
        assert "[REDACTED]" in scrub_output_secrets("password: hunter2")

    def test_scrubs_api_key(self):
        from src.llm.secret_scrubber import scrub_output_secrets
        assert "[REDACTED]" in scrub_output_secrets("api_key: sk-1234567890abcdef")

    def test_scrubs_private_key(self):
        from src.llm.secret_scrubber import scrub_output_secrets
        assert "[REDACTED]" in scrub_output_secrets("BEGIN RSA PRIVATE KEY")

    def test_scrubs_database_url(self):
        from src.llm.secret_scrubber import scrub_output_secrets
        result = scrub_output_secrets("postgres://user:pass@host/db")
        assert "[REDACTED]" in result

    def test_scrubs_sk_token(self):
        from src.llm.secret_scrubber import scrub_output_secrets
        result = scrub_output_secrets("using key sk-abcdefghijklmnopqrstuvwxyz1234")
        assert "[REDACTED]" in result

    def test_preserves_safe_text(self):
        from src.llm.secret_scrubber import scrub_output_secrets
        safe = "Hello, the server is running fine."
        assert scrub_output_secrets(safe) == safe


class TestConfigSecretHandling:
    """Verify config loading handles secrets safely."""

    def test_required_env_var_raises_when_missing(self):
        """Required env vars (no default) raise ValueError when unset."""
        import os
        from src.config.schema import _substitute_env_vars
        os.environ.pop("_NONEXISTENT_SECRET_VAR", None)
        with pytest.raises(ValueError, match="_NONEXISTENT_SECRET_VAR"):
            _substitute_env_vars("${_NONEXISTENT_SECRET_VAR}")

    def test_optional_env_var_defaults_safely(self):
        """Optional env vars use the default when unset."""
        import os
        from src.config.schema import _substitute_env_vars
        os.environ.pop("_NONEXISTENT_SECRET_VAR", None)
        result = _substitute_env_vars("${_NONEXISTENT_SECRET_VAR:-}")
        assert result == ""

    def test_config_yml_uses_env_vars_for_secrets(self):
        """config.yml should use ${VAR} syntax for all secrets, not hardcoded values."""
        content = Path("config.yml").read_text()
        assert "${DISCORD_TOKEN}" in content
        assert "${ANTHROPIC_API_KEY}" in content


class TestNoHardcodedTimezone:
    """Verify timezone is configurable, not hardcoded."""

    def test_no_hardcoded_timezone_in_source(self):
        """No source file should hardcode America/New_York."""
        for f in _get_tracked_source_files():
            content = Path(f).read_text()
            assert "America/New_York" not in content, (
                f"{f} contains hardcoded timezone"
            )

    def test_docker_compose_uses_env_var_timezone(self):
        content = Path("docker-compose.yml").read_text()
        assert "${TZ:-UTC}" in content
        assert "America/New_York" not in content
