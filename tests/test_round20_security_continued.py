"""Round 20: Security review (continued) — additional scrubbing paths, webhook payload
scrubbing, knowledge base ingestion/search scrubbing, edge cases for Round 19 patterns.

Tests cover: webhook payload secret scrubbing, knowledge base search result scrubbing,
digest/scheduled/workflow/skill output scrubbing, secret scrubber edge cases for all
10 patterns, Round 19 fix verification (read_file, incus_exec), and source structure
verification of all scrubbing call sites.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.secret_scrubber import OUTPUT_SECRET_PATTERNS, scrub_output_secrets
from src.discord.client import scrub_response_secrets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


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
            incus_host="server1",
        ),
        sessions=SimpleNamespace(max_history=50, max_age_hours=72, persist_dir="/tmp/test_sessions"),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ===========================================================================
# 1. Webhook Payload Scrubbing
# ===========================================================================


class TestWebhookPayloadScrubbing:
    """Verify webhook payloads are scrubbed before going to Discord."""

    def test_webhook_send_imports_scrubber(self):
        """__main__.py imports scrub_response_secrets."""
        source = (SRC_DIR / "__main__.py").read_text()
        assert "scrub_response_secrets" in source

    def test_webhook_send_uses_scrubber(self):
        """_webhook_send calls scrub_response_secrets before channel.send."""
        source = (SRC_DIR / "__main__.py").read_text()
        assert "channel.send(scrub_response_secrets(text))" in source

    def test_gitea_push_with_secret_in_commit(self):
        """Gitea push commit message containing a password gets scrubbed."""
        text = "**Gitea Push** — `myrepo` (`main`)\nBy: dev | 1 commit(s)\n  • `abc1234` Set password=SuperSecret123"
        scrubbed = scrub_response_secrets(text)
        assert "SuperSecret123" not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_gitea_push_with_api_key_in_commit(self):
        """Gitea push commit message containing an API key gets scrubbed."""
        text = "**Gitea Push** — `myrepo` (`main`)\n  • `abc1234` api_key=sk-1234567890abcdef1234567890abcdef"
        scrubbed = scrub_response_secrets(text)
        assert "sk-1234567890abcdef" not in scrubbed

    def test_grafana_alert_with_db_uri(self):
        """Grafana alert containing a database URI in summary gets scrubbed."""
        text = (
            "**Grafana Alerts** (1 alert(s)):\n"
            "🔴 **DBDown** (firing)\n"
            "  Connection failed: postgres://admin:s3cret@db.internal:5432/prod"
        )
        scrubbed = scrub_response_secrets(text)
        assert "s3cret" not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_generic_webhook_with_token(self):
        """Generic webhook message with a GitHub token gets scrubbed."""
        text = "**Deploy** — Token used: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl"
        scrubbed = scrub_response_secrets(text)
        assert "ghp_ABCDEFGH" not in scrubbed

    def test_clean_webhook_unchanged(self):
        """Webhook text without secrets passes through unchanged."""
        text = "**Gitea Push** — `myrepo` (`main`)\nBy: dev | 1 commit(s)\n  • `abc1234` Fix README typo"
        assert scrub_response_secrets(text) == text


# ===========================================================================
# 2. Knowledge Base Search Result Scrubbing
# ===========================================================================


class TestKnowledgeBaseSearchScrubbing:
    """Verify knowledge base search results are scrubbed before LLM sees them."""

    def test_search_knowledge_scrubs_content(self):
        """_handle_search_knowledge scrubs content via scrub_output_secrets."""
        source = (SRC_DIR / "discord" / "client.py").read_text()
        assert "scrub_output_secrets(r[\"content\"]" in source

    def test_scrub_output_catches_password_in_chunk(self):
        """Secrets in knowledge base chunks are redacted."""
        chunk = "server config: password=MyDbP@ss123 port=5432"
        scrubbed = scrub_output_secrets(chunk)
        assert "MyDbP@ss123" not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_scrub_output_catches_api_key_in_chunk(self):
        """API keys in knowledge base chunks are redacted."""
        chunk = "api_key: sk-abcdef1234567890abcdef1234567890"
        scrubbed = scrub_output_secrets(chunk)
        assert "sk-abcdef" not in scrubbed

    def test_scrub_output_catches_db_uri_in_chunk(self):
        """Database URIs in knowledge base chunks are redacted."""
        chunk = "DATABASE_URL=postgres://user:pass123@db:5432/mydb"
        scrubbed = scrub_output_secrets(chunk)
        assert "pass123" not in scrubbed

    def test_clean_chunk_unchanged(self):
        """Clean knowledge base content passes through unchanged."""
        chunk = "To restart the service, run: systemctl restart nginx"
        assert scrub_output_secrets(chunk) == chunk


# ===========================================================================
# 3. Digest Output Scrubbing
# ===========================================================================


class TestDigestOutputScrubbing:
    """Verify digest error and summary outputs are scrubbed."""

    def test_digest_error_scrubbed(self):
        """Digest error message uses scrub_response_secrets."""
        source = (SRC_DIR / "discord" / "client.py").read_text()
        assert 'scrub_response_secrets(f"**Daily Infrastructure Digest**\\n\\nFailed to collect data: {e}")' in source

    def test_digest_summary_scrubbed(self):
        """Digest summary message uses scrub_response_secrets."""
        source = (SRC_DIR / "discord" / "client.py").read_text()
        assert 'scrub_response_secrets(f"**Daily Infrastructure Digest**\\n\\n{summary}")' in source

    def test_digest_error_with_db_uri(self):
        """Digest error containing a DB URI gets scrubbed."""
        error_msg = f"**Daily Infrastructure Digest**\n\nFailed to collect data: Connection refused: postgres://admin:secret@db:5432/prod"
        scrubbed = scrub_response_secrets(error_msg)
        assert "secret" not in scrubbed.lower().split("redacted")[0]  # secret before REDACTED should be gone
        assert "[REDACTED]" in scrubbed

    def test_digest_summary_with_secret_in_output(self):
        """Digest summary containing a leaked secret gets scrubbed."""
        summary = "All services healthy. Note: api_key=sk-test1234567890abcdef1234567890 found in /etc/config"
        full = f"**Daily Infrastructure Digest**\n\n{summary}"
        scrubbed = scrub_response_secrets(full)
        assert "sk-test1234" not in scrubbed


# ===========================================================================
# 4. Scheduled Task Scrubbing
# ===========================================================================


class TestScheduledTaskScrubbing:
    """Verify scheduled check results and errors are scrubbed."""

    def test_scheduled_check_result_scrubbed(self):
        """Scheduled check result uses scrub_response_secrets."""
        source = (SRC_DIR / "discord" / "client.py").read_text()
        assert "await channel.send(scrub_response_secrets(text))" in source

    def test_scheduled_check_error_scrubbed(self):
        """Scheduled check error uses scrub_response_secrets."""
        source = (SRC_DIR / "discord" / "client.py").read_text()
        # Error path in scheduled task
        assert 'scrub_response_secrets(f"**Scheduled task failed:**' in source

    def test_scheduled_result_with_password(self):
        """Scheduled check result containing a password gets scrubbed."""
        result = "cat /etc/myapp/config.yml\ndb:\n  password=hunter2\n  host=localhost"
        text = f"**Scheduled: check config**\n```\n{result[:1800]}\n```"
        scrubbed = scrub_response_secrets(text)
        assert "hunter2" not in scrubbed

    def test_scheduled_error_with_connection_string(self):
        """Scheduled task error containing a connection string gets scrubbed."""
        error = "Connection failed: mongodb://admin:p@ssw0rd@mongo:27017/prod"
        text = f"**Scheduled task failed:** check db\nError: {error}"
        scrubbed = scrub_response_secrets(text)
        assert "p@ssw0rd" not in scrubbed


# ===========================================================================
# 5. Workflow Results Scrubbing
# ===========================================================================


class TestWorkflowResultsScrubbing:
    """Verify workflow results are scrubbed before posting to Discord."""

    def test_workflow_send_scrubbed(self):
        """Workflow results use scrub_response_secrets before channel.send."""
        source = (SRC_DIR / "discord" / "client.py").read_text()
        # The workflow send call
        assert "await channel.send(scrub_response_secrets(text))" in source

    def test_workflow_result_with_aws_key(self):
        """Workflow result containing an AWS key gets scrubbed."""
        text = "**Workflow: Deploy**\nStep 1: Using AKIAIOSFODNN7EXAMPLE for auth\nStep 2: Deployed"
        scrubbed = scrub_response_secrets(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed

    def test_workflow_result_with_stripe_key(self):
        """Workflow result containing a Stripe key gets scrubbed."""
        text = "**Workflow: Payment setup**\nConfigured: sk_live_abcdefghijklmnopqrstuvwx"
        scrubbed = scrub_response_secrets(text)
        assert "sk_live_abcdef" not in scrubbed


# ===========================================================================
# 6. Skill Message Callback Scrubbing
# ===========================================================================


class TestSkillMessageCallbackScrubbing:
    """Verify skill message callback scrubs output."""

    def test_skill_msg_callback_scrubbed(self):
        """Skill _skill_msg callback uses scrub_response_secrets."""
        source = (SRC_DIR / "discord" / "client.py").read_text()
        assert "await message.channel.send(scrub_response_secrets(text))" in source

    def test_skill_output_with_slack_token(self):
        """Skill output containing a Slack token gets scrubbed."""
        text = "Slack integration configured with token: xoxb-1234567890-abcdefghij"
        scrubbed = scrub_response_secrets(text)
        assert "xoxb-1234567890" not in scrubbed


# ===========================================================================
# 7. Secret Scrubber Edge Cases — Round 19 Patterns
# ===========================================================================


class TestSecretScrubberEdgeCases:
    """Edge cases for Round 19 secret patterns."""

    # GitHub tokens
    def test_github_ghp_min_length(self):
        """GitHub personal access token at minimum length (36 chars) is caught."""
        token = "ghp_" + "A" * 36
        assert "[REDACTED]" in scrub_output_secrets(f"token: {token}")

    def test_github_ghp_short_not_caught(self):
        """Short string starting with ghp_ is NOT caught (not a valid token)."""
        text = "ghp_short"
        assert scrub_output_secrets(text) == text

    def test_github_gho_token(self):
        """GitHub OAuth token caught."""
        token = "gho_" + "a" * 40
        assert "[REDACTED]" in scrub_output_secrets(token)

    def test_github_ghu_token(self):
        """GitHub user-to-server token caught."""
        token = "ghu_" + "B" * 36
        assert "[REDACTED]" in scrub_output_secrets(token)

    def test_github_ghs_token(self):
        """GitHub server-to-server token caught."""
        token = "ghs_" + "C" * 36
        assert "[REDACTED]" in scrub_output_secrets(token)

    def test_github_ghr_token(self):
        """GitHub refresh token caught."""
        token = "ghr_" + "D" * 36
        assert "[REDACTED]" in scrub_output_secrets(token)

    # AWS
    def test_aws_access_key_exact_length(self):
        """AWS access key ID with exactly 16 chars after AKIA is caught."""
        key = "AKIA" + "A" * 16
        assert "[REDACTED]" in scrub_output_secrets(key)

    def test_aws_access_key_lowercase_not_caught(self):
        """AKIA with lowercase chars is not a valid AWS key format."""
        key = "AKIAabcdefghijklmnop"
        assert scrub_output_secrets(key) == key

    def test_aws_access_key_in_env_var(self):
        """AWS key in environment variable assignment is caught."""
        text = "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        scrubbed = scrub_output_secrets(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed

    # Stripe
    def test_stripe_sk_live_key(self):
        """Stripe live secret key caught."""
        key = "sk_live_" + "a" * 24
        assert "[REDACTED]" in scrub_output_secrets(key)

    def test_stripe_sk_test_key(self):
        """Stripe test secret key caught."""
        key = "sk_test_" + "b" * 20
        assert "[REDACTED]" in scrub_output_secrets(key)

    def test_stripe_rk_live_key(self):
        """Stripe restricted live key caught."""
        key = "rk_live_" + "c" * 24
        assert "[REDACTED]" in scrub_output_secrets(key)

    def test_stripe_sk_short_not_caught(self):
        """Stripe key with too-short random part not caught."""
        text = "sk_live_abc"
        assert scrub_output_secrets(text) == text

    # Slack
    def test_slack_xoxb_token(self):
        """Slack bot token caught."""
        token = "xoxb-123456789012-abcdefghij"
        assert "[REDACTED]" in scrub_output_secrets(token)

    def test_slack_xoxp_token(self):
        """Slack user token caught."""
        token = "xoxp-999-888-777-abcdefghijklmno"
        assert "[REDACTED]" in scrub_output_secrets(token)

    def test_slack_xoxa_token(self):
        """Slack app-level token caught."""
        token = "xoxa-2-abcdefghijklmno"
        assert "[REDACTED]" in scrub_output_secrets(token)

    def test_slack_xoxo_token(self):
        """Slack org-level token caught."""
        token = "xoxo-123-abc"
        assert "[REDACTED]" in scrub_output_secrets(token)

    def test_slack_xoxr_token(self):
        """Slack refresh token caught."""
        token = "xoxr-abc-def-ghi"
        assert "[REDACTED]" in scrub_output_secrets(token)

    def test_slack_xoxs_token(self):
        """Slack session token caught."""
        token = "xoxs-session-abc123"
        assert "[REDACTED]" in scrub_output_secrets(token)

    # Private keys
    def test_generic_begin_private_key(self):
        """Generic BEGIN PRIVATE KEY (no type prefix) caught."""
        text = "-----BEGIN PRIVATE KEY-----\nMIIE..."
        assert "[REDACTED]" in scrub_output_secrets(text)

    def test_dsa_private_key(self):
        """DSA private key caught."""
        text = "-----BEGIN DSA PRIVATE KEY-----\nMIIE..."
        assert "[REDACTED]" in scrub_output_secrets(text)

    def test_rsa_private_key(self):
        """RSA private key still caught (pre-existing)."""
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIE..."
        assert "[REDACTED]" in scrub_output_secrets(text)

    def test_ec_private_key(self):
        """EC private key still caught (pre-existing)."""
        text = "-----BEGIN EC PRIVATE KEY-----\nMIIE..."
        assert "[REDACTED]" in scrub_output_secrets(text)

    def test_openssh_private_key(self):
        """OPENSSH private key still caught (pre-existing)."""
        text = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3Blbn..."
        assert "[REDACTED]" in scrub_output_secrets(text)

    # Database URIs
    def test_mysql_uri(self):
        """MySQL URI with credentials caught."""
        text = "mysql://root:password123@db:3306/mydb"
        assert "[REDACTED]" in scrub_output_secrets(text)

    def test_mongodb_srv_uri(self):
        """MongoDB+SRV URI with credentials caught."""
        text = "mongodb+srv://admin:secret@cluster.mongodb.net/db"
        assert "[REDACTED]" in scrub_output_secrets(text)


# ===========================================================================
# 8. Multiple Secrets in One Message
# ===========================================================================


class TestMultipleSecretsInOneMessage:
    """Verify multiple different secret types in one string are all caught."""

    def test_four_types_in_one_string(self):
        """Four different secret types in one message all get scrubbed."""
        text = (
            "Config dump:\n"
            "password=hunter2\n"
            "GITHUB_TOKEN=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij01\n"
            "AWS_KEY=AKIAIOSFODNN7EXAMPLE\n"
            "STRIPE=sk_live_abcdefghijklmnopqrstuv\n"
        )
        scrubbed = scrub_response_secrets(text)
        assert "hunter2" not in scrubbed
        assert "ghp_ABCDEF" not in scrubbed
        assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed
        assert "sk_live_abcdefgh" not in scrubbed
        assert scrubbed.count("[REDACTED]") >= 4

    def test_secrets_in_env_format(self):
        """Secrets in env-var format are caught."""
        text = 'password=secret123\napi_key=sk-abcdefghijklmnopqrstuvwxyz1234567890'
        scrubbed = scrub_output_secrets(text)
        assert "secret123" not in scrubbed
        assert "sk-abcdefghij" not in scrubbed

    def test_json_password_format_known_limitation(self):
        """JSON-quoted password format uses space separator — pattern requires := separator.
        The sk- pattern still catches API keys regardless of format."""
        text = '{"api_key": "sk-abcdefghijklmnopqrstuvwxyz1234567890"}'
        scrubbed = scrub_output_secrets(text)
        assert "sk-abcdefghij" not in scrubbed


# ===========================================================================
# 9. Round 19 Fix Verification (still working)
# ===========================================================================


class TestRound19FixesStillWorking:
    """Verify Round 19 fixes haven't regressed."""

    def test_read_file_lines_int_validation(self):
        """read_file handler validates lines parameter as int."""
        source = (SRC_DIR / "tools" / "executor.py").read_text()
        assert "int(inp.get(\"lines\"" in source or "min(int(" in source

    def test_read_file_lines_cap_1000(self):
        """read_file handler caps lines at 1000."""
        source = (SRC_DIR / "tools" / "executor.py").read_text()
        assert "1000" in source

    def test_incus_exec_shlex_quote(self):
        """incus_exec handler uses shlex.quote for command."""
        source = (SRC_DIR / "tools" / "executor.py").read_text()
        assert "shlex.quote(command)" in source

    def test_error_message_scrubbed_in_client(self):
        """Error messages to Discord are scrubbed."""
        source = (SRC_DIR / "discord" / "client.py").read_text()
        assert 'scrub_response_secrets(f"Something went wrong: {e}")' in source

    def test_monitor_alert_scrubbed(self):
        """Monitor alerts to Discord are scrubbed."""
        source = (SRC_DIR / "discord" / "client.py").read_text()
        # Line 2274
        assert "channel.send(scrub_response_secrets(message))" in source

    def test_ten_secret_patterns(self):
        """Secret scrubber has at least 10 patterns."""
        assert len(OUTPUT_SECRET_PATTERNS) >= 10


# ===========================================================================
# 10. Source Structure Verification — All Scrubbing Paths
# ===========================================================================


class TestAllScrubbingPathsPresent:
    """Verify all channel.send paths that could contain secrets use scrubbing."""

    def test_all_unscrubbed_sends_identified(self):
        """Count scrub_response_secrets calls in client.py — should be >= 8."""
        source = (SRC_DIR / "discord" / "client.py").read_text()
        count = source.count("scrub_response_secrets(")
        # Definition (1) + error (1) + response (1) + monitor (1) +
        # digest_error (1) + digest_summary (1) + scheduled_result (1) +
        # scheduled_error (1) + workflow (1) + skill_msg (1)
        assert count >= 9, f"Expected >= 9 scrub_response_secrets calls, got {count}"

    def test_scrub_output_in_knowledge_search(self):
        """scrub_output_secrets applied to knowledge search results."""
        source = (SRC_DIR / "discord" / "client.py").read_text()
        # The knowledge search handler scrubs content
        assert "scrub_output_secrets" in source

    def test_webhook_send_scrubbed_in_main(self):
        """__main__.py webhook callback scrubs before sending."""
        source = (SRC_DIR / "__main__.py").read_text()
        assert "scrub_response_secrets" in source

    def test_no_bare_channel_send_with_tool_output(self):
        """Scheduled check and workflow sends use scrub_response_secrets."""
        source = (SRC_DIR / "discord" / "client.py").read_text()
        # Find all channel.send calls and verify the ones with dynamic content are scrubbed
        # The scheduled task section should use scrubbing
        sched_section = source[source.index("_on_scheduled_task"):]
        sched_section = sched_section[:sched_section.index("async def _send_with_retry")]
        channel_sends = [line.strip() for line in sched_section.split("\n") if "channel.send(" in line]
        for line in channel_sends:
            # Static reminder messages are OK unscrubbed
            if "Scheduled reminder" in line:
                continue
            # Dynamic content should be scrubbed
            if "result" in line.lower() or "error" in line.lower() or "summary" in line.lower():
                assert "scrub_response_secrets" in line, f"Unscrubbed dynamic send: {line}"


# ===========================================================================
# 11. Webhook Payload Construction Audit
# ===========================================================================


class TestWebhookPayloadConstructionSafety:
    """Verify webhook payload construction handles potentially malicious input."""

    def test_gitea_commit_message_truncated(self):
        """Gitea commit messages are truncated at 80 chars."""
        source = (SRC_DIR / "health" / "server.py").read_text()
        assert "[:80]" in source

    def test_gitea_commits_limited(self):
        """Gitea only shows first 5 commits."""
        source = (SRC_DIR / "health" / "server.py").read_text()
        assert "[:5]" in source

    def test_grafana_summary_truncated(self):
        """Grafana alert summaries are truncated at 200 chars."""
        source = (SRC_DIR / "health" / "server.py").read_text()
        assert "[:200]" in source

    def test_grafana_alerts_limited(self):
        """Grafana only shows first 10 alerts."""
        source = (SRC_DIR / "health" / "server.py").read_text()
        assert "[:10]" in source

    def test_hmac_verification_required(self):
        """Gitea webhook requires HMAC verification."""
        source = (SRC_DIR / "health" / "server.py").read_text()
        assert "_verify_hmac_sha256" in source

    def test_no_secret_without_hmac(self):
        """Webhooks without secret configured are rejected."""
        source = (SRC_DIR / "health" / "server.py").read_text()
        assert 'return False' in source  # _verify_hmac_sha256 returns False when no secret


# ===========================================================================
# 12. Knowledge Base Ingestion — Secrets Stored But Search Scrubbed
# ===========================================================================


class TestKnowledgeBaseIngestionAudit:
    """Verify that knowledge base search results are scrubbed even if
    ingested content contains secrets."""

    def test_search_returns_scrubbed_content(self):
        """When a chunk contains a password, search result is scrubbed."""
        # Simulate what _handle_search_knowledge does
        chunk_content = "DB_PASSWORD=SuperSecret123\nDB_HOST=localhost"
        # Simulate the scrubbing step
        content = scrub_output_secrets(chunk_content.replace("\n", " ")[:500])
        assert "SuperSecret123" not in content
        assert "[REDACTED]" in content

    def test_search_returns_scrubbed_api_key(self):
        """When a chunk contains an API key, search result is scrubbed."""
        chunk_content = "config:\n  openai_key: sk-proj1234567890abcdefghijklmnop"
        content = scrub_output_secrets(chunk_content.replace("\n", " ")[:500])
        assert "sk-proj1234" not in content

    def test_search_returns_scrubbed_github_token(self):
        """When a chunk contains a GitHub token, search result is scrubbed."""
        chunk_content = "GITHUB_TOKEN=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij01"
        content = scrub_output_secrets(chunk_content.replace("\n", " ")[:500])
        assert "ghp_ABCDEF" not in content

    def test_search_clean_content_unchanged(self):
        """Clean content passes through without modification."""
        chunk_content = "To restart nginx: systemctl restart nginx"
        content = scrub_output_secrets(chunk_content.replace("\n", " ")[:500])
        assert content == chunk_content


# ===========================================================================
# 13. Cross-Round Consistency
# ===========================================================================


class TestCrossRoundConsistency:
    """Verify all previous rounds' fixes are still intact."""

    def test_no_approval_in_source(self):
        """No approval system references in source (Round 1)."""
        for path in SRC_DIR.rglob("*.py"):
            content = path.read_text()
            assert "requires_approval" not in content, f"requires_approval found in {path}"

    def test_no_classifier_in_source(self):
        """No classifier references in source (Round 2)."""
        haiku_path = SRC_DIR / "llm" / "haiku_classifier.py"
        assert not haiku_path.exists()

    def test_no_routing_module(self):
        """routing.py deleted (Round 9)."""
        routing_path = SRC_DIR / "discord" / "routing.py"
        assert not routing_path.exists()

    def test_local_execution_intact(self):
        """Local execution dispatch still works (Round 5)."""
        source = (SRC_DIR / "tools" / "ssh.py").read_text()
        assert "is_local_address" in source
        assert "run_local_command" in source

    def test_prompt_under_5000(self):
        """System prompt template under 5000 chars (Round 7)."""
        from src.llm.system_prompt import SYSTEM_PROMPT_TEMPLATE
        assert len(SYSTEM_PROMPT_TEMPLATE) < 5000

    def test_personality_present(self):
        """Personality in system prompt (Round 1)."""
        from src.llm.system_prompt import SYSTEM_PROMPT_TEMPLATE
        assert "exhausted omniscience" in SYSTEM_PROMPT_TEMPLATE.lower()

    def test_65_plus_tools(self):
        """65+ tools in registry."""
        from src.tools.registry import TOOLS
        assert len(TOOLS) >= 65


# ===========================================================================
# 14. Secret Scrubber Pattern Coverage Summary
# ===========================================================================


class TestSecretPatternCoverage:
    """Verify all 10 patterns in OUTPUT_SECRET_PATTERNS are functional."""

    @pytest.mark.parametrize("text,should_scrub", [
        ("password=test123", True),
        ("api_key=abcdef1234567890abc", True),
        ("secret=abcdef1234567890abcdef", True),
        ("sk-abcdef1234567890abcdef1234567890", True),
        ("BEGIN RSA PRIVATE KEY", True),
        ("postgres://user:pass@host/db", True),
        ("ghp_" + "A" * 36, True),
        ("AKIA" + "B" * 16, True),
        ("sk_live_" + "c" * 20, True),
        ("xoxb-abc-def", True),
        ("Hello world", False),
        ("normal text here", False),
    ])
    def test_pattern(self, text, should_scrub):
        """Each pattern in OUTPUT_SECRET_PATTERNS catches its target."""
        result = scrub_output_secrets(text)
        if should_scrub:
            assert "[REDACTED]" in result, f"Expected scrubbing for: {text!r}"
        else:
            assert result == text, f"Unexpected scrubbing for: {text!r}"


# ===========================================================================
# 15. Webhook Server Error Response Safety
# ===========================================================================


class TestWebhookServerErrorResponseSafety:
    """Verify webhook server error responses don't leak internal state."""

    def test_error_response_is_json(self):
        """Webhook error responses use json_response (structured, not raw)."""
        source = (SRC_DIR / "health" / "server.py").read_text()
        # All error returns should use web.json_response
        error_lines = [l.strip() for l in source.split("\n") if "status=4" in l or "status=5" in l]
        for line in error_lines:
            assert "json_response" in line, f"Non-JSON error response: {line}"

    def test_no_traceback_in_error_response(self):
        """Webhook _send error doesn't include traceback."""
        source = (SRC_DIR / "health" / "server.py").read_text()
        assert "traceback" not in source.lower()
        assert "exc_info" not in source.split("log.error")[1].split("\n")[0] if "log.error" in source else True
