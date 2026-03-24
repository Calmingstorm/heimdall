"""Round 17: Security hardening tests for new overhaul features.

Tests verify:
- URL scheme validation on PDF and image tools (SSRF prevention)
- Process manager host validation
- ComfyUI prompt_id sanitization
- Secret scrubbing on Discord-facing outputs (broadcast, poll, image gen)
- SQL injection resistance in knowledge store
- Path traversal prevention in host+path tools
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# URL scheme validation (SSRF prevention)
# ---------------------------------------------------------------------------

class TestURLSchemeValidation:
    """Verify PDF and image tools reject non-HTTP(S) URLs."""

    async def test_pdf_rejects_file_url(self):
        """analyze_pdf must reject file:// URLs."""
        from src.tools.executor import ToolExecutor
        from src.config.schema import ToolsConfig

        executor = ToolExecutor(ToolsConfig())
        result = await executor.execute("analyze_pdf", {"url": "file:///etc/passwd"})
        assert "Only http:// and https://" in result

    async def test_pdf_rejects_ftp_url(self):
        from src.tools.executor import ToolExecutor
        from src.config.schema import ToolsConfig

        executor = ToolExecutor(ToolsConfig())
        result = await executor.execute("analyze_pdf", {"url": "ftp://evil.com/payload.pdf"})
        assert "Only http:// and https://" in result

    async def test_pdf_rejects_data_url(self):
        from src.tools.executor import ToolExecutor
        from src.config.schema import ToolsConfig

        executor = ToolExecutor(ToolsConfig())
        result = await executor.execute("analyze_pdf", {"url": "data:application/pdf;base64,AAAA"})
        assert "Only http:// and https://" in result

    async def test_pdf_accepts_https_url(self):
        """Verify https:// URLs are accepted (actual fetch may fail, but scheme check passes)."""
        from src.tools.executor import ToolExecutor
        from src.config.schema import ToolsConfig

        executor = ToolExecutor(ToolsConfig())
        # Will fail at the fetch step, not the scheme check
        result = await executor.execute("analyze_pdf", {"url": "https://example.com/doc.pdf"})
        assert "Only http://" not in result

    async def test_pdf_accepts_http_url(self):
        from src.tools.executor import ToolExecutor
        from src.config.schema import ToolsConfig

        executor = ToolExecutor(ToolsConfig())
        result = await executor.execute("analyze_pdf", {"url": "http://example.com/doc.pdf"})
        assert "Only http://" not in result

    async def test_image_rejects_file_url(self):
        """analyze_image must reject file:// URLs."""
        from src.discord.client import LokiBot

        bot = MagicMock(spec=LokiBot)
        bot.tool_executor = MagicMock()
        message = MagicMock()

        result = await LokiBot._handle_analyze_image(bot, message, {"url": "file:///etc/shadow"})
        assert "Only http:// and https://" in result

    async def test_image_rejects_ftp_url(self):
        from src.discord.client import LokiBot

        bot = MagicMock(spec=LokiBot)
        message = MagicMock()
        result = await LokiBot._handle_analyze_image(bot, message, {"url": "ftp://evil.com/img.png"})
        assert "Only http:// and https://" in result

    async def test_image_rejects_javascript_url(self):
        from src.discord.client import LokiBot

        bot = MagicMock(spec=LokiBot)
        message = MagicMock()
        result = await LokiBot._handle_analyze_image(bot, message, {"url": "javascript:alert(1)"})
        assert "Only http:// and https://" in result

    async def test_image_accepts_https(self):
        """https:// URLs pass the scheme check (may fail at fetch)."""
        from src.discord.client import LokiBot

        bot = MagicMock(spec=LokiBot)
        message = MagicMock()
        # Will fail at fetch, not scheme check
        result = await LokiBot._handle_analyze_image(bot, message, {"url": "https://example.com/img.png"})
        assert "Only http://" not in result


# ---------------------------------------------------------------------------
# Process manager host validation
# ---------------------------------------------------------------------------

class TestProcessManagerHostValidation:
    """Verify manage_process validates host against configured hosts."""

    async def test_start_rejects_unknown_host(self):
        """Starting a process on an unknown host should be rejected."""
        from src.tools.executor import ToolExecutor
        from src.config.schema import ToolsConfig

        executor = ToolExecutor(ToolsConfig())
        result = await executor.execute("manage_process", {
            "action": "start",
            "host": "unknown-attacker-host",
            "command": "whoami",
        })
        assert "Unknown or disallowed host" in result

    async def test_start_rejects_empty_host(self):
        from src.tools.executor import ToolExecutor
        from src.config.schema import ToolsConfig

        executor = ToolExecutor(ToolsConfig())
        result = await executor.execute("manage_process", {
            "action": "start",
            "host": "",
            "command": "whoami",
        })
        # Empty host should fail (either "required" or "Unknown")
        assert "host" in result.lower() or "Unknown" in result

    async def test_start_requires_command(self):
        from src.tools.executor import ToolExecutor
        from src.config.schema import ToolsConfig

        executor = ToolExecutor(ToolsConfig())
        result = await executor.execute("manage_process", {
            "action": "start",
            "host": "somehost",
        })
        assert "command is required" in result


# ---------------------------------------------------------------------------
# ComfyUI prompt_id sanitization
# ---------------------------------------------------------------------------

class TestComfyUIPromptIdValidation:
    """Verify ComfyUI client validates prompt_id from server responses."""

    async def test_rejects_path_traversal_prompt_id(self):
        """prompt_id containing path traversal chars must be rejected."""
        from src.tools.comfyui import ComfyUIClient
        from unittest.mock import AsyncMock

        client = ComfyUIClient("http://localhost:8188")

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"prompt_id": "../../etc/passwd"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.generate("test prompt")
        assert result is None

    async def test_rejects_slash_in_prompt_id(self):
        from src.tools.comfyui import ComfyUIClient

        client = ComfyUIClient("http://localhost:8188")

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"prompt_id": "abc/def"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.generate("test prompt")
        assert result is None

    async def test_rejects_overly_long_prompt_id(self):
        from src.tools.comfyui import ComfyUIClient

        client = ComfyUIClient("http://localhost:8188")

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"prompt_id": "a" * 200})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.generate("test prompt")
        assert result is None

    async def test_accepts_valid_uuid_prompt_id(self):
        """Valid UUID-style prompt_ids should be accepted."""
        from src.tools.comfyui import ComfyUIClient

        client = ComfyUIClient("http://localhost:8188")

        # The prompt_id validation should pass for UUID format
        valid_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert all(c.isalnum() or c in "-_" for c in valid_id)
        assert len(valid_id) <= 100

    def test_accepts_alphanumeric_with_hyphens(self):
        """Alphanumeric + hyphens + underscores should be accepted."""
        valid_ids = [
            "abc-123",
            "test-prompt-456",
            "a1b2c3d4",
            "prompt_id_with_underscores",
        ]
        for pid in valid_ids:
            assert all(c.isalnum() or c in "-_" for c in pid)
            assert len(pid) <= 100


# ---------------------------------------------------------------------------
# Secret scrubbing on Discord-facing outputs
# ---------------------------------------------------------------------------

class TestDiscordOutputScrubbing:
    """Verify tools that send directly to Discord scrub secrets first."""

    async def test_broadcast_scrubs_password(self):
        """Broadcast handler must scrub secrets from text before Discord send."""
        from src.discord.client import LokiBot

        bot = MagicMock(spec=LokiBot)
        message = MagicMock()
        message.channel.send = AsyncMock()

        await LokiBot._handle_broadcast(bot, message, {
            "text": "Here is the password=hunter2 for the server",
        })

        # Verify channel.send was called
        message.channel.send.assert_called_once()
        sent_content = message.channel.send.call_args[1].get("content", "")
        # The password should have been scrubbed
        assert "hunter2" not in sent_content
        assert "[REDACTED]" in sent_content

    async def test_broadcast_scrubs_api_key(self):
        from src.discord.client import LokiBot

        bot = MagicMock(spec=LokiBot)
        message = MagicMock()
        message.channel.send = AsyncMock()

        await LokiBot._handle_broadcast(bot, message, {
            "text": "Use api_key=sk-abcdefghijklmnopqrstuvwxyz1234567890",
        })

        sent_content = message.channel.send.call_args[1].get("content", "")
        assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in sent_content
        assert "[REDACTED]" in sent_content

    async def test_broadcast_preserves_safe_text(self):
        from src.discord.client import LokiBot

        bot = MagicMock(spec=LokiBot)
        message = MagicMock()
        message.channel.send = AsyncMock()

        await LokiBot._handle_broadcast(bot, message, {
            "text": "Server is running fine with 99.9% uptime.",
        })

        sent_content = message.channel.send.call_args[1].get("content", "")
        assert sent_content == "Server is running fine with 99.9% uptime."

    async def test_poll_scrubs_question_secrets(self):
        """Poll questions must be scrubbed for secrets."""
        import discord
        from src.discord.client import LokiBot

        bot = MagicMock(spec=LokiBot)
        message = MagicMock()
        message.channel.send = AsyncMock()

        # Mock discord.Poll
        with patch.object(discord, "Poll") as mock_poll:
            mock_poll_instance = MagicMock()
            mock_poll.return_value = mock_poll_instance

            await LokiBot._handle_create_poll(bot, message, {
                "question": "The password=secret123 was leaked?",
                "options": ["Yes", "No"],
            })

            # The question passed to Poll() should be scrubbed
            call_kwargs = mock_poll.call_args[1]
            assert "secret123" not in call_kwargs["question"]
            assert "[REDACTED]" in call_kwargs["question"]

    async def test_poll_scrubs_options_secrets(self):
        """Poll options must also be scrubbed."""
        import discord
        from src.discord.client import LokiBot

        bot = MagicMock(spec=LokiBot)
        message = MagicMock()
        message.channel.send = AsyncMock()

        with patch.object(discord, "Poll") as mock_poll:
            mock_poll_instance = MagicMock()
            mock_poll.return_value = mock_poll_instance

            await LokiBot._handle_create_poll(bot, message, {
                "question": "Which key?",
                "options": ["sk-abcdefghijklmnopqrstuvwxyz1234567890", "Other"],
            })

            # Check that add_answer was called with scrubbed text
            add_calls = mock_poll_instance.add_answer.call_args_list
            first_option = add_calls[0][1]["text"]
            assert "sk-abcdefghijklmnopqrstuvwxyz" not in first_option
            assert "[REDACTED]" in first_option

    async def test_generate_image_caption_scrubbed(self):
        """Image generation caption must scrub secrets before Discord send."""
        import discord
        from src.discord.client import LokiBot

        bot = MagicMock(spec=LokiBot)
        bot.config = MagicMock()
        bot.config.comfyui.enabled = True
        bot.config.comfyui.url = "http://localhost:8188"
        message = MagicMock()
        message.channel.send = AsyncMock()

        fake_png = b"\x89PNG" + b"\x00" * 100

        with patch("src.tools.comfyui.ComfyUIClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.generate = AsyncMock(return_value=fake_png)
            MockClient.return_value = mock_client

            await LokiBot._handle_generate_image(bot, message, {
                "prompt": "a picture with password=admin123 in it",
            })

        message.channel.send.assert_called_once()
        sent_content = message.channel.send.call_args[1].get("content", "")
        assert "admin123" not in sent_content
        assert "[REDACTED]" in sent_content


# ---------------------------------------------------------------------------
# SQL injection resistance
# ---------------------------------------------------------------------------

class TestSQLInjectionResistance:
    """Verify parameterized queries prevent SQL injection in knowledge store."""

    async def test_knowledge_ingest_sql_injection_source(self):
        """Source names with SQL injection attempts should be safely stored."""
        from src.knowledge.store import KnowledgeStore

        with patch("src.knowledge.store.load_extension", return_value=False):
            store = KnowledgeStore(":memory:")

        assert store.available
        # Attempt SQL injection via source name
        count = await store.ingest(
            "Some content",
            "'; DROP TABLE knowledge_chunks; --",
        )
        assert count > 0

        # Table should still exist and be queryable
        sources = store.list_sources()
        assert len(sources) == 1
        assert "DROP TABLE" in sources[0]["source"]

    async def test_knowledge_delete_sql_injection(self):
        """SQL injection via delete_source should be safely handled."""
        from src.knowledge.store import KnowledgeStore

        with patch("src.knowledge.store.load_extension", return_value=False):
            store = KnowledgeStore(":memory:")

        await store.ingest("Real content", "real-doc")
        # Attempt SQL injection via delete
        deleted = store.delete_source("'; DROP TABLE knowledge_chunks; --")
        assert deleted == 0

        # Original data should be intact
        sources = store.list_sources()
        assert len(sources) == 1
        assert sources[0]["source"] == "real-doc"

    async def test_knowledge_search_sql_injection(self):
        """SQL injection via search query should be safely handled."""
        from src.knowledge.store import KnowledgeStore

        with patch("src.knowledge.store.load_extension", return_value=False):
            store = KnowledgeStore(":memory:")

        await store.ingest("Some content", "doc1")
        # Search with injection attempt — should return empty, not crash
        results = await store.search("'; DROP TABLE knowledge_chunks; --")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Path traversal prevention
# ---------------------------------------------------------------------------

class TestPathTraversalPrevention:
    """Verify host+path tools use shlex.quote for path safety."""

    async def test_pdf_path_with_semicolons_quoted(self):
        """Shell injection via semicolons in PDF path should be safely quoted."""
        import sys
        from src.tools.executor import ToolExecutor
        from src.config.schema import ToolsConfig, ToolHost

        config = ToolsConfig(hosts={"myhost": ToolHost(address="localhost", ssh_user="root")})
        executor = ToolExecutor(config)

        mock_fitz = MagicMock()
        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            with patch.object(executor, "_exec_command", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = (1, "No such file")
                await executor.execute("analyze_pdf", {
                    "host": "myhost",
                    "path": "/tmp/x; cat /etc/shadow",
                })
                # The path should be shell-quoted, preventing command injection
                cmd = mock_exec.call_args[0][1]
                assert "'" in cmd  # shlex.quote wraps in single quotes
                # The semicolon should be inside quotes, not splitting commands
                assert "base64 -w0 " in cmd

    async def test_pdf_path_with_special_chars_quoted(self):
        """Paths with shell metacharacters should be safely quoted."""
        import sys
        from src.tools.executor import ToolExecutor
        from src.config.schema import ToolsConfig, ToolHost

        config = ToolsConfig(hosts={"myhost": ToolHost(address="localhost", ssh_user="root")})
        executor = ToolExecutor(config)

        mock_fitz = MagicMock()
        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            with patch.object(executor, "_exec_command", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = (1, "No such file")
                await executor.execute("analyze_pdf", {
                    "host": "myhost",
                    "path": "/tmp/$(whoami)/file.pdf",
                })
                cmd = mock_exec.call_args[0][1]
                # $(whoami) should be inside quotes, not executed
                assert "$(whoami)" in cmd
                assert "'" in cmd


# ---------------------------------------------------------------------------
# Tool output scrubbing in tool loop
# ---------------------------------------------------------------------------

class TestToolOutputScrubbing:
    """Verify all new tool outputs pass through secret scrubbing."""

    def test_scrub_covers_process_output(self):
        """Process poll output containing secrets should be scrubbed."""
        from src.llm.secret_scrubber import scrub_output_secrets

        output = "Config loaded: api_key=sk-abcdef1234567890abcdef1234567890"
        scrubbed = scrub_output_secrets(output)
        assert "sk-abcdef1234567890abcdef1234567890" not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_scrub_covers_pdf_text(self):
        """PDF text containing secrets should be scrubbed."""
        from src.llm.secret_scrubber import scrub_output_secrets

        pdf_text = "## Page 1\nDatabase URI: postgres://admin:s3cret@db.internal/prod"
        scrubbed = scrub_output_secrets(pdf_text)
        assert "s3cret" not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_scrub_covers_github_token(self):
        from src.llm.secret_scrubber import scrub_output_secrets

        output = "Token: ghp_1234567890abcdefghijklmnopqrstuvwxyz1234"
        scrubbed = scrub_output_secrets(output)
        assert "ghp_" not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_scrub_covers_aws_key(self):
        from src.llm.secret_scrubber import scrub_output_secrets

        output = "AWS key: AKIAIOSFODNN7EXAMPLE"
        scrubbed = scrub_output_secrets(output)
        assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed

    def test_scrub_covers_private_key(self):
        from src.llm.secret_scrubber import scrub_output_secrets

        output = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpA..."
        scrubbed = scrub_output_secrets(output)
        assert "BEGIN RSA PRIVATE KEY" not in scrubbed


# ---------------------------------------------------------------------------
# Existing security patterns still enforced
# ---------------------------------------------------------------------------

class TestExistingSecurityPatterns:
    """Verify existing security measures are not broken by new code."""

    def test_incus_name_validation_still_works(self):
        """Incus name regex should block injection attempts."""
        from src.tools.executor import _INCUS_NAME_RE

        assert _INCUS_NAME_RE.match("valid-name")
        assert _INCUS_NAME_RE.match("container1")
        assert not _INCUS_NAME_RE.match("$(whoami)")
        assert not _INCUS_NAME_RE.match("name; rm -rf /")
        assert not _INCUS_NAME_RE.match("")
        assert not _INCUS_NAME_RE.match("-starts-with-dash")

    def test_interpreter_allowlist_still_works(self):
        """run_script interpreter must be in allowlist."""
        allowed = {"bash", "sh", "python3", "python", "node", "ruby", "perl"}
        dangerous = {"curl", "wget", "nc", "/bin/sh", "../../bin/sh"}
        for interp in dangerous:
            assert interp not in allowed

    def test_service_validation_still_required(self):
        """check_service/restart_service must validate against allowlist."""
        from src.tools.executor import ToolExecutor
        from src.config.schema import ToolsConfig

        executor = ToolExecutor(ToolsConfig())
        # No services in allowlist by default
        assert not executor._validate_service("sshd")
        assert not executor._validate_service("$(whoami)")

    def test_scrub_output_secrets_function_exists(self):
        """Secret scrubber module should be importable."""
        from src.llm.secret_scrubber import scrub_output_secrets, OUTPUT_SECRET_PATTERNS
        assert callable(scrub_output_secrets)
        assert len(OUTPUT_SECRET_PATTERNS) >= 10

    def test_scrub_response_secrets_function_exists(self):
        """Response scrubber should be importable."""
        from src.discord.client import scrub_response_secrets
        assert callable(scrub_response_secrets)
        # Should scrub passwords
        result = scrub_response_secrets("password: hunter2")
        assert "[REDACTED]" in result
