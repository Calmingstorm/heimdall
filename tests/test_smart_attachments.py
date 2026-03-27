"""Tests for smart file attachment context hints."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import HeimdallBot  # noqa: E402


# --- _get_attachment_hint tests (static method, no mocking needed) ---


class TestGetAttachmentHintPython:
    def test_python_file_suggests_skill(self):
        hint = HeimdallBot._get_attachment_hint("my_tool.py", ".py", 1000)
        assert "create_skill" in hint
        assert "SKILL_DEFINITION" in hint

    def test_python_file_suggests_ingest_fallback(self):
        hint = HeimdallBot._get_attachment_hint("utils.py", ".py", 1000)
        assert "ingest" in hint.lower()

    def test_python_large_file_both_hints(self):
        hint = HeimdallBot._get_attachment_hint("big.py", ".py", 60_000)
        assert "create_skill" in hint
        assert "large" in hint.lower()


class TestGetAttachmentHintConfig:
    def test_yaml_suggests_deploy_or_ingest(self):
        hint = HeimdallBot._get_attachment_hint("config.yml", ".yml", 500)
        assert "write_file" in hint
        assert "ingest_document" in hint

    def test_json_suggests_deploy_or_ingest(self):
        hint = HeimdallBot._get_attachment_hint("settings.json", ".json", 500)
        assert "configuration" in hint.lower()
        assert "write_file" in hint

    def test_toml_suggests_deploy_or_ingest(self):
        hint = HeimdallBot._get_attachment_hint("pyproject.toml", ".toml", 500)
        assert "configuration" in hint.lower()

    def test_ini_suggests_deploy_or_ingest(self):
        hint = HeimdallBot._get_attachment_hint("config.ini", ".ini", 500)
        assert "configuration" in hint.lower()

    def test_conf_suggests_deploy_or_ingest(self):
        hint = HeimdallBot._get_attachment_hint("nginx.conf", ".conf", 500)
        assert "configuration" in hint.lower()

    def test_cfg_suggests_deploy_or_ingest(self):
        hint = HeimdallBot._get_attachment_hint("setup.cfg", ".cfg", 500)
        assert "configuration" in hint.lower()

    def test_yaml_extension(self):
        hint = HeimdallBot._get_attachment_hint("docker-compose.yaml", ".yaml", 500)
        assert "configuration" in hint.lower()


class TestGetAttachmentHintShellScript:
    def test_sh_suggests_deploy_or_run(self):
        hint = HeimdallBot._get_attachment_hint("setup.sh", ".sh", 500)
        assert "write_file" in hint
        assert "run_command" in hint

    def test_bash_suggests_deploy_or_run(self):
        hint = HeimdallBot._get_attachment_hint("deploy.bash", ".bash", 500)
        assert "shell script" in hint.lower()


class TestGetAttachmentHintSystemd:
    def test_service_suggests_deploy(self):
        hint = HeimdallBot._get_attachment_hint("myapp.service", ".service", 500)
        assert "systemd" in hint.lower()
        assert "write_file" in hint

    def test_timer_suggests_deploy(self):
        hint = HeimdallBot._get_attachment_hint("backup.timer", ".timer", 500)
        assert "systemd" in hint.lower()


class TestGetAttachmentHintDocumentation:
    def test_markdown_suggests_ingest(self):
        hint = HeimdallBot._get_attachment_hint("README.md", ".md", 2000)
        assert "documentation" in hint.lower()
        assert "ingest_document" in hint

    def test_txt_suggests_ingest(self):
        hint = HeimdallBot._get_attachment_hint("notes.txt", ".txt", 2000)
        assert "documentation" in hint.lower()
        assert "ingest_document" in hint


class TestGetAttachmentHintLargeFile:
    def test_large_file_suggests_ingest(self):
        hint = HeimdallBot._get_attachment_hint("data.csv", ".csv", 60_000)
        assert "large" in hint.lower()
        assert "ingest" in hint.lower()

    def test_small_file_no_large_hint(self):
        hint = HeimdallBot._get_attachment_hint("data.csv", ".csv", 10_000)
        assert "large" not in hint.lower()

    def test_exactly_50kb_no_large_hint(self):
        hint = HeimdallBot._get_attachment_hint("data.csv", ".csv", 50_000)
        assert "large" not in hint.lower()

    def test_just_over_50kb_has_large_hint(self):
        hint = HeimdallBot._get_attachment_hint("data.csv", ".csv", 50_001)
        assert "large" in hint.lower()


class TestGetAttachmentHintNoHint:
    def test_generic_code_no_special_hint(self):
        hint = HeimdallBot._get_attachment_hint("main.js", ".js", 1000)
        assert hint == ""

    def test_css_no_special_hint(self):
        hint = HeimdallBot._get_attachment_hint("style.css", ".css", 1000)
        assert hint == ""

    def test_go_no_special_hint(self):
        hint = HeimdallBot._get_attachment_hint("main.go", ".go", 1000)
        assert hint == ""

    def test_unknown_extension_no_hint(self):
        hint = HeimdallBot._get_attachment_hint("file.xyz", ".xyz", 1000)
        assert hint == ""

    def test_no_extension_no_hint(self):
        hint = HeimdallBot._get_attachment_hint("Makefile", "", 1000)
        assert hint == ""


# --- _process_attachments integration tests ---


def _make_attachment(filename: str, size: int, content: bytes = b"file content",
                     content_type: str | None = None) -> MagicMock:
    """Create a mock Discord attachment."""
    att = MagicMock()
    att.filename = filename
    att.size = size
    att.content_type = content_type
    att.read = AsyncMock(return_value=content)
    return att


def _make_message(*attachments: MagicMock) -> MagicMock:
    """Create a mock Discord message with attachments."""
    msg = MagicMock()
    msg.attachments = list(attachments)
    return msg


@pytest.fixture
def bot():
    """Create a minimal stub with real _process_attachments and _get_attachment_hint."""
    stub = MagicMock()
    # Bind the real instance method and static methods to the stub
    stub._process_attachments = HeimdallBot._process_attachments.__get__(stub)
    stub._get_attachment_hint = HeimdallBot._get_attachment_hint
    stub._detect_image_type = HeimdallBot._detect_image_type
    return stub


class TestProcessAttachmentsLargeText:
    async def test_large_text_file_shows_preview(self, bot):
        content = b"x" * 200_000
        att = _make_attachment("big.log", 200_000, content)
        msg = _make_message(att)

        text, images = await bot._process_attachments(msg)
        assert "too large to fully inline" in text
        assert "showing first 2KB" in text

    async def test_large_text_file_suggests_ingest(self, bot):
        content = b"x" * 200_000
        att = _make_attachment("big.log", 200_000, content)
        msg = _make_message(att)

        text, images = await bot._process_attachments(msg)
        assert "ingest_document" in text

    async def test_large_text_preview_is_truncated(self, bot):
        # Create content where first 2000 chars are 'A' and rest are 'B'
        content = (b"A" * 2000) + (b"B" * 100_000)
        att = _make_attachment("big.txt", len(content), content)
        msg = _make_message(att)

        text, _ = await bot._process_attachments(msg)
        assert "A" * 100 in text
        # The preview should not contain the B section
        assert "BBBB" not in text

    async def test_large_text_shows_file_size(self, bot):
        content = b"y" * 150_000
        att = _make_attachment("data.csv", 150_000, content)
        msg = _make_message(att)

        text, _ = await bot._process_attachments(msg)
        assert "150,000" in text

    async def test_large_binary_still_rejected(self, bot):
        att = _make_attachment("archive.zip", 200_000, content_type="application/zip")
        msg = _make_message(att)

        text, _ = await bot._process_attachments(msg)
        assert "too large to read" in text

    async def test_large_text_read_error(self, bot):
        att = _make_attachment("big.py", 200_000)
        att.read = AsyncMock(side_effect=Exception("network error"))
        msg = _make_message(att)

        text, _ = await bot._process_attachments(msg)
        assert "failed to read" in text


class TestProcessAttachmentsSmartHints:
    async def test_python_file_gets_hint(self, bot):
        att = _make_attachment("my_skill.py", 500, b"def execute(): pass")
        msg = _make_message(att)

        text, _ = await bot._process_attachments(msg)
        assert "create_skill" in text

    async def test_yaml_file_gets_hint(self, bot):
        att = _make_attachment("config.yml", 200, b"key: value")
        msg = _make_message(att)

        text, _ = await bot._process_attachments(msg)
        assert "configuration" in text.lower()

    async def test_markdown_file_gets_hint(self, bot):
        att = _make_attachment("README.md", 1000, b"# Title\nSome docs")
        msg = _make_message(att)

        text, _ = await bot._process_attachments(msg)
        assert "documentation" in text.lower()

    async def test_service_file_gets_hint(self, bot):
        att = _make_attachment("app.service", 300, b"[Unit]\nDescription=My App")
        msg = _make_message(att)

        text, _ = await bot._process_attachments(msg)
        assert "systemd" in text.lower()

    async def test_shell_script_gets_hint(self, bot):
        att = _make_attachment("deploy.sh", 400, b"#!/bin/bash\necho hello")
        msg = _make_message(att)

        text, _ = await bot._process_attachments(msg)
        assert "shell script" in text.lower()

    async def test_generic_code_no_hint(self, bot):
        att = _make_attachment("main.js", 500, b"console.log('hello')")
        msg = _make_message(att)

        text, _ = await bot._process_attachments(msg)
        assert "[This is" not in text
        assert "main.js" in text

    async def test_image_no_hint(self, bot):
        # PNG magic bytes
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        att = _make_attachment("photo.png", len(png_data), png_data, content_type="image/png")
        msg = _make_message(att)

        text, images = await bot._process_attachments(msg)
        assert len(images) == 1
        assert "create_skill" not in text
        assert "ingest" not in text

    async def test_hint_appended_after_content(self, bot):
        att = _make_attachment("setup.sh", 200, b"#!/bin/bash\necho setup")
        msg = _make_message(att)

        text, _ = await bot._process_attachments(msg)
        # File content should appear before the hint
        code_pos = text.find("echo setup")
        hint_pos = text.find("[This is a shell script")
        assert code_pos < hint_pos

    async def test_multiple_attachments_each_get_hints(self, bot):
        att1 = _make_attachment("tool.py", 500, b"def main(): pass")
        att2 = _make_attachment("config.yml", 200, b"key: value")
        msg = _make_message(att1, att2)

        text, _ = await bot._process_attachments(msg)
        assert "create_skill" in text
        assert "configuration" in text.lower()
