"""Tests for context/loader.py."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.context.loader import ContextLoader


@pytest.fixture
def context_dir(tmp_dir: Path) -> Path:
    d = tmp_dir / "context"
    d.mkdir()
    return d


class TestLoad:
    def test_loads_md_files(self, context_dir: Path):
        (context_dir / "infra.md").write_text("## Servers\n- server1\n")
        (context_dir / "desktop.md").write_text("## Desktop\n- monitor\n")
        loader = ContextLoader(str(context_dir))
        content = loader.load()
        assert "Servers" in content
        assert "Desktop" in content

    def test_alphabetical_order(self, context_dir: Path):
        (context_dir / "b.md").write_text("BBB")
        (context_dir / "a.md").write_text("AAA")
        loader = ContextLoader(str(context_dir))
        content = loader.load()
        assert content.index("AAA") < content.index("BBB")

    def test_empty_directory(self, context_dir: Path):
        loader = ContextLoader(str(context_dir))
        content = loader.load()
        assert content == ""

    def test_missing_directory(self, tmp_dir: Path):
        loader = ContextLoader(str(tmp_dir / "nonexistent"))
        content = loader.load()
        assert content == ""

    def test_reload(self, context_dir: Path):
        (context_dir / "a.md").write_text("version 1")
        loader = ContextLoader(str(context_dir))
        loader.load()
        assert "version 1" in loader.context

        (context_dir / "a.md").write_text("version 2")
        loader.reload()
        assert "version 2" in loader.context


class TestSecretScanning:
    def test_warns_on_password(self, context_dir: Path, caplog):
        (context_dir / "bad.md").write_text("password: hunter2")
        loader = ContextLoader(str(context_dir))
        with caplog.at_level(logging.WARNING, logger="ansiblex.context"):
            loader.load()
        assert any("secret" in r.message.lower() for r in caplog.records)

    def test_warns_on_api_key(self, context_dir: Path, caplog):
        (context_dir / "bad.md").write_text("api_key: sk-1234567890abcdef1234")
        loader = ContextLoader(str(context_dir))
        with caplog.at_level(logging.WARNING, logger="ansiblex.context"):
            loader.load()
        assert any("secret" in r.message.lower() for r in caplog.records)

    def test_no_warning_on_clean(self, context_dir: Path, caplog):
        (context_dir / "clean.md").write_text("## Infrastructure\nNo secrets here.")
        loader = ContextLoader(str(context_dir))
        with caplog.at_level(logging.WARNING, logger="ansiblex.context"):
            loader.load()
        secret_warnings = [r for r in caplog.records if "secret" in r.message.lower()]
        assert len(secret_warnings) == 0
