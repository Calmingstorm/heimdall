"""Tests for audit/logger.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.audit.logger import AuditLogger


@pytest.fixture
def audit(tmp_dir: Path) -> AuditLogger:
    return AuditLogger(path=str(tmp_dir / "audit.jsonl"))


class TestLogExecution:
    @pytest.mark.asyncio
    async def test_writes_entry(self, audit: AuditLogger):
        await audit.log_execution(
            user_id="123",
            user_name="testuser",
            channel_id="ch1",
            tool_name="check_disk",
            tool_input={"host": "server"},
            approved=True,
            result_summary="Filesystem  Size  Used  Avail",
            execution_time_ms=150,
        )
        lines = audit.path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["tool_name"] == "check_disk"
        assert entry["user_name"] == "testuser"
        assert entry["approved"] is True

    @pytest.mark.asyncio
    async def test_truncates_result(self, audit: AuditLogger):
        await audit.log_execution(
            user_id="123",
            user_name="testuser",
            channel_id="ch1",
            tool_name="read_file",
            tool_input={"host": "server", "path": "/big"},
            approved=True,
            result_summary="x" * 1000,
            execution_time_ms=50,
        )
        entry = json.loads(audit.path.read_text().strip())
        assert len(entry["result_summary"]) <= 500

    @pytest.mark.asyncio
    async def test_multiple_entries(self, audit: AuditLogger):
        for i in range(5):
            await audit.log_execution(
                user_id="123",
                user_name="testuser",
                channel_id="ch1",
                tool_name=f"tool_{i}",
                tool_input={},
                approved=True,
                result_summary="ok",
                execution_time_ms=10,
            )
        lines = audit.path.read_text().strip().split("\n")
        assert len(lines) == 5


class TestSearch:
    @pytest.mark.asyncio
    async def _populate(self, audit: AuditLogger):
        await audit.log_execution(
            user_id="123", user_name="alice", channel_id="ch1",
            tool_name="check_disk", tool_input={"host": "server"},
            approved=True, result_summary="ok", execution_time_ms=10,
        )
        await audit.log_execution(
            user_id="456", user_name="bob", channel_id="ch1",
            tool_name="restart_service", tool_input={"host": "desktop", "service": "apache2"},
            approved=True, result_summary="restarted", execution_time_ms=20,
        )
        await audit.log_execution(
            user_id="123", user_name="alice", channel_id="ch2",
            tool_name="check_disk", tool_input={"host": "desktop"},
            approved=True, result_summary="ok", execution_time_ms=15,
        )

    @pytest.mark.asyncio
    async def test_search_by_tool(self, audit: AuditLogger):
        await self._populate(audit)
        results = await audit.search(tool_name="check_disk")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_by_host(self, audit: AuditLogger):
        await self._populate(audit)
        results = await audit.search(host="desktop")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_by_keyword(self, audit: AuditLogger):
        await self._populate(audit)
        results = await audit.search(keyword="restarted")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_limit(self, audit: AuditLogger):
        await self._populate(audit)
        results = await audit.search(limit=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_empty(self, audit: AuditLogger):
        results = await audit.search()
        assert results == []

    @pytest.mark.asyncio
    async def test_most_recent_first(self, audit: AuditLogger):
        await self._populate(audit)
        results = await audit.search(tool_name="check_disk")
        # Last written should come first
        assert results[0]["tool_input"]["host"] == "desktop"
