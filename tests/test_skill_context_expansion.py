"""Tests for expanded SkillContext API — knowledge base, history, scheduler, execute_tool."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.schema import ToolsConfig, ToolHost
from src.tools.executor import ToolExecutor
from src.tools.skill_context import SkillContext
from src.tools.skill_manager import SkillManager


@pytest.fixture
def executor(tools_config: ToolsConfig) -> ToolExecutor:
    return ToolExecutor(tools_config)


@pytest.fixture
def mock_knowledge_store():
    store = MagicMock()
    store.available = True
    store.search_hybrid = AsyncMock(return_value=[
        {"content": "chunk1", "source": "doc.md", "score": 0.9},
    ])
    store.ingest = AsyncMock(return_value=3)
    return store


@pytest.fixture
def mock_embedder():
    return MagicMock()


@pytest.fixture
def mock_session_manager():
    mgr = MagicMock()
    mgr.search_history = AsyncMock(return_value=[
        {"type": "user", "content": "hello", "timestamp": 1.0, "channel_id": "123"},
    ])
    return mgr


@pytest.fixture
def mock_scheduler():
    sched = MagicMock()
    sched.add = MagicMock(return_value={"id": "abc123", "description": "test"})
    sched.list_all = MagicMock(return_value=[{"id": "abc123"}])
    sched.delete = MagicMock(return_value=True)
    return sched


@pytest.fixture
def full_context(
    executor, mock_knowledge_store, mock_embedder,
    mock_session_manager, mock_scheduler,
):
    """SkillContext with all services wired."""
    return SkillContext(
        executor, "test_skill",
        knowledge_store=mock_knowledge_store,
        embedder=mock_embedder,
        session_manager=mock_session_manager,
        scheduler=mock_scheduler,
    )


@pytest.fixture
def bare_context(executor):
    """SkillContext with no optional services."""
    return SkillContext(executor, "test_skill")


# --- search_knowledge ---

class TestSearchKnowledge:
    async def test_returns_results(self, full_context, mock_knowledge_store, mock_embedder):
        results = await full_context.search_knowledge("test query")
        assert len(results) == 1
        assert results[0]["source"] == "doc.md"
        mock_knowledge_store.search_hybrid.assert_called_once_with(
            "test query", mock_embedder, limit=5,
        )

    async def test_custom_limit(self, full_context, mock_knowledge_store, mock_embedder):
        await full_context.search_knowledge("q", limit=10)
        mock_knowledge_store.search_hybrid.assert_called_once_with(
            "q", mock_embedder, limit=10,
        )

    async def test_no_knowledge_store_returns_empty(self, bare_context):
        results = await bare_context.search_knowledge("test")
        assert results == []

    async def test_no_embedder_returns_empty(self, executor, mock_knowledge_store):
        ctx = SkillContext(
            executor, "test", knowledge_store=mock_knowledge_store,
        )
        results = await ctx.search_knowledge("test")
        assert results == []


# --- ingest_document ---

class TestIngestDocument:
    async def test_ingests_and_returns_count(self, full_context, mock_knowledge_store, mock_embedder):
        count = await full_context.ingest_document("some text", "my_doc.md")
        assert count == 3
        mock_knowledge_store.ingest.assert_called_once_with(
            "some text", "my_doc.md", mock_embedder,
        )

    async def test_no_knowledge_store_returns_zero(self, bare_context):
        count = await bare_context.ingest_document("text", "src")
        assert count == 0

    async def test_no_embedder_returns_zero(self, executor, mock_knowledge_store):
        ctx = SkillContext(
            executor, "test", knowledge_store=mock_knowledge_store,
        )
        count = await ctx.ingest_document("text", "src")
        assert count == 0


# --- search_history ---

class TestSearchHistory:
    async def test_returns_results(self, full_context, mock_session_manager):
        results = await full_context.search_history("hello")
        assert len(results) == 1
        assert results[0]["content"] == "hello"
        mock_session_manager.search_history.assert_called_once_with(
            "hello", limit=10,
        )

    async def test_custom_limit(self, full_context, mock_session_manager):
        await full_context.search_history("q", limit=3)
        mock_session_manager.search_history.assert_called_once_with("q", limit=3)

    async def test_no_session_manager_returns_empty(self, bare_context):
        results = await bare_context.search_history("test")
        assert results == []


# --- schedule_task ---

class TestScheduleTask:
    def test_adds_schedule(self, full_context, mock_scheduler):
        result = full_context.schedule_task(
            "Test check", "check", "chan1",
            cron="0 * * * *", tool_name="check_disk", tool_input={"host": "server"},
        )
        assert result["id"] == "abc123"
        mock_scheduler.add.assert_called_once_with(
            "Test check", "check", "chan1",
            cron="0 * * * *", tool_name="check_disk", tool_input={"host": "server"},
        )

    def test_no_scheduler_returns_none(self, bare_context):
        result = bare_context.schedule_task("desc", "reminder", "chan1", cron="0 * * * *")
        assert result is None

    def test_passes_trigger_kwarg(self, full_context, mock_scheduler):
        full_context.schedule_task(
            "On push", "workflow", "chan1",
            trigger={"source": "gitea", "event": "push"},
            steps=[{"tool_name": "check_disk", "tool_input": {"host": "server"}}],
        )
        call_kwargs = mock_scheduler.add.call_args
        assert call_kwargs[1]["trigger"] == {"source": "gitea", "event": "push"}


# --- list_schedules ---

class TestListSchedules:
    def test_returns_list(self, full_context, mock_scheduler):
        result = full_context.list_schedules()
        assert len(result) == 1
        assert result[0]["id"] == "abc123"

    def test_no_scheduler_returns_empty(self, bare_context):
        result = bare_context.list_schedules()
        assert result == []


# --- delete_schedule ---

class TestDeleteSchedule:
    def test_deletes_and_returns_true(self, full_context, mock_scheduler):
        result = full_context.delete_schedule("abc123")
        assert result is True
        mock_scheduler.delete.assert_called_once_with("abc123")

    def test_no_scheduler_returns_false(self, bare_context):
        result = bare_context.delete_schedule("abc123")
        assert result is False


# --- execute_tool ---

class TestExecuteTool:
    async def test_calls_executor(self, full_context):
        with patch.object(
            full_context._executor, "execute", new_callable=AsyncMock, return_value="file ok",
        ) as mock_exec:
            result = await full_context.execute_tool("read_file", {"host": "server", "path": "/tmp/test"})
            assert result == "file ok"
            mock_exec.assert_called_once_with("read_file", {"host": "server", "path": "/tmp/test"})

    async def test_default_empty_input(self, full_context):
        with patch.object(
            full_context._executor, "execute", new_callable=AsyncMock, return_value="ok",
        ) as mock_exec:
            await full_context.execute_tool("search_knowledge")
            mock_exec.assert_called_once_with("search_knowledge", {})


# --- SkillManager.set_services wiring ---

class TestSkillManagerSetServices:
    def test_set_services_stores_refs(self, tmp_dir, tools_config):
        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(tmp_dir / "skills"), executor)
        ks = MagicMock()
        emb = MagicMock()
        sm = MagicMock()
        sched = MagicMock()
        mgr.set_services(
            knowledge_store=ks, embedder=emb,
            session_manager=sm, scheduler=sched,
        )
        assert mgr._knowledge_store is ks
        assert mgr._embedder is emb
        assert mgr._session_manager is sm
        assert mgr._scheduler is sched

    def test_defaults_are_none(self, tmp_dir, tools_config):
        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(tmp_dir / "skills"), executor)
        assert mgr._knowledge_store is None
        assert mgr._embedder is None
        assert mgr._session_manager is None
        assert mgr._scheduler is None

    async def test_execute_passes_services_to_context(self, tmp_dir, tools_config):
        """When a skill is executed, the SkillContext receives all wired services."""
        executor = ToolExecutor(tools_config)
        mgr = SkillManager(str(tmp_dir / "skills"), executor)

        ks = MagicMock()
        emb = MagicMock()
        sm = MagicMock()
        sched = MagicMock()
        mgr.set_services(knowledge_store=ks, embedder=emb, session_manager=sm, scheduler=sched)

        # Create a skill that inspects its context
        skill_code = '''
SKILL_DEFINITION = {
    "name": "probe_ctx",
    "description": "Probe context services",
    "input_schema": {"type": "object", "properties": {}},
}

async def execute(inp, context):
    parts = []
    parts.append(f"ks={context._knowledge_store is not None}")
    parts.append(f"emb={context._embedder is not None}")
    parts.append(f"sm={context._session_manager is not None}")
    parts.append(f"sched={context._scheduler is not None}")
    return ",".join(parts)
'''
        mgr.create_skill("probe_ctx", skill_code)
        result = await mgr.execute("probe_ctx", {})
        assert "ks=True" in result
        assert "emb=True" in result
        assert "sm=True" in result
        assert "sched=True" in result


# --- Constructor accepts new params ---

class TestSkillContextConstructor:
    def test_accepts_all_new_params(self, executor):
        """SkillContext can be created with all optional service params."""
        ks = MagicMock()
        emb = MagicMock()
        sm = MagicMock()
        sched = MagicMock()
        ctx = SkillContext(
            executor, "test",
            knowledge_store=ks,
            embedder=emb,
            session_manager=sm,
            scheduler=sched,
        )
        assert ctx._knowledge_store is ks
        assert ctx._embedder is emb
        assert ctx._session_manager is sm
        assert ctx._scheduler is sched

    def test_defaults_to_none(self, executor):
        """Without optional params, all service refs are None."""
        ctx = SkillContext(executor, "test")
        assert ctx._knowledge_store is None
        assert ctx._embedder is None
        assert ctx._session_manager is None
        assert ctx._scheduler is None
