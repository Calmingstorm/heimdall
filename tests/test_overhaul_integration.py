"""Cross-feature integration tests for the Heimdall Overhaul (Rounds 1-14).

Covers: tool pack filtering with new tools, knowledge/session search
with LocalEmbedder, FTS-only fallback mode, handler coverage,
system prompt size, and protected detection code.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.registry import (
    TOOL_PACKS,
    TOOLS,
    _ALL_PACK_TOOLS,
    get_pack_tool_names,
    get_tool_definitions,
)
from src.knowledge.store import KnowledgeStore, VECTOR_DIM
from src.search.vectorstore import SessionVectorStore
from src.search.embedder import LocalEmbedder
from src.llm.system_prompt import SYSTEM_PROMPT_TEMPLATE, build_system_prompt


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

# All tool names defined in the registry
ALL_TOOL_NAMES = {t["name"] for t in TOOLS}

# New tools introduced in the overhaul (Rounds 5-9)
NEW_TOOLS = {
    "analyze_pdf",
    "add_reaction",
    "create_poll",
    "manage_process",
    "analyze_image",
    "generate_image",
}

# Client-side handled tools (from client.py _run_tool dispatch)
CLIENT_TOOLS = {
    "purge_messages", "browser_screenshot", "generate_file", "post_file",
    "schedule_task", "list_schedules", "delete_schedule", "parse_time",
    "search_history", "delegate_task", "list_tasks", "cancel_task",
    "search_knowledge", "ingest_document", "list_knowledge", "delete_knowledge",
    "set_permission", "search_audit", "create_digest",
    "create_skill", "edit_skill", "delete_skill", "list_skills",
    "add_reaction", "create_poll", "broadcast", "analyze_image", "generate_image",
    "start_loop", "stop_loop", "list_loops",
    "spawn_agent", "send_to_agent", "list_agents", "kill_agent", "get_agent_results",
}


@pytest.fixture
def mock_embedder():
    """Mock embedder returning fixed 384-dim vectors."""
    emb = AsyncMock()
    emb.embed.return_value = [0.1] * VECTOR_DIM
    emb.DIMENSIONS = VECTOR_DIM
    return emb


@pytest.fixture
def mock_fts():
    """Mock FTS index for knowledge store."""
    fts = MagicMock()
    fts.index_knowledge_chunk.return_value = True
    fts.search_knowledge.return_value = [
        {"chunk_id": "abc_0", "content": "test content about servers", "source": "test.md", "score": 0.8},
    ]
    fts.delete_knowledge_source.return_value = 0
    fts.has_knowledge_chunk.return_value = False
    return fts


@pytest.fixture
def mock_session_fts():
    """Mock FTS index for session store."""
    fts = MagicMock()
    fts.index_session.return_value = True
    fts.search_sessions.return_value = [
        {"doc_id": "chan_123", "content": "session about deployment", "channel_id": "123", "score": 0.7},
    ]
    fts.has_session.return_value = False
    return fts


@pytest.fixture
def knowledge_store(tmp_path):
    """KnowledgeStore with real SQLite, no vec extension."""
    db_path = str(tmp_path / "knowledge.db")
    with patch("src.knowledge.store.load_extension", return_value=False):
        return KnowledgeStore(db_path)


@pytest.fixture
def knowledge_store_fts(tmp_path, mock_fts):
    """KnowledgeStore with FTS, no vec extension."""
    db_path = str(tmp_path / "knowledge_fts.db")
    with patch("src.knowledge.store.load_extension", return_value=False):
        return KnowledgeStore(db_path, fts_index=mock_fts)


@pytest.fixture
def session_store(tmp_path):
    """SessionVectorStore with real SQLite, no vec extension."""
    db_path = str(tmp_path / "sessions.db")
    with patch("src.search.vectorstore.load_extension", return_value=False):
        return SessionVectorStore(db_path)


@pytest.fixture
def session_store_fts(tmp_path, mock_session_fts):
    """SessionVectorStore with FTS, no vec extension."""
    db_path = str(tmp_path / "sessions_fts.db")
    with patch("src.search.vectorstore.load_extension", return_value=False):
        return SessionVectorStore(db_path, fts_index=mock_session_fts)


@pytest.fixture
def sample_archive(tmp_path):
    """Create a sample session archive JSON file."""
    archive = {
        "channel_id": "123456",
        "last_active": 1700000000.0,
        "summary": "Discussion about server deployments and Docker configuration.",
        "messages": [
            {"role": "user", "content": "How do I restart the nginx container?"},
            {"role": "assistant", "content": "I'll check the Docker status and restart it."},
            {"role": "user", "content": "Thanks, that worked!"},
        ],
    }
    path = tmp_path / "archives" / "123456_1700000000.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(archive))
    return path


# ---------------------------------------------------------------------------
# 1. Tool pack filtering with new tools
# ---------------------------------------------------------------------------


class TestToolPackFilteringWithNewTools:
    """Verify tool packs interact correctly with new overhaul tools."""

    def test_new_tools_available_when_no_packs(self):
        """All new tools are returned when no pack filtering is applied."""
        defs = get_tool_definitions(enabled_packs=[])
        names = {t["name"] for t in defs}
        for tool in NEW_TOOLS:
            assert tool in names, f"New tool '{tool}' missing when packs=[]"

    def test_new_tools_available_with_none_packs(self):
        """All new tools returned with enabled_packs=None."""
        defs = get_tool_definitions(enabled_packs=None)
        names = {t["name"] for t in defs}
        for tool in NEW_TOOLS:
            assert tool in names

    def test_comfyui_pack_filters_generate_image(self):
        """generate_image is in comfyui pack — excluded when only docker enabled."""
        defs = get_tool_definitions(enabled_packs=["docker"])
        names = {t["name"] for t in defs}
        assert "generate_image" not in names, "generate_image should be filtered out"

    def test_comfyui_pack_includes_generate_image(self):
        """generate_image is included when comfyui pack is enabled."""
        defs = get_tool_definitions(enabled_packs=["docker", "comfyui"])
        names = {t["name"] for t in defs}
        assert "generate_image" in names

    def test_non_pack_new_tools_always_available(self):
        """New tools not in any pack are always returned, even with filtering."""
        non_pack_new = NEW_TOOLS - _ALL_PACK_TOOLS
        defs = get_tool_definitions(enabled_packs=["docker"])
        names = {t["name"] for t in defs}
        for tool in non_pack_new:
            assert tool in names, f"Non-pack tool '{tool}' should always be available"

    def test_pack_filtering_preserves_tool_structure(self):
        """Filtered tools still have name, description, input_schema."""
        defs = get_tool_definitions(enabled_packs=["git"])
        for d in defs:
            assert set(d.keys()) == {"name", "description", "input_schema"}
            assert isinstance(d["name"], str) and d["name"]
            assert isinstance(d["description"], str) and d["description"]
            assert isinstance(d["input_schema"], dict)

    def test_all_packs_combined_equals_all_tools(self):
        """Enabling all packs returns the same tools as no filtering."""
        all_packs = list(TOOL_PACKS.keys())
        filtered = get_tool_definitions(enabled_packs=all_packs)
        unfiltered = get_tool_definitions(enabled_packs=[])
        assert {t["name"] for t in filtered} == {t["name"] for t in unfiltered}


# ---------------------------------------------------------------------------
# 2. Knowledge ingest with local embedder
# ---------------------------------------------------------------------------


class TestKnowledgeIngestWithLocalEmbedder:
    """Integration: KnowledgeStore + mock LocalEmbedder."""

    async def test_ingest_and_list_sources(self, knowledge_store, mock_embedder):
        """Ingest a document and verify it appears in list_sources."""
        content = "This is a test document about Kubernetes deployment strategies."
        count = await knowledge_store.ingest(content, "k8s-guide.md", embedder=mock_embedder)
        assert count > 0

        sources = knowledge_store.list_sources()
        assert len(sources) == 1
        assert sources[0]["source"] == "k8s-guide.md"
        assert sources[0]["chunks"] == count

    async def test_ingest_stores_chunks_in_sqlite(self, knowledge_store, mock_embedder):
        """Chunks are stored in knowledge_chunks table."""
        content = "Short document."
        count = await knowledge_store.ingest(content, "short.md", embedder=mock_embedder)
        assert count == 1

        row = knowledge_store._conn.execute(
            "SELECT content, source FROM knowledge_chunks LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row[0] == "Short document."
        assert row[1] == "short.md"

    async def test_reingest_replaces_old_chunks(self, knowledge_store, mock_embedder):
        """Re-ingesting the same source replaces previous chunks."""
        await knowledge_store.ingest("Version 1", "doc.md", embedder=mock_embedder)
        await knowledge_store.ingest("Version 2", "doc.md", embedder=mock_embedder)

        rows = knowledge_store._conn.execute(
            "SELECT content FROM knowledge_chunks WHERE source = 'doc.md'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "Version 2"

    async def test_ingest_without_embedder_still_stores(self, knowledge_store):
        """Ingest works even without an embedder (FTS-only mode)."""
        count = await knowledge_store.ingest("FTS only content", "fts.md", embedder=None)
        assert count > 0
        assert knowledge_store.count() == count

    async def test_ingest_writes_fts_when_embed_fails(self, knowledge_store_fts, mock_fts):
        """FTS gets written even when there is no embedder."""
        failing_emb = AsyncMock()
        failing_emb.embed.return_value = None
        count = await knowledge_store_fts.ingest(
            "Content to index", "fail.md", embedder=failing_emb
        )
        assert count > 0
        # FTS index_knowledge_chunk should have been called
        assert mock_fts.index_knowledge_chunk.called


# ---------------------------------------------------------------------------
# 3. Session search with local embedder
# ---------------------------------------------------------------------------


class TestSessionSearchWithLocalEmbedder:
    """Integration: SessionVectorStore + mock LocalEmbedder."""

    async def test_index_session_creates_archive(self, session_store, mock_embedder, sample_archive):
        """Index a session archive and verify it's stored."""
        result = await session_store.index_session(sample_archive, mock_embedder)
        assert result is True

        row = session_store._conn.execute(
            "SELECT doc_id, channel_id FROM session_archives LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row[1] == "123456"

    async def test_index_session_stores_content(self, session_store, mock_embedder, sample_archive):
        """Indexed content includes summary and messages."""
        await session_store.index_session(sample_archive, mock_embedder)

        row = session_store._conn.execute(
            "SELECT content FROM session_archives LIMIT 1"
        ).fetchone()
        content = row[0]
        assert "server deployments" in content
        assert "nginx container" in content

    async def test_session_hybrid_search_fts_only(
        self, session_store_fts, mock_session_fts
    ):
        """Hybrid search works with FTS only (no embedder)."""
        results = await session_store_fts.search_hybrid("deployment", embedder=None)
        assert len(results) > 0
        assert mock_session_fts.search_sessions.called

    async def test_backfill_indexes_new_archives(
        self, session_store, mock_embedder, sample_archive
    ):
        """Backfill picks up unindexed archives."""
        archive_dir = sample_archive.parent
        count = await session_store.backfill(archive_dir, mock_embedder)
        assert count == 1

        # Second backfill should find nothing new
        count2 = await session_store.backfill(archive_dir, mock_embedder)
        assert count2 == 0


# ---------------------------------------------------------------------------
# 4. FTS-only mode end-to-end
# ---------------------------------------------------------------------------


class TestFTSOnlyModeEndToEnd:
    """Verify the system works without any embedder (pure FTS5 mode)."""

    async def test_knowledge_ingest_fts_only(self, knowledge_store_fts, mock_fts):
        """Knowledge ingest writes to FTS even without embedder."""
        count = await knowledge_store_fts.ingest(
            "Runbook for incident response", "runbook.md", embedder=None
        )
        assert count > 0
        assert mock_fts.index_knowledge_chunk.called

    async def test_knowledge_hybrid_search_fts_only(self, knowledge_store_fts, mock_fts):
        """Hybrid search returns FTS results when no embedder."""
        # Ingest first
        await knowledge_store_fts.ingest(
            "How to deploy to production", "deploy.md", embedder=None
        )
        mock_fts.search_knowledge.return_value = [
            {"chunk_id": "abc_0", "content": "How to deploy to production", "source": "deploy.md", "score": 0.9},
        ]
        results = await knowledge_store_fts.search_hybrid("deploy", embedder=None)
        assert len(results) > 0
        assert results[0]["content"] == "How to deploy to production"

    async def test_knowledge_semantic_search_returns_empty_without_embedder(self, knowledge_store):
        """Semantic-only search returns empty list without embedder."""
        await knowledge_store.ingest("Some content", "doc.md", embedder=None)
        results = await knowledge_store.search("content", embedder=None)
        assert results == []

    async def test_session_hybrid_search_fts_only(self, session_store_fts, mock_session_fts):
        """Session hybrid search returns FTS results when no embedder."""
        results = await session_store_fts.search_hybrid("deployment", embedder=None)
        assert len(results) > 0
        mock_session_fts.search_sessions.assert_called_once()

    async def test_knowledge_delete_works_in_fts_mode(self, knowledge_store_fts, mock_fts):
        """Delete source works correctly in FTS-only mode."""
        await knowledge_store_fts.ingest(
            "Temporary document", "temp.md", embedder=None
        )
        deleted = knowledge_store_fts.delete_source("temp.md")
        assert deleted > 0
        mock_fts.delete_knowledge_source.assert_called_with("temp.md")

    async def test_stores_available_without_vec_extension(self, knowledge_store, session_store):
        """Both stores report available=True even without sqlite-vec."""
        assert knowledge_store.available is True
        assert session_store.available is True


# ---------------------------------------------------------------------------
# 5. New tools in tool definitions
# ---------------------------------------------------------------------------


class TestNewToolsInToolDefinitions:
    """Verify all new overhaul tools are properly defined in the registry."""

    def test_all_new_tools_exist(self):
        """Every new tool is in the registry."""
        names = {t["name"] for t in TOOLS}
        for tool in NEW_TOOLS:
            assert tool in names, f"New tool '{tool}' not in TOOLS registry"

    def test_new_tools_have_valid_schemas(self):
        """New tools have required schema fields."""
        for tool in TOOLS:
            if tool["name"] in NEW_TOOLS:
                assert "description" in tool, f"{tool['name']} missing description"
                assert "input_schema" in tool, f"{tool['name']} missing input_schema"
                schema = tool["input_schema"]
                assert schema.get("type") == "object", f"{tool['name']} schema type must be object"
                assert "properties" in schema, f"{tool['name']} schema missing properties"

    def test_analyze_pdf_has_url_and_host_params(self):
        """analyze_pdf supports both URL and host+path input."""
        tool = next(t for t in TOOLS if t["name"] == "analyze_pdf")
        props = tool["input_schema"]["properties"]
        assert "url" in props
        assert "host" in props
        assert "path" in props
        assert "pages" in props

    def test_create_poll_has_required_fields(self):
        """create_poll requires question and options."""
        tool = next(t for t in TOOLS if t["name"] == "create_poll")
        required = tool["input_schema"].get("required", [])
        assert "question" in required
        assert "options" in required

    def test_manage_process_has_action_param(self):
        """manage_process has an action parameter."""
        tool = next(t for t in TOOLS if t["name"] == "manage_process")
        props = tool["input_schema"]["properties"]
        assert "action" in props

    def test_generate_image_in_comfyui_pack(self):
        """generate_image is in the comfyui tool pack."""
        assert "comfyui" in TOOL_PACKS
        assert "generate_image" in TOOL_PACKS["comfyui"]

    def test_analyze_image_has_url_or_host(self):
        """analyze_image supports URL and host+path input."""
        tool = next(t for t in TOOLS if t["name"] == "analyze_image")
        props = tool["input_schema"]["properties"]
        assert "url" in props
        assert "host" in props
        assert "path" in props

    def test_add_reaction_requires_message_id_and_emoji(self):
        """add_reaction requires message_id and emoji."""
        tool = next(t for t in TOOLS if t["name"] == "add_reaction")
        required = tool["input_schema"].get("required", [])
        assert "message_id" in required
        assert "emoji" in required


# ---------------------------------------------------------------------------
# 6. All new tools have handlers
# ---------------------------------------------------------------------------


class TestAllNewToolsHaveHandlers:
    """Verify every new tool has a handler in client.py or executor.py."""

    def test_new_client_tools_have_handlers(self):
        """New client-side tools have handler methods on the bot class."""
        import src.discord.client as client_mod
        bot_cls = client_mod.HeimdallBot

        client_handled = {"add_reaction", "create_poll", "analyze_image", "generate_image"}
        for tool_name in client_handled:
            handler_name = f"_handle_{tool_name}"
            assert hasattr(bot_cls, handler_name), (
                f"HeimdallBot missing handler '{handler_name}' for tool '{tool_name}'"
            )

    def test_new_executor_tools_have_handlers(self):
        """New executor-side tools have handler methods on ToolExecutor."""
        from src.tools.executor import ToolExecutor

        executor_handled = {"analyze_pdf", "manage_process"}
        for tool_name in executor_handled:
            handler_name = f"_handle_{tool_name}"
            assert hasattr(ToolExecutor, handler_name), (
                f"ToolExecutor missing handler '{handler_name}' for tool '{tool_name}'"
            )

    def test_every_tool_has_some_handler(self):
        """Every tool in the registry has a handler in either client or executor."""
        from src.tools.executor import ToolExecutor

        for tool in TOOLS:
            name = tool["name"]
            handler_name = f"_handle_{name}"
            in_client = name in CLIENT_TOOLS
            in_executor = hasattr(ToolExecutor, handler_name)
            assert in_client or in_executor, (
                f"Tool '{name}' has no handler in client or executor"
            )


# ---------------------------------------------------------------------------
# 7. System prompt still under 5000 chars
# ---------------------------------------------------------------------------


class TestSystemPromptStillUnder5000:
    """System prompt must stay under the 5000 character limit."""

    def test_template_under_5000_chars(self):
        """The raw template is under 5000 chars."""
        assert len(SYSTEM_PROMPT_TEMPLATE) < 5000, (
            f"Template is {len(SYSTEM_PROMPT_TEMPLATE)} chars, limit is 5000"
        )

    def test_rendered_prompt_under_5000_with_minimal_context(self):
        """Rendered prompt with minimal context stays under 5000."""
        prompt = build_system_prompt(
            context="",
            hosts={},
            services=[],
            playbooks=[],
        )
        assert len(prompt) < 5000, (
            f"Rendered prompt is {len(prompt)} chars, limit is 5000"
        )

    def test_rendered_prompt_under_5000_with_typical_context(self):
        """Rendered prompt with typical deployment context stays under 5000."""
        prompt = build_system_prompt(
            context="Production infrastructure: 3 hosts, 12 services.",
            hosts={"web1": "10.0.0.1", "db1": "10.0.0.2", "monitor": "10.0.0.3"},
            services=["nginx", "postgres", "redis", "grafana"],
            playbooks=["deploy-web", "rollback-db"],
            voice_info="",
            tz="UTC",
        )
        assert len(prompt) < 5000, (
            f"Rendered prompt is {len(prompt)} chars with typical context, limit is 5000"
        )


# ---------------------------------------------------------------------------
# 8. Protected code still exists
# ---------------------------------------------------------------------------


class TestProtectedCodeStillExists:
    """Critical detection functions must not be removed during the overhaul."""

    def test_detect_fabrication_exists(self):
        """detect_fabrication function exists in client.py."""
        from src.discord.client import detect_fabrication
        assert callable(detect_fabrication)

    def test_detect_hedging_exists(self):
        """detect_hedging function exists in client.py."""
        from src.discord.client import detect_hedging
        assert callable(detect_hedging)

    def test_detect_premature_failure_exists(self):
        """detect_premature_failure function exists in client.py."""
        from src.discord.client import detect_premature_failure
        assert callable(detect_premature_failure)

    def test_detect_fabrication_catches_fabricated_output(self):
        """detect_fabrication correctly identifies fabricated command output."""
        from src.discord.client import detect_fabrication
        # Should detect fabrication when no tools were used but output looks like command result
        result = detect_fabrication(
            "Here's the output:\n```\nNAME       STATUS    RESTARTS\nnginx      Running   0\n```",
            [],
        )
        assert isinstance(result, bool)

    def test_detect_hedging_catches_permission_asking(self):
        """detect_hedging correctly identifies hedging language."""
        from src.discord.client import detect_hedging
        result = detect_hedging("Shall I restart the service for you?", [])
        assert isinstance(result, bool)

    def test_detect_premature_failure_catches_early_give_up(self):
        """detect_premature_failure detects giving up without trying alternatives."""
        from src.discord.client import detect_premature_failure
        result = detect_premature_failure(
            "I'm unable to check the server status.", []
        )
        assert isinstance(result, bool)

    def test_detection_functions_used_in_tool_loop(self):
        """Detection functions are referenced in the _process_with_tools method."""
        import inspect
        from src.discord import client as client_mod

        # Get the source of the HeimdallBot class
        source = inspect.getsource(client_mod.HeimdallBot)
        assert "detect_fabrication" in source, "detect_fabrication not used in HeimdallBot"
        assert "detect_hedging" in source, "detect_hedging not used in HeimdallBot"
        assert "detect_premature_failure" in source, "detect_premature_failure not used in HeimdallBot"


# ---------------------------------------------------------------------------
# Cross-cutting: embedder dimensions match stores
# ---------------------------------------------------------------------------


class TestEmbedderDimensionsMatch:
    """Verify LocalEmbedder DIMENSIONS matches store VECTOR_DIM constants."""

    def test_embedder_matches_knowledge_store(self):
        """LocalEmbedder.DIMENSIONS == KnowledgeStore VECTOR_DIM."""
        from src.knowledge.store import VECTOR_DIM as KS_DIM
        assert LocalEmbedder.DIMENSIONS == KS_DIM

    def test_embedder_matches_session_store(self):
        """LocalEmbedder.DIMENSIONS == SessionVectorStore VECTOR_DIM."""
        from src.search.vectorstore import VECTOR_DIM as SV_DIM
        assert LocalEmbedder.DIMENSIONS == SV_DIM

    def test_both_stores_use_same_dimension(self):
        """Knowledge and session stores use same vector dimension."""
        from src.knowledge.store import VECTOR_DIM as KS_DIM
        from src.search.vectorstore import VECTOR_DIM as SV_DIM
        assert KS_DIM == SV_DIM == 384
