"""Round 18 validation tests — full test pass + backward compatibility + FTS-only + tool packs.

Comprehensive tests verifying:
1. Backward compatibility: no tool_packs config = all tools loaded
2. FTS-only mode works end-to-end for knowledge and session stores
3. Tool pack filtering is correct and complete
4. System invariants hold (prompt size, detection functions, tool_choice)
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.knowledge.store import KnowledgeStore, CHUNK_SIZE, VECTOR_DIM
from src.search.fts import FullTextIndex
from src.search.vectorstore import SessionVectorStore
from src.tools.registry import (
    TOOL_PACKS,
    TOOLS,
    get_pack_tool_names,
    get_tool_definitions,
)

# ---------- helpers -------------------------------------------------------

_ALL_PACK_TOOLS = {name for tools in TOOL_PACKS.values() for name in tools}


def _patch_load_extension():
    """Patch sqlite-vec load_extension to fail (simulate missing extension)."""
    return patch("src.search.sqlite_vec.load_extension", return_value=False)


# ===========================================================================
# Section 1: Backward Compatibility — no tool_packs = all tools
# ===========================================================================


class TestBackwardCompatibility:
    """Verify that empty/absent/None tool_packs returns ALL tools."""

    def test_no_packs_returns_all_tools(self):
        all_tools = get_tool_definitions()
        assert len(all_tools) == len(TOOLS)

    def test_empty_list_returns_all_tools(self):
        all_tools = get_tool_definitions(enabled_packs=[])
        assert len(all_tools) == len(TOOLS)

    def test_none_returns_all_tools(self):
        all_tools = get_tool_definitions(enabled_packs=None)
        assert len(all_tools) == len(TOOLS)

    def test_all_tool_names_present_no_packs(self):
        all_tools = get_tool_definitions()
        names = {t["name"] for t in all_tools}
        expected = {t["name"] for t in TOOLS}
        assert names == expected

    def test_tool_definitions_have_required_fields(self):
        for tool in get_tool_definitions():
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert isinstance(tool["name"], str)
            assert isinstance(tool["description"], str)
            assert isinstance(tool["input_schema"], dict)

    def test_default_config_loads_all_tools(self):
        """Config with default tool_packs (empty list) should load all tools."""
        from src.config.schema import ToolsConfig
        tc = ToolsConfig()
        assert tc.tool_packs == []
        # Empty list = all tools
        tools = get_tool_definitions(enabled_packs=tc.tool_packs)
        assert len(tools) == len(TOOLS)


# ===========================================================================
# Section 2: Tool Pack Filtering
# ===========================================================================


class TestToolPackFiltering:
    """Verify tool pack filtering logic."""

    def test_single_pack_filters_correctly(self):
        """Enabling one pack returns core + that pack's tools."""
        docker_tools = get_tool_definitions(enabled_packs=["docker"])
        docker_names = {t["name"] for t in docker_tools}

        # All docker tools present
        for name in TOOL_PACKS["docker"]:
            assert name in docker_names, f"Docker tool {name} missing"

        # No tools from OTHER packs (except those in docker)
        for pack, tools in TOOL_PACKS.items():
            if pack == "docker":
                continue
            for name in tools:
                assert name not in docker_names, f"Non-docker tool {name} found"

    def test_multiple_packs_combine(self):
        """Enabling multiple packs returns core + both packs' tools."""
        combined = get_tool_definitions(enabled_packs=["docker", "git"])
        names = {t["name"] for t in combined}

        for name in TOOL_PACKS["docker"]:
            assert name in names
        for name in TOOL_PACKS["git"]:
            assert name in names

        # No systemd/incus/etc
        for name in TOOL_PACKS["systemd"]:
            assert name not in names

    def test_core_tools_always_present(self):
        """Core tools (not in any pack) are always present regardless of packs."""
        core_names = {t["name"] for t in TOOLS} - _ALL_PACK_TOOLS

        # With single pack
        filtered = get_tool_definitions(enabled_packs=["docker"])
        filtered_names = {t["name"] for t in filtered}

        for name in core_names:
            assert name in filtered_names, f"Core tool {name} missing with docker pack"

    def test_unknown_pack_silently_ignored(self):
        """Unknown pack names don't cause errors."""
        tools = get_tool_definitions(enabled_packs=["nonexistent"])
        names = {t["name"] for t in tools}

        # Only core tools — all pack tools filtered out
        for pack_tool in _ALL_PACK_TOOLS:
            assert pack_tool not in names

    def test_all_packs_combined_equals_all_tools(self):
        """Enabling all packs is equivalent to no packs."""
        all_packs = list(TOOL_PACKS.keys())
        with_all_packs = get_tool_definitions(enabled_packs=all_packs)
        without_packs = get_tool_definitions()

        assert len(with_all_packs) == len(without_packs)

    def test_get_pack_tool_names_returns_set(self):
        result = get_pack_tool_names(["docker", "git"])
        assert isinstance(result, set)
        expected = set(TOOL_PACKS["docker"]) | set(TOOL_PACKS["git"])
        assert result == expected

    def test_get_pack_tool_names_empty(self):
        result = get_pack_tool_names([])
        assert result == set()

    def test_no_tool_in_multiple_packs(self):
        """Each tool should be in at most one pack."""
        seen: dict[str, str] = {}
        for pack, tools in TOOL_PACKS.items():
            for tool in tools:
                assert tool not in seen, f"Tool {tool} in both {seen[tool]} and {pack}"
                seen[tool] = pack

    def test_comfyui_pack_contains_generate_image(self):
        """ComfyUI pack should contain generate_image."""
        assert "generate_image" in TOOL_PACKS["comfyui"]

    def test_pack_tools_exist_in_registry(self):
        """All tools named in TOOL_PACKS must exist in the TOOLS list."""
        all_tool_names = {t["name"] for t in TOOLS}
        for pack, tools in TOOL_PACKS.items():
            for tool in tools:
                assert tool in all_tool_names, f"Pack {pack} references nonexistent tool {tool}"


# ===========================================================================
# Section 3: FTS-Only Mode End-to-End (Knowledge Store)
# ===========================================================================


class TestFTSOnlyKnowledgeStore:
    """Verify KnowledgeStore works in FTS-only mode (no sqlite-vec)."""

    @pytest.fixture
    def fts_only_store(self):
        """Create a store with FTS but no vector extension."""
        fts = FullTextIndex(":memory:")
        with _patch_load_extension():
            store = KnowledgeStore(":memory:", fts_index=fts)
        assert store.available
        assert not store._has_vec
        return store

    async def test_ingest_without_vectors(self, fts_only_store):
        count = await fts_only_store.ingest("Python is a programming language", "test.txt")
        assert count == 1

    async def test_ingest_multiple_chunks(self, fts_only_store):
        long_text = "word " * 1000  # >CHUNK_SIZE
        count = await fts_only_store.ingest(long_text, "long.txt")
        assert count > 1

    async def test_hybrid_search_fts_only(self, fts_only_store):
        await fts_only_store.ingest("Python programming with asyncio", "python.txt")
        results = await fts_only_store.search_hybrid("Python", embedder=None)
        assert len(results) > 0
        assert any("Python" in r.get("content", "") or "python" in r.get("content", "").lower()
                    for r in results)

    async def test_hybrid_search_no_results(self, fts_only_store):
        await fts_only_store.ingest("Python programming language", "test.txt")
        results = await fts_only_store.search_hybrid("xyznonexistent", embedder=None)
        assert len(results) == 0

    async def test_semantic_search_returns_empty_without_vec(self, fts_only_store):
        """Semantic-only search returns empty when no vector extension."""
        await fts_only_store.ingest("Python programming", "test.txt")
        results = await fts_only_store.search("Python", embedder=None)
        assert results == []

    async def test_list_sources_after_ingest(self, fts_only_store):
        await fts_only_store.ingest("Content A", "docA.txt")
        await fts_only_store.ingest("Content B", "docB.txt")
        sources = fts_only_store.list_sources()
        names = [s["source"] for s in sources]
        assert "docA.txt" in names
        assert "docB.txt" in names

    async def test_delete_source_fts_only(self, fts_only_store):
        await fts_only_store.ingest("Content to delete", "deleteme.txt")
        assert fts_only_store.count() == 1
        deleted = fts_only_store.delete_source("deleteme.txt")
        assert deleted == 1
        assert fts_only_store.count() == 0

    async def test_reingest_replaces_old(self, fts_only_store):
        await fts_only_store.ingest("Version 1", "doc.txt")
        await fts_only_store.ingest("Version 2 updated", "doc.txt")
        sources = fts_only_store.list_sources()
        assert len(sources) == 1
        assert fts_only_store.count() == 1

    async def test_ingest_and_search_roundtrip(self, fts_only_store):
        """Full roundtrip: ingest, search, verify content."""
        await fts_only_store.ingest(
            "Docker containers provide lightweight virtualization", "docker.txt"
        )
        await fts_only_store.ingest(
            "Kubernetes orchestrates containerized applications", "k8s.txt"
        )
        results = await fts_only_store.search_hybrid("Docker containers", embedder=None)
        assert len(results) >= 1
        contents = [r.get("content", "") for r in results]
        assert any("Docker" in c for c in contents)

    async def test_fts_writes_even_without_embedder(self, fts_only_store):
        """FTS5 should be written to even when no embedder is provided."""
        await fts_only_store.ingest("Testing FTS persistence", "fts_test.txt")
        # Search should find it via FTS
        results = await fts_only_store.search_hybrid("Testing FTS", embedder=None)
        assert len(results) > 0


# ===========================================================================
# Section 4: FTS-Only Mode End-to-End (Session Vector Store)
# ===========================================================================


class TestFTSOnlySessionStore:
    """Verify SessionVectorStore works in FTS-only mode."""

    @pytest.fixture
    def fts_only_session_store(self):
        fts = FullTextIndex(":memory:")
        with _patch_load_extension():
            store = SessionVectorStore(":memory:", fts_index=fts)
        assert store.available
        assert not store._has_vec
        return store

    def _make_archive(self, tmp_path: Path, name: str, channel: str, summary: str,
                      messages: list[dict]) -> Path:
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({
            "channel_id": channel,
            "last_active": 1700000000.0,
            "summary": summary,
            "messages": messages,
        }))
        return p

    async def test_index_session_fts_only(self, fts_only_session_store, tmp_path):
        archive = self._make_archive(
            tmp_path, "ch1_12345", "ch1", "Discussed Python",
            [{"role": "user", "content": "How to use asyncio?"},
             {"role": "assistant", "content": "Use async/await pattern"}]
        )
        result = await fts_only_session_store.index_session(archive, embedder=None)
        assert result is True

    async def test_hybrid_search_fts_only_sessions(self, fts_only_session_store, tmp_path):
        archive = self._make_archive(
            tmp_path, "ch1_12345", "ch1", "Python discussion",
            [{"role": "user", "content": "How to use asyncio for concurrent tasks?"}]
        )
        await fts_only_session_store.index_session(archive, embedder=None)
        results = await fts_only_session_store.search_hybrid("asyncio", embedder=None)
        assert len(results) > 0

    async def test_semantic_search_empty_without_vec(self, fts_only_session_store, tmp_path):
        archive = self._make_archive(
            tmp_path, "ch1_12345", "ch1", "Test",
            [{"role": "user", "content": "Hello world"}]
        )
        await fts_only_session_store.index_session(archive, embedder=None)
        results = await fts_only_session_store.search("hello", embedder=None)
        assert results == []

    async def test_backfill_without_vec(self, fts_only_session_store, tmp_path):
        self._make_archive(
            tmp_path, "ch1_001", "ch1", "First session",
            [{"role": "user", "content": "First conversation"}]
        )
        self._make_archive(
            tmp_path, "ch2_002", "ch2", "Second session",
            [{"role": "user", "content": "Second conversation"}]
        )
        count = await fts_only_session_store.backfill(tmp_path, embedder=None)
        assert count == 2


# ===========================================================================
# Section 5: No-FTS, No-Vec Mode (graceful degradation)
# ===========================================================================


class TestNoSearchBackend:
    """Verify stores work (return empty) when neither FTS nor vec available."""

    async def test_knowledge_hybrid_no_backends(self):
        with _patch_load_extension():
            store = KnowledgeStore(":memory:")  # No FTS, no vec
        await store.ingest("Some content", "test.txt")
        results = await store.search_hybrid("content", embedder=None)
        assert results == []

    async def test_knowledge_ingest_still_works(self):
        with _patch_load_extension():
            store = KnowledgeStore(":memory:")
        count = await store.ingest("Some content", "test.txt")
        assert count == 1  # Metadata still stored

    async def test_knowledge_list_sources_still_works(self):
        with _patch_load_extension():
            store = KnowledgeStore(":memory:")
        await store.ingest("Content", "src.txt")
        sources = store.list_sources()
        assert len(sources) == 1
        assert sources[0]["source"] == "src.txt"


# ===========================================================================
# Section 6: System Invariants
# ===========================================================================


class TestSystemInvariants:
    """Verify critical system invariants still hold."""

    def test_system_prompt_under_5000_chars(self):
        from src.llm.system_prompt import SYSTEM_PROMPT_TEMPLATE
        assert len(SYSTEM_PROMPT_TEMPLATE) < 5000

    def test_detect_fabrication_exists_and_callable(self):
        from src.discord.client import detect_fabrication
        assert callable(detect_fabrication)
        result = detect_fabrication("Some text", [])
        assert isinstance(result, bool)

    def test_detect_hedging_exists_and_callable(self):
        from src.discord.client import detect_hedging
        assert callable(detect_hedging)
        result = detect_hedging("Some text", [])
        assert isinstance(result, bool)

    def test_detect_premature_failure_exists_and_callable(self):
        from src.discord.client import detect_premature_failure
        assert callable(detect_premature_failure)
        result = detect_premature_failure("Some text", [])
        assert isinstance(result, bool)

    def test_tool_choice_is_auto(self):
        """tool_choice should be 'auto' in the Codex client."""
        from src.llm.openai_codex import CodexChatClient
        import inspect
        source = inspect.getsource(CodexChatClient)
        assert '"tool_choice": "auto"' in source or "'tool_choice': 'auto'" in source

    def test_vector_dim_matches_across_stores(self):
        """Knowledge and session stores should use same vector dimensions."""
        from src.search.vectorstore import VECTOR_DIM as SESSION_DIM
        assert VECTOR_DIM == SESSION_DIM == 384

    def test_all_new_tools_exist(self):
        """All overhaul tools are present in the registry."""
        names = {t["name"] for t in TOOLS}
        new_tools = [
            "analyze_pdf", "add_reaction", "create_poll",
            "manage_process", "analyze_image", "generate_image",
        ]
        for tool in new_tools:
            assert tool in names, f"New tool {tool} not found in registry"

    def test_tool_count_is_78(self):
        """Total tool count should be 78."""
        assert len(TOOLS) == 78

    def test_seven_tool_packs(self):
        """There should be 7 tool packs."""
        assert len(TOOL_PACKS) == 7
        expected_packs = {"docker", "systemd", "incus", "ansible", "prometheus", "git", "comfyui"}
        assert set(TOOL_PACKS.keys()) == expected_packs

    def test_pack_tool_count_is_34(self):
        """34 tools total across all packs."""
        all_pack = set()
        for tools in TOOL_PACKS.values():
            all_pack.update(tools)
        assert len(all_pack) == 34

    def test_core_tool_count_is_44(self):
        """44 core tools always available."""
        all_names = {t["name"] for t in TOOLS}
        core = all_names - _ALL_PACK_TOOLS
        assert len(core) == 44

    def test_no_chromadb_imports_in_source(self):
        """No chromadb import statements should remain in source code.

        Note: chromadb_path config field name is kept for backward compat
        (renamed in Round 19). We check for actual chromadb library usage.
        """
        import os
        src_dir = Path(__file__).parent.parent / "src"
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                if f.endswith(".py"):
                    content = (Path(root) / f).read_text()
                    # Check for actual chromadb library imports
                    assert "import chromadb" not in content, \
                        f"chromadb import found in {Path(root) / f}"
                    assert "from chromadb" not in content, \
                        f"chromadb import found in {Path(root) / f}"

    def test_no_ollama_embedder_in_source(self):
        """No OllamaEmbedder references should remain in source code."""
        import os
        src_dir = Path(__file__).parent.parent / "src"
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                if f.endswith(".py"):
                    content = (Path(root) / f).read_text()
                    assert "OllamaEmbedder" not in content, \
                        f"OllamaEmbedder reference found in {Path(root) / f}"


# ===========================================================================
# Section 7: Chunk Text Edge Cases
# ===========================================================================


class TestChunkTextEdgeCases:
    """Validate _chunk_text handling of edge cases."""

    def test_empty_string(self):
        assert KnowledgeStore._chunk_text("") == []

    def test_whitespace_only(self):
        assert KnowledgeStore._chunk_text("   \n\n  ") == []

    def test_short_text_single_chunk(self):
        chunks = KnowledgeStore._chunk_text("Hello world")
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_exactly_chunk_size(self):
        text = "x" * CHUNK_SIZE
        chunks = KnowledgeStore._chunk_text(text)
        assert len(chunks) == 1

    def test_over_chunk_size_splits(self):
        text = "word " * 1000  # ~5000 chars
        chunks = KnowledgeStore._chunk_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= CHUNK_SIZE + 500  # with overlap tolerance


# ===========================================================================
# Section 8: Tool Definition Structure
# ===========================================================================


class TestToolDefinitionStructure:
    """Validate that all tool definitions have proper structure."""

    def test_all_tools_have_input_schema(self):
        for tool in TOOLS:
            assert "input_schema" in tool, f"Tool {tool['name']} missing input_schema"
            assert isinstance(tool["input_schema"], dict)

    def test_all_schemas_have_type_object(self):
        for tool in TOOLS:
            schema = tool["input_schema"]
            assert schema.get("type") == "object", \
                f"Tool {tool['name']} schema type is not 'object'"

    def test_all_schemas_have_properties(self):
        for tool in TOOLS:
            schema = tool["input_schema"]
            assert "properties" in schema, \
                f"Tool {tool['name']} schema missing 'properties'"

    def test_required_fields_are_in_properties(self):
        for tool in TOOLS:
            schema = tool["input_schema"]
            required = schema.get("required", [])
            props = schema.get("properties", {})
            for r in required:
                assert r in props, \
                    f"Tool {tool['name']} requires '{r}' but it's not in properties"
