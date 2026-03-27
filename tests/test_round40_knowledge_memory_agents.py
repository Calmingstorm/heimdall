"""
Round 40 — Knowledge + Memory + Agents Pages Redesign Tests
Visual chunk browser, tree view, live agent status cards, new CSS classes
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
KNOWLEDGE_JS = UI_DIR / "js" / "pages" / "knowledge.js"
MEMORY_JS = UI_DIR / "js" / "pages" / "memory.js"
AGENTS_JS = UI_DIR / "js" / "pages" / "agents.js"
APP_JS = UI_DIR / "js" / "app.js"
STYLE_CSS = UI_DIR / "css" / "style.css"


@pytest.fixture(scope="module")
def knowledge_js():
    return KNOWLEDGE_JS.read_text()


@pytest.fixture(scope="module")
def memory_js():
    return MEMORY_JS.read_text()


@pytest.fixture(scope="module")
def agents_js():
    return AGENTS_JS.read_text()


@pytest.fixture(scope="module")
def app_js():
    return APP_JS.read_text()


@pytest.fixture(scope="module")
def style_css():
    return STYLE_CSS.read_text()


# ---------------------------------------------------------------------------
# API test helpers
# ---------------------------------------------------------------------------

def _make_bot():
    bot = MagicMock()
    guild = MagicMock()
    guild.id = 111
    guild.name = "TestGuild"
    guild.member_count = 42
    bot.guilds = [guild]
    bot.is_ready = MagicMock(return_value=True)
    bot._start_time = time.monotonic() - 3600

    # Tools
    bot._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command", "description": "Run a shell command", "is_core": True},
    ])
    bot.sessions = MagicMock()
    bot.sessions._sessions = {}
    bot.skill_manager = MagicMock()
    bot.skill_manager.list_skills = MagicMock(return_value=[])
    bot.skill_manager._skills = {}
    bot.scheduler = MagicMock()
    bot.scheduler.list_all = MagicMock(return_value=[])
    bot.audit = MagicMock()
    bot.audit.count_by_tool = AsyncMock(return_value={})
    bot.tool_executor = MagicMock()
    bot.tool_executor.run_tool = AsyncMock(return_value="ok")
    bot.tool_executor._load_all_memory = MagicMock(return_value={
        "global": {"pref_lang": "python", "timezone": "UTC"},
        "user:123": {"name": "Alice"},
    })
    bot.tool_executor._save_all_memory = MagicMock()
    bot.config = MagicMock()
    bot.config.tools = MagicMock()
    bot.config.tools.tool_packs = []
    bot.config.monitoring = MagicMock()
    bot.config.monitoring.enabled = False
    bot._cached_merged_tools = None

    # Knowledge store
    store = MagicMock()
    store.available = True
    store.list_sources = MagicMock(return_value=[
        {"source": "docs", "chunks": 5, "uploader": "web-api", "ingested_at": "2025-01-01T00:00:00", "preview": "Hello..."},
        {"source": "faq", "chunks": 3, "uploader": "system", "ingested_at": "2025-01-02T00:00:00", "preview": "FAQ..."},
    ])
    store.get_source_chunks = MagicMock(return_value=[
        {"chunk_id": "c1", "content": "First chunk content", "chunk_index": 0, "total_chunks": 2, "ingested_at": "2025-01-01T00:00:00", "char_count": 19},
        {"chunk_id": "c2", "content": "Second chunk content here", "chunk_index": 1, "total_chunks": 2, "ingested_at": "2025-01-01T00:00:00", "char_count": 25},
    ])
    store.get_source_content = MagicMock(return_value="Full document content")
    store.delete_source = MagicMock(return_value=5)
    bot._knowledge_store = store
    bot._embedder = MagicMock()

    # Agent manager (using real-like dict)
    agent_info = MagicMock()
    agent_info.id = "abc12345"
    agent_info.label = "test-agent"
    agent_info.goal = "Deploy the container"
    agent_info.status = "running"
    agent_info.channel_id = "ch1"
    agent_info.requester_id = "u1"
    agent_info.requester_name = "Alice"
    agent_info.iteration_count = 5
    agent_info.tools_used = ["run_command", "read_file"]
    agent_info.created_at = time.time() - 120
    agent_info.ended_at = None
    agent_info.result = ""
    agent_info.error = ""
    agent_info.messages = []

    agent_info2 = MagicMock()
    agent_info2.id = "def67890"
    agent_info2.label = "backup-agent"
    agent_info2.goal = "Run database backup"
    agent_info2.status = "completed"
    agent_info2.channel_id = "ch1"
    agent_info2.requester_id = "u2"
    agent_info2.requester_name = "Bob"
    agent_info2.iteration_count = 10
    agent_info2.tools_used = ["run_command"]
    agent_info2.created_at = time.time() - 300
    agent_info2.ended_at = time.time() - 60
    agent_info2.result = "Backup completed successfully"
    agent_info2.error = ""
    agent_info2.messages = []

    mgr = MagicMock()
    mgr._agents = {"abc12345": agent_info, "def67890": agent_info2}
    mgr.kill = MagicMock(return_value="Kill signal sent to agent 'test-agent'.")
    bot.agent_manager = mgr

    # Others
    bot.process_manager = MagicMock()
    bot.process_manager.list_processes = MagicMock(return_value=[])
    bot.knowledge = MagicMock()
    bot.knowledge.list_sources = AsyncMock(return_value=[])
    bot.learning_reflector = MagicMock()
    bot.learning_reflector.get_all_lessons = MagicMock(return_value=[])
    bot._autonomous_loops = {}

    return bot


def _make_app(bot=None, *, api_token=""):
    if bot is None:
        bot = _make_bot()
    from src.web.api import setup_api
    from src.config.schema import WebConfig
    from src.health.server import (
        _make_auth_middleware,
        _make_rate_limit_middleware,
        _make_security_headers_middleware,
    )
    web_config = WebConfig(api_token=api_token)
    app = web.Application(middlewares=[
        _make_security_headers_middleware(),
        _make_rate_limit_middleware(),
        _make_auth_middleware(web_config),
    ])
    setup_api(app, bot)
    return app, bot


# ===================================================================
# Knowledge Page — Template Structure
# ===================================================================

class TestKnowledgeTemplateStructure:
    """Verify knowledge.js template has tree view and chunk browser elements."""

    def test_has_page_wrapper(self, knowledge_js):
        assert "page-fade-in" in knowledge_js

    def test_has_stats_bar(self, knowledge_js):
        assert "kb-stats-bar" in knowledge_js

    def test_has_stat_value(self, knowledge_js):
        assert "kb-stat-value" in knowledge_js

    def test_has_stat_label(self, knowledge_js):
        assert "kb-stat-label" in knowledge_js

    def test_has_total_chunks_computed(self, knowledge_js):
        assert "totalChunks" in knowledge_js

    def test_has_uploader_count_computed(self, knowledge_js):
        assert "uploaderCount" in knowledge_js

    def test_has_tree_view(self, knowledge_js):
        assert "kb-tree" in knowledge_js

    def test_has_tree_list(self, knowledge_js):
        assert "kb-tree-list" in knowledge_js

    def test_has_tree_node(self, knowledge_js):
        assert "kb-tree-node" in knowledge_js

    def test_has_tree_header(self, knowledge_js):
        assert "kb-tree-header" in knowledge_js

    def test_has_tree_arrow(self, knowledge_js):
        assert "kb-tree-arrow" in knowledge_js

    def test_has_tree_arrow_open_class(self, knowledge_js):
        assert "kb-tree-arrow-open" in knowledge_js

    def test_has_tree_name(self, knowledge_js):
        assert "kb-tree-name" in knowledge_js

    def test_has_tree_icon(self, knowledge_js):
        assert "kb-tree-icon" in knowledge_js

    def test_has_tree_actions(self, knowledge_js):
        assert "kb-tree-actions" in knowledge_js

    def test_has_tree_meta(self, knowledge_js):
        assert "kb-tree-meta" in knowledge_js

    def test_has_tree_preview(self, knowledge_js):
        assert "kb-tree-preview" in knowledge_js

    def test_has_chunk_browser(self, knowledge_js):
        assert "kb-chunk-browser" in knowledge_js

    def test_has_chunk_list(self, knowledge_js):
        assert "kb-chunk-list" in knowledge_js

    def test_has_chunk_item(self, knowledge_js):
        assert "kb-chunk-item" in knowledge_js

    def test_has_chunk_selected_class(self, knowledge_js):
        assert "kb-chunk-selected" in knowledge_js

    def test_has_chunk_index(self, knowledge_js):
        assert "kb-chunk-index" in knowledge_js

    def test_has_chunk_chars(self, knowledge_js):
        assert "kb-chunk-chars" in knowledge_js

    def test_has_chunk_bar(self, knowledge_js):
        assert "kb-chunk-bar" in knowledge_js

    def test_has_chunk_bar_fill(self, knowledge_js):
        assert "kb-chunk-bar-fill" in knowledge_js

    def test_has_chunk_content(self, knowledge_js):
        assert "kb-chunk-content" in knowledge_js

    def test_has_chunk_preview(self, knowledge_js):
        assert "kb-chunk-preview" in knowledge_js

    def test_has_score_badge(self, knowledge_js):
        assert "kb-score-badge" in knowledge_js

    def test_has_search_result_class(self, knowledge_js):
        assert "kb-search-result" in knowledge_js

    def test_has_ingest_form_class(self, knowledge_js):
        assert "kb-ingest-form" in knowledge_js


# ===================================================================
# Knowledge Page — JS Logic
# ===================================================================

class TestKnowledgeJsLogic:
    """Verify knowledge.js has proper setup logic for tree view and chunk browser."""

    def test_has_expanded_ref(self, knowledge_js):
        assert "expanded" in knowledge_js
        assert "ref({})" in knowledge_js

    def test_has_source_chunks_ref(self, knowledge_js):
        assert "sourceChunks" in knowledge_js

    def test_has_loading_chunks_ref(self, knowledge_js):
        assert "loadingChunks" in knowledge_js

    def test_has_selected_chunk_ref(self, knowledge_js):
        assert "selectedChunk" in knowledge_js

    def test_has_toggle_source_function(self, knowledge_js):
        assert "toggleSource" in knowledge_js

    def test_has_chunk_bar_width_function(self, knowledge_js):
        assert "chunkBarWidth" in knowledge_js

    def test_fetches_chunks_api(self, knowledge_js):
        assert "/api/knowledge/" in knowledge_js
        assert "/chunks" in knowledge_js

    def test_has_highlight_terms(self, knowledge_js):
        assert "highlightTerms" in knowledge_js

    def test_has_escape_html(self, knowledge_js):
        assert "escapeHtml" in knowledge_js

    def test_has_search_query_ref(self, knowledge_js):
        assert "searchQuery" in knowledge_js

    def test_has_do_search_function(self, knowledge_js):
        assert "doSearch" in knowledge_js

    def test_has_clear_search(self, knowledge_js):
        assert "clearSearch" in knowledge_js

    def test_has_do_ingest(self, knowledge_js):
        assert "doIngest" in knowledge_js

    def test_has_do_reingest(self, knowledge_js):
        assert "doReingest" in knowledge_js

    def test_has_confirm_delete(self, knowledge_js):
        assert "confirmDelete" in knowledge_js

    def test_has_do_delete(self, knowledge_js):
        assert "doDelete" in knowledge_js

    def test_returns_all_refs(self, knowledge_js):
        # Verify the return statement includes new refs
        for name in ["expanded", "sourceChunks", "loadingChunks", "selectedChunk",
                      "totalChunks", "uploaderCount", "chunkBarWidth", "toggleSource"]:
            assert name in knowledge_js


# ===================================================================
# Memory Page — Template Structure
# ===================================================================

class TestMemoryTemplateStructure:
    """Verify memory.js template has tree view elements."""

    def test_has_page_wrapper(self, memory_js):
        assert "page-fade-in" in memory_js

    def test_has_stats_bar(self, memory_js):
        assert "mem-stats-bar" in memory_js

    def test_has_stat_value(self, memory_js):
        assert "mem-stat-value" in memory_js

    def test_has_stat_label(self, memory_js):
        assert "mem-stat-label" in memory_js

    def test_has_filter_input(self, memory_js):
        assert "filterQuery" in memory_js
        assert "Filter memory keys" in memory_js

    def test_has_tree_view(self, memory_js):
        assert "mem-tree" in memory_js

    def test_has_tree_node(self, memory_js):
        assert "mem-tree-node" in memory_js

    def test_has_tree_header(self, memory_js):
        assert "mem-tree-header" in memory_js

    def test_has_tree_arrow(self, memory_js):
        assert "mem-tree-arrow" in memory_js

    def test_has_tree_arrow_open(self, memory_js):
        assert "mem-tree-arrow-open" in memory_js

    def test_has_tree_entries(self, memory_js):
        assert "mem-tree-entries" in memory_js

    def test_has_tree_entry(self, memory_js):
        assert "mem-tree-entry" in memory_js

    def test_has_tree_entry_selected(self, memory_js):
        assert "mem-tree-entry-selected" in memory_js

    def test_has_tree_entry_header(self, memory_js):
        assert "mem-tree-entry-header" in memory_js

    def test_has_tree_key(self, memory_js):
        assert "mem-tree-key" in memory_js

    def test_has_tree_value(self, memory_js):
        assert "mem-tree-value" in memory_js

    def test_has_tree_edit(self, memory_js):
        assert "mem-tree-edit" in memory_js

    def test_has_tree_entry_actions(self, memory_js):
        assert "mem-tree-entry-actions" in memory_js

    def test_has_tree_loading(self, memory_js):
        assert "mem-tree-loading" in memory_js

    def test_has_tree_empty(self, memory_js):
        assert "mem-tree-empty" in memory_js

    def test_has_add_form(self, memory_js):
        assert "mem-add-form" in memory_js

    def test_has_scope_badge(self, memory_js):
        assert "memory-scope-badge" in memory_js

    def test_has_scope_global(self, memory_js):
        assert "memory-scope-global" in memory_js

    def test_has_scope_user(self, memory_js):
        assert "memory-scope-user" in memory_js

    def test_has_stat_action(self, memory_js):
        assert "mem-stat-action" in memory_js


# ===================================================================
# Memory Page — JS Logic
# ===================================================================

class TestMemoryJsLogic:
    """Verify memory.js has proper setup logic for tree view and filtering."""

    def test_has_filter_query_ref(self, memory_js):
        assert "filterQuery" in memory_js
        assert "ref('')" in memory_js

    def test_has_filtered_entries_function(self, memory_js):
        assert "filteredEntries" in memory_js

    def test_filter_by_key_and_value(self, memory_js):
        # filteredEntries checks both key and value
        assert "e.key.toLowerCase().includes(q)" in memory_js
        assert "e.value" in memory_js

    def test_has_expanded_ref(self, memory_js):
        assert "expanded" in memory_js

    def test_has_toggle_scope(self, memory_js):
        assert "toggleScope" in memory_js

    def test_has_selection_logic(self, memory_js):
        assert "isSelected" in memory_js
        assert "toggleSelect" in memory_js
        assert "isScopeAllSelected" in memory_js
        assert "toggleSelectAll" in memory_js

    def test_has_copy_value(self, memory_js):
        assert "copyValue" in memory_js

    def test_has_bulk_delete(self, memory_js):
        assert "doBulkDelete" in memory_js
        assert "confirmBulkDelete" in memory_js

    def test_returns_filter_query(self, memory_js):
        assert "filterQuery" in memory_js


# ===================================================================
# Agents Page — Template Structure
# ===================================================================

class TestAgentsTemplateStructure:
    """Verify agents.js template has status card elements."""

    def test_has_page_wrapper(self, agents_js):
        assert "page-fade-in" in agents_js

    def test_has_stats_bar(self, agents_js):
        assert "ag-stats-bar" in agents_js

    def test_has_stat_value(self, agents_js):
        assert "ag-stat-value" in agents_js

    def test_has_stat_running_class(self, agents_js):
        assert "ag-stat-running" in agents_js

    def test_has_stat_completed_class(self, agents_js):
        assert "ag-stat-completed" in agents_js

    def test_has_stat_failed_class(self, agents_js):
        assert "ag-stat-failed" in agents_js

    def test_has_filter_bar(self, agents_js):
        assert "ag-filter-bar" in agents_js

    def test_has_filter_btn(self, agents_js):
        assert "ag-filter-btn" in agents_js

    def test_has_filter_active_class(self, agents_js):
        assert "ag-filter-active" in agents_js

    def test_has_filter_count(self, agents_js):
        assert "ag-filter-count" in agents_js

    def test_has_card_grid(self, agents_js):
        assert "ag-card-grid" in agents_js

    def test_has_card(self, agents_js):
        assert "ag-card" in agents_js

    def test_has_card_status_binding(self, agents_js):
        # Dynamic class: 'ag-card-' + agent.status
        assert "'ag-card-' + agent.status" in agents_js

    def test_has_card_header(self, agents_js):
        assert "ag-card-header" in agents_js

    def test_has_card_label(self, agents_js):
        assert "ag-card-label" in agents_js

    def test_has_card_id(self, agents_js):
        assert "ag-card-id" in agents_js

    def test_has_status_dot(self, agents_js):
        assert "ag-status-dot" in agents_js

    def test_has_status_dot_binding(self, agents_js):
        # Dynamic class: 'ag-dot-' + agent.status
        assert "'ag-dot-' + agent.status" in agents_js

    def test_has_status_badge(self, agents_js):
        assert "ag-status-badge" in agents_js

    def test_has_status_badge_binding(self, agents_js):
        # Dynamic class: 'ag-badge-' + agent.status
        assert "'ag-badge-' + agent.status" in agents_js

    def test_has_card_goal(self, agents_js):
        assert "ag-card-goal" in agents_js

    def test_has_progress_bar(self, agents_js):
        assert "ag-progress-bar" in agents_js

    def test_has_progress_fill(self, agents_js):
        assert "ag-progress-fill" in agents_js

    def test_has_card_stats(self, agents_js):
        assert "ag-card-stats" in agents_js

    def test_has_card_stat(self, agents_js):
        assert "ag-card-stat" in agents_js

    def test_has_card_stat_label(self, agents_js):
        assert "ag-card-stat-label" in agents_js

    def test_has_card_stat_value(self, agents_js):
        assert "ag-card-stat-value" in agents_js

    def test_has_tool_chips(self, agents_js):
        assert "ag-tool-chip" in agents_js

    def test_has_card_tools(self, agents_js):
        assert "ag-card-tools" in agents_js

    def test_has_card_meta(self, agents_js):
        assert "ag-card-meta" in agents_js

    def test_has_card_result(self, agents_js):
        assert "ag-card-result" in agents_js

    def test_has_card_error(self, agents_js):
        assert "ag-card-error" in agents_js

    def test_has_result_label(self, agents_js):
        assert "ag-result-label" in agents_js

    def test_has_result_text(self, agents_js):
        assert "ag-result-text" in agents_js

    def test_has_card_actions(self, agents_js):
        assert "ag-card-actions" in agents_js

    def test_has_kill_button(self, agents_js):
        assert "killAgent" in agents_js
        assert "Kill Agent" in agents_js

    def test_has_auto_refresh(self, agents_js):
        assert "autoRefresh" in agents_js
        assert "Auto-refresh" in agents_js

    def test_has_checkbox(self, agents_js):
        assert "ag-checkbox" in agents_js


# ===================================================================
# Agents Page — JS Logic
# ===================================================================

class TestAgentsJsLogic:
    """Verify agents.js has proper setup logic for status cards and filtering."""

    def test_has_agents_ref(self, agents_js):
        assert "agents" in agents_js
        assert "ref([])" in agents_js

    def test_has_status_filter(self, agents_js):
        assert "statusFilter" in agents_js
        assert "ref('all')" in agents_js

    def test_has_running_count(self, agents_js):
        assert "runningCount" in agents_js

    def test_has_completed_count(self, agents_js):
        assert "completedCount" in agents_js

    def test_has_failed_count(self, agents_js):
        assert "failedCount" in agents_js

    def test_has_status_filters_computed(self, agents_js):
        assert "statusFilters" in agents_js

    def test_has_filtered_agents(self, agents_js):
        assert "filteredAgents" in agents_js

    def test_has_format_runtime(self, agents_js):
        assert "formatRuntime" in agents_js

    def test_format_runtime_handles_minutes(self, agents_js):
        assert "Math.floor(seconds / 60)" in agents_js

    def test_has_progress_percent(self, agents_js):
        assert "progressPercent" in agents_js

    def test_has_fetch_agents(self, agents_js):
        assert "fetchAgents" in agents_js
        assert "/api/agents" in agents_js

    def test_has_kill_agent(self, agents_js):
        assert "killAgent" in agents_js

    def test_has_auto_refresh_logic(self, agents_js):
        assert "startAutoRefresh" in agents_js
        assert "stopAutoRefresh" in agents_js
        assert "setInterval" in agents_js

    def test_has_on_unmounted(self, agents_js):
        assert "onUnmounted" in agents_js

    def test_auto_refresh_interval_5s(self, agents_js):
        assert "5000" in agents_js

    def test_failed_filter_includes_timeout_killed(self, agents_js):
        # The 'failed' filter should include timeout and killed
        assert "'failed', 'timeout', 'killed'" in agents_js


# ===================================================================
# App Router — Agents Page Registration
# ===================================================================

class TestAppRouterAgents:
    """Verify agents page is registered in the router."""

    def test_imports_agents_page(self, app_js):
        assert "import AgentsPage from './pages/agents.js'" in app_js

    def test_has_agents_route(self, app_js):
        assert "/agents" in app_js
        assert "AgentsPage" in app_js

    def test_has_agents_label(self, app_js):
        assert "'Agents'" in app_js

    def test_has_agents_icon(self, app_js):
        assert "1F916" in app_js  # robot emoji unicode escape


# ===================================================================
# CSS — Knowledge Classes
# ===================================================================

class TestKnowledgeCss:
    """Verify CSS has knowledge tree and chunk browser classes."""

    def test_has_knowledge_highlight(self, style_css):
        assert ".knowledge-highlight" in style_css

    def test_has_stats_bar(self, style_css):
        assert ".kb-stats-bar" in style_css

    def test_has_stat(self, style_css):
        assert ".kb-stat" in style_css

    def test_has_stat_value(self, style_css):
        assert ".kb-stat-value" in style_css

    def test_has_stat_label(self, style_css):
        assert ".kb-stat-label" in style_css

    def test_has_score_badge(self, style_css):
        assert ".kb-score-badge" in style_css

    def test_has_search_result(self, style_css):
        assert ".kb-search-result" in style_css

    def test_has_ingest_form(self, style_css):
        assert ".kb-ingest-form" in style_css

    def test_has_tree(self, style_css):
        assert ".kb-tree" in style_css

    def test_has_tree_list(self, style_css):
        assert ".kb-tree-list" in style_css

    def test_has_tree_node(self, style_css):
        assert ".kb-tree-node" in style_css

    def test_has_tree_header(self, style_css):
        assert ".kb-tree-header" in style_css

    def test_has_tree_header_hover(self, style_css):
        assert ".kb-tree-header:hover" in style_css

    def test_has_tree_arrow(self, style_css):
        assert ".kb-tree-arrow" in style_css

    def test_has_tree_arrow_open(self, style_css):
        assert ".kb-tree-arrow-open" in style_css

    def test_has_tree_name(self, style_css):
        assert ".kb-tree-name" in style_css

    def test_has_tree_icon(self, style_css):
        assert ".kb-tree-icon" in style_css

    def test_has_tree_actions(self, style_css):
        assert ".kb-tree-actions" in style_css

    def test_has_tree_meta(self, style_css):
        assert ".kb-tree-meta" in style_css

    def test_has_tree_preview(self, style_css):
        assert ".kb-tree-preview" in style_css

    def test_has_chunk_browser(self, style_css):
        assert ".kb-chunk-browser" in style_css

    def test_has_chunk_loading(self, style_css):
        assert ".kb-chunk-loading" in style_css

    def test_has_chunk_list(self, style_css):
        assert ".kb-chunk-list" in style_css

    def test_has_chunk_header(self, style_css):
        assert ".kb-chunk-header" in style_css

    def test_has_chunk_item(self, style_css):
        assert ".kb-chunk-item" in style_css

    def test_has_chunk_item_hover(self, style_css):
        assert ".kb-chunk-item:hover" in style_css

    def test_has_chunk_selected(self, style_css):
        assert ".kb-chunk-selected" in style_css

    def test_has_chunk_item_header(self, style_css):
        assert ".kb-chunk-item-header" in style_css

    def test_has_chunk_index(self, style_css):
        assert ".kb-chunk-index" in style_css

    def test_has_chunk_chars(self, style_css):
        assert ".kb-chunk-chars" in style_css

    def test_has_chunk_bar(self, style_css):
        assert ".kb-chunk-bar" in style_css

    def test_has_chunk_bar_fill(self, style_css):
        assert ".kb-chunk-bar-fill" in style_css

    def test_has_chunk_content(self, style_css):
        assert ".kb-chunk-content" in style_css

    def test_has_chunk_preview(self, style_css):
        assert ".kb-chunk-preview" in style_css


# ===================================================================
# CSS — Memory Classes
# ===================================================================

class TestMemoryCss:
    """Verify CSS has memory tree view classes."""

    def test_has_scope_badge(self, style_css):
        assert ".memory-scope-badge" in style_css

    def test_has_scope_global(self, style_css):
        assert ".memory-scope-global" in style_css

    def test_has_scope_user(self, style_css):
        assert ".memory-scope-user" in style_css

    def test_has_checkbox(self, style_css):
        assert ".memory-checkbox" in style_css

    def test_has_stats_bar(self, style_css):
        assert ".mem-stats-bar" in style_css

    def test_has_stat(self, style_css):
        assert ".mem-stat" in style_css

    def test_has_stat_value(self, style_css):
        assert ".mem-stat-value" in style_css

    def test_has_stat_label(self, style_css):
        assert ".mem-stat-label" in style_css

    def test_has_stat_action(self, style_css):
        assert ".mem-stat-action" in style_css

    def test_has_add_form(self, style_css):
        assert ".mem-add-form" in style_css

    def test_has_tree(self, style_css):
        assert ".mem-tree" in style_css

    def test_has_tree_node(self, style_css):
        assert ".mem-tree-node" in style_css

    def test_has_tree_header(self, style_css):
        assert ".mem-tree-header" in style_css

    def test_has_tree_header_hover(self, style_css):
        assert ".mem-tree-header:hover" in style_css

    def test_has_tree_arrow(self, style_css):
        assert ".mem-tree-arrow" in style_css

    def test_has_tree_arrow_open(self, style_css):
        assert ".mem-tree-arrow-open" in style_css

    def test_has_tree_entries(self, style_css):
        assert ".mem-tree-entries" in style_css

    def test_has_tree_entry(self, style_css):
        assert ".mem-tree-entry" in style_css

    def test_has_tree_entry_hover(self, style_css):
        assert ".mem-tree-entry:hover" in style_css

    def test_has_tree_entry_selected(self, style_css):
        assert ".mem-tree-entry-selected" in style_css

    def test_has_tree_entry_header(self, style_css):
        assert ".mem-tree-entry-header" in style_css

    def test_has_tree_key(self, style_css):
        assert ".mem-tree-key" in style_css

    def test_has_tree_value(self, style_css):
        assert ".mem-tree-value" in style_css

    def test_has_tree_edit(self, style_css):
        assert ".mem-tree-edit" in style_css

    def test_has_tree_entry_actions(self, style_css):
        assert ".mem-tree-entry-actions" in style_css

    def test_has_tree_loading(self, style_css):
        assert ".mem-tree-loading" in style_css

    def test_has_tree_empty(self, style_css):
        assert ".mem-tree-empty" in style_css


# ===================================================================
# CSS — Agents Classes
# ===================================================================

class TestAgentsCss:
    """Verify CSS has agent status card classes."""

    def test_has_checkbox(self, style_css):
        assert ".ag-checkbox" in style_css

    def test_has_stats_bar(self, style_css):
        assert ".ag-stats-bar" in style_css

    def test_has_stat(self, style_css):
        assert ".ag-stat" in style_css

    def test_has_stat_value(self, style_css):
        assert ".ag-stat-value" in style_css

    def test_has_stat_running(self, style_css):
        assert ".ag-stat-running" in style_css

    def test_has_stat_completed(self, style_css):
        assert ".ag-stat-completed" in style_css

    def test_has_stat_failed(self, style_css):
        assert ".ag-stat-failed" in style_css

    def test_has_filter_bar(self, style_css):
        assert ".ag-filter-bar" in style_css

    def test_has_filter_btn(self, style_css):
        assert ".ag-filter-btn" in style_css

    def test_has_filter_btn_hover(self, style_css):
        assert ".ag-filter-btn:hover" in style_css

    def test_has_filter_active(self, style_css):
        assert ".ag-filter-active" in style_css

    def test_has_filter_count(self, style_css):
        assert ".ag-filter-count" in style_css

    def test_has_card_grid(self, style_css):
        assert ".ag-card-grid" in style_css

    def test_has_card(self, style_css):
        assert ".ag-card" in style_css

    def test_has_card_running(self, style_css):
        assert ".ag-card-running" in style_css

    def test_has_card_completed(self, style_css):
        assert ".ag-card-completed" in style_css

    def test_has_card_failed(self, style_css):
        assert ".ag-card-failed" in style_css

    def test_has_card_timeout(self, style_css):
        assert ".ag-card-timeout" in style_css

    def test_has_card_killed(self, style_css):
        assert ".ag-card-killed" in style_css

    def test_has_card_header(self, style_css):
        assert ".ag-card-header" in style_css

    def test_has_card_title_row(self, style_css):
        assert ".ag-card-title-row" in style_css

    def test_has_card_label(self, style_css):
        assert ".ag-card-label" in style_css

    def test_has_card_id(self, style_css):
        assert ".ag-card-id" in style_css

    def test_has_status_dot(self, style_css):
        assert ".ag-status-dot" in style_css

    def test_has_dot_running(self, style_css):
        assert ".ag-dot-running" in style_css

    def test_has_dot_completed(self, style_css):
        assert ".ag-dot-completed" in style_css

    def test_has_dot_failed(self, style_css):
        assert ".ag-dot-failed" in style_css

    def test_has_dot_timeout(self, style_css):
        assert ".ag-dot-timeout" in style_css

    def test_has_dot_killed(self, style_css):
        assert ".ag-dot-killed" in style_css

    def test_has_status_badge(self, style_css):
        assert ".ag-status-badge" in style_css

    def test_has_badge_running(self, style_css):
        assert ".ag-badge-running" in style_css

    def test_has_badge_completed(self, style_css):
        assert ".ag-badge-completed" in style_css

    def test_has_badge_failed(self, style_css):
        assert ".ag-badge-failed" in style_css

    def test_has_badge_timeout(self, style_css):
        assert ".ag-badge-timeout" in style_css

    def test_has_badge_killed(self, style_css):
        assert ".ag-badge-killed" in style_css

    def test_has_card_goal(self, style_css):
        assert ".ag-card-goal" in style_css

    def test_has_progress_bar(self, style_css):
        assert ".ag-progress-bar" in style_css

    def test_has_progress_fill(self, style_css):
        assert ".ag-progress-fill" in style_css

    def test_has_progress_glow_animation(self, style_css):
        assert "ag-progress-glow" in style_css

    def test_has_card_stats(self, style_css):
        assert ".ag-card-stats" in style_css

    def test_has_card_stat(self, style_css):
        assert ".ag-card-stat" in style_css

    def test_has_card_stat_label(self, style_css):
        assert ".ag-card-stat-label" in style_css

    def test_has_card_stat_value(self, style_css):
        assert ".ag-card-stat-value" in style_css

    def test_has_tool_chip(self, style_css):
        assert ".ag-tool-chip" in style_css

    def test_has_card_tools(self, style_css):
        assert ".ag-card-tools" in style_css

    def test_has_card_meta(self, style_css):
        assert ".ag-card-meta" in style_css

    def test_has_card_result(self, style_css):
        assert ".ag-card-result" in style_css

    def test_has_card_error(self, style_css):
        assert ".ag-card-error" in style_css

    def test_has_result_label(self, style_css):
        assert ".ag-result-label" in style_css

    def test_has_result_text(self, style_css):
        assert ".ag-result-text" in style_css

    def test_has_card_actions(self, style_css):
        assert ".ag-card-actions" in style_css

    def test_uses_design_tokens(self, style_css):
        # Verify agent CSS uses design tokens
        import re
        agent_section = style_css[style_css.index("AGENTS PAGE"):]
        agent_section = agent_section[:agent_section.index("================================================================", 50)] if "================================================================" in agent_section[50:] else agent_section
        tokens_found = len(re.findall(r"var\(--hm-", agent_section))
        assert tokens_found >= 20, f"Expected 20+ design token uses, found {tokens_found}"


# ===================================================================
# API — Knowledge Chunks Endpoint
# ===================================================================

class TestKnowledgeChunksApi:
    """Test the new /api/knowledge/{source}/chunks endpoint."""

    @pytest.mark.asyncio
    async def test_list_chunks_returns_array(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge/docs/chunks")
            assert resp.status == 200
            data = await resp.json()
            assert isinstance(data, list)
            assert len(data) == 2

    @pytest.mark.asyncio
    async def test_chunk_has_expected_fields(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge/docs/chunks")
            data = await resp.json()
            chunk = data[0]
            assert "chunk_id" in chunk
            assert "content" in chunk
            assert "chunk_index" in chunk
            assert "total_chunks" in chunk
            assert "char_count" in chunk

    @pytest.mark.asyncio
    async def test_chunks_ordered_by_index(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge/docs/chunks")
            data = await resp.json()
            assert data[0]["chunk_index"] == 0
            assert data[1]["chunk_index"] == 1

    @pytest.mark.asyncio
    async def test_missing_source_returns_404(self):
        app, bot = _make_app()
        bot._knowledge_store.get_source_chunks = MagicMock(return_value=[])
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge/nonexistent/chunks")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_unavailable_store_returns_503(self):
        app, bot = _make_app()
        bot._knowledge_store.available = False
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge/docs/chunks")
            assert resp.status == 503


# ===================================================================
# API — Agents Endpoint
# ===================================================================

class TestAgentsApi:
    """Test the /api/agents endpoint returns proper agent data."""

    @pytest.mark.asyncio
    async def test_list_agents_returns_array(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents")
            assert resp.status == 200
            data = await resp.json()
            assert isinstance(data, list)
            assert len(data) == 2

    @pytest.mark.asyncio
    async def test_agent_has_expected_fields(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents")
            data = await resp.json()
            agent = next(a for a in data if a["id"] == "abc12345")
            assert agent["label"] == "test-agent"
            assert agent["status"] == "running"
            assert agent["iteration_count"] == 5
            assert "tools_used" in agent
            assert "runtime_seconds" in agent
            assert "goal" in agent
            assert "requester_name" in agent

    @pytest.mark.asyncio
    async def test_kill_agent(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/agents/abc12345")
            assert resp.status == 200
            data = await resp.json()
            assert "result" in data

    @pytest.mark.asyncio
    async def test_kill_nonexistent_agent(self):
        app, bot = _make_app()
        bot.agent_manager.kill = MagicMock(return_value="Error: Agent 'zzz' not found.")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/agents/zzz")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_no_agents_returns_empty_list(self):
        app, bot = _make_app()
        bot.agent_manager._agents = {}
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents")
            data = await resp.json()
            assert data == []


# ===================================================================
# API — Memory Endpoint
# ===================================================================

class TestMemoryApi:
    """Test memory API endpoints work with tree view data."""

    @pytest.mark.asyncio
    async def test_list_memory_returns_scopes(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/memory")
            assert resp.status == 200
            data = await resp.json()
            assert "global" in data
            assert data["global"]["count"] == 2
            assert "user:123" in data

    @pytest.mark.asyncio
    async def test_get_memory_value(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/memory/global/pref_lang")
            assert resp.status == 200
            data = await resp.json()
            assert data["value"] == "python"

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/memory/global/nonexistent")
            assert resp.status == 404


# ===================================================================
# Knowledge Store — get_source_chunks Method
# ===================================================================

class TestKnowledgeStoreGetSourceChunks:
    """Test the new get_source_chunks method on KnowledgeStore."""

    def test_returns_list(self):
        from src.knowledge.store import KnowledgeStore
        store = KnowledgeStore.__new__(KnowledgeStore)
        store._conn = MagicMock()
        store._has_vec = False
        store._fts = None
        store._conn.execute.return_value.fetchall.return_value = [
            ("chunk1", "Content one", 0, 2, "2025-01-01T00:00:00"),
            ("chunk2", "Content two here", 1, 2, "2025-01-01T00:00:00"),
        ]
        result = store.get_source_chunks("test-doc")
        assert len(result) == 2
        assert result[0]["chunk_id"] == "chunk1"
        assert result[0]["chunk_index"] == 0
        assert result[0]["char_count"] == 11
        assert result[1]["chunk_id"] == "chunk2"
        assert result[1]["char_count"] == 16

    def test_returns_empty_when_not_available(self):
        from src.knowledge.store import KnowledgeStore
        store = KnowledgeStore.__new__(KnowledgeStore)
        store._conn = None
        result = store.get_source_chunks("test-doc")
        assert result == []

    def test_returns_empty_on_exception(self):
        from src.knowledge.store import KnowledgeStore
        store = KnowledgeStore.__new__(KnowledgeStore)
        store._conn = MagicMock()
        store._has_vec = False
        store._fts = None
        store._conn.execute.side_effect = Exception("db error")
        result = store.get_source_chunks("test-doc")
        assert result == []

    def test_chunk_has_all_fields(self):
        from src.knowledge.store import KnowledgeStore
        store = KnowledgeStore.__new__(KnowledgeStore)
        store._conn = MagicMock()
        store._has_vec = False
        store._fts = None
        store._conn.execute.return_value.fetchall.return_value = [
            ("c1", "Hello world", 0, 1, "2025-06-01T12:00:00"),
        ]
        result = store.get_source_chunks("src")
        chunk = result[0]
        expected_keys = {"chunk_id", "content", "chunk_index", "total_chunks", "ingested_at", "char_count"}
        assert set(chunk.keys()) == expected_keys


# ===================================================================
# Backward Compatibility
# ===================================================================

class TestBackwardCompatibility:
    """Ensure existing CSS classes are preserved."""

    def test_knowledge_highlight_preserved(self, style_css):
        assert ".knowledge-highlight" in style_css

    def test_knowledge_preview_preserved(self, style_css):
        assert ".knowledge-preview" in style_css

    def test_memory_scope_badge_preserved(self, style_css):
        assert ".memory-scope-badge" in style_css

    def test_memory_scope_global_preserved(self, style_css):
        assert ".memory-scope-global" in style_css

    def test_memory_scope_user_preserved(self, style_css):
        assert ".memory-scope-user" in style_css

    def test_memory_checkbox_preserved(self, style_css):
        assert ".memory-checkbox" in style_css


# ===================================================================
# Design Token Usage
# ===================================================================

class TestDesignTokenUsage:
    """Verify all new CSS uses design tokens consistently."""

    def test_knowledge_css_uses_tokens(self, style_css):
        import re
        # Find the knowledge section
        kb_start = style_css.index("KNOWLEDGE PAGE")
        kb_end = style_css.index("MEMORY PAGE", kb_start)
        kb_section = style_css[kb_start:kb_end]
        tokens = len(re.findall(r"var\(--hm-", kb_section))
        assert tokens >= 15, f"Knowledge CSS should use 15+ tokens, found {tokens}"

    def test_memory_css_uses_tokens(self, style_css):
        import re
        mem_start = style_css.index("MEMORY PAGE")
        mem_end = style_css.index("AGENTS PAGE", mem_start)
        mem_section = style_css[mem_start:mem_end]
        tokens = len(re.findall(r"var\(--hm-", mem_section))
        assert tokens >= 15, f"Memory CSS should use 15+ tokens, found {tokens}"

    def test_agents_css_uses_tokens(self, style_css):
        import re
        ag_start = style_css.index("AGENTS PAGE")
        ag_section = style_css[ag_start:]
        # Cut at next section if exists
        if "================================================================" in ag_section[50:]:
            next_section = ag_section.index("================================================================", 50)
            ag_section = ag_section[:next_section]
        tokens = len(re.findall(r"var\(--hm-", ag_section))
        assert tokens >= 20, f"Agents CSS should use 20+ tokens, found {tokens}"


# ===================================================================
# File Structure Checks
# ===================================================================

class TestFileStructure:
    """Verify all required files exist and are non-empty."""

    def test_knowledge_js_exists(self):
        assert KNOWLEDGE_JS.exists()
        assert KNOWLEDGE_JS.stat().st_size > 1000

    def test_memory_js_exists(self):
        assert MEMORY_JS.exists()
        assert MEMORY_JS.stat().st_size > 1000

    def test_agents_js_exists(self):
        assert AGENTS_JS.exists()
        assert AGENTS_JS.stat().st_size > 1000

    def test_agents_js_is_vue_component(self, agents_js):
        assert "export default" in agents_js
        assert "template:" in agents_js
        assert "setup()" in agents_js

    def test_knowledge_js_is_vue_component(self, knowledge_js):
        assert "export default" in knowledge_js
        assert "template:" in knowledge_js
        assert "setup()" in knowledge_js

    def test_memory_js_is_vue_component(self, memory_js):
        assert "export default" in memory_js
        assert "template:" in memory_js
        assert "setup()" in memory_js
