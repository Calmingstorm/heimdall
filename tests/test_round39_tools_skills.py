"""
Round 39 — Tools + Skills Pages Redesign Tests
Card layout, usage sparklines, code editor with line numbers + syntax highlighting
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
TOOLS_JS = UI_DIR / "js" / "pages" / "tools.js"
SKILLS_JS = UI_DIR / "js" / "pages" / "skills.js"
STYLE_CSS = UI_DIR / "css" / "style.css"


@pytest.fixture(scope="module")
def tools_js():
    return TOOLS_JS.read_text()


@pytest.fixture(scope="module")
def skills_js():
    return SKILLS_JS.read_text()


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
    bot._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command", "description": "Run a shell command", "is_core": True},
        {"name": "read_file", "description": "Read a file", "is_core": True},
        {"name": "incus_list", "description": "List containers", "is_core": False, "pack": "incus"},
    ])
    bot.sessions = MagicMock()
    bot.sessions._sessions = {}
    bot.skill_manager = MagicMock()
    bot.skill_manager.list_skills = MagicMock(return_value=[
        {"name": "joke", "description": "Tell a joke", "loaded_at": "2025-01-01T00:00:00"},
    ])
    bot.skill_manager._skills = {}  # No loaded skill objects — code will be None
    bot.scheduler = MagicMock()
    bot.scheduler.list_all = MagicMock(return_value=[])
    bot.audit = MagicMock()
    bot.audit.count_by_tool = AsyncMock(return_value={"run_command": 42, "read_file": 10})
    bot.tool_executor = MagicMock()
    bot.tool_executor.run_tool = AsyncMock(return_value="ok")
    bot.config = MagicMock()
    bot.config.tools = MagicMock()
    bot.config.monitoring = MagicMock()
    bot.config.monitoring.enabled = False
    bot._cached_merged_tools = None
    bot.agent_manager = MagicMock()
    bot.agent_manager.list_agents = MagicMock(return_value=[])
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
        SessionManager,
        _make_auth_middleware,
        _make_rate_limit_middleware,
        _make_security_headers_middleware,
    )
    web_config = WebConfig(api_token=api_token)
    app = web.Application(middlewares=[
        _make_security_headers_middleware(),
        _make_rate_limit_middleware(),
        _make_auth_middleware(web_config, SessionManager()),
    ])
    setup_api(app, bot)
    return app, bot


# ===================================================================
# Tools Page — Template Structure
# ===================================================================


class TestToolsTemplateStructure:
    """Verify tools.js template has redesigned card layout elements."""

    def test_has_page_wrapper(self, tools_js):
        assert "page-fade-in" in tools_js

    def test_has_view_toggle(self, tools_js):
        assert "tl-view-toggle" in tools_js

    def test_has_card_view_button(self, tools_js):
        assert "viewMode = 'cards'" in tools_js

    def test_has_table_view_button(self, tools_js):
        assert "viewMode = 'table'" in tools_js

    def test_has_view_active_class(self, tools_js):
        assert "tl-view-active" in tools_js

    def test_has_stat_cards(self, tools_js):
        assert "tl-stat-card" in tools_js

    def test_has_stat_value(self, tools_js):
        assert "tl-stat-value" in tools_js

    def test_has_stat_label(self, tools_js):
        assert "tl-stat-label" in tools_js

    def test_has_stat_sparkline(self, tools_js):
        assert "tl-stat-spark" in tools_js

    def test_has_total_sparkline(self, tools_js):
        assert "totalSparkline" in tools_js

    def test_has_packs_section(self, tools_js):
        assert "tl-packs-section" in tools_js

    def test_has_pack_grid(self, tools_js):
        assert "tl-pack-grid" in tools_js

    def test_has_pack_card(self, tools_js):
        assert "tl-pack-card" in tools_js

    def test_has_pack_enabled_class(self, tools_js):
        assert "tl-pack-enabled" in tools_js

    def test_has_pack_disabled_class(self, tools_js):
        assert "tl-pack-disabled" in tools_js

    def test_has_pack_name(self, tools_js):
        assert "tl-pack-name" in tools_js

    def test_has_pack_count(self, tools_js):
        assert "tl-pack-count" in tools_js

    def test_has_pack_tool_tags(self, tools_js):
        assert "tl-pack-tool-tag" in tools_js

    def test_has_category_chips(self, tools_js):
        assert "tl-category-chips" in tools_js

    def test_has_category_chip(self, tools_js):
        assert "tl-category-chip" in tools_js

    def test_has_category_active_class(self, tools_js):
        assert "tl-category-active" in tools_js

    def test_has_search_input(self, tools_js):
        assert "tl-search" in tools_js

    def test_has_group_header(self, tools_js):
        assert "tl-group-header" in tools_js

    def test_has_group_icon(self, tools_js):
        assert "tl-group-icon" in tools_js

    def test_has_group_label(self, tools_js):
        assert "tl-group-label" in tools_js

    def test_has_tool_grid(self, tools_js):
        assert "tl-tool-grid" in tools_js

    def test_has_tool_card(self, tools_js):
        assert "tl-tool-card" in tools_js

    def test_has_tool_card_active(self, tools_js):
        assert "tl-tool-card-active" in tools_js

    def test_has_tool_header(self, tools_js):
        assert "tl-tool-header" in tools_js

    def test_has_tool_name(self, tools_js):
        assert "tl-tool-name" in tools_js

    def test_has_tool_desc(self, tools_js):
        assert "tl-tool-desc" in tools_js

    def test_has_tool_footer(self, tools_js):
        assert "tl-tool-footer" in tools_js

    def test_has_tool_usage_count(self, tools_js):
        assert "tl-tool-usage-count" in tools_js

    def test_has_tool_usage_label(self, tools_js):
        assert "tl-tool-usage-label" in tools_js

    def test_has_tool_sparkline(self, tools_js):
        assert "tl-tool-spark" in tools_js

    def test_has_tool_detail_section(self, tools_js):
        assert "tl-tool-detail" in tools_js

    def test_has_tool_params(self, tools_js):
        assert "tl-tool-params" in tools_js

    def test_has_tool_param_name(self, tools_js):
        assert "tl-tool-param-name" in tools_js

    def test_has_tool_param_type(self, tools_js):
        assert "tl-tool-param-type" in tools_js

    def test_has_tool_param_req(self, tools_js):
        assert "tl-tool-param-req" in tools_js

    def test_has_tool_pack_badge(self, tools_js):
        assert "tl-tool-pack-badge" in tools_js

    def test_has_loading_skeleton(self, tools_js):
        assert "skeleton" in tools_js

    def test_has_error_state(self, tools_js):
        assert "error-state" in tools_js

    def test_has_empty_state(self, tools_js):
        assert "empty-state" in tools_js

    def test_has_table_view_block(self, tools_js):
        assert "hm-table" in tools_js

    def test_has_table_tool_expand_icon(self, tools_js):
        assert "tool-expand-icon" in tools_js


# ===================================================================
# Tools Page — Logic
# ===================================================================


class TestToolsLogic:
    """Verify tools.js JavaScript logic: data refs, computeds, functions."""

    def test_has_viewmode_ref(self, tools_js):
        assert "const viewMode = ref(" in tools_js

    def test_has_active_category_ref(self, tools_js):
        assert "const activeCategory = ref(" in tools_js

    def test_has_usage_history_ref(self, tools_js):
        assert "const usageHistory = ref(" in tools_js

    def test_has_core_count_computed(self, tools_js):
        assert "const coreCount = computed" in tools_js

    def test_has_pack_count_computed(self, tools_js):
        assert "const packCount = computed" in tools_js

    def test_has_total_usage_computed(self, tools_js):
        assert "const totalUsage = computed" in tools_js

    def test_has_tool_sparklines_computed(self, tools_js):
        assert "const toolSparklines = computed" in tools_js

    def test_has_total_sparkline_computed(self, tools_js):
        assert "const totalSparkline = computed" in tools_js

    def test_has_filtered_tools_computed(self, tools_js):
        assert "const filteredTools = computed" in tools_js

    def test_has_used_categories_computed(self, tools_js):
        assert "const usedCategories = computed" in tools_js

    def test_has_grouped_tools_computed(self, tools_js):
        assert "const groupedTools = computed" in tools_js

    def test_has_categorize_function(self, tools_js):
        assert "function categorize(" in tools_js

    def test_has_generate_usage_buckets(self, tools_js):
        assert "function generateUsageBuckets(" in tools_js

    def test_has_hash_code_function(self, tools_js):
        assert "function hashCode(" in tools_js

    def test_has_toggle_expand(self, tools_js):
        assert "function toggleExpand(" in tools_js

    def test_has_fetch_tools(self, tools_js):
        assert "async function fetchTools()" in tools_js

    def test_has_toggle_pack(self, tools_js):
        assert "async function togglePack(" in tools_js

    def test_has_truncate(self, tools_js):
        assert "function truncate(" in tools_js

    def test_has_refresh(self, tools_js):
        assert "function refresh()" in tools_js

    def test_on_mounted(self, tools_js):
        assert "onMounted" in tools_js

    def test_api_get_tools(self, tools_js):
        assert "/api/tools" in tools_js

    def test_api_get_packs(self, tools_js):
        assert "/api/tools/packs" in tools_js

    def test_api_get_stats(self, tools_js):
        assert "/api/tools/stats" in tools_js

    def test_api_put_packs(self, tools_js):
        assert "api.put('/api/tools/packs'" in tools_js


# ===================================================================
# Tools Page — Category System
# ===================================================================


class TestToolCategories:
    """Verify TOOL_CATEGORIES definitions and categorization logic."""

    def test_has_tool_categories_constant(self, tools_js):
        assert "TOOL_CATEGORIES" in tools_js

    def test_has_system_category(self, tools_js):
        assert "'system'" in tools_js

    def test_has_infra_category(self, tools_js):
        assert "'infra'" in tools_js

    def test_has_network_category(self, tools_js):
        assert "'network'" in tools_js

    def test_has_knowledge_category(self, tools_js):
        assert "'knowledge'" in tools_js

    def test_has_discord_category(self, tools_js):
        assert "'discord'" in tools_js

    def test_has_ai_category(self, tools_js):
        assert "'ai'" in tools_js

    def test_has_automation_category(self, tools_js):
        assert "'automation'" in tools_js

    def test_has_other_category(self, tools_js):
        assert "'other'" in tools_js

    def test_categories_have_labels(self, tools_js):
        assert "'System & Commands'" in tools_js
        assert "'Infrastructure'" in tools_js
        assert "'Network & Web'" in tools_js

    def test_categories_have_icons(self, tools_js):
        # Each category has an icon field
        idx = tools_js.index("TOOL_CATEGORIES")
        section = tools_js[idx:idx + 1500]
        assert "icon:" in section

    def test_categories_have_match_functions(self, tools_js):
        idx = tools_js.index("TOOL_CATEGORIES")
        section = tools_js[idx:idx + 1500]
        assert "match:" in section

    def test_other_is_catch_all(self, tools_js):
        # The 'other' category match returns true for everything
        assert "'other'" in tools_js
        idx = tools_js.index("'other'")
        section = tools_js[idx:idx + 100]
        assert "() => true" in section


# ===================================================================
# Tools Page — Sparkline
# ===================================================================


class TestToolSparklines:
    """Verify sparkline SVG generation function exists and works."""

    def test_has_sparkline_svg_function(self, tools_js):
        assert "function sparklineSVG(" in tools_js

    def test_sparkline_generates_svg(self, tools_js):
        assert "tl-sparkline" in tools_js
        assert "<svg" in tools_js
        assert "<polyline" in tools_js

    def test_sparkline_accepts_width_height(self, tools_js):
        idx = tools_js.index("function sparklineSVG(")
        sig = tools_js[idx:idx + 80]
        assert "width" in sig
        assert "height" in sig

    def test_sparkline_accepts_color(self, tools_js):
        idx = tools_js.index("function sparklineSVG(")
        sig = tools_js[idx:idx + 80]
        assert "color" in sig

    def test_sparkline_handles_empty_values(self, tools_js):
        # Should return '' for insufficient data
        idx = tools_js.index("function sparklineSVG(")
        section = tools_js[idx:idx + 200]
        assert "return ''" in section

    def test_sparkline_used_in_tool_cards(self, tools_js):
        assert "toolSparklines[t.name]" in tools_js

    def test_sparkline_used_in_table_view(self, tools_js):
        # Also shown in table view Uses column
        assert "toolSparklines[t.name]" in tools_js

    def test_total_sparkline_in_stat_card(self, tools_js):
        assert "totalSparkline" in tools_js
        assert "usageHistory" in tools_js


# ===================================================================
# Skills Page — Template Structure
# ===================================================================


class TestSkillsTemplateStructure:
    """Verify skills.js template has redesigned card layout elements."""

    def test_has_page_wrapper(self, skills_js):
        assert "page-fade-in" in skills_js

    def test_has_stat_cards(self, skills_js):
        assert "sk-stat-card" in skills_js

    def test_has_stat_value(self, skills_js):
        assert "sk-stat-value" in skills_js

    def test_has_stat_label(self, skills_js):
        assert "sk-stat-label" in skills_js

    def test_has_search_input(self, skills_js):
        assert "sk-search" in skills_js

    def test_has_card_grid(self, skills_js):
        assert "sk-card-grid" in skills_js

    def test_has_skill_card(self, skills_js):
        assert "sk-card" in skills_js

    def test_has_card_header(self, skills_js):
        assert "sk-card-header" in skills_js

    def test_has_card_title_row(self, skills_js):
        assert "sk-card-title-row" in skills_js

    def test_has_card_icon(self, skills_js):
        assert "sk-card-icon" in skills_js

    def test_has_card_name(self, skills_js):
        assert "sk-card-name" in skills_js

    def test_has_card_runs(self, skills_js):
        assert "sk-card-runs" in skills_js

    def test_has_card_actions(self, skills_js):
        assert "sk-card-actions" in skills_js

    def test_has_action_buttons(self, skills_js):
        assert "sk-action-btn" in skills_js
        assert "sk-action-test" in skills_js
        assert "sk-action-code" in skills_js
        assert "sk-action-edit" in skills_js
        assert "sk-action-delete" in skills_js

    def test_has_card_body(self, skills_js):
        assert "sk-card-body" in skills_js

    def test_has_card_desc(self, skills_js):
        assert "sk-card-desc" in skills_js

    def test_has_card_meta(self, skills_js):
        assert "sk-card-meta" in skills_js

    def test_has_card_date(self, skills_js):
        assert "sk-card-date" in skills_js

    def test_has_card_lines_count(self, skills_js):
        assert "sk-card-lines" in skills_js

    def test_has_test_result(self, skills_js):
        assert "sk-test-result" in skills_js

    def test_has_test_pass(self, skills_js):
        assert "sk-test-pass" in skills_js

    def test_has_test_fail(self, skills_js):
        assert "sk-test-fail" in skills_js

    def test_has_test_label(self, skills_js):
        assert "sk-test-label" in skills_js

    def test_has_test_output(self, skills_js):
        assert "sk-test-output" in skills_js

    def test_has_code_container(self, skills_js):
        assert "sk-code-container" in skills_js

    def test_has_code_header(self, skills_js):
        assert "sk-code-header" in skills_js

    def test_has_code_filename(self, skills_js):
        assert "sk-code-filename" in skills_js

    def test_has_code_copy_button(self, skills_js):
        assert "sk-code-copy" in skills_js

    def test_has_line_numbers(self, skills_js):
        assert "sk-line-numbers" in skills_js

    def test_has_code_block(self, skills_js):
        assert "sk-code-block" in skills_js

    def test_has_code_wrap(self, skills_js):
        assert "sk-code-wrap" in skills_js

    def test_has_editor_panel(self, skills_js):
        assert "sk-editor-panel" in skills_js

    def test_has_editor_header(self, skills_js):
        assert "sk-editor-header" in skills_js

    def test_has_editor_title(self, skills_js):
        assert "sk-editor-title" in skills_js

    def test_has_editor_wrap(self, skills_js):
        assert "sk-editor-wrap" in skills_js

    def test_has_editor_gutter(self, skills_js):
        assert "sk-editor-gutter" in skills_js

    def test_has_editor_textarea(self, skills_js):
        assert "sk-editor-textarea" in skills_js

    def test_has_editor_status_bar(self, skills_js):
        assert "sk-editor-status" in skills_js

    def test_has_editor_line_count(self, skills_js):
        assert "sk-editor-line-count" in skills_js

    def test_has_editor_char_count(self, skills_js):
        assert "sk-editor-char-count" in skills_js

    def test_has_validation_box(self, skills_js):
        assert "sk-validation-box" in skills_js

    def test_has_validation_ok(self, skills_js):
        assert "sk-validation-ok" in skills_js

    def test_has_validation_err(self, skills_js):
        assert "sk-validation-err" in skills_js

    def test_has_field_label(self, skills_js):
        assert "sk-field-label" in skills_js

    def test_has_field_hint(self, skills_js):
        assert "sk-field-hint" in skills_js

    def test_has_delete_modal(self, skills_js):
        assert "modal-overlay" in skills_js
        assert "modal-content" in skills_js

    def test_has_loading_skeleton(self, skills_js):
        assert "skeleton" in skills_js

    def test_has_error_state(self, skills_js):
        assert "error-state" in skills_js

    def test_has_empty_state(self, skills_js):
        assert "empty-state" in skills_js


# ===================================================================
# Skills Page — Logic
# ===================================================================


class TestSkillsLogic:
    """Verify skills.js JavaScript logic: data refs, computeds, functions."""

    def test_has_search_ref(self, skills_js):
        assert "const search = ref(" in skills_js

    def test_has_copied_ref(self, skills_js):
        assert "const copied = ref(" in skills_js

    def test_has_editor_ref(self, skills_js):
        assert "const editorRef = ref(" in skills_js

    def test_has_enabled_count_computed(self, skills_js):
        assert "const enabledCount = computed" in skills_js

    def test_has_total_executions_computed(self, skills_js):
        assert "const totalExecutions = computed" in skills_js

    def test_has_total_lines_computed(self, skills_js):
        assert "const totalLines = computed" in skills_js

    def test_has_displayed_skills_computed(self, skills_js):
        assert "const displayedSkills = computed" in skills_js

    def test_has_edit_line_count_computed(self, skills_js):
        assert "const editLineCount = computed" in skills_js

    def test_has_editor_line_nums_computed(self, skills_js):
        assert "const editorLineNums = computed" in skills_js

    def test_has_edit_validation_computed(self, skills_js):
        assert "const editValidation = computed" in skills_js

    def test_has_highlight_function(self, skills_js):
        assert "function highlight(" in skills_js

    def test_has_count_lines_function(self, skills_js):
        assert "function countLines(" in skills_js

    def test_has_get_line_numbers_function(self, skills_js):
        assert "function getLineNumbers(" in skills_js

    def test_has_copy_code_function(self, skills_js):
        assert "async function copyCode(" in skills_js

    def test_has_handle_editor_key(self, skills_js):
        assert "function handleEditorKey(" in skills_js

    def test_has_sync_scroll(self, skills_js):
        assert "function syncScroll(" in skills_js

    def test_has_fetch_skills(self, skills_js):
        assert "async function fetchSkills()" in skills_js

    def test_has_test_skill(self, skills_js):
        assert "async function testSkill(" in skills_js

    def test_has_show_create(self, skills_js):
        assert "function showCreate()" in skills_js

    def test_has_edit_skill(self, skills_js):
        assert "function editSkill(" in skills_js

    def test_has_cancel_edit(self, skills_js):
        assert "function cancelEdit()" in skills_js

    def test_has_save_skill(self, skills_js):
        assert "async function saveSkill()" in skills_js

    def test_has_confirm_delete(self, skills_js):
        assert "function confirmDelete(" in skills_js

    def test_has_do_delete(self, skills_js):
        assert "async function doDelete()" in skills_js

    def test_on_mounted(self, skills_js):
        assert "onMounted" in skills_js

    def test_api_get_skills(self, skills_js):
        assert "/api/skills" in skills_js

    def test_api_post_skills(self, skills_js):
        assert "api.post('/api/skills'" in skills_js

    def test_api_put_skills(self, skills_js):
        assert "api.put(" in skills_js

    def test_api_del_skills(self, skills_js):
        assert "api.del(" in skills_js


# ===================================================================
# Skills Page — Syntax Highlighting
# ===================================================================


class TestSkillsSyntaxHighlighting:
    """Verify Python syntax highlighting function and tokens."""

    def test_has_highlight_python_function(self, skills_js):
        assert "function highlightPython(" in skills_js

    def test_highlights_keywords(self, skills_js):
        assert "sk-kw" in skills_js

    def test_highlights_strings(self, skills_js):
        assert "sk-str" in skills_js

    def test_highlights_comments(self, skills_js):
        assert "sk-cmt" in skills_js

    def test_highlights_decorators(self, skills_js):
        assert "sk-dec" in skills_js

    def test_highlights_numbers(self, skills_js):
        assert "sk-num" in skills_js

    def test_highlights_builtins(self, skills_js):
        assert "sk-builtin" in skills_js

    def test_keyword_list_includes_def(self, skills_js):
        idx = skills_js.index("highlightPython")
        section = skills_js[idx:idx + 800]
        assert "def" in section
        assert "class" in section
        assert "return" in section
        assert "async" in section
        assert "await" in section

    def test_builtin_list_includes_common(self, skills_js):
        idx = skills_js.index("builtins")
        section = skills_js[idx:idx + 500]
        assert "print" in section
        assert "len" in section
        assert "range" in section
        assert "isinstance" in section

    def test_escapes_html(self, skills_js):
        idx = skills_js.index("highlightPython")
        section = skills_js[idx:idx + 300]
        assert "&amp;" in section
        assert "&lt;" in section
        assert "&gt;" in section

    def test_has_line_numbers_function(self, skills_js):
        assert "function lineNumbers(" in skills_js

    def test_line_numbers_generates_sequential(self, skills_js):
        idx = skills_js.index("function lineNumbers(")
        section = skills_js[idx:idx + 200]
        assert "split('\\n')" in section


# ===================================================================
# Skills Page — Code Editor Features
# ===================================================================


class TestSkillsCodeEditor:
    """Verify code editor features: line gutter, tab handling, validation."""

    def test_editor_has_tab_handling(self, skills_js):
        # Tab key inserts 4 spaces
        assert "e.key === 'Tab'" in skills_js
        assert "e.preventDefault()" in skills_js
        assert "'    '" in skills_js

    def test_editor_has_scroll_sync(self, skills_js):
        assert "syncScroll" in skills_js
        assert "previousElementSibling" in skills_js

    def test_validation_checks_skill_definition(self, skills_js):
        assert "SKILL_DEFINITION" in skills_js

    def test_validation_checks_execute_function(self, skills_js):
        assert "async def execute" in skills_js

    def test_validation_returns_valid(self, skills_js):
        assert "valid: true" in skills_js

    def test_validation_returns_invalid(self, skills_js):
        assert "valid: false" in skills_js

    def test_editor_shows_line_count(self, skills_js):
        assert "editLineCount" in skills_js

    def test_editor_shows_char_count(self, skills_js):
        assert "editCode.length" in skills_js

    def test_code_copy_uses_clipboard(self, skills_js):
        assert "navigator.clipboard" in skills_js

    def test_code_copy_has_feedback(self, skills_js):
        assert "copied" in skills_js


# ===================================================================
# CSS — Tools Page Styles
# ===================================================================


class TestToolsCSS:
    """Verify all tl-* CSS classes exist in style.css."""

    def test_tl_stat_card(self, style_css):
        assert ".tl-stat-card" in style_css

    def test_tl_stat_value(self, style_css):
        assert ".tl-stat-value" in style_css

    def test_tl_stat_label(self, style_css):
        assert ".tl-stat-label" in style_css

    def test_tl_stat_spark(self, style_css):
        assert ".tl-stat-spark" in style_css

    def test_tl_view_toggle(self, style_css):
        assert ".tl-view-toggle" in style_css

    def test_tl_view_btn(self, style_css):
        assert ".tl-view-btn" in style_css

    def test_tl_view_active(self, style_css):
        assert ".tl-view-active" in style_css

    def test_tl_section_title(self, style_css):
        assert ".tl-section-title" in style_css

    def test_tl_section_icon(self, style_css):
        assert ".tl-section-icon" in style_css

    def test_tl_packs_section(self, style_css):
        assert ".tl-packs-section" in style_css

    def test_tl_pack_grid(self, style_css):
        assert ".tl-pack-grid" in style_css

    def test_tl_pack_card(self, style_css):
        assert ".tl-pack-card" in style_css

    def test_tl_pack_enabled(self, style_css):
        assert ".tl-pack-enabled" in style_css

    def test_tl_pack_disabled(self, style_css):
        assert ".tl-pack-disabled" in style_css

    def test_tl_pack_name(self, style_css):
        assert ".tl-pack-name" in style_css

    def test_tl_pack_count(self, style_css):
        assert ".tl-pack-count" in style_css

    def test_tl_pack_tools(self, style_css):
        assert ".tl-pack-tools" in style_css

    def test_tl_pack_tool_tag(self, style_css):
        assert ".tl-pack-tool-tag" in style_css

    def test_tl_pack_tool_more(self, style_css):
        assert ".tl-pack-tool-more" in style_css

    def test_tl_search(self, style_css):
        assert ".tl-search" in style_css

    def test_tl_category_chips(self, style_css):
        assert ".tl-category-chips" in style_css

    def test_tl_category_chip(self, style_css):
        assert ".tl-category-chip" in style_css

    def test_tl_category_chip_hover(self, style_css):
        assert ".tl-category-chip:hover" in style_css

    def test_tl_category_active(self, style_css):
        assert ".tl-category-active" in style_css

    def test_tl_group_header(self, style_css):
        assert ".tl-group-header" in style_css

    def test_tl_group_icon(self, style_css):
        assert ".tl-group-icon" in style_css

    def test_tl_group_label(self, style_css):
        assert ".tl-group-label" in style_css

    def test_tl_tool_grid(self, style_css):
        assert ".tl-tool-grid" in style_css

    def test_tl_tool_card(self, style_css):
        assert ".tl-tool-card" in style_css

    def test_tl_tool_card_hover(self, style_css):
        assert ".tl-tool-card:hover" in style_css

    def test_tl_tool_card_active(self, style_css):
        assert ".tl-tool-card-active" in style_css

    def test_tl_tool_header(self, style_css):
        assert ".tl-tool-header" in style_css

    def test_tl_tool_name(self, style_css):
        assert ".tl-tool-name" in style_css

    def test_tl_tool_pack_badge(self, style_css):
        assert ".tl-tool-pack-badge" in style_css

    def test_tl_tool_desc(self, style_css):
        assert ".tl-tool-desc" in style_css

    def test_tl_tool_footer(self, style_css):
        assert ".tl-tool-footer" in style_css

    def test_tl_tool_usage(self, style_css):
        assert ".tl-tool-usage" in style_css

    def test_tl_tool_usage_count(self, style_css):
        assert ".tl-tool-usage-count" in style_css

    def test_tl_tool_usage_zero(self, style_css):
        assert ".tl-tool-usage-zero" in style_css

    def test_tl_tool_usage_label(self, style_css):
        assert ".tl-tool-usage-label" in style_css

    def test_tl_tool_spark(self, style_css):
        assert ".tl-tool-spark" in style_css

    def test_tl_sparkline(self, style_css):
        assert ".tl-sparkline" in style_css

    def test_tl_tool_detail(self, style_css):
        assert ".tl-tool-detail" in style_css

    def test_tl_tool_detail_desc(self, style_css):
        assert ".tl-tool-detail-desc" in style_css

    def test_tl_tool_params(self, style_css):
        assert ".tl-tool-params" in style_css

    def test_tl_tool_params_title(self, style_css):
        assert ".tl-tool-params-title" in style_css

    def test_tl_tool_param(self, style_css):
        assert ".tl-tool-param" in style_css

    def test_tl_tool_param_name(self, style_css):
        assert ".tl-tool-param-name" in style_css

    def test_tl_tool_param_type(self, style_css):
        assert ".tl-tool-param-type" in style_css

    def test_tl_tool_param_req(self, style_css):
        assert ".tl-tool-param-req" in style_css


# ===================================================================
# CSS — Skills Page Styles
# ===================================================================


class TestSkillsCSS:
    """Verify all sk-* CSS classes exist in style.css."""

    def test_sk_stat_card(self, style_css):
        assert ".sk-stat-card" in style_css

    def test_sk_stat_value(self, style_css):
        assert ".sk-stat-value" in style_css

    def test_sk_stat_label(self, style_css):
        assert ".sk-stat-label" in style_css

    def test_sk_search(self, style_css):
        assert ".sk-search" in style_css

    def test_sk_card_grid(self, style_css):
        assert ".sk-card-grid" in style_css

    def test_sk_card(self, style_css):
        assert ".sk-card" in style_css

    def test_sk_card_hover(self, style_css):
        assert ".sk-card:hover" in style_css

    def test_sk_card_tested(self, style_css):
        assert ".sk-card-tested" in style_css

    def test_sk_card_header(self, style_css):
        assert ".sk-card-header" in style_css

    def test_sk_card_title_row(self, style_css):
        assert ".sk-card-title-row" in style_css

    def test_sk_card_icon(self, style_css):
        assert ".sk-card-icon" in style_css

    def test_sk_card_name(self, style_css):
        assert ".sk-card-name" in style_css

    def test_sk_card_runs(self, style_css):
        assert ".sk-card-runs" in style_css

    def test_sk_card_actions(self, style_css):
        assert ".sk-card-actions" in style_css

    def test_sk_action_btn(self, style_css):
        assert ".sk-action-btn" in style_css

    def test_sk_action_btn_hover(self, style_css):
        assert ".sk-action-btn:hover" in style_css

    def test_sk_action_test_hover(self, style_css):
        assert ".sk-action-test:hover" in style_css

    def test_sk_action_code_hover(self, style_css):
        assert ".sk-action-code:hover" in style_css

    def test_sk_action_edit_hover(self, style_css):
        assert ".sk-action-edit:hover" in style_css

    def test_sk_action_delete_hover(self, style_css):
        assert ".sk-action-delete:hover" in style_css

    def test_sk_card_body(self, style_css):
        assert ".sk-card-body" in style_css

    def test_sk_card_desc(self, style_css):
        assert ".sk-card-desc" in style_css

    def test_sk_card_meta(self, style_css):
        assert ".sk-card-meta" in style_css

    def test_sk_test_result(self, style_css):
        assert ".sk-test-result" in style_css

    def test_sk_test_pass(self, style_css):
        assert ".sk-test-pass" in style_css

    def test_sk_test_fail(self, style_css):
        assert ".sk-test-fail" in style_css

    def test_sk_test_label(self, style_css):
        assert ".sk-test-label" in style_css

    def test_sk_test_output(self, style_css):
        assert ".sk-test-output" in style_css

    def test_sk_code_container(self, style_css):
        assert ".sk-code-container" in style_css

    def test_sk_code_header(self, style_css):
        assert ".sk-code-header" in style_css

    def test_sk_code_filename(self, style_css):
        assert ".sk-code-filename" in style_css

    def test_sk_code_copy(self, style_css):
        assert ".sk-code-copy" in style_css

    def test_sk_code_copy_hover(self, style_css):
        assert ".sk-code-copy:hover" in style_css

    def test_sk_code_wrap(self, style_css):
        assert ".sk-code-wrap" in style_css

    def test_sk_line_numbers(self, style_css):
        assert ".sk-line-numbers" in style_css

    def test_sk_code_block(self, style_css):
        assert ".sk-code-block" in style_css

    def test_sk_kw(self, style_css):
        assert ".sk-kw" in style_css

    def test_sk_str(self, style_css):
        assert ".sk-str" in style_css

    def test_sk_cmt(self, style_css):
        assert ".sk-cmt" in style_css

    def test_sk_dec(self, style_css):
        assert ".sk-dec" in style_css

    def test_sk_num(self, style_css):
        assert ".sk-num" in style_css

    def test_sk_builtin(self, style_css):
        assert ".sk-builtin" in style_css

    def test_sk_editor_panel(self, style_css):
        assert ".sk-editor-panel" in style_css

    def test_sk_editor_header(self, style_css):
        assert ".sk-editor-header" in style_css

    def test_sk_editor_title(self, style_css):
        assert ".sk-editor-title" in style_css

    def test_sk_field_label(self, style_css):
        assert ".sk-field-label" in style_css

    def test_sk_field_hint(self, style_css):
        assert ".sk-field-hint" in style_css

    def test_sk_editor_wrap(self, style_css):
        assert ".sk-editor-wrap" in style_css

    def test_sk_editor_gutter(self, style_css):
        assert ".sk-editor-gutter" in style_css

    def test_sk_editor_textarea(self, style_css):
        assert ".sk-editor-textarea" in style_css

    def test_sk_editor_status(self, style_css):
        assert ".sk-editor-status" in style_css

    def test_sk_validation_box(self, style_css):
        assert ".sk-validation-box" in style_css

    def test_sk_validation_ok(self, style_css):
        assert ".sk-validation-ok" in style_css

    def test_sk_validation_err(self, style_css):
        assert ".sk-validation-err" in style_css


# ===================================================================
# CSS — Design Token Usage
# ===================================================================


class TestDesignTokenUsage:
    """Verify new styles use design tokens from the design system."""

    def test_tools_uses_hm_surface(self, style_css):
        idx = style_css.index(".tl-stat-card")
        section = style_css[idx:idx + 200]
        assert "var(--hm-surface)" in section

    def test_tools_uses_hm_border(self, style_css):
        idx = style_css.index(".tl-pack-card")
        section = style_css[idx:idx + 200]
        assert "var(--hm-border)" in section

    def test_tools_uses_hm_accent(self, style_css):
        idx = style_css.index(".tl-tool-usage-count")
        section = style_css[idx:idx + 200]
        assert "var(--hm-accent)" in section

    def test_tools_uses_hm_radius(self, style_css):
        idx = style_css.index(".tl-tool-card {")
        section = style_css[idx:idx + 300]
        assert "var(--hm-radius" in section

    def test_tools_uses_hm_text_dim(self, style_css):
        idx = style_css.index(".tl-stat-label")
        section = style_css[idx:idx + 200]
        assert "var(--hm-text-dim)" in section

    def test_tools_uses_hm_transition(self, style_css):
        idx = style_css.index(".tl-category-chip {")
        section = style_css[idx:idx + 400]
        assert "var(--hm-transition" in section

    def test_skills_uses_hm_surface(self, style_css):
        idx = style_css.index(".sk-stat-card")
        section = style_css[idx:idx + 200]
        assert "var(--hm-surface)" in section

    def test_skills_uses_hm_success(self, style_css):
        idx = style_css.index(".sk-test-pass .sk-test-label")
        section = style_css[idx:idx + 200]
        assert "var(--hm-success)" in section

    def test_skills_uses_hm_danger(self, style_css):
        idx = style_css.index(".sk-test-fail .sk-test-label")
        section = style_css[idx:idx + 200]
        assert "var(--hm-danger)" in section

    def test_skills_uses_hm_font_mono(self, style_css):
        idx = style_css.index(".sk-editor-textarea")
        section = style_css[idx:idx + 400]
        assert "var(--hm-font-mono)" in section

    def test_skills_uses_hm_bg(self, style_css):
        idx = style_css.index(".sk-code-block")
        section = style_css[idx:idx + 200]
        assert "var(--hm-bg)" in section


# ===================================================================
# Backward Compatibility
# ===================================================================


class TestBackwardCompatibility:
    """Ensure old features are preserved after redesign."""

    def test_tools_has_search_filter(self, tools_js):
        assert "search" in tools_js
        assert "filteredTools" in tools_js

    def test_tools_has_pack_toggle(self, tools_js):
        assert "togglePack" in tools_js
        assert "packSaving" in tools_js

    def test_tools_has_expand(self, tools_js):
        assert "toggleExpand" in tools_js
        assert "expanded" in tools_js

    def test_tools_has_refresh(self, tools_js):
        assert "refresh" in tools_js

    def test_tools_api_calls_preserved(self, tools_js):
        assert "api.get('/api/tools')" in tools_js
        assert "api.get('/api/tools/packs')" in tools_js
        assert "api.get('/api/tools/stats')" in tools_js

    def test_tools_has_error_handling(self, tools_js):
        assert "error.value = e.message" in tools_js

    def test_tools_has_loading_state(self, tools_js):
        assert "loading.value = true" in tools_js
        assert "loading.value = false" in tools_js

    def test_skills_has_create_skill(self, skills_js):
        assert "showCreate" in skills_js
        assert "editMode.value = 'create'" in skills_js

    def test_skills_has_edit_skill(self, skills_js):
        assert "editSkill" in skills_js
        assert "editMode.value = 'edit'" in skills_js

    def test_skills_has_delete_skill(self, skills_js):
        assert "confirmDelete" in skills_js
        assert "doDelete" in skills_js

    def test_skills_has_test_skill(self, skills_js):
        assert "testSkill" in skills_js
        assert "testing.value = name" in skills_js

    def test_skills_api_calls_preserved(self, skills_js):
        assert "api.get('/api/skills')" in skills_js
        assert "api.post('/api/skills'" in skills_js
        assert "api.del(" in skills_js

    def test_skills_has_name_validation(self, skills_js):
        assert "'Name is required'" in skills_js

    def test_skills_has_code_validation(self, skills_js):
        assert "'Code is required'" in skills_js

    def test_skills_cancel_edit_preserved(self, skills_js):
        assert "cancelEdit" in skills_js

    def test_skills_success_auto_close(self, skills_js):
        assert "setTimeout" in skills_js

    def test_old_tool_classes_still_in_css(self, style_css):
        # Legacy table-view classes preserved
        assert ".tool-expand-icon" in style_css
        assert ".tool-detail-row" in style_css
        assert ".tool-detail-cell" in style_css

    def test_old_skill_classes_still_in_css(self, style_css):
        assert ".skill-code-block" in style_css
        assert ".skill-card" in style_css
        assert ".skill-editor" in style_css


# ===================================================================
# API Integration
# ===================================================================


class TestToolsStatsAPI:
    """Verify /api/tools/stats returns correct data for sparklines."""

    @pytest.mark.asyncio
    async def test_stats_endpoint_returns_counts(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/tools/stats")
            assert resp.status == 200
            body = await resp.json()
            assert "run_command" in body
            assert body["run_command"] == 42

    @pytest.mark.asyncio
    async def test_stats_endpoint_has_multiple_tools(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/tools/stats")
            body = await resp.json()
            assert "read_file" in body
            assert body["read_file"] == 10

    @pytest.mark.asyncio
    async def test_tools_endpoint_returns_list(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/tools")
            assert resp.status == 200
            body = await resp.json()
            assert isinstance(body, list)
            assert len(body) >= 2

    @pytest.mark.asyncio
    async def test_tools_have_name_and_description(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/tools")
            body = await resp.json()
            for tool in body:
                assert "name" in tool
                assert "description" in tool


class TestSkillsAPI:
    """Verify skills API endpoints work for the redesigned page."""

    @pytest.mark.asyncio
    async def test_skills_endpoint_returns_list(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/skills")
            assert resp.status == 200
            body = await resp.json()
            assert isinstance(body, list)

    @pytest.mark.asyncio
    async def test_skills_have_required_fields(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/skills")
            body = await resp.json()
            if body:
                skill = body[0]
                assert "name" in skill
                assert "description" in skill
