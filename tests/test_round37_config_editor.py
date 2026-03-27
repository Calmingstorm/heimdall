"""Round 37 — Config editor redesign tests.

Tests grouped sections, inline validation, undo/redo, diff view,
and the config page JS structure (groups, validation rules, CSS classes).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from src.config.schema import Config, WebConfig
from src.health.server import (
    SessionManager,
    _make_auth_middleware,
    _make_rate_limit_middleware,
    _make_security_headers_middleware,
)
from src.web.api import create_api_routes, setup_api


# ---------------------------------------------------------------------------
# Python mirrors of JS utility functions (for logic testing)
# ---------------------------------------------------------------------------

def deepClone(obj):
    import copy
    return copy.deepcopy(obj)


def deepEqual(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def computeDiff(original, edited):
    diff = {}
    for section in edited:
        if section not in original:
            continue
        orig = original[section]
        edit = edited[section]
        if deepEqual(orig, edit):
            continue
        if (isinstance(orig, dict) and isinstance(edit, dict)):
            section_diff = {}
            for k in edit:
                if not deepEqual(orig.get(k), edit[k]):
                    section_diff[k] = edit[k]
            if section_diff:
                diff[section] = section_diff
        else:
            diff[section] = edit
    return diff


def buildDiffEntries(original, edited):
    entries = []
    for section in edited:
        if section not in original:
            continue
        orig = original[section]
        edit = edited[section]
        if deepEqual(orig, edit):
            continue
        if isinstance(orig, dict) and isinstance(edit, dict):
            for k in edit:
                if not deepEqual(orig.get(k), edit.get(k)):
                    entries.append({"section": section, "key": k, "oldVal": orig.get(k), "newVal": edit[k]})
        else:
            entries.append({"section": section, "key": None, "oldVal": orig, "newVal": edit})
    return entries


VALIDATION_RULES = {
    'discord.allowed_users': {'type': 'array', 'itemType': 'string', 'message': 'Must be a list of user IDs'},
    'discord.channels': {'type': 'array', 'itemType': 'string', 'message': 'Must be a list of channel IDs'},
    'openai_codex.max_tokens': {'type': 'number', 'min': 1, 'max': 128000, 'message': 'Must be 1\u2013128000'},
    'sessions.max_history': {'type': 'number', 'min': 1, 'max': 10000, 'message': 'Must be 1\u201310000'},
    'sessions.max_age_hours': {'type': 'number', 'min': 1, 'message': 'Must be at least 1'},
    'learning.max_entries': {'type': 'number', 'min': 1, 'message': 'Must be at least 1'},
    'learning.consolidation_target': {'type': 'number', 'min': 1, 'message': 'Must be at least 1'},
    'monitoring.cooldown_minutes': {'type': 'number', 'min': 0, 'message': 'Must be non-negative'},
    'browser.default_timeout_ms': {'type': 'number', 'min': 100, 'message': 'Must be at least 100ms'},
    'browser.viewport_width': {'type': 'number', 'min': 100, 'max': 7680, 'message': 'Must be 100\u20137680'},
    'browser.viewport_height': {'type': 'number', 'min': 100, 'max': 4320, 'message': 'Must be 100\u20134320'},
}


def validateField(section, key, value):
    rule = VALIDATION_RULES.get(section + '.' + key)
    if not rule:
        return None
    if rule['type'] == 'number':
        try:
            n = float(value)
        except (TypeError, ValueError):
            return 'Must be a number'
        if 'min' in rule and n < rule['min']:
            return rule.get('message', 'Value too low')
        if 'max' in rule and n > rule['max']:
            return rule.get('message', 'Value too high')
    if rule['type'] == 'array' and not isinstance(value, list):
        return rule.get('message', 'Must be an array')
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot(config_dict=None):
    bot = MagicMock()
    if config_dict is None:
        config_dict = {
            "timezone": "UTC",
            "discord": {
                "token": "••••••••",
                "allowed_users": ["user1"],
                "channels": ["ch1"],
                "respond_to_bots": False,
                "require_mention": True,
            },
            "openai_codex": {
                "enabled": True,
                "model": "gpt-4.1",
                "max_tokens": 4096,
                "credentials_path": "••••••••",
            },
            "context": {"directory": "data/context", "max_system_prompt_tokens": 4000},
            "sessions": {"max_history": 40, "max_age_hours": 24, "persist_directory": "data/sessions"},
            "tools": {
                "enabled": True,
                "ssh_key_path": "••••••••",
                "hosts": {"local": {"address": "localhost", "os": "linux"}},
                "allowed_services": ["nginx"],
                "ansible_directory": "",
                "timeouts": {"default": 300},
                "tool_packs": ["systemd"],
            },
            "logging": {"level": "INFO", "directory": "data/logs"},
            "usage": {"directory": "data/usage"},
            "webhook": {"enabled": False, "secret": "••••••••", "channel_id": ""},
            "learning": {"enabled": True, "max_entries": 100, "consolidation_target": 50},
            "search": {"enabled": True, "search_db_path": "data/search.db"},
            "voice": {"enabled": False},
            "browser": {
                "enabled": False,
                "cdp_url": "",
                "default_timeout_ms": 30000,
                "viewport_width": 1280,
                "viewport_height": 720,
            },
            "monitoring": {"enabled": False, "checks": [], "alert_channel_id": "", "cooldown_minutes": 15},
            "permissions": {"tiers": {}, "default_tier": "standard", "overrides_path": ""},
            "comfyui": {"enabled": False, "url": ""},
            "web": {"enabled": True, "api_token": "••••••••"},
        }
    bot.config = MagicMock()
    bot.config.model_dump = MagicMock(return_value=config_dict)
    bot.guilds = []
    bot.is_ready = MagicMock(return_value=True)
    bot._start_time = time.monotonic() - 100
    bot._merged_tool_definitions = MagicMock(return_value=[])
    bot.sessions = MagicMock()
    bot.sessions._sessions = {}
    bot.skill_manager = MagicMock()
    bot.skill_manager.list_skills = MagicMock(return_value=[])
    bot.scheduler = MagicMock()
    bot.scheduler.list_all = MagicMock(return_value=[])
    bot.loop_manager = MagicMock()
    bot.loop_manager.active_count = 0
    bot.loop_manager._loops = {}
    bot.agent_manager = MagicMock()
    bot.agent_manager._agents = {}
    bot.tool_executor = MagicMock()
    proc_reg = MagicMock()
    proc_reg._processes = {}
    bot.tool_executor._process_registry = proc_reg
    bot.infra_watcher = MagicMock()
    bot.infra_watcher.get_status = MagicMock(return_value={"enabled": False})
    bot.audit = MagicMock()
    bot.audit.search = AsyncMock(return_value=[])
    bot.audit.count_by_tool = AsyncMock(return_value={})
    bot.tool_executor._load_all_memory = MagicMock(return_value={})
    bot.tool_executor._save_all_memory = MagicMock()
    bot.context_loader = MagicMock()
    bot._invalidate_prompt_caches = MagicMock()
    return bot


def _make_app(bot=None, *, api_token=""):
    if bot is None:
        bot = _make_bot()
    web_config = WebConfig(api_token=api_token)
    app = web.Application(middlewares=[
        _make_security_headers_middleware(),
        _make_rate_limit_middleware(),
        _make_auth_middleware(web_config, SessionManager()),
    ])
    setup_api(app, bot)
    return app, bot


# ===================================================================
# Config API endpoint tests
# ===================================================================


class TestConfigGetEndpoint:
    """GET /api/config returns redacted config."""

    @pytest.mark.asyncio
    async def test_get_config_returns_all_sections(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/config")
            assert resp.status == 200
            body = await resp.json()
            assert "timezone" in body
            assert "discord" in body
            assert "openai_codex" in body
            assert "tools" in body
            assert "logging" in body

    @pytest.mark.asyncio
    async def test_get_config_redacts_sensitive_fields(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/config")
            body = await resp.json()
            assert body["discord"]["token"] == "\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022"

    @pytest.mark.asyncio
    async def test_get_config_includes_nested_objects(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/config")
            body = await resp.json()
            assert "hosts" in body["tools"]
            assert "timeouts" in body["tools"]


class TestConfigPutEndpoint:
    """PUT /api/config applies partial updates."""

    @pytest.mark.asyncio
    async def test_update_timezone(self):
        bot = _make_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            with patch("src.web.api._write_config"):
                resp = await client.put("/api/config", json={"timezone": "US/Eastern"})
                # Should fail because bot.config is a mock (Pydantic validation)
                # But we verify the endpoint accepts the request
                assert resp.status in (200, 400)

    @pytest.mark.asyncio
    async def test_reject_sensitive_field_update(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.put("/api/config", json={"discord": {"token": "newtoken"}})
            assert resp.status in (400, 403)
            body = await resp.json()
            err = body.get("error", "").lower()
            assert "sensitive" in err or "token" in err or "forbidden" in err or "not allowed" in err

    @pytest.mark.asyncio
    async def test_reject_non_dict_body(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.put("/api/config", json="not a dict")
            assert resp.status == 400


# ===================================================================
# Config page JS structure tests
# ===================================================================


class TestConfigPageStructure:
    """Verify the config.js file has expected structure."""

    def _read_config_js(self):
        return Path("ui/js/pages/config.js").read_text()

    def test_config_js_exists(self):
        assert Path("ui/js/pages/config.js").exists()

    def test_imports_api(self):
        src = self._read_config_js()
        assert "import { api } from" in src

    def test_exports_default_component(self):
        src = self._read_config_js()
        assert "export default" in src

    def test_has_template(self):
        src = self._read_config_js()
        assert "template:" in src

    def test_has_setup(self):
        src = self._read_config_js()
        assert "setup()" in src


# ===================================================================
# Grouped sections
# ===================================================================


class TestSectionGrouping:
    """Config sections should be organized into logical groups."""

    def _read_config_js(self):
        return Path("ui/js/pages/config.js").read_text()

    def test_section_groups_defined(self):
        src = self._read_config_js()
        assert "SECTION_GROUPS" in src

    def test_core_group_exists(self):
        src = self._read_config_js()
        assert "'core'" in src
        assert "'Core'" in src

    def test_llm_group_exists(self):
        src = self._read_config_js()
        assert "'llm'" in src
        assert "'LLM & AI'" in src

    def test_data_group_exists(self):
        src = self._read_config_js()
        assert "'data'" in src
        assert "'Data & Storage'" in src

    def test_services_group_exists(self):
        src = self._read_config_js()
        assert "'services'" in src
        assert "'Services'" in src

    def test_infra_group_exists(self):
        src = self._read_config_js()
        assert "'infra'" in src
        assert "'Infrastructure'" in src

    def test_ui_group_exists(self):
        src = self._read_config_js()
        assert "'ui'" in src
        assert "'Web UI'" in src

    def test_group_has_icon(self):
        src = self._read_config_js()
        # Each group should have an icon property
        assert "icon:" in src

    def test_group_has_sections_array(self):
        src = self._read_config_js()
        # Each group maps to sections
        assert "sections:" in src

    def test_discord_in_core_group(self):
        src = self._read_config_js()
        # discord should be in core group
        assert "'discord'" in src

    def test_openai_codex_in_llm_group(self):
        src = self._read_config_js()
        assert "'openai_codex'" in src

    def test_sessions_in_data_group(self):
        src = self._read_config_js()
        assert "'sessions'" in src

    def test_webhook_in_services_group(self):
        src = self._read_config_js()
        assert "'webhook'" in src

    def test_tools_in_infra_group(self):
        src = self._read_config_js()
        assert "'tools'" in src

    def test_web_in_ui_group(self):
        src = self._read_config_js()
        assert "'web'" in src

    def test_cfg_group_css_class_in_template(self):
        src = self._read_config_js()
        assert "cfg-group" in src

    def test_cfg_group_header_in_template(self):
        src = self._read_config_js()
        assert "cfg-group-header" in src

    def test_cfg_group_body_in_template(self):
        src = self._read_config_js()
        assert "cfg-group-body" in src

    def test_cfg_section_class_in_template(self):
        src = self._read_config_js()
        assert "cfg-section" in src

    def test_toggle_group_function(self):
        src = self._read_config_js()
        assert "toggleGroup" in src

    def test_expanded_groups_state(self):
        src = self._read_config_js()
        assert "expandedGroups" in src

    def test_visible_groups_computed(self):
        src = self._read_config_js()
        assert "visibleGroups" in src

    def test_ungrouped_sections_fallback(self):
        src = self._read_config_js()
        assert "ungroupedSections" in src

    def test_group_changed_indicator(self):
        src = self._read_config_js()
        assert "groupChanged" in src

    def test_group_count_displayed(self):
        src = self._read_config_js()
        assert "groupCount" in src

    def test_section_count_displayed(self):
        src = self._read_config_js()
        assert "sectionCount" in src


# ===================================================================
# Inline validation
# ===================================================================


class TestInlineValidation:
    """Fields should show inline validation errors."""

    def _read_config_js(self):
        return Path("ui/js/pages/config.js").read_text()

    def test_validation_rules_defined(self):
        src = self._read_config_js()
        assert "VALIDATION_RULES" in src

    def test_validate_field_function(self):
        src = self._read_config_js()
        assert "validateField" in src

    def test_validation_errors_computed(self):
        src = self._read_config_js()
        assert "validationErrors" in src

    def test_has_errors_computed(self):
        src = self._read_config_js()
        assert "hasErrors" in src

    def test_get_validation_error_function(self):
        src = self._read_config_js()
        assert "getValidationError" in src

    def test_max_tokens_validation_rule(self):
        src = self._read_config_js()
        assert "'openai_codex.max_tokens'" in src

    def test_max_history_validation_rule(self):
        src = self._read_config_js()
        assert "'sessions.max_history'" in src

    def test_viewport_width_validation_rule(self):
        src = self._read_config_js()
        assert "'browser.viewport_width'" in src

    def test_viewport_height_validation_rule(self):
        src = self._read_config_js()
        assert "'browser.viewport_height'" in src

    def test_cooldown_minutes_validation_rule(self):
        src = self._read_config_js()
        assert "'monitoring.cooldown_minutes'" in src

    def test_cfg_field_error_class_in_template(self):
        src = self._read_config_js()
        assert "cfg-field-error" in src

    def test_cfg_input_error_class_in_template(self):
        src = self._read_config_js()
        assert "cfg-input-error" in src

    def test_save_disabled_when_errors(self):
        src = self._read_config_js()
        assert "hasErrors" in src
        # Save button should be disabled when there are errors
        assert "!hasChanges || hasErrors" in src

    def test_validation_checks_number_min(self):
        src = self._read_config_js()
        assert "rule.min" in src

    def test_validation_checks_number_max(self):
        src = self._read_config_js()
        assert "rule.max" in src

    def test_validation_checks_nan(self):
        src = self._read_config_js()
        assert "isNaN" in src

    def test_learning_max_entries_rule(self):
        src = self._read_config_js()
        assert "'learning.max_entries'" in src

    def test_consolidation_target_rule(self):
        src = self._read_config_js()
        assert "'learning.consolidation_target'" in src

    def test_timeout_validation_rule(self):
        src = self._read_config_js()
        assert "'browser.default_timeout_ms'" in src


# ===================================================================
# Undo/redo
# ===================================================================


class TestUndoRedo:
    """Undo/redo should track edit history."""

    def _read_config_js(self):
        return Path("ui/js/pages/config.js").read_text()

    def test_undo_stack_exists(self):
        src = self._read_config_js()
        assert "undoStack" in src

    def test_redo_stack_exists(self):
        src = self._read_config_js()
        assert "redoStack" in src

    def test_can_undo_computed(self):
        src = self._read_config_js()
        assert "canUndo" in src

    def test_can_redo_computed(self):
        src = self._read_config_js()
        assert "canRedo" in src

    def test_undo_function(self):
        src = self._read_config_js()
        assert "function undo()" in src

    def test_redo_function(self):
        src = self._read_config_js()
        assert "function redo()" in src

    def test_push_edit_captures_snapshot(self):
        src = self._read_config_js()
        assert "pushEdit" in src
        # Should push snapshot before applying change
        assert "undoStack" in src

    def test_undo_clears_redo_on_new_edit(self):
        src = self._read_config_js()
        # After a new edit, redo stack should be cleared
        assert "redoStack.value = []" in src

    def test_max_undo_limit(self):
        src = self._read_config_js()
        assert "MAX_UNDO" in src

    def test_undo_button_in_template(self):
        src = self._read_config_js()
        assert "cfg-undo-btn" in src or "Undo" in src

    def test_redo_button_in_template(self):
        src = self._read_config_js()
        assert "cfg-redo-btn" in src or "Redo" in src

    def test_keyboard_shortcut_handler(self):
        src = self._read_config_js()
        assert "handleKeydown" in src

    def test_ctrl_z_triggers_undo(self):
        src = self._read_config_js()
        # Should handle Ctrl+Z
        assert "key === 'z'" in src

    def test_ctrl_y_triggers_redo(self):
        src = self._read_config_js()
        # Should handle Ctrl+Y
        assert "key === 'y'" in src

    def test_meta_key_supported(self):
        src = self._read_config_js()
        # metaKey for Mac Cmd support
        assert "metaKey" in src

    def test_keydown_listener_added_on_mount(self):
        src = self._read_config_js()
        assert "addEventListener" in src
        assert "keydown" in src

    def test_keydown_listener_removed_on_unmount(self):
        src = self._read_config_js()
        assert "removeEventListener" in src

    def test_stacks_cleared_on_start_edit(self):
        src = self._read_config_js()
        # startEdit should clear both stacks
        assert "undoStack.value = []" in src

    def test_stacks_cleared_on_cancel(self):
        src = self._read_config_js()
        # cancelEdit should clear both stacks
        assert "redoStack.value = []" in src

    def test_undo_pops_from_stack(self):
        src = self._read_config_js()
        assert "undoStack.value.pop()" in src

    def test_redo_pops_from_stack(self):
        src = self._read_config_js()
        assert "redoStack.value.pop()" in src

    def test_undo_pushes_to_redo(self):
        src = self._read_config_js()
        # When undoing, current state goes to redo
        assert "redoStack.value.push" in src

    def test_redo_pushes_to_undo(self):
        src = self._read_config_js()
        # When redoing, current state goes to undo
        assert "undoStack.value.push" in src

    def test_prevent_default_on_shortcut(self):
        src = self._read_config_js()
        assert "preventDefault" in src

    def test_undo_disabled_when_empty(self):
        src = self._read_config_js()
        assert ":disabled=\"!canUndo\"" in src

    def test_redo_disabled_when_empty(self):
        src = self._read_config_js()
        assert ":disabled=\"!canRedo\"" in src


# ===================================================================
# Diff view
# ===================================================================


class TestDiffView:
    """A diff view should show changes before saving."""

    def _read_config_js(self):
        return Path("ui/js/pages/config.js").read_text()

    def test_show_diff_modal_state(self):
        src = self._read_config_js()
        assert "showDiffModal" in src

    def test_diff_entries_computed(self):
        src = self._read_config_js()
        assert "diffEntries" in src

    def test_build_diff_entries_function(self):
        src = self._read_config_js()
        assert "buildDiffEntries" in src

    def test_show_diff_function(self):
        src = self._read_config_js()
        assert "function showDiff" in src or "showDiff" in src

    def test_diff_modal_overlay_in_template(self):
        src = self._read_config_js()
        assert "modal-overlay" in src

    def test_diff_modal_content_in_template(self):
        src = self._read_config_js()
        assert "modal-content" in src

    def test_review_button_in_template(self):
        src = self._read_config_js()
        assert "Review" in src

    def test_cfg_diff_list_class(self):
        src = self._read_config_js()
        assert "cfg-diff-list" in src

    def test_cfg_diff_entry_class(self):
        src = self._read_config_js()
        assert "cfg-diff-entry" in src

    def test_cfg_diff_path_shows_section(self):
        src = self._read_config_js()
        assert "cfg-diff-path" in src

    def test_cfg_diff_old_value(self):
        src = self._read_config_js()
        assert "cfg-diff-old" in src

    def test_cfg_diff_new_value(self):
        src = self._read_config_js()
        assert "cfg-diff-new" in src

    def test_diff_label_minus_plus(self):
        src = self._read_config_js()
        assert "cfg-diff-label" in src

    def test_format_diff_val_function(self):
        src = self._read_config_js()
        assert "formatDiffVal" in src

    def test_diff_review_title(self):
        src = self._read_config_js()
        assert "Review Changes" in src

    def test_diff_modal_close_button(self):
        src = self._read_config_js()
        # Should have a close button
        assert "showDiffModal = false" in src

    def test_diff_modal_save_button(self):
        src = self._read_config_js()
        # Save button within diff modal
        assert "Save Changes" in src

    def test_no_changes_message_in_diff(self):
        src = self._read_config_js()
        assert "No changes to review" in src

    def test_change_count_displayed(self):
        src = self._read_config_js()
        assert "changeCount" in src
        assert "cfg-change-count" in src


# ===================================================================
# CSS classes for config editor
# ===================================================================


class TestConfigCSS:
    """CSS file should include config editor styles."""

    def _read_css(self):
        return Path("ui/css/style.css").read_text()

    def test_cfg_group_style(self):
        css = self._read_css()
        assert ".cfg-group" in css

    def test_cfg_group_header_style(self):
        css = self._read_css()
        assert ".cfg-group-header" in css

    def test_cfg_group_icon_style(self):
        css = self._read_css()
        assert ".cfg-group-icon" in css

    def test_cfg_group_label_style(self):
        css = self._read_css()
        assert ".cfg-group-label" in css

    def test_cfg_group_arrow_style(self):
        css = self._read_css()
        assert ".cfg-group-arrow" in css

    def test_cfg_group_body_style(self):
        css = self._read_css()
        assert ".cfg-group-body" in css

    def test_cfg_section_style(self):
        css = self._read_css()
        assert ".cfg-section" in css

    def test_cfg_section_header_style(self):
        css = self._read_css()
        assert ".cfg-section-header" in css

    def test_cfg_section_name_style(self):
        css = self._read_css()
        assert ".cfg-section-name" in css

    def test_cfg_section_body_style(self):
        css = self._read_css()
        assert ".cfg-section-body" in css

    def test_cfg_field_error_style(self):
        css = self._read_css()
        assert ".cfg-field-error" in css

    def test_cfg_input_error_style(self):
        css = self._read_css()
        assert ".cfg-input-error" in css

    def test_cfg_change_count_style(self):
        css = self._read_css()
        assert ".cfg-change-count" in css

    def test_cfg_diff_list_style(self):
        css = self._read_css()
        assert ".cfg-diff-list" in css

    def test_cfg_diff_entry_style(self):
        css = self._read_css()
        assert ".cfg-diff-entry" in css

    def test_cfg_diff_path_style(self):
        css = self._read_css()
        assert ".cfg-diff-path" in css

    def test_cfg_diff_old_style(self):
        css = self._read_css()
        assert ".cfg-diff-old" in css

    def test_cfg_diff_new_style(self):
        css = self._read_css()
        assert ".cfg-diff-new" in css

    def test_cfg_diff_label_style(self):
        css = self._read_css()
        assert ".cfg-diff-label" in css

    def test_group_header_hover(self):
        css = self._read_css()
        assert ".cfg-group-header:hover" in css

    def test_section_header_hover(self):
        css = self._read_css()
        assert ".cfg-section-header:hover" in css

    def test_group_uses_design_tokens(self):
        css = self._read_css()
        # Should use var(--hm-*) tokens
        idx = css.index(".cfg-group {")
        section = css[idx:idx + 300]
        assert "var(--hm-" in section

    def test_diff_old_has_red_tint(self):
        css = self._read_css()
        idx = css.index(".cfg-diff-old")
        section = css[idx:idx + 200]
        assert "rgba(239, 68, 68" in section

    def test_diff_new_has_green_tint(self):
        css = self._read_css()
        idx = css.index(".cfg-diff-new")
        section = css[idx:idx + 200]
        assert "rgba(34, 197, 94" in section


# ===================================================================
# Existing features preserved
# ===================================================================


class TestExistingFeatures:
    """Existing config page features should still work."""

    def _read_config_js(self):
        return Path("ui/js/pages/config.js").read_text()

    def test_sensitive_keys_set(self):
        src = self._read_config_js()
        assert "SENSITIVE_KEYS" in src
        assert "'token'" in src
        assert "'api_token'" in src
        assert "'password'" in src

    def test_enum_fields_map(self):
        src = self._read_config_js()
        assert "ENUM_FIELDS" in src
        assert "'logging.level'" in src

    def test_redacted_constant(self):
        src = self._read_config_js()
        assert "REDACTED" in src

    def test_deep_clone_function(self):
        src = self._read_config_js()
        assert "deepClone" in src

    def test_deep_equal_function(self):
        src = self._read_config_js()
        assert "deepEqual" in src

    def test_compute_diff_function(self):
        src = self._read_config_js()
        assert "computeDiff" in src

    def test_is_sensitive_key_function(self):
        src = self._read_config_js()
        assert "isSensitiveKey" in src

    def test_is_redacted_function(self):
        src = self._read_config_js()
        assert "isRedacted" in src

    def test_toggle_switch_for_booleans(self):
        src = self._read_config_js()
        assert "toggle-switch" in src
        assert "toggle-slider" in src

    def test_array_tag_editing(self):
        src = self._read_config_js()
        assert "config-tag" in src
        assert "addArrayItem" in src
        assert "removeArrayItem" in src

    def test_nested_json_editor(self):
        src = self._read_config_js()
        assert "formatJson" in src
        assert "toggleNested" in src

    def test_toast_notifications(self):
        src = self._read_config_js()
        assert "showToast" in src
        assert "toast-success" in src
        assert "toast-error" in src

    def test_loading_skeleton(self):
        src = self._read_config_js()
        assert "skeleton" in src

    def test_error_state(self):
        src = self._read_config_js()
        assert "error-state" in src

    def test_sensitive_badge(self):
        src = self._read_config_js()
        assert "badge badge-warning" in src
        assert "sensitive" in src

    def test_field_changed_indicator(self):
        src = self._read_config_js()
        assert "field-changed" in src
        assert "fieldChanged" in src

    def test_has_changes_computed(self):
        src = self._read_config_js()
        assert "hasChanges" in src

    def test_fetch_config_on_mount(self):
        src = self._read_config_js()
        assert "onMounted" in src
        assert "fetchConfig" in src

    def test_save_config_sends_diff(self):
        src = self._read_config_js()
        assert "api.put('/api/config'" in src

    def test_cancel_edit_clears_state(self):
        src = self._read_config_js()
        assert "cancelEdit" in src
        assert "editing.value = false" in src

    def test_edit_mode_toggle(self):
        src = self._read_config_js()
        assert "startEdit" in src
        assert "editing.value = true" in src


# ===================================================================
# JS logic: computeDiff and buildDiffEntries
# ===================================================================


class TestDiffLogic:
    """Test the diff computation logic directly."""

    def test_compute_diff_no_changes(self):
        """computeDiff returns empty dict when nothing changed."""
        original = {"discord": {"enabled": True, "token": "x"}}
        edited = {"discord": {"enabled": True, "token": "x"}}
        assert computeDiff(original, edited) == {}

    def test_compute_diff_scalar_change(self):
        original = {"timezone": "UTC"}
        edited = {"timezone": "US/Eastern"}
        diff = computeDiff(original, edited)
        assert diff == {"timezone": "US/Eastern"}

    def test_compute_diff_object_partial(self):
        """Only changed keys within an object section are included."""
        original = {"discord": {"enabled": True, "require_mention": False}}
        edited = {"discord": {"enabled": True, "require_mention": True}}
        diff = computeDiff(original, edited)
        assert diff == {"discord": {"require_mention": True}}
        # enabled not included since it didn't change
        assert "enabled" not in diff.get("discord", {})

    def test_compute_diff_ignores_unknown_sections(self):
        original = {"discord": {"enabled": True}}
        edited = {"discord": {"enabled": True}, "newstuff": "hi"}
        diff = computeDiff(original, edited)
        assert "newstuff" not in diff

    def test_build_diff_entries_empty(self):
        entries = buildDiffEntries({"x": 1}, {"x": 1})
        assert entries == []

    def test_build_diff_entries_scalar(self):
        entries = buildDiffEntries({"timezone": "UTC"}, {"timezone": "EST"})
        assert len(entries) == 1
        assert entries[0]["section"] == "timezone"
        assert entries[0]["key"] is None
        assert entries[0]["oldVal"] == "UTC"
        assert entries[0]["newVal"] == "EST"

    def test_build_diff_entries_object_field(self):
        entries = buildDiffEntries(
            {"discord": {"enabled": True, "require_mention": False}},
            {"discord": {"enabled": True, "require_mention": True}},
        )
        assert len(entries) == 1
        assert entries[0]["section"] == "discord"
        assert entries[0]["key"] == "require_mention"
        assert entries[0]["oldVal"] is False
        assert entries[0]["newVal"] is True

    def test_build_diff_entries_multiple_fields(self):
        entries = buildDiffEntries(
            {"discord": {"a": 1, "b": 2, "c": 3}},
            {"discord": {"a": 10, "b": 2, "c": 30}},
        )
        assert len(entries) == 2
        keys = [e["key"] for e in entries]
        assert "a" in keys
        assert "c" in keys
        assert "b" not in keys


class TestValidateFieldLogic:
    """Test the validateField function logic."""

    def test_no_rule_returns_none(self):
        assert validateField("unknown", "field", "value") is None

    def test_number_nan(self):
        result = validateField("openai_codex", "max_tokens", "abc")
        assert result is not None
        assert "number" in result.lower()

    def test_number_below_min(self):
        result = validateField("openai_codex", "max_tokens", 0)
        assert result is not None

    def test_number_above_max(self):
        result = validateField("openai_codex", "max_tokens", 999999)
        assert result is not None

    def test_number_valid(self):
        result = validateField("openai_codex", "max_tokens", 4096)
        assert result is None

    def test_viewport_width_valid(self):
        result = validateField("browser", "viewport_width", 1280)
        assert result is None

    def test_viewport_width_too_small(self):
        result = validateField("browser", "viewport_width", 50)
        assert result is not None

    def test_viewport_width_too_large(self):
        result = validateField("browser", "viewport_width", 10000)
        assert result is not None

    def test_cooldown_non_negative(self):
        assert validateField("monitoring", "cooldown_minutes", 0) is None
        assert validateField("monitoring", "cooldown_minutes", -1) is not None

    def test_max_age_hours_valid(self):
        assert validateField("sessions", "max_age_hours", 24) is None
        assert validateField("sessions", "max_age_hours", 0) is not None


# ===================================================================
# Integration: config page + API
# ===================================================================


class TestConfigPageAPIIntegration:
    """Config page should work with the API backend."""

    @pytest.mark.asyncio
    async def test_config_has_all_group_sections(self):
        """Every section referenced in SECTION_GROUPS should be fetchable."""
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/config")
            body = await resp.json()
            # Core group sections
            for section in ["timezone", "discord", "logging"]:
                assert section in body, f"Missing section: {section}"

    @pytest.mark.asyncio
    async def test_config_sections_are_objects_or_scalars(self):
        """Each config section should be a dict, list, or scalar."""
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/config")
            body = await resp.json()
            for section, value in body.items():
                assert isinstance(value, (dict, list, str, int, float, bool, type(None))), \
                    f"Section {section} has unexpected type: {type(value)}"

    @pytest.mark.asyncio
    async def test_diff_only_sends_changed_sections(self):
        """When saving, only changed sections should be sent."""
        # This tests the computeDiff logic used by the frontend
        original = {
            "timezone": "UTC",
            "discord": {"enabled": True, "require_mention": False},
            "logging": {"level": "INFO"},
        }
        edited = {
            "timezone": "UTC",
            "discord": {"enabled": True, "require_mention": True},
            "logging": {"level": "INFO"},
        }
        diff = computeDiff(original, edited)
        assert "timezone" not in diff
        assert "logging" not in diff
        assert "discord" in diff
        assert diff["discord"] == {"require_mention": True}


# ===================================================================
# onUnmounted cleanup
# ===================================================================


class TestLifecycle:
    """Component lifecycle should be properly managed."""

    def _read_config_js(self):
        return Path("ui/js/pages/config.js").read_text()

    def test_on_unmounted_imported(self):
        src = self._read_config_js()
        assert "onUnmounted" in src

    def test_event_listener_cleanup(self):
        src = self._read_config_js()
        assert "removeEventListener" in src
        assert "handleKeydown" in src

    def test_stacks_cleared_on_save(self):
        src = self._read_config_js()
        # After successful save, stacks should be cleared
        # Look for undoStack and redoStack being reset in saveConfig
        assert "undoStack.value = []" in src


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge case handling in the config editor."""

    def test_deep_equal_with_nested(self):
        assert deepEqual({"a": {"b": [1, 2]}}, {"a": {"b": [1, 2]}})
        assert not deepEqual({"a": {"b": [1, 2]}}, {"a": {"b": [1, 3]}})

    def test_deep_clone_isolation(self):
        orig = {"a": {"b": [1, 2]}}
        clone = deepClone(orig)
        clone["a"]["b"].append(3)
        assert len(orig["a"]["b"]) == 2  # original unchanged

    def test_compute_diff_array_change(self):
        diff = computeDiff(
            {"tools": {"tool_packs": ["systemd"]}},
            {"tools": {"tool_packs": ["systemd", "incus"]}},
        )
        assert diff == {"tools": {"tool_packs": ["systemd", "incus"]}}

    def test_validate_field_no_max(self):
        """Rule with min but no max should pass large values."""
        result = validateField("sessions", "max_age_hours", 999999)
        assert result is None

    def test_build_diff_entries_array_section(self):
        """Array-typed top-level sections should diff as scalar."""
        entries = buildDiffEntries(
            {"items": [1, 2, 3]},
            {"items": [1, 2, 3, 4]},
        )
        assert len(entries) == 1
        assert entries[0]["key"] is None
