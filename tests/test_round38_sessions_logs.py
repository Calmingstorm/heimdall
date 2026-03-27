"""Round 38 — Sessions + Logs redesign tests.

Covers:
- Sessions page: conversation threading, filter presets, sort options,
  source icons, expand/collapse, thread grouping, custom presets, search
- Logs page: timeline visualization, time range filtering, filter presets,
  tool badge, custom presets, parseLogEntry with _time
- CSS: all new class selectors exist in style.css
- Shared: preset chip styles, filter bar patterns
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
SESSIONS_JS = UI_DIR / "js" / "pages" / "sessions.js"
LOGS_JS = UI_DIR / "js" / "pages" / "logs.js"
STYLE_CSS = UI_DIR / "css" / "style.css"


@pytest.fixture(scope="module")
def sessions_js():
    return SESSIONS_JS.read_text()


@pytest.fixture(scope="module")
def logs_js():
    return LOGS_JS.read_text()


@pytest.fixture(scope="module")
def style_css():
    return STYLE_CSS.read_text()


# ===================================================================
# Sessions Page — Template Structure
# ===================================================================


class TestSessionsTemplateStructure:
    """Verify the sessions template contains all redesigned elements."""

    def test_has_page_header(self, sessions_js):
        assert "Sessions" in sessions_js
        assert "text-xl font-semibold" in sessions_js

    def test_has_subtitle_count(self, sessions_js):
        assert "filteredSessions.length" in sessions_js
        assert "shown" in sessions_js

    def test_has_filter_presets_bar(self, sessions_js):
        assert "sess-filter-bar" in sessions_js
        assert "filterPresets" in sessions_js

    def test_has_preset_chips(self, sessions_js):
        assert "sess-preset-chip" in sessions_js
        assert "sess-preset-active" in sessions_js
        assert "applyPreset" in sessions_js

    def test_has_search_input(self, sessions_js):
        assert "searchQuery" in sessions_js
        assert "Search channels" in sessions_js

    def test_has_sort_options(self, sessions_js):
        assert "sortBy" in sessions_js
        assert "sortAsc" in sessions_js
        assert "hm-select" in sessions_js

    def test_has_source_icons(self, sessions_js):
        assert "sess-source-icon" in sessions_js
        assert "sess-source-web" in sessions_js
        assert "sess-source-discord" in sessions_js

    def test_has_expand_icon(self, sessions_js):
        assert "sess-expand-icon" in sessions_js
        assert "sess-expanded" in sessions_js

    def test_has_summary_banner(self, sessions_js):
        assert "sess-summary-banner" in sessions_js
        assert "sess-summary-label" in sessions_js
        assert "Compacted Summary" in sessions_js

    def test_has_thread_view_toggle(self, sessions_js):
        assert "sess-view-btn" in sessions_js
        assert "sess-view-active" in sessions_js
        assert "threadView" in sessions_js
        assert "Threaded" in sessions_js
        assert "Flat" in sessions_js

    def test_has_thread_container(self, sessions_js):
        assert "sess-thread-container" in sessions_js

    def test_has_thread_elements(self, sessions_js):
        assert "sess-thread-header" in sessions_js
        assert "sess-thread-num" in sessions_js
        assert "sess-thread-arrow" in sessions_js
        assert "sess-thread-summary" in sessions_js
        assert "sess-thread-count" in sessions_js

    def test_has_thread_messages(self, sessions_js):
        assert "sess-thread-messages" in sessions_js
        assert "sess-thread-msg" in sessions_js
        assert "sess-thread-connector" in sessions_js

    def test_has_message_header(self, sessions_js):
        assert "sess-msg-header" in sessions_js
        assert "sess-role-dot" in sessions_js
        assert "sess-role-label" in sessions_js

    def test_has_role_classes(self, sessions_js):
        assert "sess-msg-user" in sessions_js
        assert "sess-msg-assistant" in sessions_js
        assert "sess-msg-system" in sessions_js
        assert "sess-dot-user" in sessions_js
        assert "sess-dot-assistant" in sessions_js

    def test_has_custom_presets(self, sessions_js):
        assert "customPresets" in sessions_js
        assert "saveCustomPreset" in sessions_js
        assert "removeCustomPreset" in sessions_js
        assert "sess-preset-custom" in sessions_js
        assert "sess-preset-remove" in sessions_js

    def test_has_save_preset_ui(self, sessions_js):
        assert "showSavePreset" in sessions_js
        assert "newPresetName" in sessions_js
        assert "Save as preset" in sessions_js

    def test_has_clear_filters(self, sessions_js):
        assert "resetFilters" in sessions_js
        assert "Clear Filters" in sessions_js

    def test_has_no_match_empty_state(self, sessions_js):
        assert "No sessions match" in sessions_js

    def test_retains_clear_session_modal(self, sessions_js):
        assert "Clear Session" in sessions_js
        assert "clearTarget" in sessions_js
        assert "modal-overlay" in sessions_js

    def test_retains_bulk_clear_modal(self, sessions_js):
        assert "Clear Selected Sessions" in sessions_js
        assert "bulkClearing" in sessions_js
        assert "doBulkClear" in sessions_js

    def test_retains_export_buttons(self, sessions_js):
        assert "exportSession" in sessions_js
        assert "JSON" in sessions_js
        assert "TXT" in sessions_js


# ===================================================================
# Sessions Page — JavaScript Logic
# ===================================================================


class TestSessionsLogic:
    """Verify sessions page logic and computed properties."""

    def test_has_filter_presets_data(self, sessions_js):
        assert "FILTER_PRESETS" in sessions_js
        assert "'all'" in sessions_js
        assert "'active'" in sessions_js
        assert "'discord'" in sessions_js
        assert "'web'" in sessions_js
        assert "'long'" in sessions_js
        assert "'compacted'" in sessions_js

    def test_has_sort_options_data(self, sessions_js):
        assert "SORT_OPTIONS" in sessions_js
        assert "'last_active'" in sessions_js
        assert "'created_at'" in sessions_js
        assert "'message_count'" in sessions_js

    def test_has_filtered_sessions_computed(self, sessions_js):
        assert "filteredSessions" in sessions_js
        assert "computed" in sessions_js

    def test_filter_by_source(self, sessions_js):
        assert "filters.source" in sessions_js

    def test_filter_by_min_messages(self, sessions_js):
        assert "filters.minMessages" in sessions_js

    def test_filter_by_compaction(self, sessions_js):
        assert "filters.hasCompaction" in sessions_js

    def test_filter_by_recent_activity(self, sessions_js):
        assert "filters.maxAge" in sessions_js

    def test_search_query_filter(self, sessions_js):
        assert "searchQuery" in sessions_js
        assert "channel_id" in sessions_js
        assert "last_user_id" in sessions_js

    def test_sort_implementation(self, sessions_js):
        assert "result.sort" in sessions_js
        assert "sortAsc" in sessions_js

    def test_threads_computed(self, sessions_js):
        assert "const threads = computed" in sessions_js

    def test_thread_grouping_on_user_role(self, sessions_js):
        # Threads group by splitting on user messages
        assert "m.role === 'user'" in sessions_js
        assert "groups.push" in sessions_js

    def test_thread_summary_function(self, sessions_js):
        assert "threadSummary" in sessions_js
        assert "slice(0, 120)" in sessions_js

    def test_toggle_thread(self, sessions_js):
        assert "toggleThread" in sessions_js
        assert "collapsedThreads" in sessions_js

    def test_localStorage_custom_presets(self, sessions_js):
        assert "localStorage.getItem" in sessions_js
        assert "localStorage.setItem" in sessions_js
        assert "heimdall-session-presets" in sessions_js

    def test_apply_preset(self, sessions_js):
        assert "applyPreset" in sessions_js
        assert "activePreset" in sessions_js

    def test_reset_filters(self, sessions_js):
        assert "function resetFilters" in sessions_js

    def test_has_active_filters_computed(self, sessions_js):
        assert "hasActiveFilters" in sessions_js

    def test_websocket_subscription(self, sessions_js):
        assert "ws.subscribe" in sessions_js
        assert "ws.unsubscribe" in sessions_js

    def test_debounced_refresh(self, sessions_js):
        assert "debounceTimer" in sessions_js
        assert "setTimeout" in sessions_js

    def test_format_age_function(self, sessions_js):
        assert "formatAge" in sessions_js
        assert "minute" in sessions_js
        assert "hour" in sessions_js
        assert "day" in sessions_js

    def test_truncate_content(self, sessions_js):
        assert "truncateContent" in sessions_js
        assert "2000" in sessions_js

    def test_thread_view_toggle_state(self, sessions_js):
        assert "threadView" in sessions_js
        assert "'threaded'" in sessions_js
        assert "'flat'" in sessions_js

    def test_select_all_uses_filtered(self, sessions_js):
        # Select all should use filtered list, not all sessions
        assert "filteredSessions.value.map" in sessions_js

    def test_all_selected_uses_filtered(self, sessions_js):
        assert "filteredSessions.value.length" in sessions_js

    def test_returns_all_required_properties(self, sessions_js):
        for prop in [
            "sessions", "loading", "error",
            "expandedId", "detail", "detailLoading",
            "filterPresets", "sortOptions",
            "filteredSessions", "threads",
            "threadView", "collapsedThreads",
            "customPresets", "activePreset",
            "searchQuery", "sortBy", "sortAsc",
        ]:
            assert prop in sessions_js, f"Missing returned property: {prop}"


# ===================================================================
# Sessions — Preset Definitions
# ===================================================================


class TestSessionsPresetDefinitions:
    """Verify all filter presets have required fields."""

    def test_all_presets_have_id(self, sessions_js):
        assert sessions_js.count("id: '") >= 6  # 6 built-in presets

    def test_all_presets_have_name(self, sessions_js):
        for name in ["All Sessions", "Recently Active", "Discord Only",
                      "Web Only", "Long Conversations", "Compacted"]:
            assert name in sessions_js

    def test_all_presets_have_icon(self, sessions_js):
        assert sessions_js.count("icon: '") >= 6

    def test_all_presets_have_filters(self, sessions_js):
        assert sessions_js.count("filters:") >= 6


# ===================================================================
# Logs Page — Template Structure
# ===================================================================


class TestLogsTemplateStructure:
    """Verify the logs template contains all redesigned elements."""

    def test_has_page_header(self, logs_js):
        assert "Logs" in logs_js
        assert "text-xl font-semibold" in logs_js

    def test_has_entry_count_subtitle(self, logs_js):
        assert "entries" in logs_js

    def test_has_filter_presets_bar(self, logs_js):
        assert "logs-filter-bar" in logs_js
        assert "logPresets" in logs_js

    def test_has_preset_chips(self, logs_js):
        assert "sess-preset-chip" in logs_js
        assert "sess-preset-active" in logs_js
        assert "applyLogPreset" in logs_js

    def test_has_level_chips(self, logs_js):
        assert "log-chip" in logs_js
        assert "log-chip-active" in logs_js
        assert "toggleLevel" in logs_js

    def test_has_time_range_select(self, logs_js):
        assert "timeRange" in logs_js
        assert "timeRanges" in logs_js
        assert "hm-select" in logs_js

    def test_has_text_filter(self, logs_js):
        assert "textFilter" in logs_js
        assert "Filter logs" in logs_js

    def test_has_regex_toggle(self, logs_js):
        assert "useRegex" in logs_js

    def test_has_auto_scroll(self, logs_js):
        assert "autoScroll" in logs_js
        assert "Auto-scroll" in logs_js

    def test_has_custom_preset_save(self, logs_js):
        assert "showSaveLogPreset" in logs_js
        assert "newLogPresetName" in logs_js
        assert "Save as preset" in logs_js
        assert "saveLogCustomPreset" in logs_js

    def test_has_custom_presets_display(self, logs_js):
        assert "customLogPresets" in logs_js
        assert "sess-preset-custom" in logs_js
        assert "removeLogCustomPreset" in logs_js

    def test_has_timeline_visualization(self, logs_js):
        assert "logs-timeline" in logs_js
        assert "logs-timeline-header" in logs_js
        assert "logs-timeline-chart" in logs_js
        assert "Activity Timeline" in logs_js

    def test_has_timeline_bars(self, logs_js):
        assert "logs-timeline-bar-wrap" in logs_js
        assert "logs-timeline-bar" in logs_js
        assert "logs-timeline-segment" in logs_js

    def test_has_timeline_colors(self, logs_js):
        assert "logs-tl-error" in logs_js
        assert "logs-tl-warning" in logs_js
        assert "logs-tl-info" in logs_js

    def test_has_timeline_labels(self, logs_js):
        assert "logs-timeline-label" in logs_js
        assert "timelineLabelSkip" in logs_js

    def test_has_tool_badge(self, logs_js):
        assert "logs-tool-badge" in logs_js

    def test_has_time_range_badge_in_status(self, logs_js):
        assert "timeRangeLabel" in logs_js
        assert "badge badge-info" in logs_js

    def test_has_status_bar(self, logs_js):
        assert "status-dot" in logs_js
        assert "Live" in logs_js
        assert "Disconnected" in logs_js

    def test_has_pause_button(self, logs_js):
        assert "togglePause" in logs_js
        assert "Pause" in logs_js
        assert "Resume" in logs_js

    def test_has_clear_button(self, logs_js):
        assert "clearLogs" in logs_js
        assert "Clear" in logs_js

    def test_has_export_button(self, logs_js):
        assert "exportLogs" in logs_js
        assert "Export" in logs_js

    def test_has_jump_to_bottom(self, logs_js):
        assert "log-jump-btn" in logs_js
        assert "jumpToBottom" in logs_js

    def test_has_empty_states(self, logs_js):
        assert "empty-state" in logs_js
        assert "Waiting for log entries" in logs_js
        assert "No entries match" in logs_js


# ===================================================================
# Logs Page — JavaScript Logic
# ===================================================================


class TestLogsLogic:
    """Verify logs page logic and computed properties."""

    def test_has_log_presets_data(self, logs_js):
        assert "LOG_PRESETS" in logs_js
        assert "'all'" in logs_js
        assert "'errors'" in logs_js
        assert "'warnings'" in logs_js
        assert "'tools'" in logs_js
        assert "'recent-errors'" in logs_js

    def test_has_time_ranges_data(self, logs_js):
        assert "TIME_RANGES" in logs_js
        assert "'last_5m'" in logs_js
        assert "'last_15m'" in logs_js
        assert "'last_1h'" in logs_js
        assert "'last_4h'" in logs_js
        assert "'last_24h'" in logs_js

    def test_time_ranges_have_seconds(self, logs_js):
        assert "seconds: 300" in logs_js
        assert "seconds: 900" in logs_js
        assert "seconds: 3600" in logs_js
        assert "seconds: 14400" in logs_js
        assert "seconds: 86400" in logs_js

    def test_filtered_logs_has_time_range(self, logs_js):
        assert "timeRange.value" in logs_js
        assert "tr.seconds" in logs_js
        assert "cutoff" in logs_js

    def test_parse_log_entry_has_time(self, logs_js):
        assert "_time" in logs_js

    def test_timeline_buckets_computed(self, logs_js):
        assert "timelineBuckets" in logs_js
        assert "computed" in logs_js
        assert "TIMELINE_BUCKETS" in logs_js

    def test_timeline_has_24_buckets(self, logs_js):
        assert "TIMELINE_BUCKETS = 24" in logs_js

    def test_timeline_max_computed(self, logs_js):
        assert "timelineMax" in logs_js

    def test_timeline_span_label(self, logs_js):
        assert "timelineSpanLabel" in logs_js
        assert "Last 24 hours" in logs_js

    def test_segment_height_function(self, logs_js):
        assert "segmentHeight" in logs_js
        assert "count / max" in logs_js

    def test_jump_to_timeline_bucket(self, logs_js):
        assert "jumpToTimelineBucket" in logs_js
        assert "scrollIntoView" in logs_js

    def test_apply_log_preset(self, logs_js):
        assert "applyLogPreset" in logs_js
        assert "activeLogPreset" in logs_js

    def test_custom_log_presets_localStorage(self, logs_js):
        assert "heimdall-log-presets" in logs_js
        assert "localStorage" in logs_js

    def test_save_log_custom_preset(self, logs_js):
        assert "saveLogCustomPreset" in logs_js

    def test_remove_log_custom_preset(self, logs_js):
        assert "removeLogCustomPreset" in logs_js

    def test_has_active_log_filters(self, logs_js):
        assert "hasActiveLogFilters" in logs_js

    def test_time_range_label_computed(self, logs_js):
        assert "timeRangeLabel" in logs_js

    def test_log_entry_all_paths_have_time(self, logs_js):
        # Every parseLogEntry return path should include _time
        # Count _time occurrences within the parseLogEntry function
        start = logs_js.index("function parseLogEntry")
        # Find the next top-level function after it
        end = logs_js.index("function onLog", start)
        func_body = logs_js[start:end]
        time_count = func_body.count("_time")
        assert time_count >= 5, f"Expected >=5 _time references in parseLogEntry, got {time_count}"

    def test_max_logs_limit(self, logs_js):
        assert "MAX_LOGS = 2000" in logs_js

    def test_pause_buffer(self, logs_js):
        assert "pauseBuffer" in logs_js

    def test_websocket_subscription(self, logs_js):
        assert "ws.subscribe('logs'" in logs_js
        assert "ws.unsubscribe('logs'" in logs_js

    def test_returns_all_required_properties(self, logs_js):
        for prop in [
            "logs", "paused", "autoScroll", "levelFilter", "textFilter",
            "filteredLogs", "pauseBuffer", "logPresets", "timeRanges",
            "timeRange", "activeLogPreset", "customLogPresets",
            "timelineBuckets", "timelineMax",
            "segmentHeight", "jumpToTimelineBucket",
        ]:
            assert prop in logs_js, f"Missing returned property: {prop}"


# ===================================================================
# Logs — Preset Definitions
# ===================================================================


class TestLogsPresetDefinitions:
    """Verify all log filter presets have required fields."""

    def test_all_presets_have_id(self, logs_js):
        assert "id: 'all'" in logs_js
        assert "id: 'errors'" in logs_js
        assert "id: 'warnings'" in logs_js
        assert "id: 'tools'" in logs_js

    def test_all_presets_have_name(self, logs_js):
        for name in ["All Logs", "Errors Only", "Warnings+", "Tool Activity", "Recent Errors"]:
            assert name in logs_js

    def test_all_presets_have_icon(self, logs_js):
        # Each preset has an icon field
        assert logs_js.count("icon: '") >= 5

    def test_recent_errors_preset_has_time(self, logs_js):
        assert "timeRange: 'last_1h'" in logs_js


# ===================================================================
# CSS — Session Redesign Classes
# ===================================================================


class TestSessionCSS:
    """Verify all new CSS selectors exist."""

    def test_source_icon(self, style_css):
        assert ".sess-source-icon" in style_css

    def test_source_discord(self, style_css):
        assert ".sess-source-discord" in style_css

    def test_source_web(self, style_css):
        assert ".sess-source-web" in style_css

    def test_expand_icon(self, style_css):
        assert ".sess-expand-icon" in style_css

    def test_expanded_class(self, style_css):
        assert ".sess-expanded" in style_css

    def test_summary_banner(self, style_css):
        assert ".sess-summary-banner" in style_css

    def test_summary_label(self, style_css):
        assert ".sess-summary-label" in style_css

    def test_view_btn(self, style_css):
        assert ".sess-view-btn" in style_css

    def test_view_active(self, style_css):
        assert ".sess-view-active" in style_css

    def test_thread_container(self, style_css):
        assert ".sess-thread-container" in style_css

    def test_thread(self, style_css):
        assert ".sess-thread" in style_css

    def test_thread_header(self, style_css):
        assert ".sess-thread-header" in style_css

    def test_thread_num(self, style_css):
        assert ".sess-thread-num" in style_css

    def test_thread_arrow(self, style_css):
        assert ".sess-thread-arrow" in style_css

    def test_thread_arrow_open(self, style_css):
        assert ".sess-thread-arrow-open" in style_css

    def test_thread_summary(self, style_css):
        assert ".sess-thread-summary" in style_css

    def test_thread_count(self, style_css):
        assert ".sess-thread-count" in style_css

    def test_thread_messages(self, style_css):
        assert ".sess-thread-messages" in style_css

    def test_thread_msg(self, style_css):
        assert ".sess-thread-msg" in style_css

    def test_msg_user(self, style_css):
        assert ".sess-msg-user" in style_css

    def test_msg_assistant(self, style_css):
        assert ".sess-msg-assistant" in style_css

    def test_msg_system(self, style_css):
        assert ".sess-msg-system" in style_css

    def test_thread_connector(self, style_css):
        assert ".sess-thread-connector" in style_css

    def test_msg_header(self, style_css):
        assert ".sess-msg-header" in style_css

    def test_role_dot(self, style_css):
        assert ".sess-role-dot" in style_css

    def test_dot_user(self, style_css):
        assert ".sess-dot-user" in style_css

    def test_dot_assistant(self, style_css):
        assert ".sess-dot-assistant" in style_css

    def test_dot_system(self, style_css):
        assert ".sess-dot-system" in style_css

    def test_role_label(self, style_css):
        assert ".sess-role-label" in style_css

    def test_msg_content(self, style_css):
        assert ".sess-msg-content" in style_css


# ===================================================================
# CSS — Filter Preset Chips
# ===================================================================


class TestPresetChipCSS:
    """Verify filter preset chip CSS classes."""

    def test_filter_bar(self, style_css):
        assert ".sess-filter-bar" in style_css

    def test_logs_filter_bar(self, style_css):
        assert ".logs-filter-bar" in style_css

    def test_preset_chip(self, style_css):
        assert ".sess-preset-chip" in style_css

    def test_preset_chip_hover(self, style_css):
        assert ".sess-preset-chip:hover" in style_css

    def test_preset_active(self, style_css):
        assert ".sess-preset-active" in style_css

    def test_preset_icon(self, style_css):
        assert ".sess-preset-icon" in style_css

    def test_preset_custom(self, style_css):
        assert ".sess-preset-custom" in style_css

    def test_preset_remove(self, style_css):
        assert ".sess-preset-remove" in style_css

    def test_preset_remove_hover(self, style_css):
        assert ".sess-preset-remove:hover" in style_css


# ===================================================================
# CSS — Log Timeline Classes
# ===================================================================


class TestLogTimelineCSS:
    """Verify all timeline CSS selectors exist."""

    def test_timeline_container(self, style_css):
        assert ".logs-timeline" in style_css

    def test_timeline_header(self, style_css):
        assert ".logs-timeline-header" in style_css

    def test_timeline_chart(self, style_css):
        assert ".logs-timeline-chart" in style_css

    def test_timeline_bar_wrap(self, style_css):
        assert ".logs-timeline-bar-wrap" in style_css

    def test_timeline_bar(self, style_css):
        assert ".logs-timeline-bar" in style_css

    def test_timeline_bar_hover(self, style_css):
        assert ".logs-timeline-bar-wrap:hover" in style_css

    def test_timeline_segment(self, style_css):
        assert ".logs-timeline-segment" in style_css

    def test_timeline_error_color(self, style_css):
        assert ".logs-tl-error" in style_css

    def test_timeline_warning_color(self, style_css):
        assert ".logs-tl-warning" in style_css

    def test_timeline_info_color(self, style_css):
        assert ".logs-tl-info" in style_css

    def test_timeline_label(self, style_css):
        assert ".logs-timeline-label" in style_css

    def test_tool_badge(self, style_css):
        assert ".logs-tool-badge" in style_css


# ===================================================================
# CSS — Design Token Usage
# ===================================================================


class TestDesignTokenUsage:
    """Verify new CSS uses design tokens instead of hardcoded values."""

    def test_session_thread_uses_tokens(self, style_css):
        # Find the sess-thread-container section and verify token usage
        assert "--hm-space" in style_css
        assert "--hm-radius" in style_css
        assert "--hm-border" in style_css

    def test_preset_chip_uses_accent(self, style_css):
        assert "--hm-accent-dim" in style_css
        assert "--hm-gold" in style_css

    def test_timeline_uses_tokens(self, style_css):
        assert "--hm-surface" in style_css
        assert "--hm-transition" in style_css

    def test_role_dots_use_semantic_colors(self, style_css):
        # User = cyan, assistant = indigo, system = gray
        assert "06b6d4" in style_css  # cyan
        assert "6366f1" in style_css  # indigo


# ===================================================================
# Sessions — Filter/Sort Presets Correctness
# ===================================================================


class TestSessionsPresetCorrectness:
    """Verify preset filter logic is implemented correctly."""

    def test_source_filter_exact_match(self, sessions_js):
        assert "s.source === filters.source" in sessions_js

    def test_min_messages_filter(self, sessions_js):
        assert "s.message_count >= filters.minMessages" in sessions_js

    def test_compacted_filter(self, sessions_js):
        assert "s.has_summary" in sessions_js

    def test_max_age_filter(self, sessions_js):
        assert "now - s.last_active" in sessions_js

    def test_sort_descending_default(self, sessions_js):
        assert "sortAsc = ref(false)" in sessions_js

    def test_sort_direction_multiplier(self, sessions_js):
        assert "(av - bv) * dir" in sessions_js


# ===================================================================
# Logs — Time Range Filter Correctness
# ===================================================================


class TestLogsTimeRangeCorrectness:
    """Verify time range filtering logic."""

    def test_cutoff_calculation(self, logs_js):
        assert "Date.now() - tr.seconds * 1000" in logs_js

    def test_filters_by_time_object(self, logs_js):
        assert "e._time" in logs_js
        assert "e._time >= cutoff" in logs_js


# ===================================================================
# Backward Compatibility
# ===================================================================


class TestBackwardCompatibility:
    """Verify existing features are preserved."""

    def test_sessions_websocket_events(self, sessions_js):
        assert "ws.subscribe('events'" in sessions_js
        assert "ws.unsubscribe('events'" in sessions_js

    def test_sessions_api_calls(self, sessions_js):
        assert "api.get('/api/sessions')" in sessions_js
        assert "api.get(`/api/sessions/" in sessions_js
        assert "api.del(`/api/sessions/" in sessions_js
        assert "api.post('/api/sessions/clear-bulk'" in sessions_js

    def test_sessions_export(self, sessions_js):
        assert "/api/sessions/" in sessions_js
        assert "export?format=" in sessions_js

    def test_logs_websocket(self, logs_js):
        assert "ws.subscribe('logs'" in logs_js

    def test_logs_regex_filter(self, logs_js):
        assert "useRegex" in logs_js
        assert "new RegExp" in logs_js

    def test_logs_pause_buffer(self, logs_js):
        assert "pauseBuffer" in logs_js
        assert "paused.value" in logs_js

    def test_logs_clipboard_copy(self, logs_js):
        assert "navigator.clipboard.writeText" in logs_js

    def test_logs_export_blob(self, logs_js):
        assert "new Blob" in logs_js
        assert "heimdall-logs-" in logs_js

    def test_logs_scroll_behavior(self, logs_js):
        assert "scrollToBottom" in logs_js
        assert "jumpToBottom" in logs_js
        assert "showJumpBottom" in logs_js

    def test_sessions_flat_view_preserved(self, sessions_js):
        # The flat (non-threaded) view is still available
        assert "session-messages" in sessions_js
        assert "session-msg" in sessions_js
        assert "messageClass" in sessions_js


# ===================================================================
# Integration — Both Pages Share Preset Pattern
# ===================================================================


class TestSharedPatterns:
    """Verify sessions and logs share consistent patterns."""

    def test_both_use_preset_chip_class(self, sessions_js, logs_js):
        assert "sess-preset-chip" in sessions_js
        assert "sess-preset-chip" in logs_js

    def test_both_use_preset_active_class(self, sessions_js, logs_js):
        assert "sess-preset-active" in sessions_js
        assert "sess-preset-active" in logs_js

    def test_both_have_custom_presets(self, sessions_js, logs_js):
        assert "customPresets" in sessions_js or "customLogPresets" in logs_js
        assert "localStorage" in sessions_js
        assert "localStorage" in logs_js

    def test_both_have_save_preset(self, sessions_js, logs_js):
        assert "Save as preset" in sessions_js
        assert "Save as preset" in logs_js

    def test_both_import_api_and_ws(self, sessions_js, logs_js):
        assert "import { api, ws } from '../api.js'" in sessions_js
        assert "import { api, ws } from '../api.js'" in logs_js

    def test_both_use_vue_lifecycle(self, sessions_js, logs_js):
        assert "onMounted" in sessions_js
        assert "onUnmounted" in sessions_js
        assert "onMounted" in logs_js
        assert "onUnmounted" in logs_js
