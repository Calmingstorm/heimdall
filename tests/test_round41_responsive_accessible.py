"""
Round 41 — Responsive + Accessible Design Tests
Fluid layout, keyboard nav, ARIA labels, WCAG AA compliance.
"""

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
INDEX_HTML = UI_DIR / "index.html"
STYLE_CSS = UI_DIR / "css" / "style.css"
APP_JS = UI_DIR / "js" / "app.js"

# Page components
DASHBOARD_JS = UI_DIR / "js" / "pages" / "dashboard.js"
CHAT_JS = UI_DIR / "js" / "pages" / "chat.js"
SESSIONS_JS = UI_DIR / "js" / "pages" / "sessions.js"
TOOLS_JS = UI_DIR / "js" / "pages" / "tools.js"
SKILLS_JS = UI_DIR / "js" / "pages" / "skills.js"
KNOWLEDGE_JS = UI_DIR / "js" / "pages" / "knowledge.js"
SCHEDULES_JS = UI_DIR / "js" / "pages" / "schedules.js"
LOOPS_JS = UI_DIR / "js" / "pages" / "loops.js"
PROCESSES_JS = UI_DIR / "js" / "pages" / "processes.js"
AUDIT_JS = UI_DIR / "js" / "pages" / "audit.js"
CONFIG_JS = UI_DIR / "js" / "pages" / "config.js"
LOGS_JS = UI_DIR / "js" / "pages" / "logs.js"
MEMORY_JS = UI_DIR / "js" / "pages" / "memory.js"
AGENTS_JS = UI_DIR / "js" / "pages" / "agents.js"


@pytest.fixture(scope="module")
def index_html():
    return INDEX_HTML.read_text()


@pytest.fixture(scope="module")
def style_css():
    return STYLE_CSS.read_text()


@pytest.fixture(scope="module")
def app_js():
    return APP_JS.read_text()


@pytest.fixture(scope="module")
def dashboard_js():
    return DASHBOARD_JS.read_text()


@pytest.fixture(scope="module")
def chat_js():
    return CHAT_JS.read_text()


@pytest.fixture(scope="module")
def sessions_js():
    return SESSIONS_JS.read_text()


@pytest.fixture(scope="module")
def tools_js():
    return TOOLS_JS.read_text()


@pytest.fixture(scope="module")
def knowledge_js():
    return KNOWLEDGE_JS.read_text()


@pytest.fixture(scope="module")
def config_js():
    return CONFIG_JS.read_text()


@pytest.fixture(scope="module")
def agents_js():
    return AGENTS_JS.read_text()


@pytest.fixture(scope="module")
def all_page_sources():
    """Load all page component sources."""
    pages = {}
    for p in (UI_DIR / "js" / "pages").glob("*.js"):
        pages[p.stem] = p.read_text()
    return pages


# ===========================================================================
# 1. Skip Navigation Link
# ===========================================================================

class TestSkipNav:
    """Skip navigation link for keyboard users."""

    def test_skip_nav_exists_in_html(self, index_html):
        assert 'class="skip-nav"' in index_html

    def test_skip_nav_href_targets_main(self, index_html):
        assert 'href="#main-content"' in index_html

    def test_skip_nav_text(self, index_html):
        assert "Skip to main content" in index_html

    def test_skip_nav_css_class(self, style_css):
        assert ".skip-nav" in style_css

    def test_skip_nav_hidden_by_default(self, style_css):
        # Should be positioned off-screen by default
        assert "top: -100%" in style_css or "top:-100%" in style_css

    def test_skip_nav_visible_on_focus(self, style_css):
        assert ".skip-nav:focus" in style_css

    def test_main_content_id_exists(self, app_js):
        assert 'id="main-content"' in app_js


# ===========================================================================
# 2. Screen Reader Only Utility
# ===========================================================================

class TestScreenReaderOnly:
    """Screen reader only (.sr-only) CSS class."""

    def test_sr_only_class_exists(self, style_css):
        assert ".sr-only" in style_css

    def test_sr_only_has_clip(self, style_css):
        # Should clip the element
        assert "clip:" in style_css or "clip-path:" in style_css

    def test_sr_only_has_width_1px(self, style_css):
        # Between sr-only definition and next rule
        idx = style_css.index(".sr-only")
        block = style_css[idx:idx + 300]
        assert "width: 1px" in block or "width:1px" in block

    def test_sr_only_has_height_1px(self, style_css):
        idx = style_css.index(".sr-only")
        block = style_css[idx:idx + 300]
        assert "height: 1px" in block or "height:1px" in block

    def test_sr_only_has_overflow_hidden(self, style_css):
        idx = style_css.index(".sr-only")
        block = style_css[idx:idx + 300]
        assert "overflow: hidden" in block or "overflow:hidden" in block

    def test_sr_only_used_for_labels(self, app_js):
        """sr-only class used in app template for screen reader labels."""
        assert 'class="sr-only"' in app_js

    def test_sr_only_used_in_chat(self, chat_js):
        assert 'class="sr-only"' in chat_js


# ===========================================================================
# 3. Reduced Motion Preference
# ===========================================================================

class TestReducedMotion:
    """Respects prefers-reduced-motion user preference."""

    def test_reduced_motion_media_query_exists(self, style_css):
        assert "prefers-reduced-motion" in style_css

    def test_reduced_motion_disables_animation(self, style_css):
        idx = style_css.index("prefers-reduced-motion")
        block = style_css[idx:idx + 600]
        assert "animation-duration: 0.01ms" in block or "animation-duration:0.01ms" in block

    def test_reduced_motion_disables_transition(self, style_css):
        idx = style_css.index("prefers-reduced-motion")
        block = style_css[idx:idx + 600]
        assert "transition-duration: 0.01ms" in block or "transition-duration:0.01ms" in block

    def test_reduced_motion_disables_scroll(self, style_css):
        idx = style_css.index("prefers-reduced-motion")
        block = style_css[idx:idx + 600]
        assert "scroll-behavior: auto" in block or "scroll-behavior:auto" in block


# ===========================================================================
# 4. Forced Colors (High Contrast Mode)
# ===========================================================================

class TestForcedColors:
    """Support for Windows high contrast mode."""

    def test_forced_colors_media_query(self, style_css):
        assert "forced-colors: active" in style_css or "forced-colors:active" in style_css

    def test_forced_colors_adjusts_status_dots(self, style_css):
        idx = style_css.index("forced-colors")
        block = style_css[idx:idx + 500]
        assert "forced-color-adjust" in block

    def test_forced_colors_button_border(self, style_css):
        idx = style_css.index("forced-colors")
        block = style_css[idx:idx + 500]
        assert "ButtonText" in block


# ===========================================================================
# 5. ARIA Roles on App Shell
# ===========================================================================

class TestAppShellARIA:
    """ARIA roles and labels on the main app shell."""

    def test_sidebar_has_navigation_role(self, app_js):
        assert 'role="navigation"' in app_js

    def test_sidebar_has_aria_label(self, app_js):
        assert 'aria-label="Main navigation"' in app_js

    def test_nav_has_aria_label(self, app_js):
        assert 'aria-label="Page navigation"' in app_js

    def test_main_has_role(self, app_js):
        assert 'role="main"' in app_js

    def test_topbar_has_banner_role(self, app_js):
        assert 'role="banner"' in app_js

    def test_sidebar_toggle_has_aria_expanded(self, app_js):
        assert "aria-expanded" in app_js
        assert "aria-controls" in app_js

    def test_sidebar_toggle_has_aria_label(self, app_js):
        assert "Expand sidebar" in app_js
        assert "Collapse sidebar" in app_js

    def test_mobile_hamburger_has_aria_label(self, app_js):
        assert 'aria-label="Open navigation menu"' in app_js

    def test_mobile_hamburger_has_aria_expanded(self, app_js):
        # Mobile open button should have aria-expanded
        assert ":aria-expanded" in app_js

    def test_nav_icons_aria_hidden(self, app_js):
        """Emoji icons should be hidden from screen readers."""
        assert 'aria-hidden="true">{{ r.meta.icon }}' in app_js or \
               "aria-hidden=\"true\">{{ r.meta.icon }}" in app_js

    def test_nav_items_have_aria_current(self, app_js):
        assert "aria-current" in app_js

    def test_ws_status_has_aria_live(self, app_js):
        assert 'aria-live="polite"' in app_js

    def test_bot_status_dot_has_aria_label(self, app_js):
        assert "Bot status:" in app_js

    def test_logout_button_has_aria_label(self, app_js):
        assert 'aria-label="Log out"' in app_js

    def test_loading_spinner_has_status_role(self, app_js):
        assert 'role="status"' in app_js
        assert "Loading application" in app_js or "Loading" in app_js

    def test_mobile_overlay_aria_hidden(self, app_js):
        assert 'aria-hidden="true"' in app_js


# ===========================================================================
# 6. Login Screen ARIA
# ===========================================================================

class TestLoginARIA:
    """ARIA attributes on the login form."""

    def test_login_form_has_aria_labelledby(self, app_js):
        assert 'aria-labelledby="login-title"' in app_js

    def test_login_title_has_id(self, app_js):
        assert 'id="login-title"' in app_js

    def test_login_input_has_label(self, app_js):
        assert 'for="login-token"' in app_js
        assert 'id="login-token"' in app_js

    def test_login_error_has_role_alert(self, app_js):
        assert 'role="alert"' in app_js

    def test_login_has_autocomplete(self, app_js):
        assert 'autocomplete="current-password"' in app_js

    def test_login_spinner_aria_hidden(self, app_js):
        # Spinner in login button should be aria-hidden
        assert 'aria-hidden="true"' in app_js


# ===========================================================================
# 7. Dashboard ARIA
# ===========================================================================

class TestDashboardARIA:
    """ARIA attributes on the dashboard page."""

    def test_dashboard_region_label(self, dashboard_js):
        assert 'aria-label="Dashboard"' in dashboard_js

    def test_loading_has_status_role(self, dashboard_js):
        assert 'role="status"' in dashboard_js

    def test_error_state_has_alert_role(self, dashboard_js):
        assert 'role="alert"' in dashboard_js

    def test_error_icon_aria_hidden(self, dashboard_js):
        assert 'aria-hidden="true"' in dashboard_js

    def test_uptime_ring_svg_has_role(self, dashboard_js):
        assert 'role="img"' in dashboard_js

    def test_uptime_ring_svg_has_aria_label(self, dashboard_js):
        assert "Uptime:" in dashboard_js

    def test_health_bar_has_region_role(self, dashboard_js):
        assert 'aria-label="System health"' in dashboard_js

    def test_health_dots_have_aria_label(self, dashboard_js):
        assert 'role="img"' in dashboard_js

    def test_action_toast_has_aria_live(self, dashboard_js):
        assert 'aria-live="polite"' in dashboard_js


# ===========================================================================
# 8. Chat ARIA
# ===========================================================================

class TestChatARIA:
    """ARIA attributes on the chat page."""

    def test_chat_region_label(self, chat_js):
        assert 'aria-label="Chat"' in chat_js

    def test_message_list_has_log_role(self, chat_js):
        assert 'role="log"' in chat_js

    def test_message_list_has_aria_live(self, chat_js):
        assert 'aria-live="polite"' in chat_js

    def test_tool_toggle_has_aria_expanded(self, chat_js):
        assert ":aria-expanded" in chat_js

    def test_tool_toggle_has_aria_label(self, chat_js):
        assert 'aria-label="Toggle tool details"' in chat_js

    def test_typing_indicator_has_status_role(self, chat_js):
        assert 'role="status"' in chat_js

    def test_typing_dots_aria_hidden(self, chat_js):
        # The animated dots should be hidden
        idx = chat_js.index("chat-typing")
        block = chat_js[idx:idx + 200]
        assert 'aria-hidden="true"' in block

    def test_input_has_label(self, chat_js):
        assert 'id="chat-message-input"' in chat_js
        assert 'for="chat-message-input"' in chat_js

    def test_send_button_has_aria_label(self, chat_js):
        assert 'aria-label="Send message"' in chat_js

    def test_send_icon_aria_hidden(self, chat_js):
        # SVG in send button should be aria-hidden
        assert 'class="chat-send-icon" aria-hidden="true"' in chat_js

    def test_input_area_has_form_role(self, chat_js):
        assert 'role="form"' in chat_js


# ===========================================================================
# 9. Knowledge ARIA
# ===========================================================================

class TestKnowledgeARIA:
    """ARIA attributes on knowledge page tree view."""

    def test_tree_header_has_role_button(self, knowledge_js):
        assert 'role="button"' in knowledge_js

    def test_tree_header_has_tabindex(self, knowledge_js):
        assert 'tabindex="0"' in knowledge_js

    def test_tree_header_has_aria_expanded(self, knowledge_js):
        assert ":aria-expanded" in knowledge_js

    def test_tree_header_keyboard_enter(self, knowledge_js):
        assert "@keydown.enter" in knowledge_js

    def test_tree_header_keyboard_space(self, knowledge_js):
        assert "@keydown.space" in knowledge_js

    def test_tree_arrow_aria_hidden(self, knowledge_js):
        # Arrow icon should be decorative
        assert 'aria-hidden="true"' in knowledge_js

    def test_delete_modal_has_dialog_role(self, knowledge_js):
        assert 'role="dialog"' in knowledge_js

    def test_delete_modal_has_aria_modal(self, knowledge_js):
        assert 'aria-modal="true"' in knowledge_js

    def test_delete_modal_has_labelledby(self, knowledge_js):
        assert 'aria-labelledby="kb-delete-title"' in knowledge_js

    def test_delete_modal_title_has_id(self, knowledge_js):
        assert 'id="kb-delete-title"' in knowledge_js

    def test_error_has_role_alert(self, knowledge_js):
        assert 'role="alert"' in knowledge_js

    def test_loading_has_status_role(self, knowledge_js):
        assert 'role="status"' in knowledge_js


# ===========================================================================
# 10. Agents ARIA
# ===========================================================================

class TestAgentsARIA:
    """ARIA attributes on agents page."""

    def test_filter_bar_has_toolbar_role(self, agents_js):
        assert 'role="toolbar"' in agents_js

    def test_filter_buttons_have_aria_pressed(self, agents_js):
        assert ":aria-pressed" in agents_js

    def test_card_grid_has_list_role(self, agents_js):
        assert 'role="list"' in agents_js

    def test_cards_have_listitem_role(self, agents_js):
        assert 'role="listitem"' in agents_js

    def test_status_dot_has_role_img(self, agents_js):
        assert 'role="img"' in agents_js

    def test_status_dot_has_aria_label(self, agents_js):
        assert "Status:" in agents_js

    def test_error_has_alert_role(self, agents_js):
        assert 'role="alert"' in agents_js


# ===========================================================================
# 11. Sessions ARIA
# ===========================================================================

class TestSessionsARIA:
    """ARIA attributes on sessions page."""

    def test_clear_modal_has_dialog_role(self, sessions_js):
        assert 'role="dialog"' in sessions_js

    def test_clear_modal_has_aria_modal(self, sessions_js):
        assert 'aria-modal="true"' in sessions_js

    def test_clear_modal_has_labelledby(self, sessions_js):
        assert 'aria-labelledby="sess-clear-title"' in sessions_js

    def test_clear_modal_title_has_id(self, sessions_js):
        assert 'id="sess-clear-title"' in sessions_js

    def test_bulk_clear_modal_has_dialog_role(self, sessions_js):
        assert 'aria-labelledby="sess-bulk-clear-title"' in sessions_js

    def test_bulk_clear_modal_title_has_id(self, sessions_js):
        assert 'id="sess-bulk-clear-title"' in sessions_js

    def test_thread_header_has_role_button(self, sessions_js):
        assert 'role="button"' in sessions_js

    def test_thread_header_has_tabindex(self, sessions_js):
        assert 'tabindex="0"' in sessions_js

    def test_thread_header_has_aria_expanded(self, sessions_js):
        assert ":aria-expanded" in sessions_js

    def test_thread_header_keyboard_nav(self, sessions_js):
        assert "@keydown.enter" in sessions_js
        assert "@keydown.space" in sessions_js


# ===========================================================================
# 12. Tools ARIA
# ===========================================================================

class TestToolsARIA:
    """ARIA attributes on tools page."""

    def test_view_toggle_has_toolbar_role(self, tools_js):
        assert 'role="toolbar"' in tools_js

    def test_view_buttons_have_aria_pressed(self, tools_js):
        assert ":aria-pressed" in tools_js

    def test_view_buttons_have_aria_labels(self, tools_js):
        assert 'aria-label="Card view"' in tools_js
        assert 'aria-label="Table view"' in tools_js

    def test_category_chips_have_toolbar(self, tools_js):
        assert 'aria-label="Filter by category"' in tools_js

    def test_category_chips_have_aria_pressed(self, tools_js):
        # Count instances of :aria-pressed — should be on both view and category
        count = tools_js.count(":aria-pressed")
        assert count >= 3  # 2 view buttons + all/category buttons

    def test_category_icon_aria_hidden(self, tools_js):
        assert 'aria-hidden="true">{{ cat.icon }}' in tools_js or \
               "aria-hidden" in tools_js


# ===========================================================================
# 13. Config ARIA
# ===========================================================================

class TestConfigARIA:
    """ARIA attributes on config page."""

    def test_group_header_has_role_button(self, config_js):
        assert 'role="button"' in config_js

    def test_group_header_has_tabindex(self, config_js):
        assert 'tabindex="0"' in config_js

    def test_group_header_has_aria_expanded(self, config_js):
        assert ":aria-expanded" in config_js

    def test_group_header_keyboard_enter(self, config_js):
        assert "@keydown.enter" in config_js

    def test_group_header_keyboard_space(self, config_js):
        assert "@keydown.space" in config_js

    def test_group_icon_aria_hidden(self, config_js):
        assert 'aria-hidden="true">{{ group.icon }}' in config_js

    def test_group_arrow_aria_hidden(self, config_js):
        # The expand/collapse arrow should be decorative
        content = config_js
        assert content.count('aria-hidden="true"') >= 2

    def test_diff_modal_has_dialog_role(self, config_js):
        assert 'role="dialog"' in config_js

    def test_diff_modal_has_aria_modal(self, config_js):
        assert 'aria-modal="true"' in config_js

    def test_diff_modal_has_labelledby(self, config_js):
        assert 'aria-labelledby="cfg-diff-title"' in config_js

    def test_diff_modal_title_has_id(self, config_js):
        assert 'id="cfg-diff-title"' in config_js

    def test_toast_has_aria_live(self, config_js):
        assert 'aria-live="polite"' in config_js


# ===========================================================================
# 14. Focus State Styles
# ===========================================================================

class TestFocusStates:
    """Focus-visible styles for interactive elements."""

    def test_btn_focus_visible(self, style_css):
        assert ".btn:focus-visible" in style_css

    def test_input_focus_visible(self, style_css):
        assert ".hm-input:focus-visible" in style_css

    def test_select_focus_visible(self, style_css):
        assert ".hm-select:focus-visible" in style_css

    def test_nav_item_focus_visible(self, style_css):
        assert ".nav-item:focus-visible" in style_css

    def test_ag_filter_btn_focus_visible(self, style_css):
        assert ".ag-filter-btn:focus-visible" in style_css

    def test_kb_tree_header_focus_visible(self, style_css):
        assert ".kb-tree-header:focus-visible" in style_css

    def test_cron_preset_focus_visible(self, style_css):
        assert ".cron-preset-btn:focus-visible" in style_css

    def test_focus_outline_uses_accent_color(self, style_css):
        # All focus outlines should use the accent color
        focus_idx = style_css.index(".btn:focus-visible")
        block = style_css[focus_idx:focus_idx + 400]
        assert "var(--hm-accent)" in block


# ===========================================================================
# 15. Responsive Layout
# ===========================================================================

class TestResponsiveLayout:
    """Responsive breakpoints and fluid layout."""

    def test_tablet_breakpoint(self, style_css):
        assert "@media (max-width: 768px)" in style_css

    def test_phone_breakpoint(self, style_css):
        assert "@media (max-width: 480px)" in style_css

    def test_sidebar_drawer_on_mobile(self, style_css):
        assert "translateX(-100%)" in style_css

    def test_sidebar_opens_on_mobile(self, style_css):
        assert ".hm-sidebar.mobile-open" in style_css

    def test_touch_friendly_buttons(self, style_css):
        # On mobile, buttons should have min-height 36px
        assert "min-height: 36px" in style_css

    def test_touch_friendly_inputs(self, style_css):
        assert "font-size: 1rem" in style_css  # 16px minimum on iOS

    def test_mobile_hide_class(self, style_css):
        assert ".mobile-hide" in style_css

    def test_fluid_max_width(self, style_css):
        assert "max-width: 1600px" in style_css

    def test_viewport_meta_tag(self, index_html):
        assert 'name="viewport"' in index_html
        assert "width=device-width" in index_html

    def test_html_lang_attribute(self, index_html):
        assert 'lang="en"' in index_html


# ===========================================================================
# 16. Global ARIA Patterns Across All Pages
# ===========================================================================

class TestGlobalARIAPatterns:
    """Verify accessibility patterns are used consistently."""

    def test_all_modals_have_dialog_role(self, all_page_sources):
        """Any modal-overlay should have role=dialog."""
        for name, src in all_page_sources.items():
            if "modal-overlay" in src:
                assert 'role="dialog"' in src, \
                    f"{name}.js has modal-overlay without role=dialog"

    def test_all_modals_have_aria_modal(self, all_page_sources):
        """Any role=dialog should have aria-modal=true."""
        for name, src in all_page_sources.items():
            if 'role="dialog"' in src:
                assert 'aria-modal="true"' in src, \
                    f"{name}.js has role=dialog without aria-modal"

    def test_all_error_states_have_alert_or_hidden(self, all_page_sources):
        """Error icons should be aria-hidden or errors have role=alert."""
        for name, src in all_page_sources.items():
            if "error-state" in src and "error-icon" in src:
                has_alert = 'role="alert"' in src
                has_hidden = 'aria-hidden="true"' in src
                assert has_alert or has_hidden, \
                    f"{name}.js has error state without accessibility attributes"

    def test_no_page_missing_page_fade_in(self, all_page_sources):
        """All pages should have the page-fade-in class for consistency."""
        for name, src in all_page_sources.items():
            assert "page-fade-in" in src, \
                f"{name}.js missing page-fade-in class"


# ===========================================================================
# 17. Keyboard Navigation
# ===========================================================================

class TestKeyboardNavigation:
    """Keyboard navigation support."""

    def test_escape_closes_mobile_sidebar(self, app_js):
        assert "Escape" in app_js

    def test_slash_focuses_search(self, app_js):
        # / key should focus search input
        assert "'/'" in app_js or '= "/"' in app_js or "key === '/'" in app_js

    def test_knowledge_tree_keyboard(self, knowledge_js):
        """Tree headers support Enter and Space for keyboard activation."""
        assert "@keydown.enter" in knowledge_js
        assert "@keydown.space" in knowledge_js

    def test_session_thread_keyboard(self, sessions_js):
        assert "@keydown.enter" in sessions_js
        assert "@keydown.space" in sessions_js

    def test_config_group_keyboard(self, config_js):
        assert "@keydown.enter" in config_js
        assert "@keydown.space" in config_js

    def test_config_undo_redo_keyboard(self, config_js):
        """Config page supports Ctrl+Z/Y for undo/redo."""
        assert "ctrlKey" in config_js or "metaKey" in config_js


# ===========================================================================
# 18. CSS Design Token Usage
# ===========================================================================

class TestDesignTokens:
    """Verify design tokens are used for accessibility-related values."""

    def test_focus_outline_uses_accent(self, style_css):
        assert "var(--hm-accent)" in style_css

    def test_transition_tokens_exist(self, style_css):
        assert "--hm-transition-fast" in style_css
        assert "--hm-transition-base" in style_css

    def test_spacing_tokens_exist(self, style_css):
        for i in [1, 2, 3, 4, 5, 6, 8]:
            assert f"--hm-space-{i}" in style_css

    def test_radius_tokens_exist(self, style_css):
        assert "--hm-radius-sm" in style_css
        assert "--hm-radius-md" in style_css
        assert "--hm-radius-lg" in style_css


# ===========================================================================
# 19. Semantic HTML
# ===========================================================================

class TestSemanticHTML:
    """Semantic HTML elements used correctly."""

    def test_main_element_used(self, app_js):
        assert "<main" in app_js

    def test_header_element_used(self, app_js):
        assert "<header" in app_js

    def test_nav_element_used(self, app_js):
        assert "<nav" in app_js

    def test_aside_element_used(self, app_js):
        assert "<aside" in app_js

    def test_html_has_doctype(self, index_html):
        assert "<!DOCTYPE html>" in index_html

    def test_title_element(self, index_html):
        assert "<title>" in index_html
        assert "Heimdall" in index_html


# ===========================================================================
# 20. Color Contrast (Design System Check)
# ===========================================================================

class TestColorContrast:
    """Verify color values meet WCAG AA contrast requirements."""

    def test_primary_text_is_light(self, style_css):
        """Primary text should be light on dark background."""
        assert "--hm-text:" in style_css
        # #f3f4f6 is very light
        assert "#f3f4f6" in style_css

    def test_background_is_dark(self, style_css):
        assert "--hm-bg:" in style_css
        assert "#030712" in style_css

    def test_muted_text_not_too_dim(self, style_css):
        """Muted text (#9ca3af) should meet AA on dark surfaces."""
        assert "--hm-text-muted:" in style_css
        assert "#9ca3af" in style_css


# ===========================================================================
# 21. Backward Compatibility
# ===========================================================================

class TestBackwardCompatibility:
    """Ensure existing classes and features are preserved."""

    def test_existing_responsive_breakpoints_preserved(self, style_css):
        assert "@media (max-width: 768px)" in style_css
        assert "@media (max-width: 480px)" in style_css

    def test_existing_focus_states_preserved(self, style_css):
        assert ".btn:focus-visible" in style_css
        assert ".nav-item:focus-visible" in style_css

    def test_sidebar_classes_preserved(self, style_css):
        assert ".hm-sidebar" in style_css
        assert ".hm-sidebar.collapsed" in style_css

    def test_status_dot_classes_preserved(self, style_css):
        assert ".status-dot" in style_css
        assert ".status-dot.online" in style_css
        assert ".status-dot.offline" in style_css

    def test_nav_item_active_preserved(self, style_css):
        assert ".nav-item.active" in style_css

    def test_page_fade_in_preserved(self, style_css):
        assert ".page-fade-in" in style_css

    def test_hm_card_preserved(self, style_css):
        assert ".hm-card" in style_css

    def test_modal_overlay_preserved(self, style_css):
        assert ".modal-overlay" in style_css or "modal-overlay" in style_css

    def test_empty_state_preserved(self, style_css):
        assert ".empty-state" in style_css


# ===========================================================================
# 22. Fluid Typography
# ===========================================================================

class TestFluidTypography:
    """Typography scales with viewport."""

    def test_large_screen_font_scaling(self, style_css):
        """Font size increases on larger screens."""
        assert "@media (min-width: 1200px)" in style_css

    def test_phone_title_scaling(self, style_css):
        """Page titles scale down on phones."""
        # The h1 rule is in the small phone @media block
        assert "max-width: 480px" in style_css
        # Find all 480px blocks
        idx = 0
        found_h1 = False
        while True:
            try:
                idx = style_css.index("480px", idx)
                block = style_css[idx:idx + 600]
                if "h1" in block:
                    found_h1 = True
                    break
                idx += 5
            except ValueError:
                break
        assert found_h1, "h1 scaling rule not found in any 480px media query"


# ===========================================================================
# 23. Accessible Images
# ===========================================================================

class TestAccessibleImages:
    """Images have alt text and decorative elements are hidden."""

    def test_chat_images_have_alt(self, chat_js):
        """Chat inline images have alt attributes."""
        assert ":alt=" in chat_js or 'alt="' in chat_js

    def test_dashboard_hero_icon_hidden(self, dashboard_js):
        """Hero icon is decorative."""
        assert 'aria-hidden="true"' in dashboard_js
