"""Round 34 — Visual Design System tests.

Validates the Heimdall design token system: CSS variables, typography,
spacing scale, color palette, shadows, component classes, and HTML integration.
"""
import re
from pathlib import Path

import pytest

CSS_PATH = Path("ui/css/style.css")
HTML_PATH = Path("ui/index.html")


@pytest.fixture(scope="module")
def css_text():
    return CSS_PATH.read_text()


@pytest.fixture(scope="module")
def html_text():
    return HTML_PATH.read_text()


# ── File existence ──────────────────────────────────────────────────────────

class TestFileStructure:
    def test_css_file_exists(self):
        assert CSS_PATH.exists()

    def test_html_file_exists(self):
        assert HTML_PATH.exists()

    def test_css_not_empty(self, css_text):
        assert len(css_text) > 500

    def test_html_references_stylesheet(self, html_text):
        assert "/ui/css/style.css" in html_text


# ── Design Token: Colors ───────────────────────────────────────────────────

class TestColorTokens:
    """Validate all required color tokens exist in :root."""

    REQUIRED_COLORS = [
        "--hm-bg",
        "--hm-surface",
        "--hm-surface-hover",
        "--hm-surface-elevated",
        "--hm-border",
        "--hm-border-subtle",
        "--hm-accent",
        "--hm-accent-hover",
        "--hm-accent-dim",
        "--hm-accent-glow",
        "--hm-gold",
        "--hm-gold-dim",
        "--hm-text",
        "--hm-text-muted",
        "--hm-text-dim",
        "--hm-success",
        "--hm-warning",
        "--hm-danger",
        "--hm-info",
    ]

    @pytest.mark.parametrize("token", REQUIRED_COLORS)
    def test_color_token_defined(self, css_text, token):
        assert token in css_text, f"Missing color token: {token}"

    def test_accent_is_amber(self, css_text):
        """Accent color should be in the amber/gold family (#d97706)."""
        match = re.search(r"--hm-accent:\s*([^;]+);", css_text)
        assert match
        assert "#d97706" in match.group(1)

    def test_gold_is_warm(self, css_text):
        """Gold token should be warm yellow (#fbbf24)."""
        match = re.search(r"--hm-gold:\s*([^;]+);", css_text)
        assert match
        assert "#fbbf24" in match.group(1)

    def test_bg_is_dark(self, css_text):
        """Background should be near-black."""
        match = re.search(r"--hm-bg:\s*([^;]+);", css_text)
        assert match
        # Should start with #0 (very dark)
        color = match.group(1).strip()
        assert color.startswith("#0")


# ── Design Token: Typography ───────────────────────────────────────────────

class TestTypographyTokens:
    """Validate typography tokens."""

    REQUIRED_TOKENS = [
        "--hm-font-sans",
        "--hm-font-mono",
        "--hm-text-xs",
        "--hm-text-sm",
        "--hm-text-base",
        "--hm-text-md",
        "--hm-text-lg",
        "--hm-text-xl",
        "--hm-text-2xl",
        "--hm-leading-tight",
        "--hm-leading-normal",
        "--hm-leading-relaxed",
    ]

    @pytest.mark.parametrize("token", REQUIRED_TOKENS)
    def test_typography_token_defined(self, css_text, token):
        assert token in css_text, f"Missing typography token: {token}"

    def test_font_sans_includes_inter(self, css_text):
        """Sans font stack should include Inter."""
        match = re.search(r"--hm-font-sans:\s*([^;]+);", css_text)
        assert match
        assert "Inter" in match.group(1)

    def test_font_mono_includes_fira(self, css_text):
        """Mono font stack should include Fira Code."""
        match = re.search(r"--hm-font-mono:\s*([^;]+);", css_text)
        assert match
        assert "Fira Code" in match.group(1)

    def test_body_uses_font_sans(self, css_text):
        """Body should use the sans font variable."""
        assert "var(--hm-font-sans)" in css_text

    def test_text_size_scale_ordered(self, css_text):
        """Font sizes should increase from xs to 2xl."""
        sizes = {}
        for token in ["--hm-text-xs", "--hm-text-sm", "--hm-text-base",
                       "--hm-text-md", "--hm-text-lg", "--hm-text-xl", "--hm-text-2xl"]:
            match = re.search(rf"{re.escape(token)}:\s*([\d.]+)rem", css_text)
            assert match, f"Cannot parse size for {token}"
            sizes[token] = float(match.group(1))
        ordered = list(sizes.values())
        assert ordered == sorted(ordered), "Font sizes must be in ascending order"

    def test_font_smoothing(self, css_text):
        """Body should enable font smoothing."""
        assert "-webkit-font-smoothing" in css_text
        assert "antialiased" in css_text


# ── Design Token: Spacing ──────────────────────────────────────────────────

class TestSpacingTokens:
    """Validate spacing scale."""

    REQUIRED_SPACING = [
        "--hm-space-1",
        "--hm-space-2",
        "--hm-space-3",
        "--hm-space-4",
        "--hm-space-5",
        "--hm-space-6",
        "--hm-space-8",
    ]

    @pytest.mark.parametrize("token", REQUIRED_SPACING)
    def test_spacing_token_defined(self, css_text, token):
        assert token in css_text, f"Missing spacing token: {token}"

    def test_spacing_scale_increases(self, css_text):
        """Spacing values should increase with token number."""
        values = {}
        for token in self.REQUIRED_SPACING:
            match = re.search(rf"{re.escape(token)}:\s*([\d.]+)rem", css_text)
            assert match, f"Cannot parse spacing for {token}"
            values[token] = float(match.group(1))
        ordered = list(values.values())
        assert ordered == sorted(ordered), "Spacing values must increase"


# ── Design Token: Border Radius ────────────────────────────────────────────

class TestRadiusTokens:
    REQUIRED = [
        "--hm-radius-sm",
        "--hm-radius-md",
        "--hm-radius-lg",
        "--hm-radius-xl",
        "--hm-radius-full",
    ]

    @pytest.mark.parametrize("token", REQUIRED)
    def test_radius_token_defined(self, css_text, token):
        assert token in css_text

    def test_radius_full_is_pill(self, css_text):
        """Radius-full should create pill shape."""
        match = re.search(r"--hm-radius-full:\s*([^;]+);", css_text)
        assert match
        assert "9999px" in match.group(1)


# ── Design Token: Shadows ──────────────────────────────────────────────────

class TestShadowTokens:
    REQUIRED = [
        "--hm-shadow-sm",
        "--hm-shadow-md",
        "--hm-shadow-lg",
        "--hm-shadow-glow",
        "--hm-shadow-glow-sm",
    ]

    @pytest.mark.parametrize("token", REQUIRED)
    def test_shadow_token_defined(self, css_text, token):
        assert token in css_text

    def test_glow_uses_accent_color(self, css_text):
        """Glow shadows should use the accent (amber) color."""
        match = re.search(r"--hm-shadow-glow:\s*([^;]+);", css_text)
        assert match
        # Should reference amber rgb values (217, 119, 6) or similar
        assert "217" in match.group(1)


# ── Design Token: Transitions ─────────────────────────────────────────────

class TestTransitionTokens:
    REQUIRED = [
        "--hm-transition-fast",
        "--hm-transition-base",
        "--hm-transition-slow",
        "--hm-transition-spring",
    ]

    @pytest.mark.parametrize("token", REQUIRED)
    def test_transition_token_defined(self, css_text, token):
        assert token in css_text

    def test_spring_uses_cubic_bezier(self, css_text):
        """Spring transition should use cubic-bezier."""
        match = re.search(r"--hm-transition-spring:\s*([^;]+);", css_text)
        assert match
        assert "cubic-bezier" in match.group(1)


# ── Component Classes ──────────────────────────────────────────────────────

class TestComponentClasses:
    """Ensure all critical component classes exist."""

    REQUIRED_CLASSES = [
        ".hm-sidebar",
        ".hm-main",
        ".hm-topbar",
        ".hm-card",
        ".hm-table",
        ".hm-input",
        ".hm-select",
        ".btn",
        ".btn-primary",
        ".btn-danger",
        ".btn-ghost",
        ".badge",
        ".badge-success",
        ".badge-warning",
        ".badge-danger",
        ".badge-info",
        ".nav-item",
        ".nav-item.active",
        ".spinner",
        ".modal-overlay",
        ".modal-content",
        ".skeleton",
        ".toast",
        ".toast-success",
        ".toast-error",
        ".chat-container",
        ".chat-bubble",
        ".chat-bubble-user",
        ".chat-bubble-bot",
        ".chat-input",
        ".chat-typing",
        ".empty-state",
        ".log-chip",
        ".log-line",
        ".session-card",
        ".skill-code-block",
        ".status-dot",
        ".toggle-switch",
    ]

    @pytest.mark.parametrize("cls", REQUIRED_CLASSES)
    def test_component_class_exists(self, css_text, cls):
        # The class should appear as a selector
        assert cls in css_text, f"Missing component class: {cls}"


# ── Token Usage ────────────────────────────────────────────────────────────

class TestTokenUsage:
    """Ensure tokens are actually used in component definitions."""

    def test_card_uses_shadow(self, css_text):
        """Cards should have shadow for elevation."""
        # Find .hm-card block
        assert "var(--hm-shadow-sm)" in css_text or "var(--hm-shadow-md)" in css_text

    def test_input_focus_uses_glow(self, css_text):
        """Input focus should use accent glow."""
        assert "var(--hm-accent-glow)" in css_text

    def test_buttons_use_transition_token(self, css_text):
        """Buttons should use transition tokens."""
        assert "var(--hm-transition-base)" in css_text

    def test_modal_uses_shadow(self, css_text):
        """Modals should have shadow for depth."""
        assert "var(--hm-shadow-lg)" in css_text

    def test_font_mono_var_used(self, css_text):
        """Mono font variable should be used in code components."""
        assert "var(--hm-font-mono)" in css_text

    def test_radius_tokens_used(self, css_text):
        """Radius tokens should be used in components."""
        radius_usages = css_text.count("var(--hm-radius-")
        assert radius_usages >= 10, f"Only {radius_usages} radius token usages found"

    def test_spacing_tokens_used(self, css_text):
        """Spacing tokens should be used in components."""
        spacing_usages = css_text.count("var(--hm-space-")
        assert spacing_usages >= 20, f"Only {spacing_usages} spacing token usages found"


# ── Visual Enhancements ───────────────────────────────────────────────────

class TestVisualEnhancements:
    """Validate design system visual flourishes."""

    def test_active_nav_has_indicator(self, css_text):
        """Active nav should have visual indicator (inset shadow or border)."""
        # Look for the active nav definition
        active_block_start = css_text.find(".nav-item.active")
        assert active_block_start >= 0
        active_block = css_text[active_block_start:active_block_start + 200]
        assert "box-shadow" in active_block or "border" in active_block

    def test_chat_user_bubble_has_gradient(self, css_text):
        """User chat bubble should use gradient for depth."""
        bubble_start = css_text.find(".chat-bubble-user")
        assert bubble_start >= 0
        bubble_block = css_text[bubble_start:bubble_start + 200]
        assert "linear-gradient" in bubble_block

    def test_stat_card_hover_accent(self, css_text):
        """Stat cards should have hover accent effect."""
        assert "stat-card" in css_text
        # Should have a pseudo-element or hover effect
        assert "stat-card:hover" in css_text or "stat-card::before" in css_text

    def test_modal_has_backdrop_blur(self, css_text):
        """Modal overlay should use backdrop-filter for blur."""
        overlay_start = css_text.find(".modal-overlay")
        assert overlay_start >= 0
        overlay_block = css_text[overlay_start:overlay_start + 300]
        assert "backdrop-filter" in overlay_block

    def test_primary_button_hover_glow(self, css_text):
        """Primary button hover should have glow effect."""
        btn_primary_hover_start = css_text.find(".btn-primary:hover")
        assert btn_primary_hover_start >= 0
        hover_block = css_text[btn_primary_hover_start:btn_primary_hover_start + 200]
        assert "shadow" in hover_block

    def test_button_active_scale(self, css_text):
        """Buttons should scale on press."""
        assert "scale(0.97)" in css_text or "scale(0.98)" in css_text

    def test_session_card_hover_glow(self, css_text):
        """Session cards should glow on hover."""
        hover_start = css_text.find(".session-card:hover")
        assert hover_start >= 0
        hover_block = css_text[hover_start:hover_start + 200]
        assert "shadow" in hover_block

    def test_toggle_checked_glow(self, css_text):
        """Toggle switch should glow when checked."""
        checked_start = css_text.find("input:checked + .toggle-slider")
        assert checked_start >= 0
        block = css_text[checked_start:checked_start + 200]
        assert "shadow" in block


# ── Utility Classes ───────────────────────────────────────────────────────

class TestUtilityClasses:
    """Validate design system utility classes."""

    def test_card_accent_class(self, css_text):
        assert ".hm-card-accent" in css_text

    def test_divider_class(self, css_text):
        assert ".hm-divider" in css_text

    def test_mono_utility(self, css_text):
        assert ".hm-mono" in css_text

    def test_section_title_class(self, css_text):
        assert ".hm-section-title" in css_text

    def test_divider_uses_gradient(self, css_text):
        """Divider should use a subtle gold gradient."""
        divider_start = css_text.find(".hm-divider")
        assert divider_start >= 0
        divider_block = css_text[divider_start:divider_start + 200]
        assert "linear-gradient" in divider_block


# ── HTML Integration ──────────────────────────────────────────────────────

class TestHTMLIntegration:
    """Validate HTML <head> has proper font loading."""

    def test_google_fonts_preconnect(self, html_text):
        """Should preconnect to Google Fonts for performance."""
        assert "fonts.googleapis.com" in html_text

    def test_google_fonts_gstatic_preconnect(self, html_text):
        """Should preconnect to gstatic for font files."""
        assert "fonts.gstatic.com" in html_text

    def test_inter_font_loaded(self, html_text):
        """Inter font should be loaded from Google Fonts."""
        assert "Inter" in html_text

    def test_fira_code_font_loaded(self, html_text):
        """Fira Code font should be loaded from Google Fonts."""
        assert "Fira+Code" in html_text or "Fira Code" in html_text

    def test_body_uses_font_variable(self, html_text):
        """Body should reference the font CSS variable."""
        assert "var(--hm-font-sans)" in html_text

    def test_font_weights_loaded(self, html_text):
        """Multiple font weights should be loaded."""
        assert "400" in html_text
        assert "500" in html_text
        assert "600" in html_text


# ── Responsive Design ─────────────────────────────────────────────────────

class TestResponsiveDesign:
    """Ensure responsive breakpoints are still intact."""

    def test_tablet_breakpoint(self, css_text):
        assert "max-width: 768px" in css_text

    def test_phone_breakpoint(self, css_text):
        assert "max-width: 480px" in css_text

    def test_sidebar_mobile_drawer(self, css_text):
        """Sidebar should transform to drawer on mobile."""
        assert "mobile-open" in css_text

    def test_mobile_hide_utility(self, css_text):
        assert ".mobile-hide" in css_text

    def test_touch_friendly_buttons(self, css_text):
        """Buttons should have minimum height on mobile."""
        assert "min-height: 36px" in css_text


# ── Animations ────────────────────────────────────────────────────────────

class TestAnimations:
    """Ensure all animations/keyframes are defined."""

    REQUIRED_ANIMATIONS = [
        "@keyframes spin",
        "@keyframes shimmer",
        "@keyframes flash-highlight",
        "@keyframes toast-in",
        "@keyframes typing-dot",
        "@keyframes jump-btn-in",
        "@keyframes loop-pulse",
        "@keyframes page-fade",
    ]

    @pytest.mark.parametrize("animation", REQUIRED_ANIMATIONS)
    def test_animation_defined(self, css_text, animation):
        assert animation in css_text, f"Missing animation: {animation}"


# ── Design System Consistency ─────────────────────────────────────────────

class TestDesignConsistency:
    """Check for consistent usage patterns."""

    def test_no_orphan_hex_colors_in_accent_spots(self, css_text):
        """Key accent properties should use vars, not raw hex."""
        # .btn-primary background should use var, not raw #d97706
        btn_primary_start = css_text.find(".btn-primary {")
        if btn_primary_start >= 0:
            block = css_text[btn_primary_start:btn_primary_start + 200]
            assert "var(--hm-accent)" in block

    def test_table_headers_uppercase(self, css_text):
        """Table headers should use uppercase for hierarchy."""
        th_start = css_text.find(".hm-table th")
        assert th_start >= 0
        block = css_text[th_start:th_start + 300]
        assert "text-transform" in block or "uppercase" in block

    def test_consistent_font_family_usage(self, css_text):
        """Mono code elements should use the font-mono variable."""
        mono_var_count = css_text.count("var(--hm-font-mono)")
        # Should be used in multiple places (textarea, code blocks, logs, etc.)
        assert mono_var_count >= 3, f"Only {mono_var_count} uses of --hm-font-mono variable"

    def test_organized_sections(self, css_text):
        """CSS should have organized section headers."""
        assert "DESIGN TOKENS" in css_text
        assert "LAYOUT" in css_text
        assert "BUTTONS" in css_text
        assert "FORMS" in css_text
        assert "RESPONSIVE" in css_text

    def test_css_variable_count(self, css_text):
        """Design system should have substantial number of tokens."""
        root_match = re.search(r":root\s*\{([^}]+)\}", css_text, re.DOTALL)
        assert root_match
        var_count = root_match.group(1).count("--hm-")
        assert var_count >= 35, f"Only {var_count} design tokens in :root"
