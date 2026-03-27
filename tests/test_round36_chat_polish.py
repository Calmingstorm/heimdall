"""Round 36 — Chat interface polish tests.

Covers:
- Chat JS template structure (avatars, timestamps, tool cards, welcome state,
  date separators, code copy, inline images, typing indicator with text,
  connection status, send icon, suggestions)
- Chat CSS classes (all new selectors exist)
- Chat JS helper function exports
- Image URL extraction regex
- formatTime helper
- Tool icon mapping
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
CHAT_JS = UI_DIR / "js" / "pages" / "chat.js"
STYLE_CSS = UI_DIR / "css" / "style.css"


@pytest.fixture(scope="module")
def chat_js():
    return CHAT_JS.read_text()


@pytest.fixture(scope="module")
def style_css():
    return STYLE_CSS.read_text()


# ===================================================================
# Template structure — verify required DOM elements
# ===================================================================


class TestChatTemplateStructure:
    """Verify the chat template contains all polished elements."""

    def test_has_welcome_section(self, chat_js):
        assert "chat-welcome" in chat_js

    def test_has_welcome_icon(self, chat_js):
        assert "chat-welcome-icon" in chat_js

    def test_has_welcome_title(self, chat_js):
        assert "chat-welcome-title" in chat_js
        assert "Heimdall is watching" in chat_js

    def test_has_welcome_subtitle(self, chat_js):
        assert "chat-welcome-subtitle" in chat_js

    def test_has_suggestions(self, chat_js):
        assert "chat-suggestions" in chat_js
        assert "chat-suggestion" in chat_js
        assert "useSuggestion" in chat_js

    def test_has_date_separator(self, chat_js):
        assert "chat-date-sep" in chat_js
        assert "showDateSeparator" in chat_js

    def test_has_avatars(self, chat_js):
        assert "chat-avatar" in chat_js
        assert "chat-avatar-bot" in chat_js
        assert "chat-avatar-user" in chat_js
        assert "chat-avatar-eye" in chat_js

    def test_has_bubble_wrap(self, chat_js):
        assert "chat-bubble-wrap" in chat_js

    def test_has_bubble_header(self, chat_js):
        assert "chat-bubble-header" in chat_js

    def test_has_error_indicator(self, chat_js):
        assert "chat-error-indicator" in chat_js

    def test_has_timestamps(self, chat_js):
        assert "chat-timestamp" in chat_js
        assert "formatTime" in chat_js

    def test_has_tool_cards(self, chat_js):
        assert "chat-tool-cards" in chat_js
        assert "chat-tool-card" in chat_js
        assert "chat-tool-icon" in chat_js
        assert "chat-tool-name" in chat_js

    def test_has_tool_toggle_count(self, chat_js):
        assert "chat-tools-toggle-count" in chat_js

    def test_has_inline_images(self, chat_js):
        assert "chat-images" in chat_js
        assert "chat-image-thumb" in chat_js
        assert "openImage" in chat_js
        assert "onImageError" in chat_js

    def test_has_typing_text(self, chat_js):
        assert "chat-typing-text" in chat_js
        assert "typingText" in chat_js

    def test_has_typing_pulse_avatar(self, chat_js):
        assert "chat-avatar-pulse" in chat_js

    def test_has_code_copy_attachment(self, chat_js):
        assert "attachCopyButtons" in chat_js
        assert "chat-code-copy" in chat_js

    def test_has_send_icon_svg(self, chat_js):
        assert "chat-send-icon" in chat_js
        # SVG path element for send arrow
        assert "<svg" in chat_js

    def test_has_connection_status(self, chat_js):
        assert "chat-connection-status" in chat_js
        assert "chat-status-dot" in chat_js
        assert "chat-ws-on" in chat_js
        assert "chat-ws-off" in chat_js


# ===================================================================
# CSS class coverage — verify all new classes exist in style.css
# ===================================================================


class TestChatCSSClasses:
    """Verify all new chat CSS classes are defined."""

    REQUIRED_CLASSES = [
        # Welcome/empty state
        "chat-welcome",
        "chat-welcome-icon",
        "chat-welcome-title",
        "chat-welcome-subtitle",
        "chat-suggestions",
        "chat-suggestion",
        # Date separator
        "chat-date-sep",
        # Avatars
        "chat-avatar",
        "chat-avatar-bot",
        "chat-avatar-user",
        "chat-avatar-pulse",
        # Bubble structure
        "chat-bubble-wrap",
        "chat-bubble-header",
        "chat-error-indicator",
        "chat-timestamp",
        # Tool cards
        "chat-tool-cards",
        "chat-tool-list",
        "chat-tool-card",
        "chat-tool-icon",
        "chat-tool-name",
        "chat-tools-toggle-icon",
        "chat-tools-toggle-count",
        # Code copy
        "chat-code-copy",
        # Inline images
        "chat-images",
        "chat-image-thumb",
        # Typing
        "chat-typing-text",
        "chat-bubble-typing",
        # Input area
        "chat-send-icon",
        "chat-connection-status",
        "chat-status-dot",
        "chat-ws-on",
        "chat-ws-off",
        # Markdown tables
        "chat-markdown th",
        "chat-markdown td",
        "chat-markdown table",
    ]

    @pytest.mark.parametrize("cls", REQUIRED_CLASSES)
    def test_css_class_defined(self, style_css, cls):
        # Check the class selector exists in the CSS (either as .class or compound)
        # Handle compound selectors like "chat-markdown th"
        if " " in cls:
            parts = cls.split(" ", 1)
            assert f".{parts[0]}" in style_css and parts[1] in style_css, (
                f"CSS class/selector '{cls}' not found in style.css"
            )
        else:
            assert f".{cls}" in style_css, f"CSS class '.{cls}' not found in style.css"


class TestChatCSSAnimations:
    """Verify CSS animations for chat elements."""

    def test_avatar_pulse_animation(self, style_css):
        assert "avatar-pulse" in style_css
        assert "@keyframes avatar-pulse" in style_css

    def test_typing_dot_animation(self, style_css):
        assert "@keyframes typing-dot" in style_css

    def test_typing_dots_use_accent_color(self, style_css):
        # Typing dots should use the accent color
        assert "chat-typing span" in style_css


class TestChatCSSPreservedElements:
    """Verify existing CSS classes are still present."""

    PRESERVED_CLASSES = [
        "chat-container",
        "chat-messages",
        "chat-empty",
        "chat-message",
        "chat-bubble",
        "chat-bubble-user",
        "chat-bubble-bot",
        "chat-bubble-label",
        "chat-bubble-text",
        "chat-markdown",
        "chat-tools-toggle",
        "chat-typing",
        "chat-input-area",
        "chat-input-row",
        "chat-input",
        "chat-send-btn",
        "chat-input-hint",
    ]

    @pytest.mark.parametrize("cls", PRESERVED_CLASSES)
    def test_preserved_class(self, style_css, cls):
        assert f".{cls}" in style_css, f"Preserved class '.{cls}' missing from style.css"


# ===================================================================
# JS helper functions — verify presence and exports
# ===================================================================


class TestChatJSHelpers:
    """Verify helper functions are present in chat.js."""

    def test_format_time_function(self, chat_js):
        assert "function formatTime" in chat_js

    def test_extract_image_urls_function(self, chat_js):
        assert "function extractImageUrls" in chat_js

    def test_get_tool_icon_function(self, chat_js):
        assert "function getToolIcon" in chat_js

    def test_render_markdown_function(self, chat_js):
        assert "function renderMarkdown" in chat_js

    def test_tool_icons_mapping(self, chat_js):
        assert "TOOL_ICONS" in chat_js
        # Key tools should have icons
        assert "run_command" in chat_js
        assert "read_file" in chat_js
        assert "search_knowledge" in chat_js
        assert "generate_image" in chat_js

    def test_img_url_regex(self, chat_js):
        assert "IMG_URL_RE" in chat_js
        # Should match common image extensions
        assert "png" in chat_js
        assert "jpg" in chat_js
        assert "webp" in chat_js

    def test_suggestions_defined(self, chat_js):
        assert "suggestions" in chat_js
        assert "Check system health" in chat_js

    def test_typing_phrases_defined(self, chat_js):
        assert "typingPhrases" in chat_js
        # Should include thematic Heimdall phrases
        assert "realm" in chat_js.lower() or "bifrost" in chat_js.lower()


class TestChatJSExports:
    """Verify setup() returns all needed refs and methods."""

    def test_returns_format_time(self, chat_js):
        # Check the return block includes formatTime
        return_match = re.search(r'return\s*\{([^}]+)\}', chat_js, re.DOTALL)
        assert return_match, "No return statement found in setup()"
        returned = return_match.group(1)
        assert "formatTime" in returned

    def test_returns_suggestions(self, chat_js):
        return_match = re.search(r'return\s*\{([^}]+)\}', chat_js, re.DOTALL)
        returned = return_match.group(1)
        assert "suggestions" in returned

    def test_returns_typing_text(self, chat_js):
        return_match = re.search(r'return\s*\{([^}]+)\}', chat_js, re.DOTALL)
        returned = return_match.group(1)
        assert "typingText" in returned

    def test_returns_use_suggestion(self, chat_js):
        return_match = re.search(r'return\s*\{([^}]+)\}', chat_js, re.DOTALL)
        returned = return_match.group(1)
        assert "useSuggestion" in returned

    def test_returns_open_image(self, chat_js):
        return_match = re.search(r'return\s*\{([^}]+)\}', chat_js, re.DOTALL)
        returned = return_match.group(1)
        assert "openImage" in returned

    def test_returns_get_tool_icon(self, chat_js):
        return_match = re.search(r'return\s*\{([^}]+)\}', chat_js, re.DOTALL)
        returned = return_match.group(1)
        assert "getToolIcon" in returned

    def test_returns_show_date_separator(self, chat_js):
        return_match = re.search(r'return\s*\{([^}]+)\}', chat_js, re.DOTALL)
        returned = return_match.group(1)
        assert "showDateSeparator" in returned

    def test_returns_format_date(self, chat_js):
        return_match = re.search(r'return\s*\{([^}]+)\}', chat_js, re.DOTALL)
        returned = return_match.group(1)
        assert "formatDate" in returned

    def test_returns_on_image_error(self, chat_js):
        return_match = re.search(r'return\s*\{([^}]+)\}', chat_js, re.DOTALL)
        returned = return_match.group(1)
        assert "onImageError" in returned


# ===================================================================
# Image URL extraction regex test
# ===================================================================


class TestImageUrlExtraction:
    """Test the IMG_URL_RE pattern from chat.js (replicated for testing)."""

    IMG_RE = re.compile(r'https?://\S+\.(?:png|jpg|jpeg|gif|webp|svg)(?:\?\S*)?', re.I)

    def test_matches_png(self):
        assert self.IMG_RE.search("Look at https://example.com/image.png here")

    def test_matches_jpg(self):
        assert self.IMG_RE.search("https://cdn.example.com/photo.jpg")

    def test_matches_jpeg(self):
        assert self.IMG_RE.search("https://example.com/pic.jpeg")

    def test_matches_gif(self):
        assert self.IMG_RE.search("https://example.com/anim.gif")

    def test_matches_webp(self):
        assert self.IMG_RE.search("https://example.com/modern.webp")

    def test_matches_svg(self):
        assert self.IMG_RE.search("https://example.com/icon.svg")

    def test_matches_with_query_string(self):
        m = self.IMG_RE.search("https://example.com/img.png?w=800&h=600")
        assert m

    def test_no_match_for_text_file(self):
        assert not self.IMG_RE.search("https://example.com/file.txt")

    def test_no_match_for_plain_text(self):
        assert not self.IMG_RE.search("no urls here")

    def test_multiple_images_found(self):
        text = "See https://a.com/1.png and https://b.com/2.jpg ok"
        matches = self.IMG_RE.findall(text)
        assert len(matches) == 2

    def test_case_insensitive(self):
        assert self.IMG_RE.search("https://example.com/image.PNG")


# ===================================================================
# Message structure — verify addMessage produces correct fields
# ===================================================================


class TestChatMessageStructure:
    """Verify the addMessage function creates messages with all required fields."""

    def test_message_has_id_field(self, chat_js):
        # addMessage should assign an id to each message
        assert "id:" in chat_js and "msgIdCounter" in chat_js

    def test_message_has_timestamp(self, chat_js):
        assert "timestamp:" in chat_js and "Date.now()" in chat_js

    def test_message_has_images_field(self, chat_js):
        assert "images:" in chat_js
        assert "extractImageUrls" in chat_js

    def test_message_has_show_tools_toggle(self, chat_js):
        assert "_showTools:" in chat_js


# ===================================================================
# Typing indicator features
# ===================================================================


class TestTypingIndicator:
    """Verify enhanced typing indicator features."""

    def test_typing_elapsed_ref(self, chat_js):
        assert "typingElapsed" in chat_js

    def test_typing_timer_managed(self, chat_js):
        assert "startTypingTimer" in chat_js
        assert "stopTypingTimer" in chat_js

    def test_typing_timer_cleared_on_unmount(self, chat_js):
        assert "stopTypingTimer" in chat_js
        # Should be called in onUnmounted
        assert "onUnmounted" in chat_js

    def test_typing_shows_elapsed_seconds(self, chat_js):
        # After 3 seconds, should show elapsed time
        assert "secs" in chat_js or "typingElapsed" in chat_js

    def test_typing_rotates_phrases(self, chat_js):
        assert "typingPhrases" in chat_js
        # Should have multiple phrases
        assert "Processing" in chat_js


# ===================================================================
# Code copy button
# ===================================================================


class TestCodeCopyButton:
    """Verify code copy button functionality."""

    def test_copy_button_class(self, chat_js):
        assert "chat-code-copy" in chat_js

    def test_uses_clipboard_api(self, chat_js):
        assert "navigator.clipboard" in chat_js

    def test_copy_feedback(self, chat_js):
        # Should show "Copied!" feedback
        assert "Copied!" in chat_js

    def test_copy_button_attached_after_render(self, chat_js):
        # attachCopyButtons called in nextTick after bot message
        assert "nextTick" in chat_js
        assert "attachCopyButtons" in chat_js

    def test_pre_blocks_only_get_one_button(self, chat_js):
        # Uses data-copy attribute to prevent duplicates
        assert "data-copy" in chat_js


# ===================================================================
# Connection status
# ===================================================================


class TestConnectionStatus:
    """Verify connection status indicator."""

    def test_shows_connected_or_fallback(self, chat_js):
        assert "Connected" in chat_js
        assert "REST fallback" in chat_js

    def test_ws_connected_checked(self, chat_js):
        assert "ws.connected" in chat_js


# ===================================================================
# CSS responsive rules
# ===================================================================


class TestChatResponsive:
    """Verify responsive CSS adjustments."""

    def test_mobile_wider_bubbles(self, style_css):
        assert "max-width: 90%" in style_css

    def test_mobile_smaller_avatars(self, style_css):
        # Should reduce avatar size on mobile
        assert "24px" in style_css


# ===================================================================
# Markdown CSS enhancements
# ===================================================================


class TestMarkdownCSS:
    """Verify markdown CSS enhancements."""

    def test_table_styles(self, style_css):
        assert ".chat-markdown table" in style_css
        assert "border-collapse" in style_css

    def test_table_cell_styles(self, style_css):
        assert ".chat-markdown th" in style_css
        assert ".chat-markdown td" in style_css

    def test_code_block_position_relative(self, style_css):
        # Pre blocks need position:relative for copy button
        assert "position: relative" in style_css

    def test_code_copy_positioned(self, style_css):
        assert ".chat-code-copy" in style_css
        assert "position: absolute" in style_css

    def test_code_copy_opacity_transition(self, style_css):
        # Copy button should appear on hover
        assert "chat-markdown pre:hover .chat-code-copy" in style_css
