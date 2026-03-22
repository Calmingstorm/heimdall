"""Tests for Playwright browser automation (browser.py).

Mocks Playwright entirely — no real browser needed. Tests verify:
- URL validation (allowed/rejected schemes)
- BrowserManager connection lifecycle (lazy connect, reconnect, shutdown)
- new_page context manager (creates context, page, cleans up)
- handle_browser_screenshot (navigation, screenshot, wait, status)
- handle_browser_read_page (full body, CSS selector, truncation)
- handle_browser_read_table (markdown formatting, no-table, empty table, truncation)
- handle_browser_click (success, click failure, post-click navigation)
- handle_browser_fill (fill, fill+submit, fill error, submit error)
- handle_browser_evaluate (dict/list/str results, JS errors, truncation)
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.browser import (
    BrowserManager,
    _validate_url,
    handle_browser_click,
    handle_browser_evaluate,
    handle_browser_fill,
    handle_browser_read_page,
    handle_browser_read_table,
    handle_browser_screenshot,
)


@pytest.fixture(autouse=True)
def _fake_playwright():
    """Inject a fake playwright module so the import inside _ensure_connected works."""
    fake_api = ModuleType("playwright.async_api")
    fake_api.async_playwright = MagicMock()  # not used when _playwright is already set
    fake_pw = ModuleType("playwright")
    fake_pw.async_api = fake_api

    with patch.dict(sys.modules, {
        "playwright": fake_pw,
        "playwright.async_api": fake_api,
    }):
        yield


# ── URL validation ─────────────────────────────────────────────────


class TestValidateUrl:
    """Test the _validate_url guard function."""

    async def test_http_allowed(self):
        """http:// URLs pass validation."""
        _validate_url("http://example.com")  # Should not raise

    async def test_https_allowed(self):
        """https:// URLs pass validation."""
        _validate_url("https://example.com/page?q=1")  # Should not raise

    async def test_case_insensitive(self):
        """Scheme check is case-insensitive."""
        _validate_url("HTTP://EXAMPLE.COM")
        _validate_url("Https://Example.com")

    async def test_ftp_rejected(self):
        """ftp:// is not an allowed scheme."""
        with pytest.raises(ValueError, match="must start with http"):
            _validate_url("ftp://files.example.com")

    async def test_javascript_rejected(self):
        """javascript: scheme is blocked."""
        with pytest.raises(ValueError, match="must start with http"):
            _validate_url("javascript:alert(1)")

    async def test_file_rejected(self):
        """file:// scheme is blocked."""
        with pytest.raises(ValueError, match="must start with http"):
            _validate_url("file:///etc/passwd")

    async def test_data_rejected(self):
        """data: scheme is blocked."""
        with pytest.raises(ValueError, match="must start with http"):
            _validate_url("data:text/html,<h1>hi</h1>")

    async def test_empty_string_rejected(self):
        """Empty string is rejected."""
        with pytest.raises(ValueError):
            _validate_url("")


# ── BrowserManager init ───────────────────────────────────────────


class TestBrowserManagerInit:
    """Test BrowserManager constructor defaults."""

    async def test_defaults(self):
        """Default values are set correctly."""
        mgr = BrowserManager()
        assert mgr._cdp_url == "ws://loki-browser:3000?token=loki-internal"
        assert mgr._default_timeout_ms == 30000
        assert mgr._viewport == {"width": 1280, "height": 720}
        assert mgr._playwright is None
        assert mgr._browser is None

    async def test_custom_params(self):
        """Custom constructor params are stored."""
        mgr = BrowserManager(
            cdp_url="ws://custom:9000",
            default_timeout_ms=5000,
            viewport_width=800,
            viewport_height=600,
        )
        assert mgr._cdp_url == "ws://custom:9000"
        assert mgr._default_timeout_ms == 5000
        assert mgr._viewport == {"width": 800, "height": 600}


# ── BrowserManager._ensure_connected ──────────────────────────────


class TestEnsureConnected:
    """Test lazy connection and reconnection logic."""

    async def test_connects_on_first_call(self):
        """First call to _ensure_connected starts playwright and connects."""
        mgr = BrowserManager()
        mock_pw_instance = AsyncMock()
        mock_new_browser = MagicMock()
        mock_new_browser.is_connected.return_value = True
        mock_pw_instance.chromium.connect_over_cdp = AsyncMock(return_value=mock_new_browser)

        mock_pw_cm = MagicMock()
        mock_pw_cm.start = AsyncMock(return_value=mock_pw_instance)
        # The import inside _ensure_connected uses the faked module; point it at our mock
        sys.modules["playwright.async_api"].async_playwright = MagicMock(return_value=mock_pw_cm)

        await mgr._ensure_connected()

        assert mgr._playwright is mock_pw_instance
        assert mgr._browser is mock_new_browser

    async def test_skips_if_already_connected(self):
        """If browser is already connected, does nothing."""
        mgr = BrowserManager()
        mock_browser = MagicMock()
        mock_browser.is_connected.return_value = True
        mgr._browser = mock_browser
        mgr._playwright = AsyncMock()

        await mgr._ensure_connected()
        mock_browser.is_connected.assert_called_once()

    async def test_reconnects_if_disconnected(self):
        """If browser connection dropped, reconnects."""
        mgr = BrowserManager()
        mock_old_browser = MagicMock()
        mock_old_browser.is_connected.return_value = False

        mock_pw_instance = AsyncMock()
        mock_new_browser = MagicMock()
        mock_pw_instance.chromium.connect_over_cdp = AsyncMock(return_value=mock_new_browser)
        mgr._browser = mock_old_browser
        mgr._playwright = mock_pw_instance

        await mgr._ensure_connected()
        assert mgr._browser is mock_new_browser

    async def test_raises_on_connect_failure(self):
        """If CDP connection fails, raises RuntimeError."""
        mgr = BrowserManager()
        mock_pw_instance = AsyncMock()
        mock_pw_instance.chromium.connect_over_cdp = AsyncMock(
            side_effect=Exception("Connection refused")
        )
        mgr._playwright = mock_pw_instance
        mgr._browser = MagicMock()
        mgr._browser.is_connected.return_value = False

        with pytest.raises(RuntimeError, match="Browser service unavailable"):
            await mgr._ensure_connected()

    async def test_raises_if_playwright_not_installed(self):
        """If playwright import fails, raises RuntimeError."""
        mgr = BrowserManager()
        mgr._browser = None

        # Override the fake module to raise ImportError
        with patch.dict(sys.modules, {
            "playwright": None,
            "playwright.async_api": None,
        }):
            with pytest.raises(RuntimeError, match="playwright is not installed"):
                await mgr._ensure_connected()


# ── BrowserManager.new_page ───────────────────────────────────────


def _make_connected_manager():
    """Create a BrowserManager with a mocked connected browser."""
    mgr = BrowserManager()
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.title = AsyncMock(return_value="Test Page")
    mock_page.screenshot = AsyncMock(return_value=b"PNG_DATA_HERE")
    mock_page.inner_text = AsyncMock(return_value="Hello world")
    mock_page.wait_for_selector = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.click = AsyncMock()
    mock_page.fill = AsyncMock()
    mock_page.press = AsyncMock()
    mock_page.evaluate = AsyncMock()
    mock_page.url = "https://example.com"

    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    mock_context.set_default_timeout = MagicMock()
    mock_context.close = AsyncMock()

    mock_browser = MagicMock()
    mock_browser.is_connected.return_value = True
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_browser.close = AsyncMock()

    mgr._browser = mock_browser
    mgr._playwright = AsyncMock()

    return mgr, mock_page, mock_context, mock_browser


class TestNewPage:
    """Test the new_page async context manager."""

    async def test_yields_page_and_cleans_up(self):
        """new_page yields a page then closes the context."""
        mgr, mock_page, mock_context, _ = _make_connected_manager()

        async with mgr.new_page() as page:
            assert page is mock_page

        mock_context.close.assert_awaited_once()

    async def test_custom_timeout(self):
        """Custom timeout_ms is passed to context."""
        mgr, _, mock_context, _ = _make_connected_manager()

        async with mgr.new_page(timeout_ms=5000):
            pass

        mock_context.set_default_timeout.assert_called_once_with(5000)

    async def test_default_timeout(self):
        """Default timeout_ms comes from manager."""
        mgr, _, mock_context, _ = _make_connected_manager()
        mgr._default_timeout_ms = 15000

        async with mgr.new_page():
            pass

        mock_context.set_default_timeout.assert_called_once_with(15000)

    async def test_context_closed_even_on_error(self):
        """Context is cleaned up even if the body raises."""
        mgr, _, mock_context, _ = _make_connected_manager()

        with pytest.raises(ValueError):
            async with mgr.new_page():
                raise ValueError("test error")

        mock_context.close.assert_awaited_once()

    async def test_context_close_error_swallowed(self):
        """If context.close() raises, it's silently swallowed."""
        mgr, _, mock_context, _ = _make_connected_manager()
        mock_context.close = AsyncMock(side_effect=Exception("already closed"))

        async with mgr.new_page():
            pass
        # Should not raise

    async def test_viewport_and_user_agent_set(self):
        """Browser context is created with viewport and user agent."""
        mgr, _, _, mock_browser = _make_connected_manager()
        mgr._viewport = {"width": 800, "height": 600}

        async with mgr.new_page():
            pass

        mock_browser.new_context.assert_awaited_once()
        call_kwargs = mock_browser.new_context.call_args[1]
        assert call_kwargs["viewport"] == {"width": 800, "height": 600}
        assert "Mozilla" in call_kwargs["user_agent"]


# ── BrowserManager.shutdown ───────────────────────────────────────


class TestShutdown:
    """Test clean disconnect."""

    async def test_shutdown_closes_browser_and_playwright(self):
        """shutdown() closes both browser and playwright."""
        mgr, _, _, mock_browser = _make_connected_manager()
        mock_pw = mgr._playwright

        await mgr.shutdown()

        mock_browser.close.assert_awaited_once()
        mock_pw.stop.assert_awaited_once()
        assert mgr._browser is None
        assert mgr._playwright is None

    async def test_shutdown_when_nothing_connected(self):
        """shutdown() on fresh manager doesn't raise."""
        mgr = BrowserManager()
        await mgr.shutdown()
        assert mgr._browser is None
        assert mgr._playwright is None

    async def test_shutdown_swallows_browser_close_error(self):
        """If browser.close() raises, shutdown still proceeds."""
        mgr, _, _, mock_browser = _make_connected_manager()
        mock_browser.close = AsyncMock(side_effect=Exception("already dead"))

        await mgr.shutdown()
        assert mgr._browser is None

    async def test_shutdown_swallows_playwright_stop_error(self):
        """If playwright.stop() raises, shutdown still completes."""
        mgr, _, _, _ = _make_connected_manager()
        mgr._playwright.stop = AsyncMock(side_effect=Exception("stop failed"))

        await mgr.shutdown()
        assert mgr._playwright is None


# ── handle_browser_screenshot ─────────────────────────────────────


class TestHandleBrowserScreenshot:
    """Test the screenshot tool handler."""

    async def test_basic_screenshot(self):
        """Takes a screenshot and returns description + bytes."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_page.goto = AsyncMock(return_value=mock_response)
        mock_page.screenshot = AsyncMock(return_value=b"X" * 2048)
        mock_page.title = AsyncMock(return_value="Example")

        text, png = await handle_browser_screenshot(mgr, {"url": "https://example.com"})

        assert "Example" in text
        assert "https://example.com" in text
        assert "200" in text
        assert "2 KB" in text
        assert png == b"X" * 2048

    async def test_full_page_screenshot(self):
        """full_page=True is passed to screenshot call."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
        mock_page.screenshot = AsyncMock(return_value=b"PNG")

        await handle_browser_screenshot(
            mgr, {"url": "https://example.com", "full_page": True}
        )

        mock_page.screenshot.assert_awaited_once_with(full_page=True, type="png")

    async def test_wait_seconds(self):
        """wait_seconds causes a delay before screenshot."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
        mock_page.screenshot = AsyncMock(return_value=b"PNG")

        await handle_browser_screenshot(
            mgr, {"url": "https://example.com", "wait_seconds": 3}
        )

        mock_page.wait_for_timeout.assert_awaited_once_with(3000)

    async def test_wait_seconds_capped_at_10(self):
        """wait_seconds is capped at 10."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
        mock_page.screenshot = AsyncMock(return_value=b"PNG")

        await handle_browser_screenshot(
            mgr, {"url": "https://example.com", "wait_seconds": 99}
        )

        mock_page.wait_for_timeout.assert_awaited_once_with(10000)

    async def test_no_wait_when_zero(self):
        """wait_seconds=0 does not call wait_for_timeout."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
        mock_page.screenshot = AsyncMock(return_value=b"PNG")

        await handle_browser_screenshot(
            mgr, {"url": "https://example.com", "wait_seconds": 0}
        )

        mock_page.wait_for_timeout.assert_not_awaited()

    async def test_none_response_shows_unknown_status(self):
        """If page.goto returns None, status shows 'unknown'."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock(return_value=None)
        mock_page.screenshot = AsyncMock(return_value=b"PNG")

        text, _ = await handle_browser_screenshot(mgr, {"url": "https://example.com"})

        assert "unknown" in text

    async def test_invalid_url_rejected(self):
        """Invalid URL scheme raises ValueError before navigation."""
        mgr, _, _, _ = _make_connected_manager()

        with pytest.raises(ValueError, match="must start with http"):
            await handle_browser_screenshot(mgr, {"url": "ftp://bad.com"})


# ── handle_browser_read_page ──────────────────────────────────────


class TestHandleBrowserReadPage:
    """Test the page-read tool handler."""

    async def test_read_full_body(self):
        """Reads body text when no selector specified."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.inner_text = AsyncMock(return_value="Page content here")
        mock_page.title = AsyncMock(return_value="My Page")

        result = await handle_browser_read_page(
            mgr, {"url": "https://example.com"}
        )

        assert "My Page" in result
        assert "Page content here" in result
        mock_page.inner_text.assert_awaited_once_with("body")

    async def test_read_with_selector(self):
        """Reads text from a specific CSS selector."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_element = AsyncMock()
        mock_element.inner_text = AsyncMock(return_value="Selected text")
        mock_page.wait_for_selector = AsyncMock(return_value=mock_element)
        mock_page.title = AsyncMock(return_value="Title")

        result = await handle_browser_read_page(
            mgr, {"url": "https://example.com", "selector": "#main"}
        )

        assert "Selected text" in result
        mock_page.wait_for_selector.assert_awaited_once_with("#main", timeout=10000)

    async def test_selector_not_found(self):
        """Returns message when CSS selector matches nothing."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_selector = AsyncMock(return_value=None)

        result = await handle_browser_read_page(
            mgr, {"url": "https://example.com", "selector": "#missing"}
        )

        assert "not found" in result
        assert "#missing" in result

    async def test_text_truncation(self):
        """Long text is truncated at max_chars."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.inner_text = AsyncMock(return_value="A" * 10000)
        mock_page.title = AsyncMock(return_value="Big Page")

        result = await handle_browser_read_page(
            mgr, {"url": "https://example.com", "max_chars": 100}
        )

        assert "truncated" in result
        # The text portion should be cut off
        assert len(result) < 10000

    async def test_max_chars_capped_at_8000(self):
        """max_chars is capped at 8000 even if higher requested."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        long_text = "B" * 9000
        mock_page.inner_text = AsyncMock(return_value=long_text)
        mock_page.title = AsyncMock(return_value="Title")

        result = await handle_browser_read_page(
            mgr, {"url": "https://example.com", "max_chars": 50000}
        )

        # Should be truncated at 8000, not 50000
        assert "truncated" in result

    async def test_wait_seconds(self):
        """wait_seconds delays before reading."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.inner_text = AsyncMock(return_value="text")
        mock_page.title = AsyncMock(return_value="T")

        await handle_browser_read_page(
            mgr, {"url": "https://example.com", "wait_seconds": 5}
        )

        mock_page.wait_for_timeout.assert_awaited_once_with(5000)

    async def test_text_stripped(self):
        """Leading/trailing whitespace is stripped."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.inner_text = AsyncMock(return_value="  spaced  \n\n ")
        mock_page.title = AsyncMock(return_value="T")

        result = await handle_browser_read_page(
            mgr, {"url": "https://example.com"}
        )

        assert "spaced" in result
        # Should not have leading/trailing whitespace in the text part
        assert "  spaced" not in result

    async def test_invalid_url_rejected(self):
        """Invalid URL raises ValueError."""
        mgr, _, _, _ = _make_connected_manager()

        with pytest.raises(ValueError):
            await handle_browser_read_page(mgr, {"url": "data:text/html,hi"})


# ── handle_browser_read_table ─────────────────────────────────────


class TestHandleBrowserReadTable:
    """Test the table-extraction tool handler."""

    async def test_basic_table(self):
        """Extracts a table and formats as markdown."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="Table Page")
        table_data = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
        # First evaluate call returns table data, second returns table count
        mock_page.evaluate = AsyncMock(side_effect=[table_data, 2])

        result = await handle_browser_read_table(
            mgr, {"url": "https://example.com"}
        )

        assert "Table Page" in result
        assert "Table 1 of 2" in result
        assert "| Name | Age |" in result
        assert "| --- | --- |" in result
        assert "| Alice | 30 |" in result
        assert "| Bob | 25 |" in result

    async def test_table_not_found(self):
        """Returns message when requested table index doesn't exist."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="No Tables")
        mock_page.evaluate = AsyncMock(side_effect=[None, 0])

        result = await handle_browser_read_table(
            mgr, {"url": "https://example.com", "table_index": 5}
        )

        assert "No table found at index 5" in result
        assert "0 table(s)" in result

    async def test_empty_table(self):
        """Returns message for an empty table."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="T")
        mock_page.evaluate = AsyncMock(side_effect=[[], 1])

        result = await handle_browser_read_table(
            mgr, {"url": "https://example.com"}
        )

        assert "Table is empty" in result

    async def test_table_index_selection(self):
        """table_index parameter selects which table to extract."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="T")
        table_data = [["Col1"], ["Data1"]]
        mock_page.evaluate = AsyncMock(side_effect=[table_data, 3])

        result = await handle_browser_read_table(
            mgr, {"url": "https://example.com", "table_index": 2}
        )

        assert "Table 3 of 3" in result
        # Verify table_index was passed to evaluate
        first_call_args = mock_page.evaluate.call_args_list[0]
        assert first_call_args[0][1] == 2

    async def test_table_truncation(self):
        """Large table markdown is truncated at 4000 chars."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="Big Table")
        # Generate a large table
        big_table = [["Col"]] + [["X" * 200] for _ in range(100)]
        mock_page.evaluate = AsyncMock(side_effect=[big_table, 1])

        result = await handle_browser_read_table(
            mgr, {"url": "https://example.com"}
        )

        assert "truncated" in result

    async def test_wait_seconds(self):
        """wait_seconds delays before extraction."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="T")
        mock_page.evaluate = AsyncMock(side_effect=[[["A"]], 1])

        await handle_browser_read_table(
            mgr, {"url": "https://example.com", "wait_seconds": 2}
        )

        mock_page.wait_for_timeout.assert_awaited_once_with(2000)

    async def test_invalid_url_rejected(self):
        """Invalid URL raises ValueError."""
        mgr, _, _, _ = _make_connected_manager()

        with pytest.raises(ValueError):
            await handle_browser_read_table(mgr, {"url": "javascript:void(0)"})


# ── handle_browser_click ──────────────────────────────────────────


class TestHandleBrowserClick:
    """Test the click tool handler."""

    async def test_successful_click(self):
        """Clicks selector and reports new page state."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.click = AsyncMock()
        mock_page.title = AsyncMock(return_value="New Page")
        mock_page.url = "https://example.com/next"

        result = await handle_browser_click(
            mgr, {"url": "https://example.com", "selector": "button.submit"}
        )

        assert "Clicked" in result
        assert "button.submit" in result
        assert "New Page" in result
        assert "https://example.com/next" in result
        mock_page.click.assert_awaited_once_with("button.submit", timeout=10000)

    async def test_click_failure(self):
        """Reports error when click fails."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.click = AsyncMock(side_effect=Exception("Element not visible"))

        result = await handle_browser_click(
            mgr, {"url": "https://example.com", "selector": "#gone"}
        )

        assert "Failed to click" in result
        assert "#gone" in result
        assert "Element not visible" in result

    async def test_click_waits_after(self):
        """After clicking, waits 1s for navigation."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.click = AsyncMock()
        mock_page.title = AsyncMock(return_value="T")

        await handle_browser_click(
            mgr, {"url": "https://example.com", "selector": "a"}
        )

        # Should wait 1000ms after click
        mock_page.wait_for_timeout.assert_any_call(1000)

    async def test_click_with_wait_seconds(self):
        """wait_seconds delays before clicking."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.click = AsyncMock()
        mock_page.title = AsyncMock(return_value="T")

        await handle_browser_click(
            mgr, {"url": "https://example.com", "selector": "a", "wait_seconds": 3}
        )

        # Should have pre-wait (3000) and post-click wait (1000)
        calls = mock_page.wait_for_timeout.call_args_list
        assert any(c[0][0] == 3000 for c in calls)
        assert any(c[0][0] == 1000 for c in calls)

    async def test_invalid_url_rejected(self):
        """Invalid URL raises ValueError."""
        mgr, _, _, _ = _make_connected_manager()

        with pytest.raises(ValueError):
            await handle_browser_click(
                mgr, {"url": "file:///etc/passwd", "selector": "a"}
            )


# ── handle_browser_fill ───────────────────────────────────────────


class TestHandleBrowserFill:
    """Test the form-fill tool handler."""

    async def test_basic_fill(self):
        """Fills a field and reports success."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.fill = AsyncMock()
        mock_page.title = AsyncMock(return_value="Form Page")
        mock_page.url = "https://example.com/form"

        result = await handle_browser_fill(
            mgr,
            {
                "url": "https://example.com/form",
                "selector": "#email",
                "value": "test@example.com",
            },
        )

        assert "Filled" in result
        assert "#email" in result
        assert "Form Page" in result
        assert "submitted" not in result
        mock_page.fill.assert_awaited_once_with("#email", "test@example.com", timeout=10000)

    async def test_fill_and_submit(self):
        """fill with submit=True presses Enter after."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.fill = AsyncMock()
        mock_page.press = AsyncMock()
        mock_page.title = AsyncMock(return_value="Result")
        mock_page.url = "https://example.com/results"

        result = await handle_browser_fill(
            mgr,
            {
                "url": "https://example.com/form",
                "selector": "#search",
                "value": "query",
                "submit": True,
            },
        )

        assert "submitted" in result
        mock_page.press.assert_awaited_once_with("#search", "Enter")
        mock_page.wait_for_timeout.assert_awaited_once_with(2000)

    async def test_fill_error(self):
        """Reports error when fill fails."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.fill = AsyncMock(side_effect=Exception("Not an input"))

        result = await handle_browser_fill(
            mgr,
            {
                "url": "https://example.com",
                "selector": "#nope",
                "value": "x",
            },
        )

        assert "Failed to fill" in result
        assert "#nope" in result

    async def test_submit_error(self):
        """Reports error when submit fails after successful fill."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.fill = AsyncMock()  # Fill succeeds
        mock_page.press = AsyncMock(side_effect=Exception("Submit blocked"))

        result = await handle_browser_fill(
            mgr,
            {
                "url": "https://example.com",
                "selector": "#input",
                "value": "val",
                "submit": True,
            },
        )

        assert "submit failed" in result
        assert "#input" in result

    async def test_invalid_url_rejected(self):
        """Invalid URL raises ValueError."""
        mgr, _, _, _ = _make_connected_manager()

        with pytest.raises(ValueError):
            await handle_browser_fill(
                mgr,
                {"url": "ftp://bad.com", "selector": "x", "value": "y"},
            )


# ── handle_browser_evaluate ───────────────────────────────────────


class TestHandleBrowserEvaluate:
    """Test the JavaScript evaluation tool handler."""

    async def test_string_result(self):
        """JS returning a string is passed through."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value="hello world")

        result = await handle_browser_evaluate(
            mgr,
            {"url": "https://example.com", "expression": "document.title"},
        )

        assert result == "hello world"

    async def test_dict_result_json(self):
        """JS returning a dict is formatted as JSON."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value={"key": "value", "num": 42})

        result = await handle_browser_evaluate(
            mgr,
            {"url": "https://example.com", "expression": "({key:'value', num:42})"},
        )

        assert '"key": "value"' in result
        assert '"num": 42' in result

    async def test_list_result_json(self):
        """JS returning a list is formatted as JSON."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=[1, 2, 3])

        result = await handle_browser_evaluate(
            mgr,
            {"url": "https://example.com", "expression": "[1,2,3]"},
        )

        assert "[" in result
        assert "1" in result

    async def test_numeric_result(self):
        """JS returning a number is converted to string."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=42)

        result = await handle_browser_evaluate(
            mgr,
            {"url": "https://example.com", "expression": "1+1"},
        )

        assert result == "42"

    async def test_boolean_result(self):
        """JS returning a boolean is converted to string."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=True)

        result = await handle_browser_evaluate(
            mgr,
            {"url": "https://example.com", "expression": "true"},
        )

        assert result == "True"

    async def test_js_error(self):
        """Reports error when JS evaluation fails."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.evaluate = AsyncMock(
            side_effect=Exception("ReferenceError: foo is not defined")
        )

        result = await handle_browser_evaluate(
            mgr,
            {"url": "https://example.com", "expression": "foo.bar()"},
        )

        assert "JavaScript evaluation failed" in result
        assert "ReferenceError" in result

    async def test_result_truncation(self):
        """Long results are truncated at 4000 chars."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value="Z" * 5000)

        result = await handle_browser_evaluate(
            mgr,
            {"url": "https://example.com", "expression": "'Z'.repeat(5000)"},
        )

        assert "truncated" in result
        assert len(result) < 5000

    async def test_wait_seconds(self):
        """wait_seconds delays before evaluation."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value="ok")

        await handle_browser_evaluate(
            mgr,
            {"url": "https://example.com", "expression": "1", "wait_seconds": 4},
        )

        mock_page.wait_for_timeout.assert_awaited_once_with(4000)

    async def test_no_wait_when_zero(self):
        """wait_seconds=0 skips the delay."""
        mgr, mock_page, _, _ = _make_connected_manager()
        mock_page.goto = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value="ok")

        await handle_browser_evaluate(
            mgr,
            {"url": "https://example.com", "expression": "1", "wait_seconds": 0},
        )

        mock_page.wait_for_timeout.assert_not_awaited()

    async def test_invalid_url_rejected(self):
        """Invalid URL raises ValueError."""
        mgr, _, _, _ = _make_connected_manager()

        with pytest.raises(ValueError):
            await handle_browser_evaluate(
                mgr, {"url": "javascript:alert(1)", "expression": "1"}
            )
