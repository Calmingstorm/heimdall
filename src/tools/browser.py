"""Browser automation via Playwright connecting to a Browserless Chromium sidecar.

Provides headless browser capabilities: screenshots, page reading, table extraction,
clicking, form filling, and JavaScript evaluation. All operations use isolated
browser contexts (incognito) that are cleaned up after each call.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from ..logging import get_logger

log = get_logger("browser")

ALLOWED_SCHEMES = ("http://", "https://")
_CONNECTION_ERROR_PATTERNS = (
    "connection closed",
    "target closed",
    "browser has been closed",
    "browser closed",
    "websocket is closed",
    "not connected",
    "connection refused",
    "target page, context or browser has been closed",
)
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _validate_url(url: str) -> None:
    """Reject dangerous URL schemes."""
    if not any(url.lower().startswith(s) for s in ALLOWED_SCHEMES):
        raise ValueError(
            f"URL must start with http:// or https:// (got: {url[:50]})"
        )


class BrowserManager:
    """Manages Playwright connection to a Browserless Chromium container via CDP."""

    def __init__(
        self,
        cdp_url: str = "ws://heimdall-browser:3000?token=heimdall-internal",
        default_timeout_ms: int = 30000,
        viewport_width: int = 1280,
        viewport_height: int = 720,
    ) -> None:
        self._cdp_url = cdp_url
        self._default_timeout_ms = default_timeout_ms
        self._viewport = {"width": viewport_width, "height": viewport_height}
        self._playwright = None
        self._browser = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """Check if an exception indicates a dead browser connection."""
        msg = str(exc).lower()
        return any(p in msg for p in _CONNECTION_ERROR_PATTERNS)

    def _on_browser_disconnected(self) -> None:
        """Callback when the browser fires a 'disconnected' event."""
        log.warning("Browser disconnected (container may have restarted)")
        self._browser = None

    async def _force_reconnect(self) -> None:
        """Force-drop the current connection and reconnect."""
        async with self._lock:
            try:
                if self._browser:
                    await self._browser.close()
            except Exception:
                pass
            self._browser = None
        await self._ensure_connected()

    async def _ensure_connected(self) -> None:
        """Lazy-connect to the browser, reconnecting if the connection dropped."""
        async with self._lock:
            if self._browser and self._browser.is_connected():
                return
            try:
                from playwright.async_api import async_playwright
            except ImportError:
                raise RuntimeError(
                    "playwright is not installed. "
                    "Add 'playwright' to pyproject.toml dependencies."
                )
            if not self._playwright:
                self._playwright = await async_playwright().start()
            try:
                self._browser = await self._playwright.chromium.connect_over_cdp(
                    self._cdp_url
                )
                self._browser.on("disconnected", self._on_browser_disconnected)
                log.info("Connected to browser at %s", self._cdp_url.split("?")[0])
            except Exception as e:
                raise RuntimeError(
                    f"Browser service unavailable. Is the heimdall-browser container running? ({e})"
                )

    async def _create_page(self, timeout_ms: int | None = None):
        """Create a new browser context and page. Returns (context, page)."""
        context = await self._browser.new_context(
            viewport=self._viewport,
            user_agent=DEFAULT_USER_AGENT,
        )
        context.set_default_timeout(timeout_ms or self._default_timeout_ms)
        page = await context.new_page()
        # CDP connections may ignore context viewport — force it via CDP protocol
        cdp = await page.context.new_cdp_session(page)
        await cdp.send("Emulation.setDeviceMetricsOverride", {
            "width": self._viewport["width"],
            "height": self._viewport["height"],
            "deviceScaleFactor": 1,
            "mobile": False,
        })
        return context, page

    @asynccontextmanager
    async def new_page(self, timeout_ms: int | None = None) -> AsyncIterator:
        """Yield a fresh page in an isolated context. Auto-cleans up.

        Self-heals stale CDP connections: if the browser container restarted,
        the first page creation attempt will fail. We catch connection errors,
        force a reconnect, and retry once.
        """
        await self._ensure_connected()
        try:
            context, page = await self._create_page(timeout_ms)
        except Exception as e:
            if self._is_connection_error(e):
                log.warning("Stale CDP connection, reconnecting: %s", e)
                await self._force_reconnect()
                context, page = await self._create_page(timeout_ms)
            else:
                raise
        try:
            yield page
        finally:
            try:
                await context.close()
            except Exception:
                pass  # Context may already be closed if browser disconnected

    async def shutdown(self) -> None:
        """Clean disconnect from the browser."""
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        self._browser = None
        self._playwright = None
        log.info("Browser manager shut down")


# --- Tool handler functions ---
# Each returns a string (tool result) or a tuple of (string, bytes) for screenshot.


async def handle_browser_screenshot(
    manager: BrowserManager, inp: dict,
) -> tuple[str, bytes | None]:
    """Navigate to a URL, take a screenshot, return (description, png_bytes)."""
    url = inp["url"]
    full_page = inp.get("full_page", False)
    wait_seconds = min(inp.get("wait_seconds", 0), 10)

    _validate_url(url)

    async with manager.new_page() as page:
        response = await page.goto(url, wait_until="domcontentloaded")
        if wait_seconds:
            await page.wait_for_timeout(wait_seconds * 1000)
        screenshot_bytes = await page.screenshot(full_page=full_page, type="png")
        title = await page.title()
        status = response.status if response else "unknown"

    size_kb = len(screenshot_bytes) // 1024
    text = f"Screenshot of **{title}** ({url}) — HTTP {status}, {size_kb} KB"
    return text, screenshot_bytes


async def handle_browser_read_page(
    manager: BrowserManager, inp: dict,
) -> str:
    """Navigate to a URL, extract visible text content."""
    url = inp["url"]
    selector = inp.get("selector")
    max_chars = min(inp.get("max_chars", 4000), 8000)
    wait_seconds = min(inp.get("wait_seconds", 0), 10)

    _validate_url(url)

    async with manager.new_page() as page:
        await page.goto(url, wait_until="domcontentloaded")
        if wait_seconds:
            await page.wait_for_timeout(wait_seconds * 1000)

        if selector:
            element = await page.wait_for_selector(selector, timeout=10000)
            if not element:
                return f"Selector `{selector}` not found on page."
            text = await element.inner_text()
        else:
            text = await page.inner_text("body")
        title = await page.title()

    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n... (content truncated)"

    return f"**{title}** ({url})\n\n{text}"


async def handle_browser_read_table(
    manager: BrowserManager, inp: dict,
) -> str:
    """Navigate to a URL, extract a table as markdown."""
    url = inp["url"]
    table_index = inp.get("table_index", 0)
    wait_seconds = min(inp.get("wait_seconds", 0), 10)

    _validate_url(url)

    async with manager.new_page() as page:
        await page.goto(url, wait_until="domcontentloaded")
        if wait_seconds:
            await page.wait_for_timeout(wait_seconds * 1000)

        table_data = await page.evaluate("""(index) => {
            const tables = document.querySelectorAll('table');
            if (index >= tables.length) return null;
            const table = tables[index];
            const rows = [];
            for (const tr of table.querySelectorAll('tr')) {
                const cells = [];
                for (const td of tr.querySelectorAll('th, td')) {
                    cells.push(td.innerText.trim());
                }
                if (cells.length > 0) rows.push(cells);
            }
            return rows;
        }""", table_index)

        title = await page.title()
        table_count = await page.evaluate("document.querySelectorAll('table').length")

    if table_data is None:
        return f"No table found at index {table_index}. Page has {table_count} table(s)."

    if not table_data:
        return "Table is empty."

    # Format as markdown table
    lines = []
    for i, row in enumerate(table_data):
        line = "| " + " | ".join(str(cell) for cell in row) + " |"
        lines.append(line)
        if i == 0:
            # Add header separator
            lines.append("| " + " | ".join("---" for _ in row) + " |")

    md = "\n".join(lines)
    if len(md) > 4000:
        md = md[:4000] + "\n... (table truncated)"

    return f"**{title}** — Table {table_index + 1} of {table_count}\n\n{md}"


async def handle_browser_click(
    manager: BrowserManager, inp: dict,
) -> str:
    """Click an element on a page by CSS selector."""
    url = inp["url"]
    selector = inp["selector"]
    wait_seconds = min(inp.get("wait_seconds", 0), 10)

    _validate_url(url)

    async with manager.new_page() as page:
        await page.goto(url, wait_until="domcontentloaded")
        if wait_seconds:
            await page.wait_for_timeout(wait_seconds * 1000)

        try:
            await page.click(selector, timeout=10000)
        except Exception as e:
            return f"Failed to click `{selector}`: {e}"

        # Wait for any navigation or rendering after click
        await page.wait_for_timeout(1000)
        new_url = page.url
        title = await page.title()

    return f"Clicked `{selector}`. Page is now: **{title}** ({new_url})"


async def handle_browser_fill(
    manager: BrowserManager, inp: dict,
) -> str:
    """Fill a form field on a page by CSS selector."""
    url = inp["url"]
    selector = inp["selector"]
    value = inp["value"]
    submit = inp.get("submit", False)

    _validate_url(url)

    async with manager.new_page() as page:
        await page.goto(url, wait_until="domcontentloaded")

        try:
            await page.fill(selector, value, timeout=10000)
        except Exception as e:
            return f"Failed to fill `{selector}`: {e}"

        if submit:
            try:
                await page.press(selector, "Enter")
                await page.wait_for_timeout(2000)
            except Exception as e:
                return f"Filled `{selector}` but submit failed: {e}"

        title = await page.title()
        new_url = page.url

    result = f"Filled `{selector}` with value. Page: **{title}** ({new_url})"
    if submit:
        result += " (submitted)"
    return result


async def handle_browser_evaluate(
    manager: BrowserManager, inp: dict,
) -> str:
    """Run JavaScript on a page and return the result."""
    url = inp["url"]
    expression = inp["expression"]
    wait_seconds = min(inp.get("wait_seconds", 0), 10)

    _validate_url(url)

    async with manager.new_page() as page:
        await page.goto(url, wait_until="domcontentloaded")
        if wait_seconds:
            await page.wait_for_timeout(wait_seconds * 1000)

        try:
            result = await page.evaluate(expression)
        except Exception as e:
            return f"JavaScript evaluation failed: {e}"

    # Convert result to string
    if isinstance(result, (dict, list)):
        import json
        text = json.dumps(result, indent=2, default=str)
    else:
        text = str(result)

    if len(text) > 4000:
        text = text[:4000] + "\n... (result truncated)"

    return text
