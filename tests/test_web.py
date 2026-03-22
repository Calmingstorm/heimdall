"""Tests for tools/web.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pytest

from src.tools.web import fetch_url, web_search, _html_to_text, _parse_ddg_results


class TestHtmlToText:
    def test_basic_html(self):
        html = "<html><body><p>Hello</p><p>World</p></body></html>"
        text = _html_to_text(html)
        assert "Hello" in text
        assert "World" in text

    def test_strips_script_style(self):
        html = "<html><script>var x = 1;</script><style>.a{}</style><p>Content</p></html>"
        text = _html_to_text(html)
        assert "var x" not in text
        assert ".a" not in text
        assert "Content" in text

    def test_strips_nav_footer(self):
        html = "<nav>Menu</nav><p>Main content</p><footer>Footer</footer>"
        text = _html_to_text(html)
        assert "Menu" not in text
        assert "Footer" not in text
        assert "Main content" in text

    def test_newlines_for_block_elements(self):
        html = "<h1>Title</h1><p>Para 1</p><p>Para 2</p>"
        text = _html_to_text(html)
        assert "\n" in text

    def test_nested_skip_tags(self):
        """Nested skip tags should keep skipping until all are closed."""
        html = "<nav>menu<footer>deep</footer>still in nav</nav><p>visible</p>"
        text = _html_to_text(html)
        assert "menu" not in text
        assert "deep" not in text
        assert "still in nav" not in text
        assert "visible" in text

    def test_adjacent_skip_tags(self):
        """Adjacent (non-nested) skip tags should each skip independently."""
        html = "<script>js code</script><p>between</p><style>css code</style><p>after</p>"
        text = _html_to_text(html)
        assert "js code" not in text
        assert "css code" not in text
        assert "between" in text
        assert "after" in text

    def test_deeply_nested_skip_tags(self):
        """Multiple levels of nesting should all be skipped."""
        html = "<nav><header><noscript>deep</noscript>mid</header>outer</nav><p>content</p>"
        text = _html_to_text(html)
        assert "deep" not in text
        assert "mid" not in text
        assert "outer" not in text
        assert "content" in text


class TestParseDdgResults:
    def test_parses_results(self):
        html = '''
        <div class="result">
            <a class="result__a" href="https://example.com">Example Title</a>
            <span class="result__snippet">This is a snippet</span>
        </div>
        <div class="result">
            <a class="result__a" href="https://other.com">Other Title</a>
            <span class="result__snippet">Another snippet</span>
        </div>
        '''
        result = _parse_ddg_results(html, 5)
        assert "Example Title" in result
        assert "Other Title" in result
        assert "This is a snippet" in result

    def test_respects_max_results(self):
        html = '''
        <a class="result__a" href="https://a.com">A</a>
        <span class="result__snippet">s1</span>
        <a class="result__a" href="https://b.com">B</a>
        <span class="result__snippet">s2</span>
        <a class="result__a" href="https://c.com">C</a>
        <span class="result__snippet">s3</span>
        '''
        result = _parse_ddg_results(html, 2)
        assert "A" in result
        assert "B" in result
        assert "C" not in result

    def test_no_results(self):
        result = _parse_ddg_results("<html></html>", 5)
        assert "No results" in result


class TestFetchUrl:
    @pytest.mark.asyncio
    async def test_fetch_html(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = AsyncMock(return_value="<html><body><p>Hello World</p></body></html>")

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(),
        ))

        with patch("src.tools.web.aiohttp.ClientSession", return_value=mock_session):
            result = await fetch_url("https://example.com")

        assert "Hello World" in result

    @pytest.mark.asyncio
    async def test_fetch_json(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.text = AsyncMock(return_value='{"key": "value"}')

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(),
        ))

        with patch("src.tools.web.aiohttp.ClientSession", return_value=mock_session):
            result = await fetch_url("https://api.example.com/data")

        assert '"key": "value"' in result

    @pytest.mark.asyncio
    async def test_fetch_truncation(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Type": "text/plain"}
        mock_resp.text = AsyncMock(return_value="x" * 10000)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(),
        ))

        with patch("src.tools.web.aiohttp.ClientSession", return_value=mock_session):
            result = await fetch_url("https://example.com/big")

        assert "truncated" in result

    @pytest.mark.asyncio
    async def test_fetch_error_status(self):
        mock_resp = AsyncMock()
        mock_resp.status = 404
        mock_resp.reason = "Not Found"

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(),
        ))

        with patch("src.tools.web.aiohttp.ClientSession", return_value=mock_session):
            result = await fetch_url("https://example.com/missing")

        assert "404" in result


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        ddg_html = '''
        <a class="result__a" href="https://example.com">Test Result</a>
        <span class="result__snippet">A test snippet about the query</span>
        '''
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.text = AsyncMock(return_value=ddg_html)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(),
        ))

        with patch("src.tools.web.aiohttp.ClientSession", return_value=mock_session):
            result = await web_search("test query")

        assert "Test Result" in result
