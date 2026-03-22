"""Tests for web.py coverage gaps.

Targets uncovered lines: 95-99, 116, 121-125, 155-158.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.tools.web import fetch_url, web_search, _parse_ddg_results


def _make_session_that_raises_on_get(error):
    """Create a mock aiohttp session where .get() raises the given error."""
    mock_session = MagicMock()

    # The session is used as `async with ClientSession() as session:`
    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    # When session.get() is called, it raises the error
    mock_session.get = MagicMock(side_effect=error)

    return mock_session_ctx


class TestFetchUrlErrors:
    @pytest.mark.asyncio
    async def test_fetch_client_error(self):
        """aiohttp.ClientError returns error message (lines 95-96)."""
        mock_ctx = _make_session_that_raises_on_get(
            aiohttp.ClientError("connection failed")
        )
        with patch("src.tools.web.aiohttp.ClientSession", return_value=mock_ctx):
            result = await fetch_url("https://example.com")
        assert "Fetch error" in result

    @pytest.mark.asyncio
    async def test_fetch_general_exception(self):
        """General exception returns error message (lines 97-99)."""
        mock_ctx = _make_session_that_raises_on_get(
            RuntimeError("unexpected")
        )
        with patch("src.tools.web.aiohttp.ClientSession", return_value=mock_ctx):
            result = await fetch_url("https://example.com")
        assert "Error" in result


class TestWebSearchErrors:
    @pytest.mark.asyncio
    async def test_search_non_200(self):
        """Non-200 response returns error (line 116)."""
        mock_resp = AsyncMock()
        mock_resp.status = 503

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(),
        ))

        with patch("src.tools.web.aiohttp.ClientSession", return_value=mock_session):
            result = await web_search("test query")
        assert "503" in result

    @pytest.mark.asyncio
    async def test_search_client_error(self):
        """aiohttp.ClientError returns error message (lines 121-122)."""
        mock_ctx = _make_session_that_raises_on_get(
            aiohttp.ClientError("timeout")
        )
        with patch("src.tools.web.aiohttp.ClientSession", return_value=mock_ctx):
            result = await web_search("test query")
        assert "Search error" in result

    @pytest.mark.asyncio
    async def test_search_general_exception(self):
        """General exception returns error message (lines 123-125)."""
        mock_ctx = _make_session_that_raises_on_get(
            RuntimeError("unexpected")
        )
        with patch("src.tools.web.aiohttp.ClientSession", return_value=mock_ctx):
            result = await web_search("test query")
        assert "Error" in result


class TestParseDdgRedirectUrl:
    def test_ddg_redirect_url_extraction(self):
        """DuckDuckGo uddg= redirect URLs are extracted (lines 155-158)."""
        html = '''
        <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc">Example Page</a>
        <span class="result__snippet">A test page</span>
        '''
        result = _parse_ddg_results(html, 5)
        assert "example.com/page" in result
        # Should NOT contain the DDG redirect wrapper
        assert "uddg=" not in result

    def test_ddg_non_redirect_url_kept(self):
        """Direct URLs without uddg= are kept as-is (lines 155-158 not triggered)."""
        html = '''
        <a class="result__a" href="https://example.com/direct">Direct Link</a>
        <span class="result__snippet">Direct snippet</span>
        '''
        result = _parse_ddg_results(html, 5)
        assert "https://example.com/direct" in result
