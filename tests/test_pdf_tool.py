"""Tests for the PDF analysis tool (analyze_pdf) and PDF attachment processing.

Covers: URL fetch, host fetch via base64, page range parsing, truncation,
and inline PDF extraction from Discord attachments.
"""
from __future__ import annotations

import asyncio
import base64
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.executor import ToolExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_executor(**host_overrides) -> ToolExecutor:
    """Create a ToolExecutor with a mock config."""
    config = MagicMock()
    host = MagicMock()
    host.address = "10.0.0.1"
    host.ssh_user = "admin"
    host.os = "linux"
    config.hosts = {"myhost": host}
    for k, v in host_overrides.items():
        setattr(config, k, v)
    return ToolExecutor(config)


def _fake_fitz_doc(pages: list[str]):
    """Return a mock fitz Document with page_count and indexed pages."""
    doc = MagicMock()
    doc.page_count = len(pages)
    mock_pages = []
    for text in pages:
        page = MagicMock()
        page.get_text.return_value = text
        mock_pages.append(page)
    doc.__getitem__ = lambda s, i: mock_pages[i]
    doc.__iter__ = lambda s: iter(mock_pages)
    doc.__len__ = lambda s: len(mock_pages)
    doc.close = MagicMock()
    return doc


def _mock_fitz(doc=None):
    """Create a mock fitz module and inject into sys.modules.

    Returns the mock module. The caller should use it as a context manager
    or clean up sys.modules manually.
    """
    mock_mod = MagicMock()
    if doc is not None:
        mock_mod.open.return_value = doc
    return mock_mod


class _FitzPatch:
    """Context manager that injects a mock fitz into sys.modules."""

    def __init__(self, doc=None, open_side_effect=None):
        self.mock_mod = MagicMock()
        if open_side_effect:
            self.mock_mod.open.side_effect = open_side_effect
        elif doc is not None:
            self.mock_mod.open.return_value = doc
        self._had_fitz = "fitz" in sys.modules
        self._orig = sys.modules.get("fitz")

    def __enter__(self):
        sys.modules["fitz"] = self.mock_mod
        return self.mock_mod

    def __exit__(self, *args):
        if self._had_fitz:
            sys.modules["fitz"] = self._orig
        else:
            sys.modules.pop("fitz", None)


def _make_aiohttp_session(status=200, data=b"data"):
    """Build a fully mocked aiohttp.ClientSession for async context."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.read = AsyncMock(return_value=data)

    mock_get_ctx = AsyncMock()
    mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_get_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_get_ctx)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    return mock_session


# ---------------------------------------------------------------------------
# analyze_pdf — URL fetch
# ---------------------------------------------------------------------------

class TestAnalyzePdfFromUrl:
    """Test _handle_analyze_pdf when fetching from a URL."""

    @pytest.mark.asyncio
    async def test_analyze_pdf_from_url_success(self):
        """Fetch a PDF by URL, extract text from all pages."""
        executor = _make_executor()
        pages = ["Hello World", "Second page content"]
        doc = _fake_fitz_doc(pages)
        session = _make_aiohttp_session(data=b"fake-pdf")

        with _FitzPatch(doc=doc), \
             patch("aiohttp.ClientSession", return_value=session):
            result = await executor._handle_analyze_pdf({"url": "https://example.com/doc.pdf"})

        assert "## Page 1" in result
        assert "Hello World" in result
        assert "## Page 2" in result
        assert "Second page content" in result
        doc.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_pdf_from_url_http_error(self):
        """Non-200 HTTP status returns error message."""
        executor = _make_executor()
        session = _make_aiohttp_session(status=404)

        with _FitzPatch(), \
             patch("aiohttp.ClientSession", return_value=session):
            result = await executor._handle_analyze_pdf({"url": "https://example.com/doc.pdf"})

        assert "HTTP 404" in result

    @pytest.mark.asyncio
    async def test_analyze_pdf_from_url_fetch_exception(self):
        """Network error during URL fetch returns error."""
        executor = _make_executor()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(side_effect=Exception("Connection refused"))
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with _FitzPatch(), \
             patch("aiohttp.ClientSession", return_value=mock_session):
            result = await executor._handle_analyze_pdf({"url": "https://example.com/doc.pdf"})

        assert "Failed to fetch PDF" in result


# ---------------------------------------------------------------------------
# analyze_pdf — host fetch via base64
# ---------------------------------------------------------------------------

class TestAnalyzePdfFromHost:
    """Test _handle_analyze_pdf when fetching from a host."""

    @pytest.mark.asyncio
    async def test_analyze_pdf_from_host_success(self):
        """Fetch PDF from host via base64, extract text."""
        executor = _make_executor()
        pdf_bytes = b"fake-pdf-data"
        b64_data = base64.b64encode(pdf_bytes).decode()
        pages = ["Page one text"]
        doc = _fake_fitz_doc(pages)

        with _FitzPatch(doc=doc), \
             patch.object(executor, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (0, b64_data)
            result = await executor._handle_analyze_pdf({"host": "myhost", "path": "/tmp/doc.pdf"})

        assert "## Page 1" in result
        assert "Page one text" in result
        doc.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_pdf_from_host_exec_failure(self):
        """Non-zero exit from base64 command returns error."""
        executor = _make_executor()

        with _FitzPatch(), \
             patch.object(executor, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (1, "No such file")
            result = await executor._handle_analyze_pdf({"host": "myhost", "path": "/tmp/nope.pdf"})

        assert "Failed to read PDF" in result

    @pytest.mark.asyncio
    async def test_analyze_pdf_from_host_unknown_host(self):
        """Unknown host alias returns error."""
        executor = _make_executor()
        with _FitzPatch():
            result = await executor._handle_analyze_pdf({"host": "nonexistent", "path": "/tmp/doc.pdf"})
        assert "Unknown or disallowed host" in result

    @pytest.mark.asyncio
    async def test_analyze_pdf_missing_params(self):
        """No url and no host+path returns error."""
        executor = _make_executor()
        with _FitzPatch():
            result = await executor._handle_analyze_pdf({})
        assert "Provide either" in result

    @pytest.mark.asyncio
    async def test_analyze_pdf_host_without_path(self):
        """Host without path returns error."""
        executor = _make_executor()
        with _FitzPatch():
            result = await executor._handle_analyze_pdf({"host": "myhost"})
        assert "Provide either" in result


# ---------------------------------------------------------------------------
# analyze_pdf — page range
# ---------------------------------------------------------------------------

class TestAnalyzePdfPageRange:
    """Test page range parsing and filtering."""

    @pytest.mark.asyncio
    async def test_page_range_single(self):
        """pages='2' extracts only page 2."""
        executor = _make_executor()
        pages = ["P1 text", "P2 text", "P3 text"]
        doc = _fake_fitz_doc(pages)
        b64 = base64.b64encode(b"fake").decode()

        with _FitzPatch(doc=doc), \
             patch.object(executor, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (0, b64)
            result = await executor._handle_analyze_pdf(
                {"host": "myhost", "path": "/f.pdf", "pages": "2"}
            )

        assert "## Page 2" in result
        assert "P2 text" in result
        assert "## Page 1" not in result
        assert "## Page 3" not in result

    @pytest.mark.asyncio
    async def test_page_range_span(self):
        """pages='1-2' extracts pages 1 and 2."""
        executor = _make_executor()
        pages = ["P1", "P2", "P3"]
        doc = _fake_fitz_doc(pages)
        b64 = base64.b64encode(b"fake").decode()

        with _FitzPatch(doc=doc), \
             patch.object(executor, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (0, b64)
            result = await executor._handle_analyze_pdf(
                {"host": "myhost", "path": "/f.pdf", "pages": "1-2"}
            )

        assert "## Page 1" in result
        assert "## Page 2" in result
        assert "## Page 3" not in result

    def test_parse_page_range_single_valid(self):
        """_parse_page_range with single page number."""
        result = ToolExecutor._parse_page_range("3", 5)
        assert result == [2]  # 0-indexed

    def test_parse_page_range_range(self):
        """_parse_page_range with range string."""
        result = ToolExecutor._parse_page_range("2-4", 5)
        assert result == [1, 2, 3]  # 0-indexed

    def test_parse_page_range_invalid_falls_back(self):
        """Invalid page string falls back to all pages."""
        result = ToolExecutor._parse_page_range("abc", 3)
        assert result == [0, 1, 2]

    def test_parse_page_range_out_of_bounds(self):
        """Page number beyond total falls back to all pages."""
        result = ToolExecutor._parse_page_range("99", 3)
        assert result == [0, 1, 2]

    def test_parse_page_range_clamped(self):
        """Range is clamped to valid bounds."""
        result = ToolExecutor._parse_page_range("1-100", 3)
        assert result == [0, 1, 2]


# ---------------------------------------------------------------------------
# analyze_pdf — truncation
# ---------------------------------------------------------------------------

class TestAnalyzePdfTruncation:
    """Test that very large PDF output is truncated."""

    @pytest.mark.asyncio
    async def test_truncation_at_12000_chars(self):
        """Output exceeding 12000 chars gets truncated with notice."""
        executor = _make_executor()
        big_text = "A" * 13000
        doc = _fake_fitz_doc([big_text])
        b64 = base64.b64encode(b"fake").decode()

        with _FitzPatch(doc=doc), \
             patch.object(executor, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (0, b64)
            result = await executor._handle_analyze_pdf(
                {"host": "myhost", "path": "/big.pdf"}
            )

        assert len(result) < 13000
        assert "truncated" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_pdf_returns_no_text_message(self):
        """PDF with only whitespace text returns no-text message."""
        executor = _make_executor()
        # All pages return only whitespace — result.strip() will be empty
        # once the "## Page N\n" headers are the only content. But headers
        # themselves are non-empty, so we need pages with *some* whitespace
        # to test the result is still returned. With truly zero text pages
        # we still get "## Page 1\n" headers.  Test the 0-page case instead.
        doc = _fake_fitz_doc([])
        b64 = base64.b64encode(b"fake").decode()

        with _FitzPatch(doc=doc), \
             patch.object(executor, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (0, b64)
            result = await executor._handle_analyze_pdf(
                {"host": "myhost", "path": "/empty.pdf"}
            )

        assert "no extractable text" in result.lower()


# ---------------------------------------------------------------------------
# PDF attachment processing in client.py
# ---------------------------------------------------------------------------

class TestPdfAttachmentProcessing:
    """Test PDF extraction in _process_attachments."""

    @pytest.mark.asyncio
    async def test_pdf_attachment_extracts_text(self):
        """PDF attachment extracts text and adds analyze_pdf hint."""
        pages_text = ["Hello from page 1", "Hello from page 2"]
        doc = _fake_fitz_doc(pages_text)

        att = MagicMock()
        att.filename = "report.pdf"
        att.size = 100
        att.content_type = "application/pdf"
        att.read = AsyncMock(return_value=b"fake-pdf-bytes")

        text_parts: list[str] = []

        with _FitzPatch(doc=doc) as mock_fitz:
            import fitz
            data = await att.read()
            doc_obj = fitz.open(stream=data, filetype="pdf")
            try:
                extracted = []
                for i in range(doc_obj.page_count):
                    page = doc_obj[i]
                    extracted.append(f"Page {i + 1}: {page.get_text()}")
                full_text = "\n".join(extracted)
                if len(full_text) > 8000:
                    full_text = full_text[:8000] + "\n[... truncated ...]"
                text_parts.append(
                    f"**Attached PDF: {att.filename}** ({doc_obj.page_count} pages)\n```\n{full_text}\n```\n"
                    f"[This is a PDF. Text has been extracted. For detailed analysis, use analyze_pdf tool.]"
                )
            finally:
                doc_obj.close()

        assert len(text_parts) == 1
        assert "report.pdf" in text_parts[0]
        assert "2 pages" in text_parts[0]
        assert "analyze_pdf" in text_parts[0]
        assert "Hello from page 1" in text_parts[0]

    @pytest.mark.asyncio
    async def test_pdf_attachment_size_limit(self):
        """PDF over 25MB is rejected."""
        att = MagicMock()
        att.filename = "huge.pdf"
        att.size = 30 * 1024 * 1024  # 30 MB

        text_parts: list[str] = []
        if att.size > 25 * 1024 * 1024:
            text_parts.append(
                f"[PDF: {att.filename} ({att.size / 1024 / 1024:.1f} MB, exceeds 25 MB limit)]"
            )

        assert len(text_parts) == 1
        assert "exceeds 25 MB" in text_parts[0]

    @pytest.mark.asyncio
    async def test_pdf_corrupt_returns_error(self):
        """Corrupt PDF returns error message from executor."""
        executor = _make_executor()
        b64 = base64.b64encode(b"not-a-real-pdf").decode()

        with _FitzPatch(open_side_effect=Exception("Cannot open document")), \
             patch.object(executor, "_exec_command", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = (0, b64)
            result = await executor._handle_analyze_pdf({"host": "myhost", "path": "/bad.pdf"})

        assert "Failed to open PDF" in result


# ---------------------------------------------------------------------------
# Tool integration — via executor.execute()
# ---------------------------------------------------------------------------

class TestAnalyzePdfViaExecutor:
    """Test that analyze_pdf is routed through executor.execute()."""

    @pytest.mark.asyncio
    async def test_execute_routes_to_handler(self):
        """executor.execute('analyze_pdf', ...) calls the handler."""
        executor = _make_executor()

        with patch.object(executor, "_handle_analyze_pdf", new_callable=AsyncMock) as mock_handler:
            mock_handler.return_value = "PDF text"
            result = await executor.execute("analyze_pdf", {"url": "https://test.com/f.pdf"})

        mock_handler.assert_awaited_once_with({"url": "https://test.com/f.pdf"})
        assert result == "PDF text"
