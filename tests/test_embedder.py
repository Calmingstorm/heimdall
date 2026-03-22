"""Tests for OllamaEmbedder (embedder.py).

Mocks aiohttp.ClientSession to avoid real Ollama calls. Tests verify:
- Successful embedding returns vector
- Input truncation to MAX_INPUT_CHARS
- HTTP error handling (non-200 status)
- Network/exception handling (returns None)
- Empty/malformed response handling
- URL construction and model parameter
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.search.embedder import OllamaEmbedder, MAX_INPUT_CHARS


def _make_mock_response(status: int = 200, json_data: dict | None = None):
    """Create a mock aiohttp response with given status and JSON body."""
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    return resp


def _patch_session(response):
    """Build mocks for the double async-context-manager aiohttp pattern.

    Returns (patcher, inner_session_mock) where inner_session_mock.post
    captures the call args.

    Usage::

        async with aiohttp.ClientSession() as session:   # outer CM
            async with session.post(...) as resp:          # inner CM
    """
    # Inner CM: session.post(...) -> async context manager yielding response
    post_cm = AsyncMock()
    post_cm.__aenter__.return_value = response

    # The session that lives inside the outer `async with`
    inner_session = MagicMock()
    inner_session.post.return_value = post_cm

    # Outer CM: aiohttp.ClientSession() -> async context manager yielding session
    outer_cm = AsyncMock()
    outer_cm.__aenter__.return_value = inner_session

    patcher = patch("aiohttp.ClientSession", return_value=outer_cm)
    return patcher, inner_session


def _patch_session_error(exc):
    """Build mocks where session.post() raises an exception."""
    inner_session = MagicMock()
    inner_session.post.side_effect = exc

    outer_cm = AsyncMock()
    outer_cm.__aenter__.return_value = inner_session

    patcher = patch("aiohttp.ClientSession", return_value=outer_cm)
    return patcher


# ── Successful embedding ────────────────────────────────────────────


class TestEmbedSuccess:
    async def test_returns_embedding_vector(self):
        """Successful API call returns the first embedding vector."""
        embedding = [0.1] * 768
        resp = _make_mock_response(200, {"embeddings": [embedding]})
        patcher, _ = _patch_session(resp)

        with patcher:
            embedder = OllamaEmbedder("http://localhost:11434")
            result = await embedder.embed("test text")

        assert result == embedding

    async def test_passes_correct_url_and_payload(self):
        """API call uses correct URL, model, and input text."""
        embedding = [0.5] * 768
        resp = _make_mock_response(200, {"embeddings": [embedding]})
        patcher, inner = _patch_session(resp)

        with patcher:
            embedder = OllamaEmbedder("http://localhost:11434", model="nomic-embed-text")
            await embedder.embed("hello world")

        call_args = inner.post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/embed"
        assert call_args[1]["json"]["model"] == "nomic-embed-text"
        assert call_args[1]["json"]["input"] == "hello world"

    async def test_strips_trailing_slash_from_base_url(self):
        """Trailing slash in base_url is stripped before building endpoint URL."""
        embedding = [0.1] * 768
        resp = _make_mock_response(200, {"embeddings": [embedding]})
        patcher, inner = _patch_session(resp)

        with patcher:
            embedder = OllamaEmbedder("http://localhost:11434/")
            await embedder.embed("test")

        call_url = inner.post.call_args[0][0]
        assert call_url == "http://localhost:11434/api/embed"
        assert "//" not in call_url.split("://")[1]


# ── Input truncation ───────────────────────────────────────────────


class TestInputTruncation:
    async def test_long_input_truncated(self):
        """Input longer than MAX_INPUT_CHARS is truncated before sending."""
        embedding = [0.2] * 768
        resp = _make_mock_response(200, {"embeddings": [embedding]})
        patcher, inner = _patch_session(resp)

        long_text = "x" * (MAX_INPUT_CHARS + 5000)

        with patcher:
            embedder = OllamaEmbedder("http://localhost:11434")
            result = await embedder.embed(long_text)

        sent_text = inner.post.call_args[1]["json"]["input"]
        assert len(sent_text) == MAX_INPUT_CHARS
        assert result == embedding

    async def test_short_input_not_truncated(self):
        """Input shorter than MAX_INPUT_CHARS is sent as-is."""
        embedding = [0.3] * 768
        resp = _make_mock_response(200, {"embeddings": [embedding]})
        patcher, inner = _patch_session(resp)

        with patcher:
            embedder = OllamaEmbedder("http://localhost:11434")
            await embedder.embed("short text")

        sent_text = inner.post.call_args[1]["json"]["input"]
        assert sent_text == "short text"


# ── Error handling ──────────────────────────────────────────────────


class TestErrorHandling:
    async def test_non_200_returns_none(self):
        """Non-200 HTTP status returns None."""
        resp = _make_mock_response(500, {})
        patcher, _ = _patch_session(resp)

        with patcher:
            embedder = OllamaEmbedder("http://localhost:11434")
            result = await embedder.embed("test")

        assert result is None

    async def test_404_returns_none(self):
        """404 response (model not found) returns None."""
        resp = _make_mock_response(404, {"error": "model not found"})
        patcher, _ = _patch_session(resp)

        with patcher:
            embedder = OllamaEmbedder("http://localhost:11434")
            result = await embedder.embed("test")

        assert result is None

    async def test_connection_error_returns_none(self):
        """Network connection error returns None."""
        patcher = _patch_session_error(ConnectionError("Connection refused"))

        with patcher:
            embedder = OllamaEmbedder("http://localhost:11434")
            result = await embedder.embed("test")

        assert result is None

    async def test_timeout_returns_none(self):
        """Timeout returns None."""
        import asyncio
        patcher = _patch_session_error(asyncio.TimeoutError())

        with patcher:
            embedder = OllamaEmbedder("http://localhost:11434")
            result = await embedder.embed("test")

        assert result is None

    async def test_generic_exception_returns_none(self):
        """Any unexpected exception returns None."""
        patcher = _patch_session_error(RuntimeError("unexpected"))

        with patcher:
            embedder = OllamaEmbedder("http://localhost:11434")
            result = await embedder.embed("test")

        assert result is None


# ── Malformed response handling ─────────────────────────────────────


class TestMalformedResponses:
    async def test_empty_embeddings_list(self):
        """Empty embeddings list returns None."""
        resp = _make_mock_response(200, {"embeddings": []})
        patcher, _ = _patch_session(resp)

        with patcher:
            embedder = OllamaEmbedder("http://localhost:11434")
            result = await embedder.embed("test")

        assert result is None

    async def test_missing_embeddings_key(self):
        """Response without 'embeddings' key returns None."""
        resp = _make_mock_response(200, {"result": "something"})
        patcher, _ = _patch_session(resp)

        with patcher:
            embedder = OllamaEmbedder("http://localhost:11434")
            result = await embedder.embed("test")

        assert result is None

    async def test_none_embeddings_value(self):
        """Response with embeddings=None returns None."""
        resp = _make_mock_response(200, {"embeddings": None})
        patcher, _ = _patch_session(resp)

        with patcher:
            embedder = OllamaEmbedder("http://localhost:11434")
            result = await embedder.embed("test")

        assert result is None


# ── Custom model ────────────────────────────────────────────────────


class TestCustomModel:
    async def test_custom_model_name(self):
        """Custom model name is passed in the request payload."""
        embedding = [0.4] * 384
        resp = _make_mock_response(200, {"embeddings": [embedding]})
        patcher, inner = _patch_session(resp)

        with patcher:
            embedder = OllamaEmbedder("http://localhost:11434", model="all-minilm")
            result = await embedder.embed("test")

        assert inner.post.call_args[1]["json"]["model"] == "all-minilm"
        assert result == embedding
