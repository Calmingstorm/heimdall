"""Tests for LocalEmbedder (src/search/embedder.py).

Covers: embedding dimensions, import failure handling, text truncation,
lazy model initialization.
Mocks fastembed to avoid real model downloads.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.search.embedder import LocalEmbedder, MAX_INPUT_CHARS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_model(dim: int = 384, value: float = 0.1):
    """Create a mock fastembed model that returns a fixed vector."""
    model = MagicMock()
    model.embed.return_value = iter([np.array([value] * dim, dtype=np.float32)])
    return model


# ---------------------------------------------------------------------------
# Embedding output
# ---------------------------------------------------------------------------


class TestEmbedReturns384DimList:
    """embed() returns a 384-dimension float list."""

    async def test_returns_384_dim_list(self):
        model = _fake_model()
        with patch.object(LocalEmbedder, "_ensure_model"):
            embedder = LocalEmbedder()
            embedder._model = model
            result = await embedder.embed("test input")

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 384

    async def test_returns_floats(self):
        model = _fake_model()
        with patch.object(LocalEmbedder, "_ensure_model"):
            embedder = LocalEmbedder()
            embedder._model = model
            result = await embedder.embed("test")

        assert all(isinstance(v, float) for v in result)

    async def test_returns_correct_values(self):
        expected_val = 0.42
        model = _fake_model(value=expected_val)
        with patch.object(LocalEmbedder, "_ensure_model"):
            embedder = LocalEmbedder()
            embedder._model = model
            result = await embedder.embed("hello")

        assert result is not None
        for v in result:
            assert abs(v - expected_val) < 1e-5

    async def test_dimensions_constant_matches(self):
        """DIMENSIONS constant matches the actual output length."""
        model = _fake_model(dim=LocalEmbedder.DIMENSIONS)
        with patch.object(LocalEmbedder, "_ensure_model"):
            embedder = LocalEmbedder()
            embedder._model = model
            result = await embedder.embed("test")

        assert len(result) == LocalEmbedder.DIMENSIONS


# ---------------------------------------------------------------------------
# Import failure
# ---------------------------------------------------------------------------


class TestEmbedNoneOnImportError:
    """embed() returns None when fastembed is not available."""

    async def test_import_error_returns_none(self):
        """If fastembed import fails, embed returns None."""
        embedder = LocalEmbedder()
        with patch.object(embedder, "_ensure_model", side_effect=ImportError("no fastembed")):
            result = await embedder.embed("test")
        assert result is None

    async def test_runtime_error_returns_none(self):
        """If model raises RuntimeError, embed returns None."""
        model = MagicMock()
        model.embed.side_effect = RuntimeError("ONNX error")
        with patch.object(LocalEmbedder, "_ensure_model"):
            embedder = LocalEmbedder()
            embedder._model = model
            result = await embedder.embed("test")
        assert result is None

    async def test_generic_exception_returns_none(self):
        """Any unexpected exception returns None."""
        embedder = LocalEmbedder()
        with patch.object(embedder, "_ensure_model", side_effect=Exception("boom")):
            result = await embedder.embed("test")
        assert result is None

    async def test_model_returns_empty_iterator(self):
        """If model.embed returns empty iterator, embed returns None (via IndexError)."""
        model = MagicMock()
        model.embed.return_value = iter([])
        with patch.object(LocalEmbedder, "_ensure_model"):
            embedder = LocalEmbedder()
            embedder._model = model
            result = await embedder.embed("test")
        assert result is None


# ---------------------------------------------------------------------------
# Text truncation
# ---------------------------------------------------------------------------


class TestEmbedTruncatesLongText:
    """Long text is truncated to MAX_INPUT_CHARS before embedding."""

    async def test_long_text_truncated(self):
        """Input longer than 32K chars is clipped."""
        model = _fake_model()
        long_text = "a" * (MAX_INPUT_CHARS + 10_000)

        with patch.object(LocalEmbedder, "_ensure_model"):
            embedder = LocalEmbedder()
            embedder._model = model
            await embedder.embed(long_text)

        # The text passed to model.embed([text]) should be truncated
        sent_text = model.embed.call_args[0][0][0]
        assert len(sent_text) == MAX_INPUT_CHARS

    async def test_short_text_not_truncated(self):
        """Input shorter than limit is passed unchanged."""
        model = _fake_model()
        with patch.object(LocalEmbedder, "_ensure_model"):
            embedder = LocalEmbedder()
            embedder._model = model
            await embedder.embed("short input")

        sent_text = model.embed.call_args[0][0][0]
        assert sent_text == "short input"

    async def test_exact_limit_not_truncated(self):
        """Input exactly at MAX_INPUT_CHARS is not truncated."""
        model = _fake_model()
        exact_text = "b" * MAX_INPUT_CHARS

        with patch.object(LocalEmbedder, "_ensure_model"):
            embedder = LocalEmbedder()
            embedder._model = model
            await embedder.embed(exact_text)

        sent_text = model.embed.call_args[0][0][0]
        assert len(sent_text) == MAX_INPUT_CHARS


# ---------------------------------------------------------------------------
# Lazy initialization
# ---------------------------------------------------------------------------


class TestLazyModelInit:
    """Model is not loaded until first embed call."""

    def test_model_none_at_init(self):
        """_model is None after construction."""
        embedder = LocalEmbedder()
        assert embedder._model is None

    def test_ensure_model_creates_model(self):
        """_ensure_model imports and creates TextEmbedding."""
        mock_cls = MagicMock()
        with patch.dict(sys.modules, {"fastembed": MagicMock(TextEmbedding=mock_cls)}):
            embedder = LocalEmbedder()
            embedder._ensure_model()
        assert embedder._model is not None
        mock_cls.assert_called_once_with(LocalEmbedder.MODEL)

    def test_ensure_model_idempotent(self):
        """Calling _ensure_model twice doesn't reload."""
        embedder = LocalEmbedder()
        sentinel = MagicMock()
        embedder._model = sentinel
        embedder._ensure_model()
        assert embedder._model is sentinel

    async def test_first_embed_triggers_model_load(self):
        """First embed() call triggers _ensure_model."""
        model = _fake_model()
        embedder = LocalEmbedder()

        with patch.object(embedder, "_ensure_model") as mock_ensure:
            embedder._model = model
            await embedder.embed("test")
            mock_ensure.assert_called_once()
