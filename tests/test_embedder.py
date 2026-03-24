"""Tests for LocalEmbedder (embedder.py).

Mocks fastembed to avoid real model downloads. Tests verify:
- Successful embedding returns 384-dim vector
- Input truncation to MAX_INPUT_CHARS
- Exception handling (returns None)
- Lazy model initialization
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.search.embedder import LocalEmbedder, MAX_INPUT_CHARS


# ── Successful embedding ────────────────────────────────────────────


class TestEmbedSuccess:
    async def test_returns_embedding_vector(self):
        """Successful embed returns a list of floats."""
        fake_model = MagicMock()
        fake_model.embed.return_value = iter([np.array([0.1] * 384, dtype=np.float32)])

        with patch("src.search.embedder.LocalEmbedder._ensure_model"):
            embedder = LocalEmbedder()
            embedder._model = fake_model
            result = await embedder.embed("test text")

        assert result is not None
        assert len(result) == 384
        assert isinstance(result[0], float)

    async def test_returns_correct_values(self):
        """Embed returns the exact values from the model."""
        expected = [float(i) / 384 for i in range(384)]
        fake_model = MagicMock()
        fake_model.embed.return_value = iter([np.array(expected, dtype=np.float32)])

        with patch("src.search.embedder.LocalEmbedder._ensure_model"):
            embedder = LocalEmbedder()
            embedder._model = fake_model
            result = await embedder.embed("test")

        assert result is not None
        for a, b in zip(result, expected):
            assert abs(a - b) < 1e-5


# ── Input truncation ───────────────────────────────────────────────


class TestInputTruncation:
    async def test_long_input_truncated(self):
        """Input longer than MAX_INPUT_CHARS is truncated before embedding."""
        fake_model = MagicMock()
        fake_model.embed.return_value = iter([np.array([0.2] * 384, dtype=np.float32)])

        long_text = "x" * (MAX_INPUT_CHARS + 5000)

        with patch("src.search.embedder.LocalEmbedder._ensure_model"):
            embedder = LocalEmbedder()
            embedder._model = fake_model
            result = await embedder.embed(long_text)

        assert result is not None
        # Verify the text passed to model.embed was truncated
        sent_text = fake_model.embed.call_args[0][0][0]
        assert len(sent_text) == MAX_INPUT_CHARS

    async def test_short_input_not_truncated(self):
        """Input shorter than MAX_INPUT_CHARS is passed as-is."""
        fake_model = MagicMock()
        fake_model.embed.return_value = iter([np.array([0.3] * 384, dtype=np.float32)])

        with patch("src.search.embedder.LocalEmbedder._ensure_model"):
            embedder = LocalEmbedder()
            embedder._model = fake_model
            await embedder.embed("short text")

        sent_text = fake_model.embed.call_args[0][0][0]
        assert sent_text == "short text"


# ── Error handling ──────────────────────────────────────────────────


class TestErrorHandling:
    async def test_model_load_failure_returns_none(self):
        """If fastembed import fails, embed returns None."""
        embedder = LocalEmbedder()
        with patch.object(embedder, "_ensure_model", side_effect=ImportError("no fastembed")):
            result = await embedder.embed("test")
        assert result is None

    async def test_embed_exception_returns_none(self):
        """If model.embed raises, returns None."""
        fake_model = MagicMock()
        fake_model.embed.side_effect = RuntimeError("ONNX error")

        with patch("src.search.embedder.LocalEmbedder._ensure_model"):
            embedder = LocalEmbedder()
            embedder._model = fake_model
            result = await embedder.embed("test")

        assert result is None

    async def test_generic_exception_returns_none(self):
        """Any unexpected exception returns None."""
        embedder = LocalEmbedder()
        with patch.object(embedder, "_ensure_model", side_effect=Exception("unexpected")):
            result = await embedder.embed("test")
        assert result is None


# ── Lazy initialization ────────────────────────────────────────────


class TestLazyInit:
    def test_model_not_loaded_at_init(self):
        """Model is not loaded until first embed call."""
        embedder = LocalEmbedder()
        assert embedder._model is None

    def test_ensure_model_loads_on_first_call(self):
        """_ensure_model imports and creates TextEmbedding."""
        mock_text_embedding = MagicMock()
        with patch.dict("sys.modules", {"fastembed": MagicMock(TextEmbedding=mock_text_embedding)}):
            embedder = LocalEmbedder()
            embedder._ensure_model()
            assert embedder._model is not None

    def test_ensure_model_idempotent(self):
        """Calling _ensure_model twice doesn't reload."""
        embedder = LocalEmbedder()
        embedder._model = MagicMock()  # Already loaded
        existing = embedder._model
        embedder._ensure_model()
        assert embedder._model is existing  # Same object


# ── Constants ───────────────────────────────────────────────────────


class TestConstants:
    def test_dimensions(self):
        """DIMENSIONS should be 384 for bge-small-en-v1.5."""
        assert LocalEmbedder.DIMENSIONS == 384

    def test_model_name(self):
        """MODEL should be BAAI/bge-small-en-v1.5."""
        assert LocalEmbedder.MODEL == "BAAI/bge-small-en-v1.5"

    def test_max_input_chars(self):
        """MAX_INPUT_CHARS should be 32000."""
        assert MAX_INPUT_CHARS == 32_000
