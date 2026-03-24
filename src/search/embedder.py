from __future__ import annotations

import asyncio
from functools import partial

from ..logging import get_logger

log = get_logger("search.embedder")

# bge-small context is ~512 tokens; truncate input to ~32K chars to be safe
MAX_INPUT_CHARS = 32_000


class LocalEmbedder:
    """In-process text embeddings via fastembed (ONNX, CPU, no server needed)."""

    MODEL = "BAAI/bge-small-en-v1.5"
    DIMENSIONS = 384

    def __init__(self) -> None:
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(self.MODEL)

    async def embed(self, text: str) -> list[float] | None:
        """Embed text. Returns 384-dim float list or None on failure."""
        try:
            self._ensure_model()
            text = text[:MAX_INPUT_CHARS]
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, partial(self._embed_sync, text)
            )
            return result
        except Exception as e:
            log.warning("Embed failed: %s", e)
            return None

    def _embed_sync(self, text: str) -> list[float]:
        vectors = list(self._model.embed([text]))
        return vectors[0].tolist()
