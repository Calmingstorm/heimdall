from __future__ import annotations

import aiohttp

from ..logging import get_logger

log = get_logger("search.embedder")

# nomic-embed-text context limit is ~8192 tokens; truncate input to ~32K chars
MAX_INPUT_CHARS = 32_000


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str = "nomic-embed-text") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def embed(self, text: str) -> list[float] | None:
        """Embed text via Ollama /api/embed endpoint. Returns None on failure."""
        text = text[:MAX_INPUT_CHARS]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": text},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        log.warning("Ollama embed returned %d", resp.status)
                        return None
                    data = await resp.json()
                    embeddings = data.get("embeddings")
                    if embeddings and len(embeddings) > 0:
                        return embeddings[0]
                    return None
        except Exception as e:
            log.warning("Ollama embed failed: %s", e)
            return None
