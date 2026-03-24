"""Tool Use Memory — tracks which tool sequences work for which query types.

After a successful tool loop, records {query, tools_used, embedding, timestamp}.
Before starting a tool loop, finds similar past queries and suggests tool
sequences that worked well. Uses semantic similarity (local embeddings via fastembed)
when available, falls back to Jaccard keyword overlap otherwise.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..search.embedder import LocalEmbedder

MAX_ENTRIES = 200
EXPIRY_DAYS = 30
MIN_SEMANTIC_SCORE = 0.65  # cosine similarity threshold for embeddings
MIN_JACCARD_SCORE = 0.15   # Jaccard threshold (fallback)

# Common words that don't help distinguish queries
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "so", "if", "then", "than", "too", "very", "just",
    "about", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "that", "this",
    "these", "those", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them",
    "their", "up", "out", "please", "hey", "hi", "thanks", "thank",
})

_WORD_RE = re.compile(r"[a-z0-9_]+")


def extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from a query string."""
    words = _WORD_RE.findall(text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two keyword sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class ToolMemory:
    """Tool pattern memory backed by a JSON file with optional semantic matching."""

    def __init__(self, data_path: str | None = None):
        self._path = Path(data_path) if data_path else None
        self._entries: list[dict] = []
        # Cache for format_hints results — avoids re-computing embeddings per message
        self._hints_cache: dict[str, tuple[float, str]] = {}
        self._hints_cache_ttl: float = 30.0  # seconds
        self._load()

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            self._entries = []
            return
        try:
            raw = self._path.read_text()
            data = json.loads(raw)
            if isinstance(data, list):
                self._entries = data
            else:
                self._entries = []
        except (json.JSONDecodeError, OSError):
            self._entries = []
        self._expire()

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._entries, indent=2))

    def _expire(self) -> None:
        """Remove entries older than EXPIRY_DAYS."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=EXPIRY_DAYS)).isoformat()
        self._entries = [
            e for e in self._entries
            if e.get("timestamp", "") >= cutoff
        ]

    async def record(
        self,
        query: str,
        tools_used: list[str],
        success: bool = True,
        embedder: LocalEmbedder | None = None,
    ) -> None:
        """Record a tool use pattern after a tool loop completes."""
        if not tools_used:
            return

        keywords = extract_keywords(query)
        if not keywords:
            return

        entry: dict = {
            "query": query[:200],
            "keywords": keywords,
            "tools_used": tools_used,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Embed the query for semantic matching
        if embedder:
            vector = await embedder.embed(query[:500])
            if vector:
                entry["embedding"] = vector

        self._entries.append(entry)
        self._expire()

        if len(self._entries) > MAX_ENTRIES:
            self._entries = self._entries[-MAX_ENTRIES:]

        self._save()

    async def find_patterns(
        self,
        query: str,
        limit: int = 3,
        allowed_tools: set[str] | None = None,
        embedder: LocalEmbedder | None = None,
    ) -> list[dict]:
        """Find successful tool patterns similar to the given query.

        Uses semantic similarity (cosine on embeddings) when available,
        falls back to Jaccard keyword overlap otherwise.
        """
        query_kw = set(extract_keywords(query))
        if not query_kw:
            return []

        # Try to embed the query for semantic matching
        query_embedding: list[float] | None = None
        if embedder:
            query_embedding = await embedder.embed(query[:500])

        scored: list[tuple[float, dict]] = []
        for entry in self._entries:
            if not entry.get("success"):
                continue
            entry_tools = entry.get("tools_used", [])
            if len(entry_tools) < 2:
                continue
            if allowed_tools is not None and not all(
                t in allowed_tools for t in entry_tools
            ):
                continue

            # Semantic similarity if both have embeddings
            entry_emb = entry.get("embedding")
            if query_embedding and entry_emb:
                score = _cosine(query_embedding, entry_emb)
                if score >= MIN_SEMANTIC_SCORE:
                    scored.append((score, entry))
                    continue
                # Cosine below threshold — fall through to Jaccard

            # Fallback to Jaccard
            entry_kw = set(entry.get("keywords", []))
            score = _jaccard(query_kw, entry_kw)
            if score >= MIN_JACCARD_SCORE:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate by tool sequence
        seen_sequences: set[tuple[str, ...]] = set()
        results: list[dict] = []
        for _score, entry in scored:
            seq = tuple(entry["tools_used"])
            if seq not in seen_sequences:
                seen_sequences.add(seq)
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    async def format_hints(
        self,
        query: str,
        allowed_tools: set[str] | None = None,
        embedder: LocalEmbedder | None = None,
    ) -> str:
        """Format tool pattern hints for system prompt injection.

        Results are cached per query string with a short TTL to avoid
        re-computing embeddings for repeated/similar queries.
        """
        import time as _time
        now = _time.monotonic()
        cache_key = query[:200]  # normalize to avoid huge keys
        cached = self._hints_cache.get(cache_key)
        if cached and now - cached[0] < self._hints_cache_ttl:
            return cached[1]

        patterns = await self.find_patterns(
            query, allowed_tools=allowed_tools, embedder=embedder,
        )
        if not patterns:
            result = ""
        else:
            lines = []
            for p in patterns:
                chain = " -> ".join(f"`{t}`" for t in p["tools_used"])
                lines.append(f"- {chain}")
            result = (
                "## Tool Use Patterns\n"
                "For similar queries, these tool sequences worked well:\n"
                + "\n".join(lines)
            )

        self._hints_cache[cache_key] = (now, result)

        # Evict stale entries periodically to prevent unbounded growth
        if len(self._hints_cache) > 100:
            self._hints_cache = {
                k: v for k, v in self._hints_cache.items()
                if now - v[0] < self._hints_cache_ttl
            }

        return result
