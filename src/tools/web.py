"""Built-in web search and URL fetching tools.

Replaces the fragile fetch_url and search_news skills with proper built-in tools.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import quote_plus

import aiohttp

from ..logging import get_logger

log = get_logger("tools.web")

MAX_CONTENT_CHARS = 4000
FETCH_TIMEOUT = aiohttp.ClientTimeout(total=15)
SEARCH_TIMEOUT = aiohttp.ClientTimeout(total=10)

# User-Agent to avoid bot blocking
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


class _HTMLToText(HTMLParser):
    """Minimal HTML to text converter — strips tags, keeps structure."""

    def __init__(self):
        super().__init__()
        self._text: list[str] = []
        self._skip_depth = 0
        self._skip_tags = {"script", "style", "noscript", "head", "nav", "footer", "header"}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag in self._skip_tags:
            self._skip_depth += 1
        if tag in ("br", "p", "div", "h1", "h2", "h3", "h4", "li", "tr"):
            self._text.append("\n")

    def handle_endtag(self, tag: str):
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str):
        if self._skip_depth == 0:
            self._text.append(data)

    def get_text(self) -> str:
        text = "".join(self._text)
        # Collapse excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


def _html_to_text(html: str) -> str:
    parser = _HTMLToText()
    parser.feed(html)
    return parser.get_text()


async def fetch_url(url: str, max_chars: int = MAX_CONTENT_CHARS) -> str:
    """Fetch a URL and return its content as text.

    Handles HTML (converts to markdown-like text), JSON (returns raw),
    and plain text. Truncates to max_chars.
    """
    try:
        async with aiohttp.ClientSession(timeout=FETCH_TIMEOUT) as session:
            async with session.get(
                url,
                headers={"User-Agent": USER_AGENT},
                allow_redirects=True,
                ssl=False,  # Disable SSL verification — internal infra may use self-signed certs
            ) as resp:
                if resp.status != 200:
                    return f"HTTP {resp.status}: {resp.reason}"

                content_type = resp.headers.get("Content-Type", "")
                body = await resp.text(errors="replace")

                if "json" in content_type:
                    # Return raw JSON
                    result = body
                elif "html" in content_type:
                    result = _html_to_text(body)
                else:
                    result = body

                if len(result) > max_chars:
                    result = result[:max_chars] + "\n\n... (content truncated)"

                return result

    except aiohttp.ClientError as e:
        return f"Fetch error: {e}"
    except Exception as e:
        log.error("fetch_url failed for %s: %s", url, e)
        return f"Error: {e}"


async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo HTML and return results.

    Returns a formatted list of results with titles, URLs, and snippets.
    """
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        async with aiohttp.ClientSession(timeout=SEARCH_TIMEOUT) as session:
            async with session.get(
                search_url,
                headers={"User-Agent": USER_AGENT},
                allow_redirects=True,
            ) as resp:
                if resp.status != 200:
                    return f"Search failed: HTTP {resp.status}"

                html = await resp.text(errors="replace")
                return _parse_ddg_results(html, max_results)

    except aiohttp.ClientError as e:
        return f"Search error: {e}"
    except Exception as e:
        log.error("web_search failed for %s: %s", query, e)
        return f"Error: {e}"


def _parse_ddg_results(html: str, max_results: int) -> str:
    """Parse DuckDuckGo HTML search results into a readable format."""
    results: list[str] = []

    # DuckDuckGo HTML results use class="result__a" for links
    # and class="result__snippet" for snippets
    link_pattern = re.compile(
        r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'class="result__snippet"[^>]*>(.*?)</(?:td|span|div)',
        re.DOTALL,
    )

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (url, title) in enumerate(links[:max_results]):
        # Clean HTML from title and snippet
        clean_title = re.sub(r"<[^>]+>", "", title).strip()
        clean_snippet = ""
        if i < len(snippets):
            clean_snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()

        # DuckDuckGo wraps URLs in a redirect — extract the actual URL
        if "uddg=" in url:
            actual = re.search(r"uddg=([^&]+)", url)
            if actual:
                from urllib.parse import unquote
                url = unquote(actual.group(1))

        result = f"**{i+1}. {clean_title}**\n{url}"
        if clean_snippet:
            result += f"\n{clean_snippet}"
        results.append(result)

    if not results:
        return "No results found."

    return "\n\n".join(results)
