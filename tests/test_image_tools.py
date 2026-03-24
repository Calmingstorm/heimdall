"""Tests for the image analysis tool (analyze_image) and vision block format.

Covers: URL fetch, host fetch, vision block format, error handling,
content-type validation, size limit, and image type detection.
"""
from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal valid PNG header (8 bytes magic + enough for detection)
PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

# Minimal JPEG header
JPEG_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 100

# Minimal GIF header
GIF_HEADER = b"GIF89a" + b"\x00" * 100

# Minimal WEBP header
WEBP_HEADER = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 100


def _make_bot():
    """Create a minimal mock bot with the _handle_analyze_image method.

    Instead of instantiating the full LokiBot, we create a mock that has
    the relevant methods from client.py.
    """
    from src.discord.client import LokiBot

    bot = MagicMock(spec=LokiBot)
    # Bind the real static method
    bot._detect_image_type = LokiBot._detect_image_type
    return bot


def _make_mock_aiohttp_session(status=200, content_type="image/png", data=PNG_HEADER):
    """Create mock aiohttp session, response context managers."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.headers = {"Content-Type": content_type}
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
# _detect_image_type (static method on LokiBot)
# ---------------------------------------------------------------------------

class TestDetectImageType:
    """Test the _detect_image_type static method."""

    def test_detect_png(self):
        from src.discord.client import LokiBot
        assert LokiBot._detect_image_type(PNG_HEADER) == "image/png"

    def test_detect_jpeg(self):
        from src.discord.client import LokiBot
        assert LokiBot._detect_image_type(JPEG_HEADER) == "image/jpeg"

    def test_detect_gif(self):
        from src.discord.client import LokiBot
        assert LokiBot._detect_image_type(GIF_HEADER) == "image/gif"

    def test_detect_webp(self):
        from src.discord.client import LokiBot
        assert LokiBot._detect_image_type(WEBP_HEADER) == "image/webp"

    def test_detect_unknown_returns_none(self):
        from src.discord.client import LokiBot
        assert LokiBot._detect_image_type(b"\x00\x00\x00\x00") is None

    def test_detect_empty_returns_none(self):
        from src.discord.client import LokiBot
        assert LokiBot._detect_image_type(b"") is None

    def test_detect_short_data_returns_none(self):
        from src.discord.client import LokiBot
        assert LokiBot._detect_image_type(b"\x89P") is None


# ---------------------------------------------------------------------------
# analyze_image — URL fetch
# ---------------------------------------------------------------------------

class TestAnalyzeImageFromUrl:
    """Test _handle_analyze_image when fetching from a URL."""

    @pytest.mark.asyncio
    async def test_success_returns_image_block(self):
        """Successful URL fetch returns a dict with __image_block__."""
        from src.discord.client import LokiBot

        # Create a minimal mock of LokiBot with required methods
        bot = MagicMock()
        bot._detect_image_type = LokiBot._detect_image_type
        message = MagicMock()
        mock_session = _make_mock_aiohttp_session(data=PNG_HEADER)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await LokiBot._handle_analyze_image(bot, message, {
                "url": "https://example.com/image.png"
            })

        assert isinstance(result, dict)
        assert "__image_block__" in result
        block = result["__image_block__"]
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/png"
        assert "__prompt__" in result

    @pytest.mark.asyncio
    async def test_url_http_error(self):
        """Non-200 status returns error string."""
        from src.discord.client import LokiBot

        bot = MagicMock()
        bot._detect_image_type = LokiBot._detect_image_type
        message = MagicMock()
        mock_session = _make_mock_aiohttp_session(status=404)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await LokiBot._handle_analyze_image(bot, message, {
                "url": "https://example.com/missing.png"
            })

        assert isinstance(result, str)
        assert "HTTP 404" in result

    @pytest.mark.asyncio
    async def test_url_non_image_content_type(self):
        """Non-image Content-Type returns error."""
        from src.discord.client import LokiBot

        bot = MagicMock()
        bot._detect_image_type = LokiBot._detect_image_type
        message = MagicMock()
        mock_session = _make_mock_aiohttp_session(content_type="text/html")

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await LokiBot._handle_analyze_image(bot, message, {
                "url": "https://example.com/page.html"
            })

        assert isinstance(result, str)
        assert "not point to an image" in result

    @pytest.mark.asyncio
    async def test_url_connection_error(self):
        """Network error returns error string."""
        from src.discord.client import LokiBot

        bot = MagicMock()
        bot._detect_image_type = LokiBot._detect_image_type
        message = MagicMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(side_effect=Exception("Connection refused"))
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await LokiBot._handle_analyze_image(bot, message, {
                "url": "https://unreachable.example.com/img.png"
            })

        assert isinstance(result, str)
        assert "Failed to fetch" in result

    @pytest.mark.asyncio
    async def test_custom_prompt(self):
        """Custom prompt is passed through in the result."""
        from src.discord.client import LokiBot

        bot = MagicMock()
        bot._detect_image_type = LokiBot._detect_image_type
        message = MagicMock()
        mock_session = _make_mock_aiohttp_session(data=PNG_HEADER)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await LokiBot._handle_analyze_image(bot, message, {
                "url": "https://example.com/img.png",
                "prompt": "Count the cats in this image."
            })

        assert result["__prompt__"] == "Count the cats in this image."


# ---------------------------------------------------------------------------
# analyze_image — host fetch
# ---------------------------------------------------------------------------

class TestAnalyzeImageFromHost:
    """Test _handle_analyze_image when fetching from a host."""

    @pytest.mark.asyncio
    async def test_host_success(self):
        """Successful host fetch returns vision block."""
        from src.discord.client import LokiBot

        bot = MagicMock()
        bot._detect_image_type = LokiBot._detect_image_type
        message = MagicMock()

        # Mock tool_executor._resolve_host and _exec_command
        b64_data = base64.b64encode(PNG_HEADER).decode()
        bot.tool_executor = MagicMock()
        bot.tool_executor._resolve_host = MagicMock(return_value=("10.0.0.1", "admin", "linux"))
        bot.tool_executor._exec_command = AsyncMock(return_value=(0, b64_data))

        result = await LokiBot._handle_analyze_image(bot, message, {
            "host": "myhost", "path": "/tmp/image.png"
        })

        assert isinstance(result, dict)
        assert "__image_block__" in result

    @pytest.mark.asyncio
    async def test_host_unknown(self):
        """Unknown host returns error."""
        from src.discord.client import LokiBot

        bot = MagicMock()
        bot._detect_image_type = LokiBot._detect_image_type
        message = MagicMock()
        bot.tool_executor = MagicMock()
        bot.tool_executor._resolve_host = MagicMock(return_value=None)

        result = await LokiBot._handle_analyze_image(bot, message, {
            "host": "badhost", "path": "/tmp/img.png"
        })

        assert isinstance(result, str)
        assert "Unknown or disallowed host" in result

    @pytest.mark.asyncio
    async def test_host_exec_failure(self):
        """Non-zero exit from host command returns error."""
        from src.discord.client import LokiBot

        bot = MagicMock()
        bot._detect_image_type = LokiBot._detect_image_type
        message = MagicMock()
        bot.tool_executor = MagicMock()
        bot.tool_executor._resolve_host = MagicMock(return_value=("10.0.0.1", "admin", "linux"))
        bot.tool_executor._exec_command = AsyncMock(return_value=(1, "file not found"))

        result = await LokiBot._handle_analyze_image(bot, message, {
            "host": "myhost", "path": "/tmp/gone.png"
        })

        assert isinstance(result, str)
        assert "Failed to read image" in result

    @pytest.mark.asyncio
    async def test_host_bad_base64(self):
        """Invalid base64 from host returns error."""
        from src.discord.client import LokiBot

        bot = MagicMock()
        bot._detect_image_type = LokiBot._detect_image_type
        message = MagicMock()
        bot.tool_executor = MagicMock()
        bot.tool_executor._resolve_host = MagicMock(return_value=("10.0.0.1", "admin", "linux"))
        bot.tool_executor._exec_command = AsyncMock(return_value=(0, "not-valid-base64!!!"))

        result = await LokiBot._handle_analyze_image(bot, message, {
            "host": "myhost", "path": "/tmp/img.png"
        })

        assert isinstance(result, str)
        assert "Failed to decode" in result


# ---------------------------------------------------------------------------
# analyze_image — validation
# ---------------------------------------------------------------------------

class TestAnalyzeImageValidation:
    """Test input validation and edge cases."""

    @pytest.mark.asyncio
    async def test_missing_params(self):
        """No url and no host+path returns error."""
        from src.discord.client import LokiBot

        bot = MagicMock()
        bot._detect_image_type = LokiBot._detect_image_type
        message = MagicMock()

        result = await LokiBot._handle_analyze_image(bot, message, {})

        assert isinstance(result, str)
        assert "Provide either" in result

    @pytest.mark.asyncio
    async def test_size_limit_5mb(self):
        """Image over 5MB is rejected."""
        from src.discord.client import LokiBot

        bot = MagicMock()
        bot._detect_image_type = LokiBot._detect_image_type
        message = MagicMock()

        # Create mock that returns >5MB of data
        big_data = PNG_HEADER + b"\x00" * (6 * 1024 * 1024)
        mock_session = _make_mock_aiohttp_session(data=big_data)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await LokiBot._handle_analyze_image(bot, message, {
                "url": "https://example.com/huge.png"
            })

        assert isinstance(result, str)
        assert "5MB" in result

    @pytest.mark.asyncio
    async def test_unsupported_format(self):
        """Unsupported image format returns error."""
        from src.discord.client import LokiBot

        bot = MagicMock()
        bot._detect_image_type = LokiBot._detect_image_type
        message = MagicMock()

        # BMP data (not supported)
        bmp_data = b"BM" + b"\x00" * 200
        mock_session = _make_mock_aiohttp_session(
            data=bmp_data, content_type="image/bmp"
        )

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await LokiBot._handle_analyze_image(bot, message, {
                "url": "https://example.com/image.bmp"
            })

        assert isinstance(result, str)
        assert "Unsupported image format" in result

    @pytest.mark.asyncio
    async def test_default_prompt(self):
        """Default prompt is 'Describe this image in detail.'."""
        from src.discord.client import LokiBot

        bot = MagicMock()
        bot._detect_image_type = LokiBot._detect_image_type
        message = MagicMock()
        mock_session = _make_mock_aiohttp_session(data=PNG_HEADER)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await LokiBot._handle_analyze_image(bot, message, {
                "url": "https://example.com/img.png"
            })

        assert result["__prompt__"] == "Describe this image in detail."


# ---------------------------------------------------------------------------
# Vision block format
# ---------------------------------------------------------------------------

class TestVisionBlockFormat:
    """Verify the vision block structure matches what the tool loop expects."""

    @pytest.mark.asyncio
    async def test_block_structure(self):
        """Vision block has correct nested structure for OpenAI API."""
        from src.discord.client import LokiBot

        bot = MagicMock()
        bot._detect_image_type = LokiBot._detect_image_type
        message = MagicMock()
        mock_session = _make_mock_aiohttp_session(data=JPEG_HEADER, content_type="image/jpeg")

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await LokiBot._handle_analyze_image(bot, message, {
                "url": "https://example.com/photo.jpg"
            })

        block = result["__image_block__"]
        assert block["type"] == "image"
        source = block["source"]
        assert source["type"] == "base64"
        assert source["media_type"] == "image/jpeg"
        # Verify the base64 data is valid
        decoded = base64.b64decode(source["data"])
        assert decoded[:2] == b"\xff\xd8"  # JPEG magic bytes

    @pytest.mark.asyncio
    async def test_different_image_types_produce_correct_media_type(self):
        """Each image format produces the correct media_type in the block."""
        from src.discord.client import LokiBot

        test_cases = [
            (PNG_HEADER, "image/png"),
            (JPEG_HEADER, "image/jpeg"),
            (GIF_HEADER, "image/gif"),
            (WEBP_HEADER, "image/webp"),
        ]

        for img_data, expected_type in test_cases:
            bot = MagicMock()
            bot._detect_image_type = LokiBot._detect_image_type
            message = MagicMock()
            mock_session = _make_mock_aiohttp_session(
                data=img_data, content_type=expected_type
            )

            with patch("aiohttp.ClientSession", return_value=mock_session):
                result = await LokiBot._handle_analyze_image(bot, message, {
                    "url": "https://example.com/img"
                })

            assert result["__image_block__"]["source"]["media_type"] == expected_type, \
                f"Expected {expected_type} but got {result['__image_block__']['source']['media_type']}"
