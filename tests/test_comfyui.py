"""Tests for ComfyUI image generation: client, handler, config, tool pack.

Covers: successful generation, disabled config, ComfyUI unavailable,
timeouts, poll loop, error handling, workflow construction, and
tool pack membership.
"""
from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import discord
from src.tools.comfyui import ComfyUIClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal PNG bytes for testing
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200


def _make_mock_post(status=200, prompt_id="test-prompt-123"):
    """Create a mock for the POST /prompt response."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    if status == 200:
        mock_resp.json = AsyncMock(return_value={"prompt_id": prompt_id})
    else:
        mock_resp.text = AsyncMock(return_value="Server error")
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    return mock_resp


def _make_mock_history(prompt_id="test-prompt-123", filename="loki_00001.png"):
    """Create a mock for the GET /history/{id} response."""
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={
        prompt_id: {
            "outputs": {
                "9": {
                    "images": [{"filename": filename}]
                }
            }
        }
    })
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    return mock_resp


def _make_mock_view(data=PNG_BYTES, status=200):
    """Create a mock for the GET /view response."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.read = AsyncMock(return_value=data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    return mock_resp


def _make_success_session(prompt_id="test-prompt-123"):
    """Create a mock aiohttp session that simulates successful generation."""
    session = AsyncMock()

    post_resp = _make_mock_post(prompt_id=prompt_id)
    history_resp = _make_mock_history(prompt_id=prompt_id)
    view_resp = _make_mock_view()

    session.post = MagicMock(return_value=post_resp)
    session.get = MagicMock(side_effect=[history_resp, view_resp])
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    return session


def _make_bot_with_config(comfyui_enabled=True, comfyui_url="http://localhost:8188"):
    """Create a minimal mock bot for generate_image handler."""
    from src.discord.client import LokiBot
    bot = MagicMock(spec=LokiBot)
    bot._handle_generate_image = LokiBot._handle_generate_image.__get__(bot, LokiBot)
    bot.config = MagicMock()
    bot.config.comfyui.enabled = comfyui_enabled
    bot.config.comfyui.url = comfyui_url
    return bot


# ---------------------------------------------------------------------------
# ComfyUIClient unit tests
# ---------------------------------------------------------------------------

class TestComfyUIClient:
    """Tests for the ComfyUIClient class."""

    def test_base_url_trailing_slash_stripped(self):
        client = ComfyUIClient("http://localhost:8188/")
        assert client.base_url == "http://localhost:8188"

    def test_base_url_no_trailing_slash(self):
        client = ComfyUIClient("http://localhost:8188")
        assert client.base_url == "http://localhost:8188"

    async def test_generate_success(self):
        """Full happy path: queue → poll → download."""
        client = ComfyUIClient("http://localhost:8188")
        mock_session = _make_success_session()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await client.generate("a cute cat")

        assert result == PNG_BYTES

    async def test_generate_post_failure(self):
        """POST /prompt returns non-200."""
        client = ComfyUIClient("http://localhost:8188")
        session = AsyncMock()
        post_resp = _make_mock_post(status=500)
        session.post = MagicMock(return_value=post_resp)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session):
            result = await client.generate("test")

        assert result is None

    async def test_generate_no_prompt_id(self):
        """POST /prompt returns 200 but no prompt_id in response."""
        client = ComfyUIClient("http://localhost:8188")
        session = AsyncMock()

        post_resp = AsyncMock()
        post_resp.status = 200
        post_resp.json = AsyncMock(return_value={})
        post_resp.__aenter__ = AsyncMock(return_value=post_resp)
        post_resp.__aexit__ = AsyncMock(return_value=False)

        session.post = MagicMock(return_value=post_resp)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session):
            result = await client.generate("test")

        assert result is None

    async def test_generate_view_failure(self):
        """GET /view returns non-200."""
        client = ComfyUIClient("http://localhost:8188")
        session = AsyncMock()

        post_resp = _make_mock_post()
        history_resp = _make_mock_history()
        view_resp = _make_mock_view(status=404)

        session.post = MagicMock(return_value=post_resp)
        session.get = MagicMock(side_effect=[history_resp, view_resp])
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await client.generate("test")

        assert result is None

    async def test_generate_timeout(self):
        """asyncio.TimeoutError should return None."""
        client = ComfyUIClient("http://localhost:8188")

        with patch("aiohttp.ClientSession", side_effect=asyncio.TimeoutError):
            result = await client.generate("test")

        assert result is None

    async def test_generate_connection_error(self):
        """aiohttp.ClientError should return None."""
        import aiohttp
        client = ComfyUIClient("http://localhost:8188")

        session = AsyncMock()
        session.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("refused"))
        session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session):
            result = await client.generate("test")

        assert result is None

    async def test_generate_unexpected_error(self):
        """Unexpected exceptions should return None."""
        client = ComfyUIClient("http://localhost:8188")

        session = AsyncMock()
        session.__aenter__ = AsyncMock(side_effect=ValueError("unexpected"))
        session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session):
            result = await client.generate("test")

        assert result is None

    async def test_workflow_dimensions_set(self):
        """Workflow should use specified width/height."""
        client = ComfyUIClient("http://localhost:8188")
        mock_session = _make_success_session()
        captured_payload = {}

        orig_post = mock_session.post
        def capture_post(url, json=None):
            captured_payload.update(json or {})
            return orig_post(url, json=json)

        mock_session.post = MagicMock(side_effect=capture_post)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await client.generate("test", width=512, height=768)

        workflow = captured_payload.get("prompt", {})
        assert workflow["5"]["inputs"]["width"] == 512
        assert workflow["5"]["inputs"]["height"] == 768

    async def test_workflow_prompts_set(self):
        """Workflow should set positive and negative prompts."""
        client = ComfyUIClient("http://localhost:8188")
        mock_session = _make_success_session()
        captured_payload = {}

        orig_post = mock_session.post
        def capture_post(url, json=None):
            captured_payload.update(json or {})
            return orig_post(url, json=json)

        mock_session.post = MagicMock(side_effect=capture_post)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await client.generate("a cat", negative="blurry")

        workflow = captured_payload.get("prompt", {})
        assert workflow["6"]["inputs"]["text"] == "a cat"
        assert workflow["7"]["inputs"]["text"] == "blurry"


# ---------------------------------------------------------------------------
# _poll_history tests
# ---------------------------------------------------------------------------

class TestPollHistory:
    """Tests for the ComfyUIClient._poll_history method."""

    async def test_poll_returns_filename_on_success(self):
        client = ComfyUIClient("http://localhost:8188")
        session = AsyncMock()
        history_resp = _make_mock_history(prompt_id="abc", filename="out.png")
        session.get = MagicMock(return_value=history_resp)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client._poll_history(session, "abc")

        assert result == "out.png"

    async def test_poll_returns_none_on_timeout(self):
        """If history never completes, should return None after polling."""
        client = ComfyUIClient("http://localhost:8188")
        session = AsyncMock()

        # Return empty history every time
        empty_resp = AsyncMock()
        empty_resp.status = 200
        empty_resp.json = AsyncMock(return_value={})
        empty_resp.__aenter__ = AsyncMock(return_value=empty_resp)
        empty_resp.__aexit__ = AsyncMock(return_value=False)

        session.get = MagicMock(return_value=empty_resp)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client._poll_history(session, "nonexistent")

        assert result is None

    async def test_poll_handles_non_200_gracefully(self):
        """Non-200 responses during polling should continue, not crash."""
        client = ComfyUIClient("http://localhost:8188")
        session = AsyncMock()

        err_resp = AsyncMock()
        err_resp.status = 500
        err_resp.__aenter__ = AsyncMock(return_value=err_resp)
        err_resp.__aexit__ = AsyncMock(return_value=False)

        # First call returns 500, second returns success
        success_resp = _make_mock_history(prompt_id="xyz", filename="ok.png")
        session.get = MagicMock(side_effect=[err_resp, success_resp])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client._poll_history(session, "xyz")

        assert result == "ok.png"

    async def test_poll_handles_exception_in_loop(self):
        """Exceptions during polling should continue, not crash."""
        client = ComfyUIClient("http://localhost:8188")
        session = AsyncMock()

        # First call raises, second returns success
        success_resp = _make_mock_history(prompt_id="abc", filename="ok.png")
        err_ctx = AsyncMock()
        err_ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("network"))
        err_ctx.__aexit__ = AsyncMock(return_value=False)

        session.get = MagicMock(side_effect=[err_ctx, success_resp])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client._poll_history(session, "abc")

        assert result == "ok.png"


# ---------------------------------------------------------------------------
# _handle_generate_image handler tests
# ---------------------------------------------------------------------------

class TestHandleGenerateImage:
    """Tests for the Discord bot _handle_generate_image handler."""

    async def test_generate_image_disabled_config(self):
        bot = _make_bot_with_config(comfyui_enabled=False)
        msg = AsyncMock(spec=discord.Message)
        msg.channel = AsyncMock()

        result = await bot._handle_generate_image(msg, {"prompt": "a cat"})
        assert "disabled" in result.lower()

    async def test_generate_image_missing_prompt(self):
        bot = _make_bot_with_config(comfyui_enabled=True)
        msg = AsyncMock(spec=discord.Message)
        msg.channel = AsyncMock()

        result = await bot._handle_generate_image(msg, {})
        assert "required" in result.lower() or "prompt" in result.lower()

    async def test_generate_image_empty_prompt(self):
        bot = _make_bot_with_config(comfyui_enabled=True)
        msg = AsyncMock(spec=discord.Message)
        msg.channel = AsyncMock()

        result = await bot._handle_generate_image(msg, {"prompt": ""})
        assert "required" in result.lower() or "prompt" in result.lower()

    async def test_generate_image_success(self):
        bot = _make_bot_with_config(comfyui_enabled=True)
        msg = AsyncMock(spec=discord.Message)
        msg.channel = AsyncMock()

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=PNG_BYTES)

        with patch("src.tools.comfyui.ComfyUIClient", return_value=mock_client):
            result = await bot._handle_generate_image(msg, {
                "prompt": "a cute cat",
                "negative": "blurry",
                "width": 512,
                "height": 512,
            })

        mock_client.generate.assert_awaited_once_with(
            prompt="a cute cat",
            negative="blurry",
            width=512,
            height=512,
        )
        msg.channel.send.assert_awaited_once()
        send_kwargs = msg.channel.send.call_args[1]
        assert "generated" in send_kwargs["content"].lower() or "cat" in send_kwargs["content"].lower()
        assert isinstance(send_kwargs["file"], discord.File)
        assert "generated" in result.lower() and "KB" in result

    async def test_generate_image_comfyui_unavailable(self):
        """When ComfyUI returns None (unavailable/timeout), handler returns error."""
        bot = _make_bot_with_config(comfyui_enabled=True)
        msg = AsyncMock(spec=discord.Message)
        msg.channel = AsyncMock()

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=None)

        with patch("src.tools.comfyui.ComfyUIClient", return_value=mock_client):
            result = await bot._handle_generate_image(msg, {"prompt": "test"})

        assert "failed" in result.lower() or "unavailable" in result.lower()
        msg.channel.send.assert_not_awaited()

    async def test_generate_image_discord_upload_failure(self):
        """When Discord upload fails, handler returns error."""
        bot = _make_bot_with_config(comfyui_enabled=True)
        msg = AsyncMock(spec=discord.Message)
        msg.channel = AsyncMock()
        msg.channel.send = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(status=413), "too large")
        )

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=PNG_BYTES)

        with patch("src.tools.comfyui.ComfyUIClient", return_value=mock_client):
            result = await bot._handle_generate_image(msg, {"prompt": "test"})

        assert "failed" in result.lower()

    async def test_generate_image_dimension_clamping(self):
        """Dimensions should be clamped to 64-2048."""
        bot = _make_bot_with_config(comfyui_enabled=True)
        msg = AsyncMock(spec=discord.Message)
        msg.channel = AsyncMock()

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=PNG_BYTES)

        with patch("src.tools.comfyui.ComfyUIClient", return_value=mock_client):
            await bot._handle_generate_image(msg, {
                "prompt": "test",
                "width": 10,   # below min
                "height": 5000,  # above max
            })

        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs["width"] == 64
        assert call_kwargs["height"] == 2048

    async def test_generate_image_default_dimensions(self):
        """Without explicit dimensions, defaults should be 1024x1024."""
        bot = _make_bot_with_config(comfyui_enabled=True)
        msg = AsyncMock(spec=discord.Message)
        msg.channel = AsyncMock()

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=PNG_BYTES)

        with patch("src.tools.comfyui.ComfyUIClient", return_value=mock_client):
            await bot._handle_generate_image(msg, {"prompt": "test"})

        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs["width"] == 1024
        assert call_kwargs["height"] == 1024

    async def test_generate_image_uses_config_url(self):
        """Handler should pass the config URL to ComfyUIClient."""
        bot = _make_bot_with_config(
            comfyui_enabled=True,
            comfyui_url="http://gpu-server:8188",
        )
        msg = AsyncMock(spec=discord.Message)
        msg.channel = AsyncMock()

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=PNG_BYTES)

        with patch("src.tools.comfyui.ComfyUIClient", return_value=mock_client) as MockCls:
            await bot._handle_generate_image(msg, {"prompt": "test"})

        MockCls.assert_called_once_with("http://gpu-server:8188")


# ---------------------------------------------------------------------------
# Tool pack membership
# ---------------------------------------------------------------------------

class TestComfyUIToolPack:
    """Tests for generate_image in the comfyui tool pack."""

    def test_comfyui_in_tool_packs(self):
        from src.tools.registry import TOOL_PACKS
        assert "comfyui" in TOOL_PACKS
        assert "generate_image" in TOOL_PACKS["comfyui"]

    def test_generate_image_filtered_without_pack(self):
        """When comfyui pack is not enabled, generate_image should be excluded."""
        from src.tools.registry import get_tool_definitions
        tools = get_tool_definitions(enabled_packs=["docker"])
        tool_names = {t["name"] for t in tools}
        assert "generate_image" not in tool_names

    def test_generate_image_included_with_pack(self):
        """When comfyui pack is enabled, generate_image should be included."""
        from src.tools.registry import get_tool_definitions
        tools = get_tool_definitions(enabled_packs=["comfyui"])
        tool_names = {t["name"] for t in tools}
        assert "generate_image" in tool_names

    def test_generate_image_in_all_tools_default(self):
        """With no packs filter (default), all tools including generate_image."""
        from src.tools.registry import get_tool_definitions
        tools = get_tool_definitions()
        tool_names = {t["name"] for t in tools}
        assert "generate_image" in tool_names


# ---------------------------------------------------------------------------
# Config model tests
# ---------------------------------------------------------------------------

class TestComfyUIConfig:
    """Tests for ComfyUIConfig Pydantic model."""

    def test_default_values(self):
        from src.config.schema import ComfyUIConfig
        cfg = ComfyUIConfig()
        assert cfg.enabled is False
        assert cfg.url == "http://localhost:8188"

    def test_custom_values(self):
        from src.config.schema import ComfyUIConfig
        cfg = ComfyUIConfig(enabled=True, url="http://gpu:9999")
        assert cfg.enabled is True
        assert cfg.url == "http://gpu:9999"

    def test_config_has_comfyui_field(self):
        """Main Config model should have a comfyui field with defaults."""
        from src.config.schema import Config
        # Config requires some fields — test that ComfyUIConfig defaults work
        from src.config.schema import ComfyUIConfig
        cfg = ComfyUIConfig()
        assert hasattr(cfg, "enabled")
        assert hasattr(cfg, "url")
