"""Tests for discord/voice.py — voice channel manager.

Covers:
- VoiceState enum values
- VoiceMessageProxy: reply() and add_reaction() delegation
- PCMStreamSource: read() frame extraction, is_opus(), buffer exhaustion
- VoiceManager:
  - is_connected / current_channel properties
  - _get_transcript_channel: cached, config-based, fallback
  - _get_transcript_channel_any: cached, config, guild fallback
  - join_channel: disabled, already connected, existing voice client cleanup, error
  - leave_channel: not connected, success
  - _start_listening / _stop_listening
  - _on_audio_receive: forwards PCM, skips when speaking
  - _ws_send_json / _ws_send_binary: connected vs disconnected
  - _handle_service_message: wake_word, transcription, tts_start, tts_done, error
  - _handle_service_audio: TTS buffer accumulation
  - _on_playback_done: speaking state reset
  - _handle_transcription: leave commands, stop commands, normal transcription routing
  - speak: sends TTS JSON
  - shutdown: disconnects voice
  - _ws_disconnect: closes WS and cancels task
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# Must mock voice_recv before importing anything from src.discord
mock_voice_recv = MagicMock()
mock_voice_recv.VoiceRecvClient = MagicMock
mock_voice_recv.BasicSink = MagicMock
sys.modules["discord.ext.voice_recv"] = mock_voice_recv
# Also mock davey and the opus decoder to prevent the module-level patch from failing
sys.modules.setdefault("davey", MagicMock())
sys.modules.setdefault("websockets", MagicMock())

import pytest  # noqa: E402

from src.discord.voice import (  # noqa: E402
    VoiceState,
    VoiceMessageProxy,
    PCMStreamSource,
    VoiceManager,
    DISCORD_FRAME_BYTES,
    DISCORD_SAMPLE_RATE,
    DISCORD_CHANNELS,
    DISCORD_FRAME_MS,
    DISCORD_FRAME_SAMPLES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_voice_config(enabled=True, transcript_channel_id=None, voice_service_url="ws://localhost:9000", default_voice="nova"):
    """Mock VoiceConfig."""
    cfg = MagicMock()
    cfg.enabled = enabled
    cfg.transcript_channel_id = transcript_channel_id
    cfg.voice_service_url = voice_service_url
    cfg.default_voice = default_voice
    return cfg


def _make_bot():
    """Mock Discord bot/client."""
    bot = MagicMock()
    bot.user = MagicMock()
    bot.user.id = 999
    bot.get_channel = MagicMock(return_value=None)
    return bot


def _make_voice_client(connected=True, channel=None):
    """Mock VoiceRecvClient."""
    vc = MagicMock()
    vc.is_connected = MagicMock(return_value=connected)
    vc.channel = channel or MagicMock()
    vc.channel.name = "voice-channel"
    vc.channel.guild = MagicMock()
    vc.listen = MagicMock()
    vc.stop_listening = MagicMock()
    vc.is_playing = MagicMock(return_value=False)
    vc.stop = MagicMock()
    vc.play = MagicMock()
    vc.disconnect = AsyncMock()
    return vc


def _make_manager(enabled=True, transcript_channel_id=None):
    """Create a VoiceManager with mocked config and bot."""
    cfg = _make_voice_config(enabled=enabled, transcript_channel_id=transcript_channel_id)
    bot = _make_bot()
    return VoiceManager(cfg, bot), cfg, bot


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_frame_bytes_calculation(self):
        """DISCORD_FRAME_BYTES is 48000 * 20/1000 * 2 * 2 = 3840."""
        assert DISCORD_SAMPLE_RATE == 48000
        assert DISCORD_CHANNELS == 2
        assert DISCORD_FRAME_MS == 20
        assert DISCORD_FRAME_SAMPLES == 960
        assert DISCORD_FRAME_BYTES == 3840


# ---------------------------------------------------------------------------
# VoiceState
# ---------------------------------------------------------------------------

class TestVoiceState:
    def test_has_all_states(self):
        """VoiceState has IDLE, LISTENING, PROCESSING, SPEAKING."""
        assert VoiceState.IDLE
        assert VoiceState.LISTENING
        assert VoiceState.PROCESSING
        assert VoiceState.SPEAKING

    def test_states_are_distinct(self):
        """All four states are distinct values."""
        states = {VoiceState.IDLE, VoiceState.LISTENING, VoiceState.PROCESSING, VoiceState.SPEAKING}
        assert len(states) == 4


# ---------------------------------------------------------------------------
# VoiceMessageProxy
# ---------------------------------------------------------------------------

class TestVoiceMessageProxy:
    async def test_reply_sends_to_channel(self):
        """reply() delegates to channel.send()."""
        channel = AsyncMock()
        channel.send = AsyncMock(return_value=MagicMock())
        member = MagicMock()
        guild = MagicMock()
        proxy = VoiceMessageProxy(author=member, channel=channel, id=12345, guild=guild)

        await proxy.reply("Hello!")
        channel.send.assert_called_once_with("Hello!")

    def test_attributes(self):
        """Proxy stores author, channel, id, guild."""
        member = MagicMock()
        channel = MagicMock()
        guild = MagicMock()
        proxy = VoiceMessageProxy(author=member, channel=channel, id=42, guild=guild)
        assert proxy.author is member
        assert proxy.channel is channel
        assert proxy.id == 42
        assert proxy.guild is guild


# ---------------------------------------------------------------------------
# PCMStreamSource
# ---------------------------------------------------------------------------

class TestPCMStreamSource:
    def test_read_returns_frame_sized_chunks(self):
        """read() returns exactly DISCORD_FRAME_BYTES per call."""
        data = b"\x00" * (DISCORD_FRAME_BYTES * 3)  # exactly 3 frames
        source = PCMStreamSource(data)

        frame1 = source.read()
        assert len(frame1) == DISCORD_FRAME_BYTES

        frame2 = source.read()
        assert len(frame2) == DISCORD_FRAME_BYTES

        frame3 = source.read()
        assert len(frame3) == DISCORD_FRAME_BYTES

    def test_read_returns_empty_when_exhausted(self):
        """read() returns empty bytes when buffer is exhausted."""
        source = PCMStreamSource(b"\x00" * DISCORD_FRAME_BYTES)
        source.read()  # consume the single frame
        assert source.read() == b""

    def test_partial_frame_returns_empty(self):
        """read() returns empty if less than one frame remains."""
        source = PCMStreamSource(b"\x00" * (DISCORD_FRAME_BYTES - 1))
        assert source.read() == b""

    def test_is_opus_returns_false(self):
        """PCM source is not opus-encoded."""
        source = PCMStreamSource(b"")
        assert source.is_opus() is False

    def test_empty_buffer(self):
        """Empty buffer returns empty on first read."""
        source = PCMStreamSource(b"")
        assert source.read() == b""

    def test_buffer_consumed_progressively(self):
        """Each read() removes the consumed data from the buffer."""
        source = PCMStreamSource(b"\x00" * (DISCORD_FRAME_BYTES * 2 + 100))
        source.read()
        assert len(source._buffer) == DISCORD_FRAME_BYTES + 100
        source.read()
        assert len(source._buffer) == 100
        assert source.read() == b""  # 100 < DISCORD_FRAME_BYTES


# ---------------------------------------------------------------------------
# VoiceManager — properties
# ---------------------------------------------------------------------------

class TestVoiceManagerProperties:
    def test_is_connected_no_client(self):
        """is_connected is False when no voice client."""
        mgr, _, _ = _make_manager()
        assert mgr.is_connected is False

    def test_is_connected_with_client(self):
        """is_connected delegates to voice_client.is_connected()."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = _make_voice_client(connected=True)
        assert mgr.is_connected is True

    def test_is_connected_client_disconnected(self):
        """is_connected is False when client exists but is disconnected."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = _make_voice_client(connected=False)
        assert mgr.is_connected is False

    def test_current_channel_none_when_no_client(self):
        """current_channel is None when not connected."""
        mgr, _, _ = _make_manager()
        assert mgr.current_channel is None

    def test_current_channel_none_when_disconnected(self):
        """current_channel is None when voice client is disconnected."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = _make_voice_client(connected=False)
        assert mgr.current_channel is None

    def test_current_channel_returns_channel(self):
        """current_channel returns the voice client's channel when connected."""
        mgr, _, _ = _make_manager()
        ch = MagicMock()
        mgr._voice_client = _make_voice_client(connected=True, channel=ch)
        assert mgr.current_channel is ch


# ---------------------------------------------------------------------------
# VoiceManager — transcript channel
# ---------------------------------------------------------------------------

class TestGetTranscriptChannel:
    def test_returns_cached(self):
        """Returns cached transcript channel if set."""
        mgr, _, _ = _make_manager()
        cached_ch = MagicMock()
        mgr._transcript_channel = cached_ch
        guild = MagicMock()
        assert mgr._get_transcript_channel(guild) is cached_ch

    def test_looks_up_by_config_id(self):
        """Looks up channel by ID from config."""
        mgr, cfg, bot = _make_manager(transcript_channel_id="12345")
        ch = MagicMock()
        bot.get_channel = MagicMock(return_value=ch)
        guild = MagicMock()

        result = mgr._get_transcript_channel(guild)
        bot.get_channel.assert_called_with(12345)
        assert result is ch
        assert mgr._transcript_channel is ch  # cached

    def test_returns_none_when_not_found(self):
        """Returns None when config channel ID doesn't resolve."""
        mgr, cfg, bot = _make_manager(transcript_channel_id="99999")
        bot.get_channel = MagicMock(return_value=None)
        guild = MagicMock()

        assert mgr._get_transcript_channel(guild) is None

    def test_returns_none_when_no_config(self):
        """Returns None when no transcript_channel_id configured."""
        mgr, cfg, _ = _make_manager()
        cfg.transcript_channel_id = None
        guild = MagicMock()
        assert mgr._get_transcript_channel(guild) is None


class TestGetTranscriptChannelAny:
    def test_returns_cached(self):
        """Returns cached transcript channel if set."""
        mgr, _, _ = _make_manager()
        cached = MagicMock()
        mgr._transcript_channel = cached
        assert mgr._get_transcript_channel_any() is cached

    def test_looks_up_by_config_id(self):
        """Looks up by config ID when not cached."""
        mgr, cfg, bot = _make_manager(transcript_channel_id="55555")
        ch = MagicMock()
        bot.get_channel = MagicMock(return_value=ch)

        result = mgr._get_transcript_channel_any()
        assert result is ch
        assert mgr._transcript_channel is ch

    def test_falls_back_to_guild_text_channel(self):
        """Falls back to first text channel in the voice channel's guild."""
        mgr, cfg, bot = _make_manager()
        cfg.transcript_channel_id = None
        bot.get_channel = MagicMock(return_value=None)

        vc = _make_voice_client(connected=True)
        text_ch = MagicMock()
        vc.channel.guild.text_channels = [text_ch]
        mgr._voice_client = vc

        result = mgr._get_transcript_channel_any()
        assert result is text_ch

    def test_returns_none_when_all_fail(self):
        """Returns None when no channel can be found."""
        mgr, cfg, bot = _make_manager()
        cfg.transcript_channel_id = None
        bot.get_channel = MagicMock(return_value=None)
        mgr._voice_client = None

        assert mgr._get_transcript_channel_any() is None


# ---------------------------------------------------------------------------
# VoiceManager — join/leave
# ---------------------------------------------------------------------------

class TestJoinChannel:
    async def test_disabled(self):
        """Returns disabled message when voice is disabled."""
        mgr, _, _ = _make_manager(enabled=False)
        channel = MagicMock()
        result = await mgr.join_channel(channel)
        assert "disabled" in result.lower()

    async def test_already_in_same_channel(self):
        """Returns 'already in' when already connected to the same channel."""
        mgr, _, _ = _make_manager()
        channel = MagicMock()
        channel.name = "general-voice"
        vc = _make_voice_client(connected=True, channel=channel)
        mgr._voice_client = vc

        result = await mgr.join_channel(channel)
        assert "Already in" in result

    async def test_disconnects_stale_client(self):
        """Disconnects stale voice client before joining new channel."""
        mgr, _, _ = _make_manager()
        old_channel = MagicMock()
        new_channel = MagicMock()
        new_channel.name = "new-voice"
        new_channel.guild = MagicMock()
        new_channel.guild.voice_client = None

        old_vc = _make_voice_client(connected=True, channel=old_channel)
        mgr._voice_client = old_vc

        # Mock channel.connect to return a new voice client
        new_vc = _make_voice_client(connected=True, channel=new_channel)
        new_channel.connect = AsyncMock(return_value=new_vc)

        with patch.object(mgr, '_start_listening'), \
             patch.object(mgr, '_ws_loop', new_callable=AsyncMock):
            result = await mgr.join_channel(new_channel)

        old_vc.disconnect.assert_called_once_with(force=True)
        assert "Joined" in result

    async def test_disconnects_guild_voice_client(self):
        """Disconnects existing guild.voice_client before joining."""
        mgr, _, _ = _make_manager()
        channel = MagicMock()
        channel.name = "voice"
        guild_vc = AsyncMock()
        guild_vc.disconnect = AsyncMock()
        guild = MagicMock()
        guild.voice_client = guild_vc
        channel.guild = guild

        # Don't pass channel to _make_voice_client — it would overwrite channel.guild
        new_vc = _make_voice_client(connected=True)
        channel.connect = AsyncMock(return_value=new_vc)

        with patch.object(mgr, '_start_listening'), \
             patch.object(mgr, '_ws_loop', new_callable=AsyncMock):
            result = await mgr.join_channel(channel)

        guild_vc.disconnect.assert_called_once_with(force=True)
        assert "Joined" in result

    async def test_connect_failure(self):
        """Returns error message when channel.connect fails."""
        mgr, _, _ = _make_manager()
        channel = MagicMock()
        channel.name = "voice"
        channel.guild = MagicMock()
        channel.guild.voice_client = None
        channel.connect = AsyncMock(side_effect=Exception("Connection timeout"))

        result = await mgr.join_channel(channel)
        assert "Failed" in result
        assert mgr._voice_client is None


class TestLeaveChannel:
    async def test_not_connected(self):
        """Returns 'not in' message when not connected."""
        mgr, _, _ = _make_manager()
        result = await mgr.leave_channel()
        assert "Not in" in result

    async def test_successful_leave(self):
        """Successfully leaves and cleans up state."""
        mgr, _, _ = _make_manager()
        vc = _make_voice_client(connected=True)
        vc.channel.name = "general-voice"
        mgr._voice_client = vc
        mgr._speaking = True
        mgr._ws = MagicMock()
        mgr._connected = True

        with patch.object(mgr, '_stop_listening') as stop_mock, \
             patch.object(mgr, '_ws_disconnect', new_callable=AsyncMock) as ws_mock:
            result = await mgr.leave_channel()

        assert "Left" in result
        assert "general-voice" in result
        assert mgr._voice_client is None
        assert mgr._speaking is False
        stop_mock.assert_called_once()
        ws_mock.assert_called_once()

    async def test_disconnect_error_handled(self):
        """Handles errors during disconnect gracefully."""
        mgr, _, _ = _make_manager()
        vc = _make_voice_client(connected=True)
        vc.disconnect = AsyncMock(side_effect=Exception("Already disconnected"))
        mgr._voice_client = vc

        with patch.object(mgr, '_stop_listening'), \
             patch.object(mgr, '_ws_disconnect', new_callable=AsyncMock):
            result = await mgr.leave_channel()

        assert "Left" in result
        assert mgr._voice_client is None


# ---------------------------------------------------------------------------
# VoiceManager — listening
# ---------------------------------------------------------------------------

class TestListening:
    def test_start_listening_no_client(self):
        """_start_listening returns early if no voice client."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = None
        # Should not raise
        mgr._start_listening()

    def test_start_listening_sets_up_sink(self):
        """_start_listening creates a BasicSink and calls listen()."""
        mgr, _, bot = _make_manager()
        vc = _make_voice_client()
        mgr._voice_client = vc

        mgr._start_listening()
        vc.listen.assert_called_once()

    def test_start_listening_handles_error(self):
        """_start_listening catches and logs errors."""
        mgr, _, _ = _make_manager()
        vc = _make_voice_client()
        vc.listen = MagicMock(side_effect=Exception("Sink error"))
        mgr._voice_client = vc

        # Should not raise
        mgr._start_listening()

    def test_stop_listening(self):
        """_stop_listening calls stop_listening on voice client."""
        mgr, _, _ = _make_manager()
        vc = _make_voice_client()
        mgr._voice_client = vc

        mgr._stop_listening()
        vc.stop_listening.assert_called_once()

    def test_stop_listening_no_client(self):
        """_stop_listening is safe with no voice client."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = None
        # Should not raise
        mgr._stop_listening()

    def test_stop_listening_handles_error(self):
        """_stop_listening catches errors."""
        mgr, _, _ = _make_manager()
        vc = _make_voice_client()
        vc.stop_listening = MagicMock(side_effect=Exception("Already stopped"))
        mgr._voice_client = vc

        # Should not raise
        mgr._stop_listening()


# ---------------------------------------------------------------------------
# VoiceManager — audio receive
# ---------------------------------------------------------------------------

class TestOnAudioReceive:
    async def test_forwards_to_ws(self):
        """_on_audio_receive forwards PCM data via websocket."""
        mgr, _, _ = _make_manager()
        mgr._speaking = False
        mgr._ws_send_binary = AsyncMock()

        await mgr._on_audio_receive(b"\x00" * 100, "user1")
        mgr._ws_send_binary.assert_called_once_with(b"\x00" * 100)

    async def test_skips_when_speaking(self):
        """_on_audio_receive skips when bot is speaking."""
        mgr, _, _ = _make_manager()
        mgr._speaking = True
        mgr._ws_send_binary = AsyncMock()

        await mgr._on_audio_receive(b"\x00" * 100, "user1")
        mgr._ws_send_binary.assert_not_called()


# ---------------------------------------------------------------------------
# VoiceManager — WebSocket send
# ---------------------------------------------------------------------------

class TestWsSend:
    async def test_send_json_when_connected(self):
        """_ws_send_json sends JSON when connected."""
        mgr, _, _ = _make_manager()
        ws = AsyncMock()
        mgr._ws = ws
        mgr._connected = True

        await mgr._ws_send_json({"type": "test"})
        ws.send.assert_called_once()
        import json
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "test"

    async def test_send_json_noop_when_disconnected(self):
        """_ws_send_json is a no-op when not connected."""
        mgr, _, _ = _make_manager()
        mgr._ws = None
        mgr._connected = False
        # Should not raise
        await mgr._ws_send_json({"type": "test"})

    async def test_send_json_handles_error(self):
        """_ws_send_json catches send errors."""
        mgr, _, _ = _make_manager()
        ws = AsyncMock()
        ws.send = AsyncMock(side_effect=Exception("WS error"))
        mgr._ws = ws
        mgr._connected = True
        # Should not raise
        await mgr._ws_send_json({"type": "test"})

    async def test_send_binary_when_connected(self):
        """_ws_send_binary sends bytes when connected."""
        mgr, _, _ = _make_manager()
        ws = AsyncMock()
        mgr._ws = ws
        mgr._connected = True

        await mgr._ws_send_binary(b"\x00\x01")
        ws.send.assert_called_once_with(b"\x00\x01")

    async def test_send_binary_noop_when_disconnected(self):
        """_ws_send_binary is a no-op when not connected."""
        mgr, _, _ = _make_manager()
        mgr._ws = None
        mgr._connected = False
        await mgr._ws_send_binary(b"\x00\x01")

    async def test_send_binary_handles_error(self):
        """_ws_send_binary catches send errors."""
        mgr, _, _ = _make_manager()
        ws = AsyncMock()
        ws.send = AsyncMock(side_effect=Exception("WS error"))
        mgr._ws = ws
        mgr._connected = True
        # Should not raise
        await mgr._ws_send_binary(b"\x00\x01")


# ---------------------------------------------------------------------------
# VoiceManager — handle_service_message
# ---------------------------------------------------------------------------

class TestHandleServiceMessage:
    async def test_wake_word_detected(self):
        """wake_word_detected message is handled without error."""
        mgr, _, _ = _make_manager()
        await mgr._handle_service_message({"type": "wake_word_detected", "user_id": "user1"})
        # No state change expected, just logging

    async def test_transcription_final(self):
        """Final transcription triggers _handle_transcription."""
        mgr, _, _ = _make_manager()
        mgr._handle_transcription = AsyncMock()

        await mgr._handle_service_message({
            "type": "transcription",
            "text": "Hello bot",
            "user_id": "user1",
            "is_final": True,
        })
        mgr._handle_transcription.assert_called_once_with("Hello bot", "user1")

    async def test_transcription_not_final_ignored(self):
        """Non-final transcriptions are ignored."""
        mgr, _, _ = _make_manager()
        mgr._handle_transcription = AsyncMock()

        await mgr._handle_service_message({
            "type": "transcription",
            "text": "Hel",
            "user_id": "user1",
            "is_final": False,
        })
        mgr._handle_transcription.assert_not_called()

    async def test_transcription_empty_text_ignored(self):
        """Empty transcription text is ignored even if final."""
        mgr, _, _ = _make_manager()
        mgr._handle_transcription = AsyncMock()

        await mgr._handle_service_message({
            "type": "transcription",
            "text": "",
            "user_id": "user1",
            "is_final": True,
        })
        mgr._handle_transcription.assert_not_called()

    async def test_tts_start(self):
        """tts_start sets speaking state and resets buffer."""
        mgr, _, _ = _make_manager()
        mgr._speaking = False
        mgr._tts_buffer = bytearray(b"old data")

        await mgr._handle_service_message({"type": "tts_start"})
        assert mgr._speaking is True
        assert mgr._tts_buffer == bytearray()

    async def test_tts_done_plays_audio(self):
        """tts_done plays buffered audio through voice client."""
        mgr, _, _ = _make_manager()
        vc = _make_voice_client()
        mgr._voice_client = vc
        mgr._speaking = True
        mgr._tts_buffer = bytearray(b"\x00" * 1000)

        await mgr._handle_service_message({"type": "tts_done"})
        vc.play.assert_called_once()
        # Buffer should be cleared
        assert mgr._tts_buffer == bytearray()

    async def test_tts_done_stops_current_playback(self):
        """tts_done stops current playback before starting new."""
        mgr, _, _ = _make_manager()
        vc = _make_voice_client()
        vc.is_playing = MagicMock(return_value=True)
        mgr._voice_client = vc
        mgr._speaking = True
        mgr._tts_buffer = bytearray(b"\x00" * 100)

        await mgr._handle_service_message({"type": "tts_done"})
        vc.stop.assert_called_once()
        vc.play.assert_called_once()

    async def test_tts_done_empty_buffer(self):
        """tts_done with empty buffer resets speaking without playing."""
        mgr, _, _ = _make_manager()
        vc = _make_voice_client()
        mgr._voice_client = vc
        mgr._speaking = True
        mgr._tts_buffer = bytearray()

        await mgr._handle_service_message({"type": "tts_done"})
        vc.play.assert_not_called()
        assert mgr._speaking is False

    async def test_tts_done_no_voice_client(self):
        """tts_done without voice client resets speaking."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = None
        mgr._speaking = True
        mgr._tts_buffer = bytearray(b"\x00" * 100)

        await mgr._handle_service_message({"type": "tts_done"})
        assert mgr._speaking is False

    async def test_error_resets_state(self):
        """Error message resets speaking state and TTS buffer."""
        mgr, _, _ = _make_manager()
        mgr._speaking = True
        mgr._tts_buffer = bytearray(b"partial data")

        await mgr._handle_service_message({"type": "error", "message": "TTS failed"})
        assert mgr._speaking is False
        assert mgr._tts_buffer == bytearray()

    async def test_unknown_message_type(self):
        """Unknown message types are handled without error."""
        mgr, _, _ = _make_manager()
        await mgr._handle_service_message({"type": "unknown_event"})


# ---------------------------------------------------------------------------
# VoiceManager — handle_service_audio
# ---------------------------------------------------------------------------

class TestHandleServiceAudio:
    async def test_accumulates_tts_buffer(self):
        """Audio data is accumulated in _tts_buffer when speaking."""
        mgr, _, _ = _make_manager()
        mgr._speaking = True
        mgr._tts_buffer = bytearray()

        await mgr._handle_service_audio(b"\x01\x02\x03")
        assert mgr._tts_buffer == bytearray(b"\x01\x02\x03")

        await mgr._handle_service_audio(b"\x04\x05")
        assert mgr._tts_buffer == bytearray(b"\x01\x02\x03\x04\x05")

    async def test_ignores_when_not_speaking(self):
        """Audio data is ignored when not speaking."""
        mgr, _, _ = _make_manager()
        mgr._speaking = False
        mgr._tts_buffer = bytearray()

        await mgr._handle_service_audio(b"\x01\x02\x03")
        assert mgr._tts_buffer == bytearray()


# ---------------------------------------------------------------------------
# VoiceManager — playback done
# ---------------------------------------------------------------------------

class TestOnPlaybackDone:
    def test_resets_speaking_on_success(self):
        """_on_playback_done resets speaking flag."""
        mgr, _, _ = _make_manager()
        mgr._speaking = True
        mgr._on_playback_done(None)
        assert mgr._speaking is False

    def test_resets_speaking_on_error(self):
        """_on_playback_done resets speaking even on error."""
        mgr, _, _ = _make_manager()
        mgr._speaking = True
        mgr._on_playback_done(Exception("Playback error"))
        assert mgr._speaking is False


# ---------------------------------------------------------------------------
# VoiceManager — handle_transcription
# ---------------------------------------------------------------------------

class TestHandleTranscription:
    async def test_leave_command(self):
        """'leave voice' triggers leave_channel."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = _make_voice_client(connected=True)
        mgr.leave_channel = AsyncMock(return_value="Left **voice-channel**.")
        mgr._get_transcript_channel_any = MagicMock(return_value=AsyncMock())

        await mgr._handle_transcription("leave voice", "user1")
        mgr.leave_channel.assert_called_once()

    async def test_disconnect_command(self):
        """'disconnect' triggers leave_channel."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = _make_voice_client(connected=True)
        mgr.leave_channel = AsyncMock(return_value="Left.")
        mgr._get_transcript_channel_any = MagicMock(return_value=AsyncMock())

        await mgr._handle_transcription("disconnect", "user1")
        mgr.leave_channel.assert_called_once()

    async def test_leave_the_channel_command(self):
        """'leave the channel' triggers leave_channel."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = _make_voice_client(connected=True)
        mgr.leave_channel = AsyncMock(return_value="Left.")
        mgr._get_transcript_channel_any = MagicMock(return_value=AsyncMock())

        await mgr._handle_transcription("please leave the channel", "user1")
        mgr.leave_channel.assert_called_once()

    async def test_stop_command_while_speaking(self):
        """'stop' cancels playback when speaking."""
        mgr, _, _ = _make_manager()
        vc = _make_voice_client()
        mgr._voice_client = vc
        mgr._speaking = True

        await mgr._handle_transcription("stop", "user1")
        vc.stop.assert_called_once()
        assert mgr._speaking is False

    async def test_shut_up_command(self):
        """'shut up' cancels playback."""
        mgr, _, _ = _make_manager()
        vc = _make_voice_client()
        mgr._voice_client = vc
        mgr._speaking = True

        await mgr._handle_transcription("shut up", "user1")
        vc.stop.assert_called_once()

    async def test_stop_command_not_speaking_ignored(self):
        """'stop' is ignored when not speaking — falls through to transcription."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = _make_voice_client()
        mgr._speaking = False
        mgr.on_transcription = AsyncMock()

        # Set up the guild member lookup
        guild = mgr._voice_client.channel.guild
        member = MagicMock()
        member.bot = False
        guild.get_member = MagicMock(return_value=member)
        mgr._get_transcript_channel = MagicMock(return_value=MagicMock())

        await mgr._handle_transcription("stop", "123")
        mgr.on_transcription.assert_called_once()

    async def test_normal_transcription_with_callback(self):
        """Normal text triggers on_transcription callback with member and channel."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = _make_voice_client(connected=True)
        member = MagicMock()
        member.bot = False
        guild = mgr._voice_client.channel.guild
        guild.get_member = MagicMock(return_value=member)
        transcript_ch = MagicMock()
        mgr._get_transcript_channel = MagicMock(return_value=transcript_ch)
        mgr.on_transcription = AsyncMock()

        await mgr._handle_transcription("check server status", "12345")

        mgr.on_transcription.assert_called_once_with(
            "check server status", member, transcript_ch,
        )

    async def test_transcription_no_callback(self):
        """No error when on_transcription is not set."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = _make_voice_client()
        mgr.on_transcription = None

        # Should not raise
        await mgr._handle_transcription("hello", "user1")

    async def test_transcription_invalid_user_id_fallback(self):
        """Falls back to first non-bot member when user_id is not an int."""
        mgr, _, _ = _make_manager()
        vc = _make_voice_client(connected=True)
        mgr._voice_client = vc

        member = MagicMock()
        member.bot = False
        bot_member = MagicMock()
        bot_member.bot = True
        vc.channel.members = [bot_member, member]

        transcript_ch = MagicMock()
        mgr._get_transcript_channel = MagicMock(return_value=transcript_ch)
        mgr.on_transcription = AsyncMock()

        await mgr._handle_transcription("hello", "not-a-number")

        mgr.on_transcription.assert_called_once_with("hello", member, transcript_ch)

    async def test_transcription_no_transcript_channel(self):
        """Logs warning when no transcript channel configured."""
        mgr, _, _ = _make_manager()
        vc = _make_voice_client(connected=True)
        mgr._voice_client = vc
        member = MagicMock()
        vc.channel.guild.get_member = MagicMock(return_value=member)
        mgr._get_transcript_channel = MagicMock(return_value=None)
        mgr.on_transcription = AsyncMock()

        await mgr._handle_transcription("hello", "123")
        mgr.on_transcription.assert_not_called()

    async def test_transcription_no_member(self):
        """No callback when member cannot be resolved."""
        mgr, _, _ = _make_manager()
        vc = _make_voice_client(connected=True)
        mgr._voice_client = vc
        vc.channel.guild.get_member = MagicMock(return_value=None)
        vc.channel.members = []  # no fallback members
        mgr.on_transcription = AsyncMock()

        await mgr._handle_transcription("hello", "999")
        mgr.on_transcription.assert_not_called()


# ---------------------------------------------------------------------------
# VoiceManager — speak
# ---------------------------------------------------------------------------

class TestSpeak:
    async def test_sends_tts_json(self):
        """speak() sends TTS request via websocket."""
        mgr, cfg, _ = _make_manager()
        vc = _make_voice_client(connected=True)
        mgr._voice_client = vc
        mgr._connected = True
        mgr._ws_send_json = AsyncMock()

        await mgr.speak("Hello world")
        mgr._ws_send_json.assert_called_once_with({
            "type": "tts",
            "text": "Hello world",
            "voice": cfg.default_voice,
        })

    async def test_speak_not_connected(self):
        """speak() is a no-op when not connected."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = None
        mgr._connected = False
        mgr._ws_send_json = AsyncMock()

        await mgr.speak("Hello")
        mgr._ws_send_json.assert_not_called()

    async def test_speak_voice_connected_ws_not(self):
        """speak() is a no-op when voice connected but WS not."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = _make_voice_client(connected=True)
        mgr._connected = False
        mgr._ws_send_json = AsyncMock()

        await mgr.speak("Hello")
        mgr._ws_send_json.assert_not_called()


# ---------------------------------------------------------------------------
# VoiceManager — shutdown / ws_disconnect
# ---------------------------------------------------------------------------

class TestShutdown:
    async def test_shutdown_when_connected(self):
        """shutdown() leaves channel and disconnects WS."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = _make_voice_client(connected=True)
        mgr.leave_channel = AsyncMock(return_value="Left.")
        mgr._ws_disconnect = AsyncMock()

        await mgr.shutdown()
        mgr.leave_channel.assert_called_once()
        mgr._ws_disconnect.assert_called_once()

    async def test_shutdown_when_not_connected(self):
        """shutdown() just disconnects WS when not in voice."""
        mgr, _, _ = _make_manager()
        mgr._voice_client = None
        mgr._ws_disconnect = AsyncMock()

        await mgr.shutdown()
        mgr._ws_disconnect.assert_called_once()


class TestWsDisconnect:
    async def test_closes_ws_and_cancels_task(self):
        """_ws_disconnect closes WS, cancels task, resets state."""
        mgr, _, _ = _make_manager()
        ws = AsyncMock()
        mgr._ws = ws
        mgr._connected = True

        task = asyncio.Future()
        task.set_result(None)
        mgr._ws_task = task

        await mgr._ws_disconnect()

        ws.close.assert_called_once()
        assert mgr._ws is None
        assert mgr._connected is False
        assert mgr._shutting_down is False

    async def test_handles_ws_close_error(self):
        """_ws_disconnect handles WS close errors."""
        mgr, _, _ = _make_manager()
        ws = AsyncMock()
        ws.close = AsyncMock(side_effect=Exception("Already closed"))
        mgr._ws = ws
        mgr._connected = True
        mgr._ws_task = None

        await mgr._ws_disconnect()
        assert mgr._ws is None

    async def test_cancels_running_task(self):
        """_ws_disconnect cancels a running asyncio task."""
        mgr, _, _ = _make_manager()
        mgr._ws = None
        mgr._connected = False

        # Use a mock task to avoid real asyncio scheduling issues
        mock_task = MagicMock()
        mock_task.done = MagicMock(return_value=False)
        mock_task.cancel = MagicMock()
        # Make awaiting the cancelled task raise CancelledError
        mock_task.__await__ = MagicMock(return_value=iter([]))
        mgr._ws_task = mock_task

        await mgr._ws_disconnect()
        mock_task.cancel.assert_called_once()

    async def test_noop_when_nothing_to_close(self):
        """_ws_disconnect is safe when nothing is connected."""
        mgr, _, _ = _make_manager()
        mgr._ws = None
        mgr._connected = False
        mgr._ws_task = None

        await mgr._ws_disconnect()
        assert mgr._shutting_down is False
