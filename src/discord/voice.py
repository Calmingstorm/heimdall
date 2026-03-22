"""Voice channel manager for Loki.

Uses discord.py 2.7+ with davey (DAVE protocol) for voice connections,
and discord-ext-voice-recv for receiving audio from users.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Awaitable

import discord
from discord.ext import voice_recv

if TYPE_CHECKING:
    from ..config.schema import VoiceConfig

log = logging.getLogger("loki.voice")


def _patch_voice_recv_dave():
    """Monkey-patch discord-ext-voice-recv to decrypt DAVE frames before opus decode.

    voice_recv was written before Discord's DAVE E2EE protocol. Without this patch,
    the opus decoder receives still-encrypted frames and fails with 'corrupted stream'.
    """
    try:
        import davey
        from discord.ext.voice_recv.opus import PacketDecoder

        _original_decode = PacketDecoder._decode_packet

        _dave_fail_count = [0]
        _dave_ok_count = [0]

        def _patched_decode(self, packet):
            if not packet or not packet.decrypted_data:
                return _original_decode(self, packet)

            try:
                vc = self.sink.voice_client
                connection = getattr(vc, '_connection', None)
                dave_session = getattr(connection, 'dave_session', None) if connection else None
                if dave_session and dave_session.ready:
                    user_id = self._cached_id or 0
                    decrypted = dave_session.decrypt(
                        user_id,
                        davey.MediaType.audio,
                        packet.decrypted_data,
                    )
                    if decrypted is not None:
                        packet.decrypted_data = decrypted
                        _dave_ok_count[0] += 1
                        if _dave_ok_count[0] <= 3 or _dave_ok_count[0] % 500 == 0:
                            log.info("DAVE decrypt OK #%d (data=%d bytes)", _dave_ok_count[0], len(decrypted))
                    else:
                        _dave_fail_count[0] += 1
                        if _dave_fail_count[0] <= 5 or _dave_fail_count[0] % 100 == 0:
                            log.warning("DAVE decrypt returned None #%d", _dave_fail_count[0])
                        pcm = self._decoder.decode(None, fec=False) if self._decoder else b''
                        return packet, pcm
                else:
                    # No DAVE session — try decoding directly (unencrypted)
                    pass
            except Exception as e:
                _dave_fail_count[0] += 1
                if _dave_fail_count[0] <= 5 or _dave_fail_count[0] % 100 == 0:
                    log.warning("DAVE decrypt error #%d: %s", _dave_fail_count[0], e)
                pcm = self._decoder.decode(None, fec=False) if self._decoder else b''
                return packet, pcm

            return _original_decode(self, packet)

        PacketDecoder._decode_packet = _patched_decode
        log.info("Patched voice_recv with DAVE decryption support")
    except ImportError:
        log.warning("Could not patch voice_recv for DAVE (missing davey or voice_recv)")
    except Exception as e:
        log.warning("Failed to patch voice_recv for DAVE: %s", e)


_patch_voice_recv_dave()

# Discord voice: 48kHz stereo s16le, 20ms frames
DISCORD_SAMPLE_RATE = 48000
DISCORD_CHANNELS = 2
DISCORD_FRAME_MS = 20
DISCORD_FRAME_SAMPLES = DISCORD_SAMPLE_RATE * DISCORD_FRAME_MS // 1000  # 960
DISCORD_FRAME_BYTES = DISCORD_FRAME_SAMPLES * DISCORD_CHANNELS * 2  # 3840 bytes


class VoiceState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


@dataclass
class VoiceMessageProxy:
    """Lightweight stand-in for discord.Message when input came from voice."""
    author: discord.Member
    channel: discord.TextChannel
    id: int
    guild: discord.Guild

    async def reply(self, content: str, **kwargs) -> discord.Message:
        return await self.channel.send(content, **kwargs)


class PCMStreamSource(discord.AudioSource):
    """Plays raw PCM audio bytes through Discord voice."""

    def __init__(self, data: bytes) -> None:
        self._buffer = bytearray(data)

    def read(self) -> bytes:
        if len(self._buffer) >= DISCORD_FRAME_BYTES:
            frame = bytes(self._buffer[:DISCORD_FRAME_BYTES])
            del self._buffer[:DISCORD_FRAME_BYTES]
            return frame
        # Done — no more audio
        return b""

    def is_opus(self) -> bool:
        return False


class VoiceManager:
    """Manages voice channel connection and full audio pipeline.

    Uses discord-ext-voice-recv for audio receive (VoiceRecvClient + BasicSink),
    streams audio to the voice processing service over WebSocket for wake word
    detection and STT, and plays TTS audio back via discord.py.
    """

    def __init__(
        self,
        config: "VoiceConfig",
        bot: discord.Client,
    ) -> None:
        self._config = config
        self._bot = bot
        self._voice_client: voice_recv.VoiceRecvClient | None = None
        self._ws = None
        self._ws_task: asyncio.Task | None = None
        self._tts_buffer: bytearray = bytearray()
        self._speaking = False
        self._connected = False
        self._transcript_channel: discord.TextChannel | None = None
        self._reconnect_delay = 1
        self._shutting_down = False
        self._loop: asyncio.AbstractEventLoop | None = None

        # Callback set by LokiBot to route transcriptions
        self.on_transcription: Callable[[str, discord.Member, discord.TextChannel], Awaitable[None]] | None = None

    @property
    def is_connected(self) -> bool:
        return self._voice_client is not None and self._voice_client.is_connected()

    @property
    def current_channel(self) -> discord.VoiceChannel | None:
        if self._voice_client and self._voice_client.is_connected():
            return self._voice_client.channel
        return None

    def _get_transcript_channel(self, guild: discord.Guild) -> discord.TextChannel | None:
        if self._transcript_channel:
            return self._transcript_channel
        if self._config.transcript_channel_id:
            ch = self._bot.get_channel(int(self._config.transcript_channel_id))
            if ch:
                self._transcript_channel = ch
                return ch
        return None

    async def join_channel(self, channel: discord.VoiceChannel) -> str:
        """Join a voice channel and start the full audio pipeline."""
        if not self._config.enabled:
            return "Voice support is disabled in config."

        self._loop = asyncio.get_event_loop()

        # Clean up stale state
        if self._voice_client is not None:
            if self._voice_client.is_connected() and self._voice_client.channel == channel:
                return f"Already in {channel.name}."
            try:
                await self._voice_client.disconnect(force=True)
            except Exception:
                pass
            self._voice_client = None

        guild = channel.guild
        if guild.voice_client:
            try:
                await guild.voice_client.disconnect(force=True)
            except Exception:
                pass

        try:
            # Connect using VoiceRecvClient for audio receive support
            self._voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient)
            log.info("Joined voice channel: %s (is_connected=%s)", channel.name, self._voice_client.is_connected())
        except Exception as e:
            log.error("Failed to join voice channel: %s", e, exc_info=True)
            self._voice_client = None
            return f"Failed to join voice channel: {e}"

        # Start WebSocket connection to voice service
        self._ws_task = asyncio.create_task(self._ws_loop())

        # Start listening for audio via voice_recv BasicSink
        self._start_listening()

        return f"Joined **{channel.name}**."

    async def leave_channel(self) -> str:
        if not self.is_connected:
            return "Not in a voice channel."

        channel_name = self._voice_client.channel.name
        self._stop_listening()
        await self._ws_disconnect()

        try:
            await self._voice_client.disconnect()
        except Exception as e:
            log.warning("Error disconnecting voice: %s", e)

        self._voice_client = None
        self._speaking = False
        log.info("Left voice channel: %s", channel_name)
        return f"Left **{channel_name}**."

    def _start_listening(self) -> None:
        """Start receiving audio via discord-ext-voice-recv."""
        if not self._voice_client:
            return
        try:
            self._audio_frame_count = 0

            bot_user_id = self._bot.user.id if self._bot.user else None

            def on_audio(user, data: voice_recv.VoiceData):
                if self._speaking:
                    return
                if user is None:
                    return
                # Skip audio from the bot itself
                uid = user.id if hasattr(user, "id") else None
                if uid and uid == bot_user_id:
                    return
                self._audio_frame_count += 1
                if self._audio_frame_count <= 3 or self._audio_frame_count % 500 == 0:
                    log.info(
                        "Audio frame #%d from %s, pcm=%d bytes",
                        self._audio_frame_count,
                        user,
                        len(data.pcm) if data.pcm else 0,
                    )
                pcm = data.pcm
                if pcm and self._loop:
                    user_id = str(uid) if uid else str(user)
                    asyncio.run_coroutine_threadsafe(
                        self._on_audio_receive(pcm, user_id),
                        self._loop,
                    )

            sink = voice_recv.BasicSink(on_audio)
            self._voice_client.listen(sink)
            log.info("Audio listening started via voice_recv")
        except Exception as e:
            log.error("Failed to start listening: %s", e, exc_info=True)

    def _stop_listening(self) -> None:
        if self._voice_client:
            try:
                self._voice_client.stop_listening()
            except Exception:
                pass

    async def _on_audio_receive(self, pcm_data: bytes, user_id: str) -> None:
        """Forward received PCM audio to the voice service."""
        if self._speaking:
            return
        await self._ws_send_binary(pcm_data)

    # --- WebSocket connection to voice service ---

    async def _ws_loop(self) -> None:
        import websockets

        while not self._shutting_down and self.is_connected:
            try:
                log.info("Connecting to voice service: %s", self._config.voice_service_url)
                async with websockets.connect(
                    self._config.voice_service_url,
                    max_size=10 * 1024 * 1024,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    self._reconnect_delay = 1
                    log.info("Connected to voice service")

                    await self._ws_send_json({
                        "type": "stt_start",
                        "sample_rate": DISCORD_SAMPLE_RATE,
                        "channels": DISCORD_CHANNELS,
                        "user_id": "stream",
                    })

                    async for raw in ws:
                        if isinstance(raw, str):
                            await self._handle_service_message(json.loads(raw))
                        elif isinstance(raw, bytes):
                            await self._handle_service_audio(raw)

            except Exception as e:
                if self._shutting_down:
                    break
                log.warning(
                    "Voice service connection lost: %s (reconnecting in %ds)",
                    e, self._reconnect_delay,
                )
                self._ws = None
                self._connected = False
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30)

    async def _ws_disconnect(self) -> None:
        self._shutting_down = True
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._connected = False

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
        self._shutting_down = False

    async def _ws_send_json(self, data: dict) -> None:
        if self._ws and self._connected:
            try:
                await self._ws.send(json.dumps(data))
            except Exception as e:
                log.warning("Failed to send JSON to voice service: %s", e)

    async def _ws_send_binary(self, data: bytes) -> None:
        if self._ws and self._connected:
            try:
                await self._ws.send(data)
            except Exception:
                pass

    # --- Handle messages from voice service ---

    async def _handle_service_message(self, msg: dict) -> None:
        msg_type = msg.get("type", "")

        if msg_type == "wake_word_detected":
            log.info("Wake word detected (user: %s)", msg.get("user_id", ""))

        elif msg_type == "transcription":
            text = msg.get("text", "")
            user_id = msg.get("user_id", "")
            is_final = msg.get("is_final", True)
            if is_final and text:
                await self._handle_transcription(text, user_id)

        elif msg_type == "tts_start":
            self._speaking = True
            self._tts_buffer = bytearray()
            log.info("TTS buffering audio")

        elif msg_type == "tts_done":
            # All TTS audio received — now play it
            if self._tts_buffer and self._voice_client:
                audio_data = bytes(self._tts_buffer)
                self._tts_buffer = bytearray()
                # Stop any current playback first
                if self._voice_client.is_playing():
                    self._voice_client.stop()
                source = PCMStreamSource(audio_data)
                self._voice_client.play(
                    source,
                    after=lambda e: self._on_playback_done(e),
                )
                log.info("TTS playback started (%d bytes)", len(audio_data))
            else:
                self._tts_buffer = bytearray()
                self._speaking = False

        elif msg_type == "error":
            log.error("Voice service error: %s", msg.get("message", ""))
            # Reset speaking state so audio receive isn't permanently blocked
            self._speaking = False
            self._tts_buffer = bytearray()

    async def _handle_service_audio(self, data: bytes) -> None:
        if self._speaking and hasattr(self, '_tts_buffer'):
            self._tts_buffer.extend(data)

    def _on_playback_done(self, error: Exception | None) -> None:
        self._speaking = False
        if error:
            log.error("Playback error: %s", error)
        else:
            log.info("TTS playback finished")

    async def _handle_transcription(self, text: str, user_id: str) -> None:
        log.info("Transcription from %s: %r", user_id, text)

        lower = text.lower().strip()
        if any(cmd in lower for cmd in ["leave voice", "disconnect", "leave the channel"]):
            result = await self.leave_channel()
            ch = self._get_transcript_channel_any()
            if ch:
                await ch.send(f"*[Voice command]* {result}")
            return

        if any(cmd in lower for cmd in ["stop", "shut up", "be quiet", "cancel"]):
            if self._speaking and self._voice_client:
                self._voice_client.stop()
                self._speaking = False
                return

        if self.on_transcription:
            member = None
            guild = None
            if self._voice_client and self._voice_client.channel:
                guild = self._voice_client.channel.guild
                try:
                    uid = int(user_id)
                    member = guild.get_member(uid)
                except (ValueError, TypeError):
                    for m in self._voice_client.channel.members:
                        if not m.bot:
                            member = m
                            break

            if member and guild:
                transcript_ch = self._get_transcript_channel(guild)
                if transcript_ch:
                    await self.on_transcription(text, member, transcript_ch)
                else:
                    log.warning("No transcript channel configured")

    def _get_transcript_channel_any(self) -> discord.TextChannel | None:
        if self._transcript_channel:
            return self._transcript_channel
        if self._config.transcript_channel_id:
            ch = self._bot.get_channel(int(self._config.transcript_channel_id))
            if ch:
                self._transcript_channel = ch
                return ch
        if self._voice_client and self._voice_client.channel:
            for ch in self._voice_client.channel.guild.text_channels:
                return ch
        return None

    async def speak(self, text: str) -> None:
        if not self.is_connected or not self._connected:
            return
        await self._ws_send_json({
            "type": "tts",
            "text": text,
            "voice": self._config.default_voice,
        })

    async def shutdown(self) -> None:
        if self.is_connected:
            await self.leave_channel()
        await self._ws_disconnect()
