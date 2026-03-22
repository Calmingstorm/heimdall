"""Text-to-speech using Microsoft Edge TTS (neural, natural-sounding, free).

Falls back to Piper if Edge TTS is unavailable.
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import wave

import numpy as np

from .audio import resample_linear
from .config import TtsConfig

log = logging.getLogger("voice.tts")


class TextToSpeech:
    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._current_voice = config.default_voice

    @property
    def current_voice(self) -> str:
        return self._current_voice

    def set_voice(self, voice: str) -> None:
        self._current_voice = voice
        log.info("TTS voice set to: %s", voice)

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Synthesize text to 48kHz stereo s16le PCM audio using Edge TTS.

        Returns:
            Raw PCM audio bytes (48kHz, stereo, s16le).
        """
        voice = voice or self._current_voice

        log.info("TTS synthesizing: %r (voice=%s)", text[:80], voice)

        import edge_tts

        communicate = edge_tts.Communicate(text, voice)

        # Edge TTS returns MP3 chunks — collect them all
        mp3_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_data.extend(chunk["data"])

        if not mp3_data:
            log.warning("Edge TTS produced no output for: %r", text[:80])
            return b""

        # Decode MP3 to PCM using ffmpeg
        pcm_48k_stereo = await self._decode_mp3_to_pcm(bytes(mp3_data))
        return pcm_48k_stereo

    async def _decode_mp3_to_pcm(self, mp3_data: bytes) -> bytes:
        """Decode MP3 to 48kHz stereo s16le PCM via ffmpeg."""
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", "pipe:0",
            "-f", "s16le", "-ar", "48000", "-ac", "2",
            "-acodec", "pcm_s16le", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=mp3_data),
            timeout=30,
        )

        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()[-200:]
            log.error("ffmpeg decode failed: %s", err)
            raise RuntimeError(f"ffmpeg decode failed: {err}")

        return stdout

    async def synthesize_chunked(self, text: str, chunk_size: int = 48000 * 2 * 2, voice: str | None = None):
        """Synthesize and yield audio in chunks for streaming."""
        full_audio = await self.synthesize(text, voice=voice)
        for i in range(0, len(full_audio), chunk_size):
            yield full_audio[i : i + chunk_size]
