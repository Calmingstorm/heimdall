"""Speech-to-text wrapper around faster-whisper.

Lazy-loads the model on first request and unloads after idle timeout
to free VRAM for Ollama.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

from .config import WhisperConfig

log = logging.getLogger("voice.stt")


class SpeechToText:
    def __init__(self, config: WhisperConfig) -> None:
        self._config = config
        self._model: WhisperModel | None = None
        self._last_used: float = 0
        self._lock = asyncio.Lock()
        self._unload_task: asyncio.Task | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> None:
        """Load Whisper model into VRAM (blocking, run in executor)."""
        from faster_whisper import WhisperModel

        log.info(
            "Loading Whisper model: size=%s device=%s compute=%s",
            self._config.model_size,
            self._config.device,
            self._config.compute_type,
        )
        t0 = time.monotonic()
        self._model = WhisperModel(
            self._config.model_size,
            device=self._config.device,
            compute_type=self._config.compute_type,
        )
        elapsed = time.monotonic() - t0
        log.info("Whisper model loaded in %.1fs", elapsed)
        self._last_used = time.monotonic()

    def _unload_model(self) -> None:
        """Unload Whisper model to free VRAM."""
        if self._model is not None:
            log.info("Unloading Whisper model (idle timeout)")
            del self._model
            self._model = None
            # Help free VRAM
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    async def _ensure_loaded(self) -> None:
        """Ensure model is loaded, loading in executor if needed."""
        if self._model is None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)
        self._last_used = time.monotonic()
        self._schedule_unload()

    def _schedule_unload(self) -> None:
        """Schedule model unload after idle timeout. 0 = never unload."""
        if self._config.idle_unload_seconds <= 0:
            return

        if self._unload_task and not self._unload_task.done():
            self._unload_task.cancel()

        async def _check_idle():
            await asyncio.sleep(self._config.idle_unload_seconds)
            if self._model and (time.monotonic() - self._last_used) >= self._config.idle_unload_seconds:
                self._unload_model()

        self._unload_task = asyncio.create_task(_check_idle())

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe 16kHz mono s16le PCM audio to text.

        Args:
            audio_bytes: Raw PCM audio (s16le, mono)
            sample_rate: Sample rate of input audio (default 16000)

        Returns:
            Transcribed text string.
        """
        async with self._lock:
            await self._ensure_loaded()

            import numpy as np
            from .audio import pcm_to_float32

            audio_float = pcm_to_float32(audio_bytes)

            # Run transcription in executor (it's CPU/GPU bound)
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self._do_transcribe, audio_float)
            return text

    def _do_transcribe(self, audio: "np.ndarray") -> str:
        """Run whisper transcription (blocking)."""
        segments, info = self._model.transcribe(
            audio,
            language=self._config.language,
            vad_filter=self._config.vad_filter,
            beam_size=5,
        )
        parts = []
        for segment in segments:
            parts.append(segment.text.strip())
        text = " ".join(parts)
        log.info("Transcribed: %r (lang=%s prob=%.2f)", text[:100], info.language, info.language_probability)
        return text

    async def shutdown(self) -> None:
        """Clean shutdown — unload model."""
        if self._unload_task and not self._unload_task.done():
            self._unload_task.cancel()
        self._unload_model()
