"""FastAPI server with WebSocket handler for voice processing.

Continuous listen mode: incoming audio is segmented using a simple
voice activity detector, then transcribed with Whisper.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .audio import pcm_48k_stereo_to_16k_mono
from .config import load_config
from .stt import SpeechToText
from .tts import TextToSpeech

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("voice.server")

config = load_config()
app = FastAPI(title="Heimdall Voice Service")

stt = SpeechToText(config.whisper)
tts = TextToSpeech(config.tts)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "whisper_loaded": stt.loaded,
    })


@app.on_event("shutdown")
async def shutdown() -> None:
    await stt.shutdown()


class SpeechSegmenter:
    """Segments audio into speech chunks using energy-based VAD.

    Handles the fact that Discord only sends audio packets when a user
    is speaking — during silence, no packets arrive. Uses last_feed_time
    to detect when audio has stopped arriving.
    """

    def __init__(
        self,
        energy_threshold: float = 150,
        silence_duration: float = 0.8,
        min_speech_duration: float = 0.3,
        max_speech_duration: float = 10.0,
    ) -> None:
        self._threshold = energy_threshold
        self._silence_dur = silence_duration
        self._min_speech = min_speech_duration
        self._max_speech = max_speech_duration

        self._buffer = bytearray()
        self._active = False
        self._speech_start: float = 0
        self._last_feed: float = 0  # last time feed() was called at all

    def feed(self, audio_16k_mono: bytes) -> bytes | None:
        samples = np.frombuffer(audio_16k_mono, dtype=np.int16)
        if len(samples) == 0:
            return None

        energy = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        now = time.monotonic()
        self._last_feed = now

        if energy > self._threshold:
            if not self._active:
                self._active = True
                self._speech_start = now
                self._buffer.clear()
            self._buffer.extend(audio_16k_mono)
            return None

        if not self._active:
            return None

        # Below threshold while active — this is a quiet frame
        self._buffer.extend(audio_16k_mono)

        speech_elapsed = now - self._speech_start
        if speech_elapsed >= self._max_speech:
            return self._emit(speech_elapsed)

        return None

    def check_timeout(self) -> bytes | None:
        """Check if speech should be emitted due to no audio arriving.

        Discord stops sending packets when a user stops speaking.
        Call this periodically to detect end-of-speech.
        """
        if not self._active:
            return None

        now = time.monotonic()
        since_last_feed = now - self._last_feed

        # No audio packets for silence_duration → speech is done
        if since_last_feed >= self._silence_dur:
            speech_elapsed = now - self._speech_start
            return self._emit(speech_elapsed)

        return None

    def _emit(self, duration: float) -> bytes | None:
        self._active = False
        if duration < self._min_speech:
            self._buffer.clear()
            return None
        audio = bytes(self._buffer)
        self._buffer.clear()
        log.info("Speech segment: %.1fs", duration)
        return audio


class ConnectionState:
    def __init__(self) -> None:
        self.streaming: bool = False
        self.current_user_id: str = ""
        self.tts_voice: str = config.tts.default_voice
        self.segmenter: SpeechSegmenter = SpeechSegmenter(
            energy_threshold=config.listen.energy_threshold,
            silence_duration=config.listen.silence_duration,
            min_speech_duration=config.listen.min_speech_duration,
            max_speech_duration=config.listen.max_speech_duration,
        )
        self.frame_count: int = 0
        self.last_log_time: float = 0


@app.websocket("/ws")
async def websocket_handler(ws: WebSocket) -> None:
    await ws.accept()
    state = ConnectionState()
    log.info("WebSocket client connected")

    # Background task: check for speech timeout (Discord sends no packets during silence)
    async def _check_segmenter_timeout():
        while True:
            await asyncio.sleep(0.3)
            if not state.streaming:
                continue
            completed = state.segmenter.check_timeout()
            if completed:
                asyncio.create_task(
                    _handle_transcription(ws, state.current_user_id, completed)
                )

    timeout_task = asyncio.create_task(_check_segmenter_timeout())

    try:
        while True:
            raw = await ws.receive()

            if "bytes" in raw and raw["bytes"]:
                await _handle_audio_frame(ws, state, raw["bytes"])
                continue

            if "text" in raw and raw["text"]:
                try:
                    msg = json.loads(raw["text"])
                except json.JSONDecodeError:
                    await _send_error(ws, "Invalid JSON")
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "stt_start":
                    state.streaming = True
                    state.current_user_id = msg.get("user_id", "unknown")
                    state.segmenter = SpeechSegmenter(
                        energy_threshold=config.listen.energy_threshold,
                        silence_duration=config.listen.silence_duration,
                        min_speech_duration=config.listen.min_speech_duration,
                        max_speech_duration=config.listen.max_speech_duration,
                    )
                    log.info("STT start: user=%s, threshold=%.0f, silence=%.1fs",
                             state.current_user_id, config.listen.energy_threshold,
                             config.listen.silence_duration)

                elif msg_type == "stt_stop":
                    state.streaming = False
                    log.info("STT stop")

                elif msg_type == "tts":
                    text = msg.get("text", "")
                    voice = msg.get("voice", state.tts_voice)
                    if text:
                        asyncio.create_task(_handle_tts(ws, text, voice))

                elif msg_type == "set_voice":
                    voice = msg.get("voice", "")
                    if voice:
                        state.tts_voice = voice
                        tts.set_voice(voice)

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as e:
        log.error("WebSocket error: %s", e, exc_info=True)
    finally:
        timeout_task.cancel()


async def _handle_audio_frame(ws: WebSocket, state: ConnectionState, data: bytes) -> None:
    if not state.streaming:
        return

    audio_16k = pcm_48k_stereo_to_16k_mono(data)
    completed = state.segmenter.feed(audio_16k)

    if completed:
        asyncio.create_task(
            _handle_transcription(ws, state.current_user_id, completed)
        )


async def _handle_transcription(ws: WebSocket, user_id: str, audio: bytes) -> None:
    try:
        text = await stt.transcribe(audio)
        text = text.strip()
        # Filter out noise/filler that Whisper hallucinates on short audio
        if text and len(text) > 1 and text.lower() not in (
            "you", "thanks", "thank you", "bye", "hmm", "um", "uh",
            ".", "...", "thank you for watching.", "thanks for watching.",
        ):
            await ws.send_text(json.dumps({
                "type": "transcription",
                "text": text,
                "user_id": user_id,
                "is_final": True,
            }))
            log.info("Transcription: user=%s text=%r", user_id, text[:80])
        else:
            log.debug("Filtered: %r", text)
    except Exception as e:
        log.error("Transcription failed: %s", e, exc_info=True)
        await _send_error(ws, f"Transcription failed: {e}")


async def _handle_tts(ws: WebSocket, text: str, voice: str) -> None:
    try:
        await ws.send_text(json.dumps({"type": "tts_start", "sample_rate": 48000, "channels": 2}))

        async for chunk in tts.synthesize_chunked(text, voice=voice):
            await ws.send_bytes(chunk)

        await ws.send_text(json.dumps({"type": "tts_done"}))
        log.info("TTS complete: %r", text[:80])

    except Exception as e:
        log.error("TTS failed: %s", e, exc_info=True)
        await _send_error(ws, f"TTS failed: {e}")


async def _send_error(ws: WebSocket, message: str) -> None:
    try:
        await ws.send_text(json.dumps({"type": "error", "message": message}))
    except Exception:
        pass
