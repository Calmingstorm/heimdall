"""Wake word detection using openwakeword.

Runs on CPU, lightweight (~10ms per inference). Tracks per-user state
for multi-user voice channels.
"""

from __future__ import annotations

import logging
import time
from enum import Enum, auto

import numpy as np

from .config import WakeWordConfig

log = logging.getLogger("voice.wake_word")


class UserState(Enum):
    IDLE = auto()        # Listening for wake word only
    LISTENING = auto()   # Wake word detected, buffering for Whisper
    PROCESSING = auto()  # Audio sent to Whisper, waiting for result


class UserSession:
    """Per-user voice state and audio buffer."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.state = UserState.IDLE
        self.audio_buffer = bytearray()
        self.last_speech_time: float = 0
        self.silence_start: float = 0
        # Max buffer: ~30 seconds of 16kHz mono s16le = 960,000 bytes
        self.max_buffer_bytes = 960_000

    def reset(self) -> None:
        self.state = UserState.IDLE
        self.audio_buffer.clear()
        self.last_speech_time = 0
        self.silence_start = 0

    def append_audio(self, data: bytes) -> None:
        self.audio_buffer.extend(data)
        # Prevent unbounded growth
        if len(self.audio_buffer) > self.max_buffer_bytes:
            self.audio_buffer = self.audio_buffer[-self.max_buffer_bytes :]


class WakeWordDetector:
    def __init__(self, config: WakeWordConfig) -> None:
        self._config = config
        self._model = None
        self._users: dict[str, UserSession] = {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        from openwakeword.model import Model

        model_paths = []
        if self._config.model_path:
            model_paths = [self._config.model_path]
            log.info("Loading custom wake word model: %s", self._config.model_path)
        else:
            log.info("Loading default openwakeword models (hey_jarvis)")

        self._model = Model(
            wakeword_models=model_paths if model_paths else [],
            inference_framework=self._config.inference_framework,
        )
        self._initialized = True
        log.info("Wake word detector initialized, models: %s", list(self._model.models.keys()))

    def get_session(self, user_id: str) -> UserSession:
        if user_id not in self._users:
            self._users[user_id] = UserSession(user_id)
        return self._users[user_id]

    def remove_session(self, user_id: str) -> None:
        self._users.pop(user_id, None)

    def detect(self, audio_16k_mono: bytes, user_id: str) -> bool:
        """Check if wake word is detected in audio chunk.

        Args:
            audio_16k_mono: 16kHz mono s16le PCM audio chunk.
            user_id: Discord user ID for per-user state tracking.

        Returns:
            True if wake word was detected.
        """
        self._ensure_initialized()

        session = self.get_session(user_id)

        # Only check for wake word when idle
        if session.state != UserState.IDLE:
            return False

        # Convert to int16 numpy array as expected by openwakeword
        samples = np.frombuffer(audio_16k_mono, dtype=np.int16)

        # openwakeword expects audio in chunks of 1280 samples (80ms at 16kHz)
        # Process in chunks
        chunk_size = 1280
        detected = False

        for i in range(0, len(samples) - chunk_size + 1, chunk_size):
            chunk = samples[i : i + chunk_size]
            prediction = self._model.predict(chunk)

            for model_name, score in prediction.items():
                if score >= self._config.threshold:
                    log.info(
                        "Wake word detected! model=%s score=%.3f user=%s",
                        model_name, score, user_id,
                    )
                    detected = True
                    break

            if detected:
                break

        if detected:
            # Reset the model's internal state after detection
            self._model.reset()

        return detected

    def process_audio(
        self, audio_16k_mono: bytes, user_id: str, silence_threshold: float = 1.5,
    ) -> tuple[UserState, bytes | None]:
        """Process audio for a user, managing wake word → listening → end-of-speech flow.

        Args:
            audio_16k_mono: 16kHz mono s16le PCM audio chunk.
            user_id: Discord user ID.
            silence_threshold: Seconds of silence before considering speech done.

        Returns:
            (new_state, completed_audio_or_None)
            If state transitions to PROCESSING, returns the buffered audio.
        """
        session = self.get_session(user_id)

        if session.state == UserState.IDLE:
            if self.detect(audio_16k_mono, user_id):
                session.state = UserState.LISTENING
                session.audio_buffer.clear()
                session.last_speech_time = time.monotonic()
                session.silence_start = 0
                log.info("User %s: IDLE → LISTENING (wake word)", user_id)
                return UserState.LISTENING, None
            return UserState.IDLE, None

        if session.state == UserState.LISTENING:
            session.append_audio(audio_16k_mono)

            # Simple energy-based speech detection
            samples = np.frombuffer(audio_16k_mono, dtype=np.int16)
            energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

            if energy > 500:  # Speech threshold
                session.last_speech_time = time.monotonic()
                session.silence_start = 0
            else:
                if session.silence_start == 0:
                    session.silence_start = time.monotonic()
                elif time.monotonic() - session.silence_start >= silence_threshold:
                    # End of speech detected
                    audio = bytes(session.audio_buffer)
                    session.state = UserState.PROCESSING
                    log.info(
                        "User %s: LISTENING → PROCESSING (%.1fs audio)",
                        user_id, len(audio) / (16000 * 2),
                    )
                    return UserState.PROCESSING, audio

            # Safety: if we've been listening too long (30s), force end
            if time.monotonic() - session.last_speech_time > 30:
                audio = bytes(session.audio_buffer)
                session.state = UserState.PROCESSING
                log.info("User %s: LISTENING → PROCESSING (timeout)", user_id)
                return UserState.PROCESSING, audio

            return UserState.LISTENING, None

        # PROCESSING state — don't accept more audio
        return session.state, None

    def finish_processing(self, user_id: str) -> None:
        """Reset user state back to IDLE after transcription is complete."""
        session = self.get_session(user_id)
        session.reset()
        log.info("User %s: → IDLE", user_id)
