from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class WhisperConfig(BaseModel):
    model_size: str = "medium"
    device: str = "cuda"
    compute_type: str = "float16"
    idle_unload_seconds: int = 300  # 5 minutes
    vad_filter: bool = True
    language: str = "en"


class TtsConfig(BaseModel):
    default_voice: str = "en_US-lessac-medium"
    voices_directory: str = "./voices"
    output_sample_rate: int = 48000
    output_channels: int = 2


class WakeWordConfig(BaseModel):
    model_path: str = ""  # empty = use built-in "hey_jarvis" as fallback
    threshold: float = 0.5
    inference_framework: str = "onnx"


class ListenConfig(BaseModel):
    mode: str = "continuous"  # "continuous" or "wake_word"
    energy_threshold: float = 300
    silence_duration: float = 1.5
    min_speech_duration: float = 0.5
    max_speech_duration: float = 30.0


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 3940
    log_level: str = "info"


class VoiceServiceConfig(BaseModel):
    whisper: WhisperConfig = WhisperConfig()
    tts: TtsConfig = TtsConfig()
    wake_word: WakeWordConfig = WakeWordConfig()
    listen: ListenConfig = ListenConfig()
    server: ServerConfig = ServerConfig()


def load_config(path: str | Path = "config.yml") -> VoiceServiceConfig:
    path = Path(path)
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        return VoiceServiceConfig(**data)
    return VoiceServiceConfig()
