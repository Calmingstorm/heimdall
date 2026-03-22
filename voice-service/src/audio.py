"""Audio resampling utilities.

Handles conversion between Discord's 48kHz stereo PCM and the formats
required by Whisper (16kHz mono) and Piper TTS output → Discord.
"""

from __future__ import annotations

import numpy as np


def pcm_48k_stereo_to_16k_mono(data: bytes) -> bytes:
    """Convert 48kHz stereo s16le PCM to 16kHz mono s16le PCM.

    Steps:
    1. Decode s16le stereo interleaved samples
    2. Average left/right channels → mono
    3. Downsample 48kHz → 16kHz (factor of 3)
    """
    samples = np.frombuffer(data, dtype=np.int16)
    if len(samples) == 0:
        return b""

    # Stereo interleaved: [L, R, L, R, ...] → reshape to (N, 2)
    if len(samples) % 2 != 0:
        samples = samples[: len(samples) - 1]
    stereo = samples.reshape(-1, 2)

    # Average channels to mono (use int32 to avoid overflow)
    mono = stereo.astype(np.int32).mean(axis=1).astype(np.int16)

    # Downsample 48kHz → 16kHz (take every 3rd sample)
    downsampled = mono[::3]

    return downsampled.tobytes()


def pcm_16k_mono_to_48k_stereo(data: bytes) -> bytes:
    """Convert 16kHz mono s16le PCM to 48kHz stereo s16le PCM.

    Steps:
    1. Upsample 16kHz → 48kHz (repeat each sample 3x)
    2. Duplicate mono → stereo interleaved
    """
    samples = np.frombuffer(data, dtype=np.int16)
    if len(samples) == 0:
        return b""

    # Upsample: repeat each sample 3 times
    upsampled = np.repeat(samples, 3)

    # Mono → stereo interleaved: [S, S, S, S, ...] → [S, S, S, S, ...]
    stereo = np.stack([upsampled, upsampled], axis=1).flatten()

    return stereo.astype(np.int16).tobytes()


def pcm_to_float32(data: bytes) -> np.ndarray:
    """Convert s16le PCM bytes to float32 array in [-1.0, 1.0] range."""
    samples = np.frombuffer(data, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def float32_to_pcm(data: np.ndarray) -> bytes:
    """Convert float32 array in [-1.0, 1.0] range to s16le PCM bytes."""
    clipped = np.clip(data, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def resample_linear(data: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Simple linear interpolation resampling for arbitrary rate conversion."""
    if from_rate == to_rate:
        return data
    ratio = to_rate / from_rate
    new_len = int(len(data) * ratio)
    indices = np.linspace(0, len(data) - 1, new_len)
    return np.interp(indices, np.arange(len(data)), data).astype(data.dtype)
