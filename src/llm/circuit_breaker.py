"""Lightweight circuit breaker for API provider health tracking.

Prevents cascading failures when an API provider is down by failing fast
instead of waiting through retries on every request. After a configurable
number of consecutive failures, the breaker "opens" and rejects requests
immediately for a recovery period, then allows a single probe request to
check if the provider has recovered.

States:
- closed: Requests flow normally. Failures increment a counter.
- open: Requests fail immediately with CircuitOpenError. Transitions to
  half_open after recovery_timeout seconds.
- half_open: One probe request is allowed through. Success → closed,
  failure → open (with reset recovery timer).
"""
from __future__ import annotations

import threading
import time


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open and requests should not be attempted."""

    def __init__(self, provider: str, retry_after: float) -> None:
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(
            f"{provider} is temporarily unavailable (retry in {retry_after:.0f}s)"
        )


class CircuitBreaker:
    """Simple circuit breaker for API provider health tracking.

    Thread-safe: state mutations are protected by a lock so that concurrent
    callers of record_success/record_failure do not race.

    Parameters
    ----------
    name : str
        Human-readable provider name (e.g. "codex_api").
    failure_threshold : int
        Consecutive failures before the breaker opens. Default 3.
    recovery_timeout : float
        Seconds to wait in open state before allowing a probe. Default 60.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._state = "closed"
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Current breaker state, accounting for recovery timeout expiry."""
        with self._lock:
            if self._state == "open":
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    return "half_open"
            return self._state

    def check(self) -> None:
        """Raise CircuitOpenError if the breaker is open.

        Call before making an API request. Does nothing when closed or
        half_open (allowing a probe request through).
        """
        with self._lock:
            current = self._state
            if current == "open":
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed < self.recovery_timeout:
                    remaining = self.recovery_timeout - elapsed
                    raise CircuitOpenError(self.name, max(0.0, remaining))
                # Recovery timeout passed — allow probe (half_open)

    def record_success(self) -> None:
        """Record a successful API call. Resets failure count, closes breaker."""
        with self._lock:
            self._failure_count = 0
            self._state = "closed"

    def record_failure(self) -> None:
        """Record a failed API call. Opens breaker after threshold is reached."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = "open"
