"""Tests for circuit breaker: core state machine and CodexChatClient integration."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from src.llm.circuit_breaker import CircuitBreaker, CircuitOpenError


# ---------------------------------------------------------------------------
# Core CircuitBreaker state machine
# ---------------------------------------------------------------------------


class TestCircuitBreakerStates:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker("test")
        assert cb.state == "closed"

    def test_check_passes_when_closed(self):
        cb = CircuitBreaker("test")
        cb.check()  # Should not raise

    def test_single_failure_stays_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        assert cb.state == "closed"
        cb.check()  # Should still pass

    def test_opens_after_reaching_failure_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

    def test_check_raises_when_open(self):
        cb = CircuitBreaker("test", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.check()
        assert exc_info.value.provider == "test"
        assert exc_info.value.retry_after >= 0

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Should be reset — two more failures should not open
        cb.record_failure()
        assert cb.state == "closed"

    def test_success_closes_open_breaker(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == "open"
        # Simulate recovery timeout passing
        cb._last_failure_time = time.monotonic() - 999
        assert cb.state == "half_open"
        cb.record_success()
        assert cb.state == "closed"

    def test_transitions_to_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=10.0)
        cb.record_failure()
        assert cb.state == "open"
        # Simulate time passing beyond recovery timeout
        cb._last_failure_time = time.monotonic() - 11.0
        assert cb.state == "half_open"

    def test_half_open_allows_check(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.0)
        cb.record_failure()
        # recovery_timeout=0 means it immediately transitions to half_open
        cb._last_failure_time = time.monotonic() - 1.0
        assert cb.state == "half_open"
        cb.check()  # Should not raise — allows one probe

    def test_failure_in_half_open_reopens(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=10.0)
        cb.record_failure()
        # Advance past recovery timeout to half_open
        cb._last_failure_time = time.monotonic() - 11.0
        assert cb.state == "half_open"
        cb.record_failure()
        assert cb.state == "open"
        # Last failure time was just updated, so still open
        with pytest.raises(CircuitOpenError):
            cb.check()

    def test_circuit_open_error_message(self):
        err = CircuitOpenError("codex_api", 42.0)
        assert "codex_api" in str(err)
        assert "42" in str(err)

    def test_retry_after_is_non_negative(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60.0)
        cb.record_failure()
        try:
            cb.check()
        except CircuitOpenError as e:
            assert e.retry_after >= 0

    def test_custom_threshold_and_timeout(self):
        cb = CircuitBreaker("custom", failure_threshold=5, recovery_timeout=120.0)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == "closed"  # Not yet at threshold
        cb.record_failure()
        assert cb.state == "open"
        assert cb.recovery_timeout == 120.0

    def test_has_threading_lock(self):
        """CircuitBreaker should have a threading.Lock for thread safety."""
        cb = CircuitBreaker("test")
        assert hasattr(cb, "_lock")
        assert isinstance(cb._lock, type(threading.Lock()))

    def test_concurrent_record_failure_thread_safety(self):
        """Concurrent record_failure calls from multiple threads should not lose counts."""
        cb = CircuitBreaker("test", failure_threshold=1000)
        barrier = threading.Barrier(10)
        errors = []

        def worker():
            try:
                barrier.wait(timeout=5)
                for _ in range(100):
                    cb.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        # 10 threads × 100 failures = 1000 total failures
        assert cb._failure_count == 1000

    def test_concurrent_success_and_failure_thread_safety(self):
        """Concurrent record_success and record_failure should not corrupt state."""
        cb = CircuitBreaker("test", failure_threshold=100)
        barrier = threading.Barrier(2)

        def fail_worker():
            barrier.wait(timeout=5)
            for _ in range(50):
                cb.record_failure()

        def success_worker():
            barrier.wait(timeout=5)
            for _ in range(50):
                cb.record_success()

        t1 = threading.Thread(target=fail_worker)
        t2 = threading.Thread(target=success_worker)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # State should be valid (either closed or open, not corrupted)
        assert cb.state in ("closed", "open", "half_open")
        assert cb._failure_count >= 0


# ---------------------------------------------------------------------------
# CodexChatClient integration
# ---------------------------------------------------------------------------


class TestCodexCircuitBreaker:
    def test_codex_client_has_breaker(self):
        """CodexChatClient should have a circuit breaker."""
        from src.llm.openai_codex import CodexChatClient

        auth = MagicMock()
        client = CodexChatClient(auth=auth, model="test", max_tokens=100)
        assert hasattr(client, "breaker")
        assert isinstance(client.breaker, CircuitBreaker)
        assert client.breaker.name == "codex_api"

    async def test_codex_stream_request_checks_breaker(self):
        """_stream_request should raise CircuitOpenError when breaker is open."""
        from src.llm.openai_codex import CodexChatClient

        auth = MagicMock()
        client = CodexChatClient(auth=auth, model="test", max_tokens=100)

        for _ in range(3):
            client.breaker.record_failure()

        with pytest.raises(CircuitOpenError):
            await client._stream_request({}, {})


# ---------------------------------------------------------------------------
# CircuitOpenError is importable from llm package
# ---------------------------------------------------------------------------


class TestCircuitOpenErrorExport:
    def test_importable_from_llm_package(self):
        from src.llm import CircuitOpenError as exported
        assert exported is CircuitOpenError

    def test_importable_from_circuit_breaker_module(self):
        from src.llm.circuit_breaker import CircuitOpenError as direct
        assert direct is CircuitOpenError
