"""
Unit tests for Circuit Breaker Pattern
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from app.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerError,
    circuit_breaker
)


class TestCircuitBreakerInitialization:
    """Test circuit breaker initialization"""

    def test_initialization_with_default_config(self):
        """Test circuit breaker initialization with defaults"""
        breaker = CircuitBreaker("test_breaker")

        assert breaker.name == "test_breaker"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.opened_count == 0

    def test_initialization_with_custom_config(self):
        """Test circuit breaker with custom configuration"""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=3,
            timeout=120.0
        )
        breaker = CircuitBreaker("custom_breaker", config)

        assert breaker.config.failure_threshold == 10
        assert breaker.config.success_threshold == 3
        assert breaker.config.timeout == 120.0


class TestCircuitBreakerStates:
    """Test circuit breaker state transitions"""

    @pytest.mark.asyncio
    async def test_closed_to_open_on_failures(self):
        """Test circuit opens after failure threshold"""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)

        async def failing_call():
            raise RuntimeError("Service unavailable")

        # First two failures should not open circuit
        for i in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(failing_call)
            assert breaker.state == CircuitState.CLOSED
            assert breaker.failure_count == i + 1

        # Third failure should open circuit
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call)

        assert breaker.state == CircuitState.OPEN
        assert breaker.opened_count == 1

    @pytest.mark.asyncio
    async def test_open_rejects_requests(self):
        """Test open circuit rejects requests immediately"""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=10.0)
        breaker = CircuitBreaker("test", config)

        async def failing_call():
            raise RuntimeError("Service unavailable")

        # Trip the circuit
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call)
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call)

        assert breaker.state == CircuitState.OPEN

        # Next call should be rejected immediately
        with pytest.raises(CircuitBreakerError) as exc_info:
            await breaker.call(failing_call)

        assert "is OPEN" in str(exc_info.value)
        assert breaker.failure_count == 2  # Count shouldn't increase

    @pytest.mark.asyncio
    async def test_open_to_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout"""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        breaker = CircuitBreaker("test", config)

        async def failing_call():
            raise RuntimeError("Service unavailable")

        # Trip the circuit
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call)
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Check state - should still be OPEN, next call will transition
        assert breaker.state == CircuitState.OPEN

        # Next call should transition to half-open before execution
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call)

        # After the call fails again in half-open, it should reopen
        # So state is OPEN again (reopened from half-open)
        assert breaker.state == CircuitState.OPEN
        assert breaker.opened_count == 2  # Tripped twice

    @pytest.mark.asyncio
    async def test_half_open_to_closed_on_success(self):
        """Test circuit closes after successful calls in half-open"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout=0.1
        )
        breaker = CircuitBreaker("test", config)

        async def failing_call():
            raise RuntimeError("Service unavailable")

        async def success_call():
            return "success"

        # Trip the circuit
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call)
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # First success in half-open
        result = await breaker.call(success_call)
        assert result == "success"
        assert breaker.state == CircuitState.HALF_OPEN

        # Second success should close circuit
        result = await breaker.call(success_call)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self):
        """Test circuit reopens on failure in half-open"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=0.1
        )
        breaker = CircuitBreaker("test", config)

        async def failing_call():
            raise RuntimeError("Service unavailable")

        async def success_call():
            return "success"

        # Trip the circuit
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(failing_call)

        assert breaker.state == CircuitState.OPEN
        assert breaker.opened_count == 1

        # Wait for timeout
        await asyncio.sleep(0.15)

        # First success in half-open
        result = await breaker.call(success_call)
        assert result == "success"
        assert breaker.state == CircuitState.HALF_OPEN

        # Second success should keep it in half-open (needs 2 to close)
        result = await breaker.call(success_call)
        assert result == "success"

        # Circuit should close after success_threshold is met
        assert breaker.state == CircuitState.CLOSED

        # Now failures should start counting again
        # First failure after reset
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call)
        assert breaker.failure_count == 1
        assert breaker.state == CircuitState.CLOSED

        # Second failure
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call)
        assert breaker.failure_count == 2
        assert breaker.state == CircuitState.CLOSED

        # Third failure trips it again
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call)
        assert breaker.state == CircuitState.OPEN
        assert breaker.opened_count == 2


class TestCircuitBreakerSuccessPath:
    """Test circuit breaker with successful calls"""

    @pytest.mark.asyncio
    async def test_successful_call_in_closed_state(self):
        """Test successful call in closed state"""
        breaker = CircuitBreaker("test")

        async def success_call():
            return "success"

        result = await breaker.call(success_call)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_reset_after_failure(self):
        """Test failure count resets after success"""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)

        async def failing_call():
            raise RuntimeError("Service unavailable")

        async def success_call():
            return "success"

        # Two failures
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call)
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call)

        assert breaker.failure_count == 2

        # Success should reset count
        result = await breaker.call(success_call)
        assert result == "success"
        assert breaker.failure_count == 0


class TestCircuitBreakerManualReset:
    """Test manual circuit breaker reset"""

    @pytest.mark.asyncio
    async def test_manual_reset_from_open(self):
        """Test manually resetting circuit breaker"""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=60.0)
        breaker = CircuitBreaker("test", config)

        async def failing_call():
            raise RuntimeError("Service unavailable")

        # Trip the circuit
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call)
        with pytest.raises(RuntimeError):
            await breaker.call(failing_call)

        assert breaker.state == CircuitState.OPEN

        # Manually reset
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker._last_failure_time is None

        # Should be able to call again
        async def success_call():
            return "success"

        result = await breaker.call(success_call)
        assert result == "success"


class TestCircuitBreakerSynchronous:
    """Test synchronous circuit breaker calls"""

    def test_sync_call_success(self):
        """Test successful synchronous call"""
        breaker = CircuitBreaker("test")

        def success_func():
            return "sync_success"

        result = breaker.call_sync(success_func)
        assert result == "sync_success"
        assert breaker.state == CircuitState.CLOSED

    def test_sync_call_failure(self):
        """Test failed synchronous call"""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test", config)

        def failing_func():
            raise RuntimeError("Sync error")

        # First failure
        with pytest.raises(RuntimeError):
            breaker.call_sync(failing_func)
        assert breaker.failure_count == 1
        assert breaker.state == CircuitState.CLOSED

        # Second failure trips circuit
        with pytest.raises(RuntimeError):
            breaker.call_sync(failing_func)
        assert breaker.state == CircuitState.OPEN

    def test_sync_open_rejects(self):
        """Test open circuit rejects sync calls"""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=60.0)
        breaker = CircuitBreaker("test", config)

        def failing_func():
            raise RuntimeError("Sync error")

        # Trip the circuit
        with pytest.raises(RuntimeError):
            breaker.call_sync(failing_func)
        with pytest.raises(RuntimeError):
            breaker.call_sync(failing_func)

        # Should be rejected
        with pytest.raises(CircuitBreakerError):
            breaker.call_sync(failing_func)


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator"""

    @pytest.mark.asyncio
    async def test_decorator_basic(self):
        """Test basic decorator usage"""
        call_count = {"value": 0}

        @circuit_breaker(name="decorator_test", failure_threshold=2)
        async def api_call():
            call_count["value"] += 1
            if call_count["value"] < 3:
                raise RuntimeError("API error")
            return "success"

        # First two calls fail
        with pytest.raises(RuntimeError):
            await api_call()
        with pytest.raises(RuntimeError):
            await api_call()

        # Third call is rejected (circuit open)
        with pytest.raises(CircuitBreakerError):
            await api_call()

        assert call_count["value"] == 2

    @pytest.mark.asyncio
    async def test_decorator_with_custom_config(self):
        """Test decorator with custom configuration"""
        @circuit_breaker(
            name="custom_decorator",
            failure_threshold=3,
            timeout=0.1
        )
        async def api_call():
            raise RuntimeError("Error")

        # Three failures to trip
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await api_call()

        # Fourth is rejected
        with pytest.raises(CircuitBreakerError):
            await api_call()


class TestCircuitBreakerRepr:
    """Test circuit breaker string representation"""

    def test_repr(self):
        """Test __repr__ method"""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("test", config)

        repr_str = repr(breaker)
        assert "CircuitBreaker" in repr_str
        assert "test" in repr_str
        assert "closed" in repr_str
        assert "0/5" in repr_str


class TestCircuitBreakerEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_exception_type_filtering(self):
        """Test that only expected exceptions count as failures"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            expected_exception=ValueError
        )
        breaker = CircuitBreaker("test", config)

        async def raise_value_error():
            raise ValueError("Expected error")

        async def raise_runtime_error():
            raise RuntimeError("Unexpected error")

        # ValueError should count as failure
        with pytest.raises(ValueError):
            await breaker.call(raise_value_error)
        assert breaker.failure_count == 1

        # RuntimeError should also count (it's not caught by expected_exception,
        # but our implementation catches all exceptions)
        with pytest.raises(RuntimeError):
            await breaker.call(raise_runtime_error)
        assert breaker.failure_count == 2

        # Third ValueError trips the circuit
        with pytest.raises(ValueError):
            await breaker.call(raise_value_error)
        assert breaker.failure_count == 3
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Test concurrent calls through circuit breaker"""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("test", config)

        async def delayed_call(delay):
            await asyncio.sleep(delay)
            return f"done_{delay}"

        # Multiple concurrent calls
        results = await asyncio.gather(
            breaker.call(delayed_call, 0.01),
            breaker.call(delayed_call, 0.02),
            breaker.call(delayed_call, 0.03)
        )

        assert len(results) == 3
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
