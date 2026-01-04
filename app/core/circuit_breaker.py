"""
Circuit Breaker Pattern Implementation

This module implements the circuit breaker pattern to prevent cascading failures
when calling external services. It provides fault tolerance and graceful degradation.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Circuit is tripped, requests fail immediately
- HALF_OPEN: Testing if service has recovered
"""

import time
import asyncio
from enum import Enum
from typing import Callable, Optional, Any, TypeVar
from functools import wraps
from dataclasses import dataclass

from app.core.logging_config import get_logger
from app.core import metrics

logger = get_logger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit tripped, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5        # Failures before tripping
    success_threshold: int = 2        # Successes to close circuit
    timeout: float = 60.0             # Seconds before attempting recovery
    expected_exception: Exception = Exception  # Exception type to catch


class CircuitBreakerError(Exception):
    """Raised when circuit is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for external service calls.

    Usage:
        breaker = CircuitBreaker(
            name="openai_api",
            failure_threshold=5,
            timeout=60.0
        )

        @breaker.call
        async def external_api_call():
            # Your API call here
            pass
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker

        Args:
            name: Unique identifier for this circuit breaker
            config: Configuration options
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._opened_count = 0  # Track how many times circuit opened

        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"timeout={self.config.timeout}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current state"""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count"""
        return self._failure_count

    @property
    def opened_count(self) -> int:
        """Get number of times circuit has been opened"""
        return self._opened_count

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self._last_failure_time is None:
            return False

        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.config.timeout

    def _on_success(self):
        """Handle successful call"""
        self._failure_count = 0
        self._success_count += 1

        if self._state == CircuitState.HALF_OPEN:
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._success_count = 0
                logger.info(f"Circuit breaker '{self.name}' reset to CLOSED state")
                metrics.circuit_breaker_state_change.labels(
                    name=self.name,
                    from_state=CircuitState.HALF_OPEN.value,
                    to_state=CircuitState.CLOSED.value
                ).inc()

    def _on_failure(self):
        """Handle failed call"""
        self._failure_count += 1
        self._success_count = 0
        self._last_failure_time = time.time()

        if self._failure_count >= self.config.failure_threshold:
            if self._state != CircuitState.OPEN:
                self._state = CircuitState.OPEN
                self._opened_count += 1
                logger.warning(
                    f"Circuit breaker '{self.name}' tripped to OPEN state "
                    f"after {self._failure_count} failures"
                )
                metrics.circuit_breaker_state_change.labels(
                    name=self.name,
                    from_state=CircuitState.CLOSED.value,
                    to_state=CircuitState.OPEN.value
                ).inc()

    def _call_before(self):
        """Check circuit state before making call"""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(
                    f"Circuit breaker '{self.name}' attempting recovery "
                    f"(timeout elapsed)"
                )
                self._state = CircuitState.HALF_OPEN
                metrics.circuit_breaker_state_change.labels(
                    name=self.name,
                    from_state=CircuitState.OPEN.value,
                    to_state=CircuitState.HALF_OPEN.value
                ).inc()
            else:
                metrics.circuit_breaker_rejected.labels(name=self.name).inc()
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN - "
                    f"rejecting request"
                )

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If function raises exception
        """
        self._call_before()

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            metrics.circuit_breaker_failure.labels(
                name=self.name,
                exception_type=type(e).__name__
            ).inc()
            raise
        except Exception as e:
            # Unexpected exceptions still count as failures
            self._on_failure()
            raise

    def call_sync(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute synchronous function through circuit breaker

        Args:
            func: Synchronous function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        self._call_before()

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            metrics.circuit_breaker_failure.labels(
                name=self.name,
                exception_type=type(e).__name__
            ).inc()
            raise
        except Exception as e:
            self._on_failure()
            raise

    def reset(self):
        """Manually reset circuit breaker to closed state"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        logger.info(f"Circuit breaker '{self.name}' manually reset")

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name='{self.name}', "
            f"state={self._state.value}, "
            f"failures={self._failure_count}/{self.config.failure_threshold})"
        )


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout: float = 60.0,
    expected_exception: Exception = Exception
):
    """
    Decorator for applying circuit breaker to async functions

    Args:
        name: Circuit breaker name
        failure_threshold: Failures before tripping
        success_threshold: Successes to close circuit
        timeout: Seconds before attempting recovery
        expected_exception: Exception type to catch

    Returns:
        Decorated function

    Example:
        @circuit_breaker(name="api_call", failure_threshold=3)
        async def my_api_call():
            return await external_service.call()
    """
    # Global registry of circuit breakers
    _registry = {}

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Get or create circuit breaker
        if name not in _registry:
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                timeout=timeout,
                expected_exception=expected_exception
            )
            _registry[name] = CircuitBreaker(name, config)

        breaker = _registry[name]

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        # Attach circuit breaker to function for access
        wrapper.circuit_breaker = breaker
        return wrapper

    return decorator
