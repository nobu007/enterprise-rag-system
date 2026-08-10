"""
Concurrency control module for managing concurrent request processing

This module provides semaphore-based concurrency control to limit the number
of simultaneous requests and prevent resource exhaustion.
"""

import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ConcurrencyStats:
    """Statistics for concurrency tracking"""
    total_requests: int = 0
    active_requests: int = 0
    rejected_requests: int = 0
    completed_requests: int = 0
    peak_concurrent: int = 0
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            "total_requests": self.total_requests,
            "active_requests": self.active_requests,
            "rejected_requests": self.rejected_requests,
            "completed_requests": self.completed_requests,
            "peak_concurrent": self.peak_concurrent,
            "last_reset": self.last_reset.isoformat(),
        }


class ConcurrencyLimiter:
    """
    Semaphore-based concurrency limiter for controlling simultaneous requests.

    This class uses asyncio.Semaphore to limit the number of concurrent
    operations and prevent resource exhaustion.

    Attributes:
        max_concurrent: Maximum number of concurrent requests allowed
        semaphore: Asyncio semaphore for concurrency control
        stats: Concurrency statistics tracking

    Example:
        ```python
        limiter = ConcurrencyLimiter(max_concurrent=10)

        async def handle_request():
            async with limiter:
                # Your request handling logic here
                await process_request()
        ```
    """

    def __init__(self, max_concurrent: int = 10):
        """
        Initialize the concurrency limiter.

        Args:
            max_concurrent: Maximum number of concurrent requests (default: 10)

        Raises:
            ValueError: If max_concurrent is less than 1
        """
        if max_concurrent < 1:
            raise ValueError(
                f"max_concurrent must be at least 1, got {max_concurrent}"
            )

        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.stats = ConcurrencyStats()
        self._lock = asyncio.Lock()

        logger.info(
            f"ConcurrencyLimiter initialized with max_concurrent={max_concurrent}"
        )

    async def __aenter__(self):
        """
        Acquire semaphore and update statistics.

        Returns:
            Self for context manager usage

        Raises:
            ConcurrencyLimitExceeded: If limit would be exceeded (rare, handled by semaphore)
        """
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release semaphore and update statistics"""
        await self.release()

    async def acquire(self) -> bool:
        """
        Acquire a slot for concurrent processing.

        This method will block until a slot is available.

        Returns:
            True if slot was acquired successfully

        Example:
            ```python
            limiter = ConcurrencyLimiter(max_concurrent=5)
            acquired = await limiter.acquire()
            if acquired:
                try:
                    # Process request
                    pass
                finally:
                    await limiter.release()
            ```
        """
        await self.semaphore.acquire()

        async with self._lock:
            self.stats.total_requests += 1
            self.stats.active_requests += 1

            # Update peak concurrent
            if self.stats.active_requests > self.stats.peak_concurrent:
                self.stats.peak_concurrent = self.stats.active_requests

            logger.debug(
                f"Acquired concurrency slot: {self.stats.active_requests}/{self.max_concurrent} active"
            )

        return True

    async def release(self):
        """
        Release a slot after processing is complete.

        Example:
            ```python
            await limiter.release()
            ```
        """
        async with self._lock:
            self.stats.active_requests -= 1
            self.stats.completed_requests += 1

            logger.debug(
                f"Released concurrency slot: {self.stats.active_requests}/{self.max_concurrent} active"
            )

        self.semaphore.release()

    async def wait_for_slot(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for a slot to become available with optional timeout.

        Args:
            timeout: Maximum time to wait in seconds (None = wait indefinitely)

        Returns:
            True if slot was acquired, False if timeout was reached

        Example:
            ```python
            limiter = ConcurrencyLimiter(max_concurrent=5)
            acquired = await limiter.wait_for_slot(timeout=5.0)
            if acquired:
                try:
                    # Process request
                    pass
                finally:
                    await limiter.release()
            else:
                # Handle timeout
                logger.warning("Timeout waiting for concurrency slot")
            ```
        """
        try:
            if timeout is None:
                await self.acquire()
                return True
            else:
                await asyncio.wait_for(self.acquire(), timeout=timeout)
                return True
        except asyncio.TimeoutError:
            async with self._lock:
                self.stats.rejected_requests += 1
            logger.warning(f"Timeout waiting for concurrency slot after {timeout}s")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current concurrency statistics.

        Returns:
            Dictionary containing concurrency stats

        Example:
            ```python
            limiter = ConcurrencyLimiter(max_concurrent=10)
            stats = limiter.get_stats()
            print(f"Active requests: {stats['active_requests']}")
            print(f"Peak concurrent: {stats['peak_concurrent']}")
            ```
        """
        return self.stats.to_dict()

    def reset_stats(self):
        """
        Reset concurrency statistics.

        Example:
            ```python
            limiter = ConcurrencyLimiter(max_concurrent=10)
            limiter.reset_stats()
            ```
        """
        self.stats = ConcurrencyStats()
        logger.info("Concurrency statistics reset")

    @property
    def available_slots(self) -> int:
        """
        Get the number of available slots.

        Returns:
            Number of available slots for concurrent processing

        Example:
            ```python
            limiter = ConcurrencyLimiter(max_concurrent=10)
            available = limiter.available_slots
            print(f"Available slots: {available}")
            ```
        """
        return self.max_concurrent - self.stats.active_requests

    @property
    def utilization(self) -> float:
        """
        Get current utilization ratio.

        Returns:
            Utilization ratio between 0.0 and 1.0

        Example:
            ```python
            limiter = ConcurrencyLimiter(max_concurrent=10)
            utilization = limiter.utilization
            print(f"Utilization: {utilization * 100:.1f}%")
            ```
        """
        return self.stats.active_requests / self.max_concurrent if self.max_concurrent > 0 else 0.0


class ConcurrencyLimitExceeded(Exception):
    """
    Exception raised when concurrency limit would be exceeded.

    This exception is typically handled internally by the semaphore,
    but can be raised in specific error conditions.
    """

    def __init__(self, message: str = "Concurrency limit exceeded"):
        self.message = message
        super().__init__(self.message)


# Global limiter instance (can be configured at application startup)
_global_limiter: Optional[ConcurrencyLimiter] = None


def get_concurrency_limiter(max_concurrent: int = 10) -> ConcurrencyLimiter:
    """
    Get or create the global concurrency limiter.

    Args:
        max_concurrent: Maximum concurrent requests (only used on first call)

    Returns:
        Global ConcurrencyLimiter instance

    Example:
        ```python
        # In application startup
        limiter = get_concurrency_limiter(max_concurrent=20)

        # In request handlers
        async with get_concurrency_limiter():
            # Process request
            pass
        ```
    """
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = ConcurrencyLimiter(max_concurrent=max_concurrent)
    return _global_limiter


def set_concurrency_limiter(limiter: ConcurrencyLimiter):
    """
    Set the global concurrency limiter.

    Args:
        limiter: ConcurrencyLimiter instance to use as global

    Example:
        ```python
        custom_limiter = ConcurrencyLimiter(max_concurrent=15)
        set_concurrency_limiter(custom_limiter)
        ```
    """
    global _global_limiter
    _global_limiter = limiter
    logger.info(f"Global concurrency limiter set to max_concurrent={limiter.max_concurrent}")
