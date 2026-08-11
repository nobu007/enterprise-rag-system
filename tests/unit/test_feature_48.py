"""
Unit tests for concurrent request handling (Feature 48)

Tests cover:
- ConcurrencyLimiter initialization and validation
- Semaphore-based concurrent request processing
- Statistics tracking and reporting
- Context manager usage
- Timeout handling
- Error handling and edge cases
- Global limiter instance management
"""

import pytest
import asyncio
from datetime import datetime, timezone

from app.core.concurrency import (
    ConcurrencyLimiter,
    ConcurrencyStats,
    ConcurrencyLimitExceeded,
    get_concurrency_limiter,
    set_concurrency_limiter,
)


# ============================================================================
# Test ConcurrencyStats
# ============================================================================

class TestConcurrencyStats:
    """Test ConcurrencyStats dataclass"""

    def test_create_stats_with_defaults(self):
        """Test creating stats with default values"""
        stats = ConcurrencyStats()

        assert stats.total_requests == 0
        assert stats.active_requests == 0
        assert stats.rejected_requests == 0
        assert stats.completed_requests == 0
        assert stats.peak_concurrent == 0
        assert isinstance(stats.last_reset, datetime)

    def test_create_stats_with_values(self):
        """Test creating stats with specific values"""
        now = datetime.now(timezone.utc)
        stats = ConcurrencyStats(
            total_requests=100,
            active_requests=5,
            rejected_requests=2,
            completed_requests=95,
            peak_concurrent=10,
            last_reset=now
        )

        assert stats.total_requests == 100
        assert stats.active_requests == 5
        assert stats.rejected_requests == 2
        assert stats.completed_requests == 95
        assert stats.peak_concurrent == 10
        assert stats.last_reset == now

    def test_to_dict(self):
        """Test converting stats to dictionary"""
        stats = ConcurrencyStats(
            total_requests=50,
            active_requests=3,
            rejected_requests=1,
            completed_requests=47,
            peak_concurrent=8
        )

        result = stats.to_dict()

        assert isinstance(result, dict)
        assert result["total_requests"] == 50
        assert result["active_requests"] == 3
        assert result["rejected_requests"] == 1
        assert result["completed_requests"] == 47
        assert result["peak_concurrent"] == 8
        assert "last_reset" in result
        assert isinstance(result["last_reset"], str)


# ============================================================================
# Test ConcurrencyLimiter Initialization
# ============================================================================

class TestConcurrencyLimiterInit:
    """Test ConcurrencyLimiter initialization"""

    def test_init_with_valid_max_concurrent(self):
        """Test initialization with valid max_concurrent value"""
        limiter = ConcurrencyLimiter(max_concurrent=5)

        assert limiter.max_concurrent == 5
        assert limiter.stats.total_requests == 0
        assert limiter.stats.active_requests == 0
        assert isinstance(limiter.semaphore, asyncio.Semaphore)

    def test_init_with_default_max_concurrent(self):
        """Test initialization with default max_concurrent value"""
        limiter = ConcurrencyLimiter()

        assert limiter.max_concurrent == 10

    def test_init_with_invalid_max_concurrent_zero(self):
        """Test that max_concurrent=0 raises ValueError"""
        with pytest.raises(ValueError, match="max_concurrent must be at least 1"):
            ConcurrencyLimiter(max_concurrent=0)

    def test_init_with_invalid_max_concurrent_negative(self):
        """Test that negative max_concurrent raises ValueError"""
        with pytest.raises(ValueError, match="max_concurrent must be at least 1"):
            ConcurrencyLimiter(max_concurrent=-1)


# ============================================================================
# Test ConcurrencyLimiter Context Manager
# ============================================================================

class TestConcurrencyLimiterContextManager:
    """Test ConcurrencyLimiter context manager usage"""

    @pytest.mark.asyncio
    async def test_context_manager_acquire_and_release(self):
        """Test acquiring and releasing via context manager"""
        limiter = ConcurrencyLimiter(max_concurrent=2)

        async with limiter:
            assert limiter.stats.active_requests == 1
            assert limiter.stats.total_requests == 1

        assert limiter.stats.active_requests == 0
        assert limiter.stats.completed_requests == 1

    @pytest.mark.asyncio
    async def test_context_manager_multiple_concurrent(self):
        """Test multiple concurrent context managers"""
        limiter = ConcurrencyLimiter(max_concurrent=3)

        async def task():
            async with limiter:
                await asyncio.sleep(0.01)
                return limiter.stats.active_requests

        # Run 3 concurrent tasks
        results = await asyncio.gather(*[task() for _ in range(3)])

        # All should complete
        assert len(results) == 3
        assert limiter.stats.completed_requests == 3
        assert limiter.stats.peak_concurrent == 3

    @pytest.mark.asyncio
    async def test_context_manager_blocks_when_full(self):
        """Test that context manager blocks when limit is reached"""
        limiter = ConcurrencyLimiter(max_concurrent=1)

        execution_order = []
        task2_started = asyncio.Event()

        async def task(name):
            execution_order.append(f"{name}-start")
            if name == "task2":
                task2_started.set()
            async with limiter:
                execution_order.append(f"{name}-acquired")
                await asyncio.sleep(0.01)
            execution_order.append(f"{name}-released")

        # Start first task
        task1 = asyncio.create_task(task("task1"))
        await asyncio.sleep(0.001)  # Let task1 acquire

        # Start second task (should block)
        task2 = asyncio.create_task(task("task2"))

        # Wait for task2 to start
        await task2_started.wait()

        await asyncio.gather(task1, task2)

        # Verify task1 acquired before task2
        task1_acquired_idx = execution_order.index("task1-acquired")
        task2_acquired_idx = execution_order.index("task2-acquired")
        assert task1_acquired_idx < task2_acquired_idx

        # Verify both completed
        assert "task1-released" in execution_order
        assert "task2-released" in execution_order


# ============================================================================
# Test ConcurrencyLimiter Acquire and Release
# ============================================================================

class TestConcurrencyLimiterAcquireRelease:
    """Test ConcurrencyLimiter acquire and release methods"""

    @pytest.mark.asyncio
    async def test_acquire_increments_stats(self):
        """Test that acquire increments statistics"""
        limiter = ConcurrencyLimiter(max_concurrent=5)

        await limiter.acquire()

        assert limiter.stats.active_requests == 1
        assert limiter.stats.total_requests == 1

    @pytest.mark.asyncio
    async def test_release_decrements_active(self):
        """Test that release decrements active requests"""
        limiter = ConcurrencyLimiter(max_concurrent=5)

        await limiter.acquire()
        assert limiter.stats.active_requests == 1

        await limiter.release()
        assert limiter.stats.active_requests == 0
        assert limiter.stats.completed_requests == 1

    @pytest.mark.asyncio
    async def test_acquire_release_balance(self):
        """Test that acquire and release maintain balance"""
        limiter = ConcurrencyLimiter(max_concurrent=10)

        for _ in range(5):
            await limiter.acquire()

        assert limiter.stats.active_requests == 5

        for _ in range(5):
            await limiter.release()

        assert limiter.stats.active_requests == 0
        assert limiter.stats.completed_requests == 5


# ============================================================================
# Test ConcurrencyLimiter Wait for Slot
# ============================================================================

class TestConcurrencyLimiterWaitForSlot:
    """Test ConcurrencyLimiter wait_for_slot method"""

    @pytest.mark.asyncio
    async def test_wait_for_slot_immediate(self):
        """Test wait_for_slot when slot is immediately available"""
        limiter = ConcurrencyLimiter(max_concurrent=5)

        acquired = await limiter.wait_for_slot()

        assert acquired is True
        assert limiter.stats.active_requests == 1

        await limiter.release()

    @pytest.mark.asyncio
    async def test_wait_for_slot_with_timeout_success(self):
        """Test wait_for_slot with timeout when slot becomes available"""
        limiter = ConcurrencyLimiter(max_concurrent=1)

        # Acquire the only slot
        await limiter.acquire()

        # Try to acquire with timeout (in background)
        async def wait_task():
            acquired = await limiter.wait_for_slot(timeout=0.1)
            return acquired

        task = asyncio.create_task(wait_task())
        await asyncio.sleep(0.01)

        # Release the slot
        await limiter.release()

        # Wait task should complete successfully
        result = await task
        assert result is True

        # Clean up
        await limiter.release()

    @pytest.mark.asyncio
    async def test_wait_for_slot_timeout_failure(self):
        """Test wait_for_slot timeout when slot doesn't become available"""
        limiter = ConcurrencyLimiter(max_concurrent=1)

        # Acquire the only slot
        await limiter.acquire()

        # Try to acquire with short timeout
        acquired = await limiter.wait_for_slot(timeout=0.01)

        assert acquired is False
        assert limiter.stats.rejected_requests == 1

        # Clean up
        await limiter.release()

    @pytest.mark.asyncio
    async def test_wait_for_slot_no_timeout(self):
        """Test wait_for_slot without timeout (waits indefinitely)"""
        limiter = ConcurrencyLimiter(max_concurrent=1)

        # Acquire the only slot
        await limiter.acquire()

        # Start task that waits indefinitely
        async def wait_task():
            acquired = await limiter.wait_for_slot(timeout=None)
            return acquired

        task = asyncio.create_task(wait_task())
        await asyncio.sleep(0.01)

        # Release the slot
        await limiter.release()

        # Task should complete
        result = await task
        assert result is True

        # Clean up
        await limiter.release()


# ============================================================================
# Test ConcurrencyLimiter Statistics
# ============================================================================

class TestConcurrencyLimiterStatistics:
    """Test ConcurrencyLimiter statistics tracking"""

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting current statistics"""
        limiter = ConcurrencyLimiter(max_concurrent=5)

        # Perform some operations
        await limiter.acquire()
        await limiter.acquire()
        await limiter.release()

        stats = limiter.get_stats()

        assert stats["total_requests"] == 2
        assert stats["active_requests"] == 1
        assert stats["completed_requests"] == 1
        assert stats["rejected_requests"] == 0
        assert isinstance(stats["last_reset"], str)

        # Clean up
        await limiter.release()

    @pytest.mark.asyncio
    async def test_peak_concurrent_tracking(self):
        """Test that peak concurrent is tracked correctly"""
        limiter = ConcurrencyLimiter(max_concurrent=5)

        # Simulate concurrent operations
        tasks = []
        for _ in range(3):
            tasks.append(limiter.acquire())

        await asyncio.gather(*tasks)

        assert limiter.stats.peak_concurrent == 3

        # Clean up
        for _ in range(3):
            await limiter.release()

    @pytest.mark.asyncio
    async def test_reset_stats(self):
        """Test resetting statistics"""
        limiter = ConcurrencyLimiter(max_concurrent=5)

        # Perform some operations
        await limiter.acquire()
        await limiter.release()

        assert limiter.stats.total_requests > 0

        # Reset
        limiter.reset_stats()

        assert limiter.stats.total_requests == 0
        assert limiter.stats.active_requests == 0
        assert limiter.stats.completed_requests == 0
        assert limiter.stats.peak_concurrent == 0
        assert limiter.stats.rejected_requests == 0


# ============================================================================
# Test ConcurrencyLimiter Properties
# ============================================================================

class TestConcurrencyLimiterProperties:
    """Test ConcurrencyLimiter properties"""

    @pytest.mark.asyncio
    async def test_available_slots_property(self):
        """Test available_slots property"""
        limiter = ConcurrencyLimiter(max_concurrent=10)

        assert limiter.available_slots == 10

        await limiter.acquire()
        assert limiter.available_slots == 9

        await limiter.acquire()
        assert limiter.available_slots == 8

        await limiter.release()
        assert limiter.available_slots == 9

        # Clean up
        await limiter.release()

    @pytest.mark.asyncio
    async def test_utilization_property(self):
        """Test utilization property"""
        limiter = ConcurrencyLimiter(max_concurrent=10)

        assert limiter.utilization == 0.0

        await limiter.acquire()
        assert limiter.utilization == 0.1

        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()
        assert limiter.utilization == 0.5

        # Clean up
        for _ in range(5):
            await limiter.release()

    @pytest.mark.asyncio
    async def test_utilization_at_capacity(self):
        """Test utilization when at full capacity"""
        limiter = ConcurrencyLimiter(max_concurrent=5)

        for _ in range(5):
            await limiter.acquire()

        assert limiter.utilization == 1.0

        # Clean up
        for _ in range(5):
            await limiter.release()


# ============================================================================
# Test ConcurrencyLimiter Error Handling
# ============================================================================

class TestConcurrencyLimiterErrorHandling:
    """Test ConcurrencyLimiter error handling"""

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self):
        """Test that context manager releases on exception"""
        limiter = ConcurrencyLimiter(max_concurrent=5)

        with pytest.raises(ValueError):
            async with limiter:
                assert limiter.stats.active_requests == 1
                raise ValueError("Test error")

        # Should still release
        assert limiter.stats.active_requests == 0
        assert limiter.stats.completed_requests == 1


# ============================================================================
# Test Global Limiter Instance
# ============================================================================

class TestGlobalLimiter:
    """Test global limiter instance management"""

    def test_get_concurrency_limiter_creates_instance(self):
        """Test that get_concurrency_limiter creates instance on first call"""
        # Reset global instance
        import app.core.concurrency
        app.core.concurrency._global_limiter = None

        limiter = get_concurrency_limiter(max_concurrent=15)

        assert isinstance(limiter, ConcurrencyLimiter)
        assert limiter.max_concurrent == 15

    def test_get_concurrency_limiter_returns_same_instance(self):
        """Test that get_concurrency_limiter returns same instance on subsequent calls"""
        # Reset and create instance
        import app.core.concurrency
        app.core.concurrency._global_limiter = None

        limiter1 = get_concurrency_limiter(max_concurrent=20)
        limiter2 = get_concurrency_limiter()

        assert limiter1 is limiter2

    def test_set_concurrency_limiter(self):
        """Test set_concurrency_limiter"""
        # Reset global instance
        import app.core.concurrency
        app.core.concurrency._global_limiter = None

        custom_limiter = ConcurrencyLimiter(max_concurrent=25)
        set_concurrency_limiter(custom_limiter)

        limiter = get_concurrency_limiter()
        assert limiter is custom_limiter
        assert limiter.max_concurrent == 25


# ============================================================================
# Test ConcurrencyLimitExceeded Exception
# ============================================================================

class TestConcurrencyLimitExceededException:
    """Test ConcurrencyLimitExceeded exception"""

    def test_exception_with_default_message(self):
        """Test exception with default message"""
        exc = ConcurrencyLimitExceeded()

        assert str(exc) == "Concurrency limit exceeded"
        assert exc.message == "Concurrency limit exceeded"

    def test_exception_with_custom_message(self):
        """Test exception with custom message"""
        exc = ConcurrencyLimitExceeded("Custom error message")

        assert str(exc) == "Custom error message"
        assert exc.message == "Custom error message"


# ============================================================================
# Integration Tests
# ============================================================================

class TestConcurrencyLimiterIntegration:
    """Integration tests for ConcurrencyLimiter"""

    @pytest.mark.asyncio
    async def test_concurrent_requests_under_limit(self):
        """Test handling concurrent requests under the limit"""
        limiter = ConcurrencyLimiter(max_concurrent=5)

        results = []

        async def request_handler(request_id):
            async with limiter:
                await asyncio.sleep(0.01)
                results.append(request_id)
                return request_id

        # Launch 3 concurrent requests (under limit of 5)
        tasks = [request_handler(i) for i in range(3)]
        await asyncio.gather(*tasks)

        assert len(results) == 3
        assert limiter.stats.completed_requests == 3
        assert limiter.stats.peak_concurrent == 3

    @pytest.mark.asyncio
    async def test_concurrent_requests_at_limit(self):
        """Test handling concurrent requests at the limit"""
        limiter = ConcurrencyLimiter(max_concurrent=3)

        results = []

        async def request_handler(request_id):
            async with limiter:
                await asyncio.sleep(0.01)
                results.append(request_id)
                return request_id

        # Launch 5 concurrent requests (limit is 3)
        tasks = [request_handler(i) for i in range(5)]
        await asyncio.gather(*tasks)

        assert len(results) == 5
        assert limiter.stats.completed_requests == 5
        assert limiter.stats.peak_concurrent == 3

    @pytest.mark.asyncio
    async def test_sequential_batches(self):
        """Test processing sequential batches of requests"""
        limiter = ConcurrencyLimiter(max_concurrent=2)

        batch_results = []

        async def process_batch(batch_id):
            async with limiter:
                await asyncio.sleep(0.01)
                batch_results.append(batch_id)

        # Process 3 batches sequentially
        for i in range(3):
            await process_batch(i)

        assert len(batch_results) == 3
        assert limiter.stats.total_requests == 3
        assert limiter.stats.completed_requests == 3
        assert limiter.stats.peak_concurrent == 1  # Sequential, so peak is 1

    @pytest.mark.asyncio
    async def test_stats_accuracy_after_many_operations(self):
        """Test statistics accuracy after many operations"""
        limiter = ConcurrencyLimiter(max_concurrent=10)

        # Perform 20 acquire/release cycles
        for _ in range(20):
            await limiter.acquire()
            await limiter.release()

        stats = limiter.get_stats()

        assert stats["total_requests"] == 20
        assert stats["active_requests"] == 0
        assert stats["completed_requests"] == 20
        assert stats["rejected_requests"] == 0
        assert stats["peak_concurrent"] == 1  # Sequential operations
