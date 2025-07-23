#!/usr/bin/env python3
"""
Test suite for Performance Optimization and Scalability features.

This test suite verifies the functionality of asynchronous operations,
intelligent caching, resource monitoring, and background execution.
"""

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

# Add CLI to path
cli_path = Path(__file__).parent
if str(cli_path) not in sys.path:
    sys.path.insert(0, str(cli_path))

from core.performance import AsyncOperationManager
from core.performance import BackgroundExecutor
from core.performance import IntelligentCache
from core.performance import PerformanceManager
from core.performance import ResourceMonitor
from core.performance import ResourceUsage


class TestAsyncOperationManager:
    """Test asynchronous operation management."""

    @pytest.fixture
    def async_manager(self):
        return AsyncOperationManager(max_concurrent_operations=3)

    @pytest.mark.asyncio
    async def test_execute_async_operation(self, async_manager):
        """Test basic async operation execution."""

        async def mock_operation(value):
            await asyncio.sleep(0.1)
            return value * 2

        result = await async_manager.execute_async(
            operation_id="test_op",
            coro_func=mock_operation,
            value=5,
        )

        assert result == 10

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, async_manager):
        """Test concurrent operation execution."""

        async def mock_operation(duration, value):
            await asyncio.sleep(duration)
            return value

        # Start multiple operations
        tasks = []
        for i in range(3):
            task = async_manager.execute_async(
                operation_id=f"op_{i}",
                coro_func=mock_operation,
                duration=0.1,
                value=i,
            )
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        assert results == [0, 1, 2]
        assert duration < 0.2  # Should run concurrently, not sequentially

    @pytest.mark.asyncio
    async def test_operation_timeout(self, async_manager):
        """Test operation timeout handling."""

        async def slow_operation():
            await asyncio.sleep(1.0)
            return "completed"

        with pytest.raises(asyncio.TimeoutError):
            await async_manager.execute_async(
                operation_id="timeout_test",
                coro_func=slow_operation,
                timeout=0.1,
            )

    @pytest.mark.asyncio
    async def test_operation_cancellation(self, async_manager):
        """Test operation cancellation."""

        async def long_operation():
            await asyncio.sleep(1.0)
            return "completed"

        # Start operation
        task = asyncio.create_task(
            async_manager.execute_async(
                operation_id="cancel_test",
                coro_func=long_operation,
            ),
        )

        # Give it a moment to start
        await asyncio.sleep(0.1)

        # Cancel operation
        success = async_manager.cancel_operation("cancel_test")
        assert success

        # Task should be cancelled
        with pytest.raises(asyncio.CancelledError):
            await task


class TestIntelligentCache:
    """Test intelligent caching system."""

    @pytest.fixture
    def cache(self):
        return IntelligentCache(max_size_mb=1, max_entries=10, default_ttl_seconds=60)

    def test_cache_put_get(self, cache):
        """Test basic cache put and get operations."""
        test_data = {"key": "value", "number": 42}

        # Put data in cache
        success = cache.put("test_key", test_data)
        assert success

        # Get data from cache
        retrieved = cache.get("test_key")
        assert retrieved == test_data

    def test_cache_miss(self, cache):
        """Test cache miss for non-existent key."""
        result = cache.get("nonexistent_key")
        assert result is None

    def test_cache_expiration(self, cache):
        """Test cache entry expiration."""
        test_data = "test_value"

        # Put data with short TTL
        cache.put("expire_test", test_data, ttl_seconds=1)

        # Should be available immediately
        assert cache.get("expire_test") == test_data

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired now
        assert cache.get("expire_test") is None

    def test_cache_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        for i in range(10):
            cache.put(f"key_{i}", f"value_{i}")

        # Add one more item (should evict LRU)
        cache.put("new_key", "new_value")

        # First item should be evicted
        assert cache.get("key_0") is None

        # New item should be present
        assert cache.get("new_key") == "new_value"

    def test_cache_statistics(self, cache):
        """Test cache statistics tracking."""
        # Add some data
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Access data (hits)
        cache.get("key1")
        cache.get("key1")

        # Try to access non-existent data (miss)
        cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats["entries"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate_percent"] > 0

    def test_cache_key_generation(self, cache):
        """Test cache key generation."""
        key1 = cache.generate_key("arg1", "arg2", kwarg1="value1")
        key2 = cache.generate_key("arg1", "arg2", kwarg1="value1")
        key3 = cache.generate_key("arg1", "arg2", kwarg1="value2")

        assert key1 == key2  # Same arguments should generate same key
        assert key1 != key3  # Different arguments should generate different key

    def test_cache_cleanup_expired(self, cache):
        """Test cleanup of expired entries."""
        # Add entries with different TTLs
        cache.put("short_ttl", "value1", ttl_seconds=1)
        cache.put("long_ttl", "value2", ttl_seconds=60)

        # Wait for short TTL to expire
        time.sleep(1.1)

        # Cleanup expired entries
        cleaned = cache.cleanup_expired()

        assert cleaned == 1
        assert cache.get("short_ttl") is None
        assert cache.get("long_ttl") == "value2"


class TestResourceMonitor:
    """Test resource monitoring system."""

    @pytest.fixture
    def monitor(self):
        return ResourceMonitor(monitoring_interval=0.1)

    def test_get_current_usage(self, monitor):
        """Test getting current resource usage."""
        usage = monitor.get_current_usage()

        assert isinstance(usage, ResourceUsage)
        assert usage.cpu_percent >= 0
        assert usage.memory_percent >= 0
        assert usage.memory_used_mb >= 0
        assert usage.disk_usage_percent >= 0

    def test_monitoring_start_stop(self, monitor):
        """Test starting and stopping monitoring."""
        assert not monitor._monitoring

        monitor.start_monitoring()
        assert monitor._monitoring

        # Give it a moment to collect some data
        time.sleep(0.2)

        monitor.stop_monitoring()
        assert not monitor._monitoring

    def test_optimization_suggestions(self, monitor):
        """Test optimization suggestions generation."""
        suggestions = monitor.get_optimization_suggestions()
        assert isinstance(suggestions, list)
        # Suggestions depend on current system state, so just verify it's a list

    def test_warning_callbacks(self, monitor):
        """Test resource warning callbacks."""
        callback_called = threading.Event()
        received_warnings = []

        def warning_callback(warnings, usage):
            received_warnings.extend(warnings)
            callback_called.set()

        monitor.add_warning_callback(warning_callback)

        # Simulate high resource usage
        high_usage = ResourceUsage(
            cpu_percent=90.0,
            memory_percent=95.0,
            memory_used_mb=8000,
            memory_available_mb=400,
            disk_usage_percent=95.0,
            disk_free_gb=1.0,
        )

        monitor._check_resource_warnings(high_usage)

        # Should have triggered warnings
        assert len(received_warnings) > 0

    def test_garbage_collection(self, monitor):
        """Test forced garbage collection."""
        stats = monitor.force_garbage_collection()

        assert isinstance(stats, dict)
        assert "objects_before" in stats
        assert "objects_after" in stats
        assert "objects_collected" in stats


class TestBackgroundExecutor:
    """Test background execution system."""

    @pytest.fixture
    def executor(self):
        return BackgroundExecutor(max_workers=2)

    def test_submit_task(self, executor):
        """Test task submission."""

        def simple_task(value):
            return value * 2

        task_id = executor.submit_task(
            task_id="test_task",
            name="Simple Task",
            func=simple_task,
            value=5,
        )

        assert task_id == "test_task"

        # Wait for completion
        time.sleep(0.5)

        task = executor.get_task_status(task_id)
        assert task is not None
        assert task.status in ["completed", "running"]

    def test_task_completion(self, executor):
        """Test task completion handling."""
        completion_event = threading.Event()
        completion_result = {}

        def completion_callback(task_id, status, progress, result, error):
            completion_result.update(
                {
                    "task_id": task_id,
                    "status": status,
                    "result": result,
                    "error": error,
                },
            )
            if status == "completed":
                completion_event.set()

        def simple_task():
            time.sleep(0.1)
            return "task_result"

        executor.submit_task(
            task_id="completion_test",
            name="Completion Test",
            func=simple_task,
            completion_callback=completion_callback,
        )

        # Wait for completion
        assert completion_event.wait(timeout=2.0)
        assert completion_result["status"] == "completed"
        assert completion_result["result"] == "task_result"

    def test_task_cancellation(self, executor):
        """Test task cancellation."""

        def long_task():
            time.sleep(2.0)
            return "completed"

        task_id = executor.submit_task(
            task_id="cancel_test",
            name="Cancellation Test",
            func=long_task,
        )

        # Give task a moment to start
        time.sleep(0.1)

        # Cancel task
        success = executor.cancel_task(task_id)
        assert success

        # Check task status
        time.sleep(0.1)
        task = executor.get_task_status(task_id)
        assert task.status == "cancelled"

    def test_cleanup_completed_tasks(self, executor):
        """Test cleanup of completed tasks."""

        def quick_task():
            return "done"

        # Submit and complete a task
        task_id = executor.submit_task(
            task_id="cleanup_test",
            name="Cleanup Test",
            func=quick_task,
        )

        # Wait for completion
        time.sleep(0.5)

        # Verify task exists
        assert executor.get_task_status(task_id) is not None

        # Cleanup (with 0 hours to clean everything)
        cleaned = executor.cleanup_completed_tasks(older_than_hours=0)
        assert cleaned >= 1

        # Task should be gone
        assert executor.get_task_status(task_id) is None


class TestPerformanceManager:
    """Test performance manager coordination."""

    @pytest.fixture
    def manager(self):
        config = {
            "cache_max_size_mb": 10,
            "cache_max_entries": 100,
            "max_concurrent_operations": 5,
            "max_background_workers": 2,
            "resource_monitoring_interval": 0.1,
            "auto_cleanup_interval_minutes": 1,
        }
        return PerformanceManager(config)

    def test_manager_initialization(self, manager):
        """Test performance manager initialization."""
        assert manager.async_manager is not None
        assert manager.cache is not None
        assert manager.resource_monitor is not None
        assert manager.background_executor is not None

    def test_performance_report(self, manager):
        """Test performance report generation."""
        report = manager.get_performance_report()

        assert "resource_usage" in report
        assert "cache_stats" in report
        assert "active_operations" in report
        assert "background_tasks" in report
        assert "optimization_suggestions" in report
        assert "config" in report

    def test_maintenance_tasks(self, manager):
        """Test maintenance task execution."""
        # Add some test data to cache
        manager.cache.put("test_key", "test_value", ttl_seconds=1)

        # Wait for expiration
        time.sleep(1.1)

        # Run maintenance
        stats = manager.perform_maintenance()

        assert isinstance(stats, dict)
        assert "cache_expired_cleaned" in stats
        assert "background_tasks_cleaned" in stats
        assert "garbage_collection" in stats

    def test_emergency_cleanup(self, manager):
        """Test emergency cleanup procedures."""
        # Add some data to cache
        manager.cache.put("test_key", "test_value")

        # Perform emergency cleanup
        stats = manager.emergency_cleanup()

        assert isinstance(stats, dict)
        assert "cache_cleared" in stats

        # Cache should be empty
        assert manager.cache.get("test_key") is None

    def test_config_update(self, manager):
        """Test configuration updates."""
        new_config = {"cache_max_size_mb": 20}
        manager.update_config(new_config)

        assert manager._config["cache_max_size_mb"] == 20

    def test_start_stop(self, manager):
        """Test starting and stopping performance management."""
        manager.start()
        assert manager.resource_monitor._monitoring

        manager.stop()
        assert not manager.resource_monitor._monitoring


@pytest.mark.asyncio
async def test_integration_scenario():
    """Test integrated performance optimization scenario."""
    # Initialize performance manager
    config = {
        "cache_max_size_mb": 5,
        "cache_max_entries": 50,
        "max_concurrent_operations": 3,
        "max_background_workers": 2,
    }

    manager = PerformanceManager(config)
    manager.start()

    try:
        # Test async operations with caching
        async def cached_operation(key, value):
            # Check cache first
            cached = manager.cache.get(key)
            if cached:
                return cached

            # Simulate work
            await asyncio.sleep(0.1)
            result = value * 2

            # Cache result
            manager.cache.put(key, result)
            return result

        # First call (cache miss)
        start_time = time.time()
        result1 = await manager.async_manager.execute_async(
            operation_id="cached_op_1",
            coro_func=cached_operation,
            key="test_key",
            value=10,
        )
        duration1 = time.time() - start_time

        # Second call (cache hit)
        start_time = time.time()
        result2 = await manager.async_manager.execute_async(
            operation_id="cached_op_2",
            coro_func=cached_operation,
            key="test_key",
            value=10,
        )
        duration2 = time.time() - start_time

        # Results should be the same
        assert result1 == result2 == 20

        # Second call should be faster (cache hit)
        assert duration2 < duration1

        # Check cache statistics
        cache_stats = manager.cache.get_stats()
        assert cache_stats["hits"] >= 1
        assert cache_stats["hit_rate_percent"] > 0

        # Test background execution
        def bg_task():
            time.sleep(0.2)
            return "background_result"

        task_id = manager.background_executor.submit_task(
            task_id="integration_bg_task",
            name="Integration Background Task",
            func=bg_task,
        )

        # Wait for completion
        for _ in range(10):  # Wait up to 1 second
            await asyncio.sleep(0.1)
            task = manager.background_executor.get_task_status(task_id)
            if task and task.status == "completed":
                break

        # Verify task completed
        final_task = manager.background_executor.get_task_status(task_id)
        assert final_task.status == "completed"
        assert final_task.result == "background_result"

        # Get performance report
        report = manager.get_performance_report()
        assert report["cache_stats"]["entries"] > 0

    finally:
        manager.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
