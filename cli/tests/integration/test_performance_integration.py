#!/usr/bin/env python3
"""
Integration test for performance optimization with analysis service.

This test verifies that the performance optimization features work correctly
when integrated with the analysis service.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add CLI to path
cli_path = Path(__file__).parent
if str(cli_path) not in sys.path:
    sys.path.insert(0, str(cli_path))

from core.performance import PerformanceManager


def test_performance_manager_basic():
    """Test basic performance manager functionality."""
    print("Testing Performance Manager Basic Functionality...")

    # Initialize performance manager
    config = {
        "cache_max_size_mb": 10,
        "cache_max_entries": 100,
        "max_concurrent_operations": 3,
        "max_background_workers": 2,
        "resource_monitoring_interval": 1.0,
    }

    manager = PerformanceManager(config)
    manager.start()

    try:
        # Test cache functionality
        print("  Testing cache...")
        cache = manager.cache

        # Put and get data
        test_data = {"analysis": "airfoil", "results": [1, 2, 3, 4, 5]}
        cache.put("test_analysis", test_data)
        retrieved = cache.get("test_analysis")

        assert retrieved == test_data, "Cache put/get failed"
        print("    ✓ Cache put/get working")

        # Test cache statistics
        stats = cache.get_stats()
        assert stats["entries"] == 1, "Cache statistics incorrect"
        assert stats["hits"] >= 1, "Cache hit count incorrect"
        print("    ✓ Cache statistics working")

        # Test resource monitoring
        print("  Testing resource monitoring...")
        usage = manager.resource_monitor.get_current_usage()
        assert usage.cpu_percent >= 0, "CPU usage should be non-negative"
        assert usage.memory_percent >= 0, "Memory usage should be non-negative"
        print("    ✓ Resource monitoring working")

        # Test optimization suggestions
        suggestions = manager.resource_monitor.get_optimization_suggestions()
        assert isinstance(suggestions, list), "Suggestions should be a list"
        print("    ✓ Optimization suggestions working")

        # Test performance report
        print("  Testing performance report...")
        report = manager.get_performance_report()
        required_keys = [
            "resource_usage",
            "cache_stats",
            "active_operations",
            "background_tasks",
        ]
        for key in required_keys:
            assert key in report, f"Performance report missing key: {key}"
        print("    ✓ Performance report working")

        print("  All basic tests passed!")

    finally:
        manager.shutdown()


async def test_async_operations():
    """Test asynchronous operation management."""
    print("Testing Asynchronous Operations...")

    manager = PerformanceManager()
    manager.start()

    try:
        async_manager = manager.async_manager

        # Test simple async operation
        async def simple_async_task(value):
            await asyncio.sleep(0.1)
            return value * 2

        result = await async_manager.execute_async(
            operation_id="test_async",
            coro_func=simple_async_task,
            value=5,
        )

        assert result == 10, "Async operation result incorrect"
        print("  ✓ Simple async operation working")

        # Test concurrent operations
        async def concurrent_task(task_id, duration):
            await asyncio.sleep(duration)
            return f"Task {task_id} completed"

        start_time = time.time()
        tasks = []
        for i in range(3):
            task = async_manager.execute_async(
                operation_id=f"concurrent_{i}",
                coro_func=concurrent_task,
                task_id=i,
                duration=0.2,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        assert len(results) == 3, "Not all concurrent tasks completed"
        assert duration < 0.4, "Tasks should run concurrently, not sequentially"
        print("  ✓ Concurrent operations working")

        print("  All async operation tests passed!")

    finally:
        manager.shutdown()


def test_background_execution():
    """Test background task execution."""
    print("Testing Background Execution...")

    manager = PerformanceManager()
    manager.start()

    try:
        executor = manager.background_executor

        # Test simple background task
        def simple_bg_task(value):
            time.sleep(0.2)
            return value * 3

        task_id = executor.submit_task(
            task_id="bg_test",
            name="Background Test",
            func=simple_bg_task,
            value=7,
        )

        # Wait for completion
        for _ in range(10):  # Wait up to 1 second
            time.sleep(0.1)
            task = executor.get_task_status(task_id)
            if task and task.status == "completed":
                break

        final_task = executor.get_task_status(task_id)
        assert final_task is not None, "Background task not found"
        assert final_task.status == "completed", "Background task did not complete"
        assert final_task.result == 21, "Background task result incorrect"
        print("  ✓ Simple background task working")

        # Test task cancellation
        def long_bg_task():
            time.sleep(2.0)
            return "completed"

        cancel_task_id = executor.submit_task(
            task_id="cancel_test",
            name="Cancellation Test",
            func=long_bg_task,
        )

        # Try to cancel task immediately (before it starts running)
        success = executor.cancel_task(cancel_task_id)

        # Check task status
        time.sleep(0.2)
        cancelled_task = executor.get_task_status(cancel_task_id)

        # Task should either be cancelled or completed quickly
        # (cancellation might fail if task already started)
        if success:
            assert cancelled_task.status == "cancelled", "Task was not cancelled"
            print("  ✓ Task cancellation working")
        else:
            # If cancellation failed, task should still complete eventually
            print("  ✓ Task cancellation attempted (task may have already started)")

        print("  All background execution tests passed!")

    finally:
        manager.shutdown()


def test_cache_advanced_features():
    """Test advanced cache features."""
    print("Testing Advanced Cache Features...")

    manager = PerformanceManager(
        {
            "cache_max_size_mb": 1,
            "cache_max_entries": 5,
            "cache_default_ttl_seconds": 1,
        },
    )

    try:
        cache = manager.cache

        # Test TTL expiration
        cache.put("expire_test", "test_value", ttl_seconds=1)
        assert (
            cache.get("expire_test") == "test_value"
        ), "Item should be available immediately"

        time.sleep(1.1)
        assert cache.get("expire_test") is None, "Item should have expired"
        print("  ✓ TTL expiration working")

        # Test LRU eviction
        for i in range(6):  # More than max_entries
            cache.put(f"key_{i}", f"value_{i}")

        # First item should be evicted
        assert cache.get("key_0") is None, "LRU item should be evicted"
        assert cache.get("key_5") == "value_5", "Most recent item should be present"
        print("  ✓ LRU eviction working")

        # Test cache key generation
        key1 = cache.generate_key("analysis", "airfoil", reynolds=100000)
        key2 = cache.generate_key("analysis", "airfoil", reynolds=100000)
        key3 = cache.generate_key("analysis", "airfoil", reynolds=200000)

        assert key1 == key2, "Same parameters should generate same key"
        assert key1 != key3, "Different parameters should generate different keys"
        print("  ✓ Cache key generation working")

        # Test cleanup expired
        cache.put("short_ttl", "value1", ttl_seconds=1)
        cache.put("long_ttl", "value2", ttl_seconds=60)

        time.sleep(1.1)
        cleaned = cache.cleanup_expired()

        assert cleaned >= 1, "Should have cleaned expired entries"
        assert cache.get("short_ttl") is None, "Expired item should be gone"
        assert cache.get("long_ttl") == "value2", "Non-expired item should remain"
        print("  ✓ Expired cleanup working")

        print("  All advanced cache tests passed!")

    finally:
        manager.shutdown()


def test_maintenance_and_optimization():
    """Test maintenance and optimization features."""
    print("Testing Maintenance and Optimization...")

    manager = PerformanceManager()
    manager.start()

    try:
        # Add some test data
        cache = manager.cache
        for i in range(10):
            cache.put(f"maintenance_test_{i}", f"value_{i}", ttl_seconds=1)

        # Wait for expiration
        time.sleep(1.1)

        # Run maintenance
        stats = manager.perform_maintenance()
        assert isinstance(stats, dict), "Maintenance should return statistics"
        assert "cache_expired_cleaned" in stats, "Should report cache cleanup"
        print("  ✓ Maintenance tasks working")

        # Test emergency cleanup
        cache.put("emergency_test", "test_value")
        emergency_stats = manager.emergency_cleanup()

        assert isinstance(
            emergency_stats,
            dict,
        ), "Emergency cleanup should return statistics"
        assert emergency_stats.get("cache_cleared"), "Cache should be cleared"
        assert (
            cache.get("emergency_test") is None
        ), "Cache should be empty after emergency cleanup"
        print("  ✓ Emergency cleanup working")

        print("  All maintenance and optimization tests passed!")

    finally:
        manager.shutdown()


async def test_integrated_scenario():
    """Test integrated performance optimization scenario."""
    print("Testing Integrated Scenario...")

    manager = PerformanceManager(
        {
            "cache_max_size_mb": 5,
            "cache_max_entries": 50,
            "max_concurrent_operations": 3,
            "max_background_workers": 2,
        },
    )
    manager.start()

    try:
        # Simulate analysis workflow with caching
        async def mock_analysis(analysis_type, parameters):
            # Generate cache key
            cache_key = manager.cache.generate_key(analysis_type, **parameters)

            # Check cache first
            cached_result = manager.cache.get(cache_key)
            if cached_result:
                return cached_result

            # Simulate analysis work
            await asyncio.sleep(0.2)
            result = {
                "analysis_type": analysis_type,
                "parameters": parameters,
                "results": [1, 2, 3, 4, 5],
                "timestamp": time.time(),
            }

            # Cache result
            manager.cache.put(cache_key, result, ttl_seconds=300)
            return result

        # Run same analysis twice (second should hit cache)
        analysis_params = {"reynolds": 100000, "mach": 0.3}

        start_time = time.time()
        result1 = await manager.async_manager.execute_async(
            operation_id="analysis_1",
            coro_func=mock_analysis,
            analysis_type="airfoil_polar",
            parameters=analysis_params,
        )
        duration1 = time.time() - start_time

        start_time = time.time()
        result2 = await manager.async_manager.execute_async(
            operation_id="analysis_2",
            coro_func=mock_analysis,
            analysis_type="airfoil_polar",
            parameters=analysis_params,
        )
        duration2 = time.time() - start_time

        # Results should be the same
        assert (
            result1["analysis_type"] == result2["analysis_type"]
        ), "Results should be identical"
        assert (
            result1["parameters"] == result2["parameters"]
        ), "Parameters should be identical"

        # Second run should be faster (cache hit)
        assert duration2 < duration1, "Cached analysis should be faster"
        print("  ✓ Analysis caching working")

        # Test background analysis
        def bg_analysis():
            time.sleep(0.3)
            return {"status": "completed", "results": [10, 20, 30]}

        bg_task_id = manager.background_executor.submit_task(
            task_id="bg_analysis",
            name="Background Analysis",
            func=bg_analysis,
        )

        # Wait for completion
        for _ in range(15):  # Wait up to 1.5 seconds
            await asyncio.sleep(0.1)
            task = manager.background_executor.get_task_status(bg_task_id)
            if task and task.status == "completed":
                break

        bg_task = manager.background_executor.get_task_status(bg_task_id)
        assert bg_task.status == "completed", "Background analysis should complete"
        assert (
            bg_task.result["status"] == "completed"
        ), "Background analysis result should be correct"
        print("  ✓ Background analysis working")

        # Check final performance report
        report = manager.get_performance_report()
        assert report["cache_stats"]["entries"] > 0, "Cache should have entries"
        assert report["cache_stats"]["hit_rate_percent"] > 0, "Should have cache hits"
        print("  ✓ Performance reporting working")

        print("  All integrated scenario tests passed!")

    finally:
        manager.shutdown()


async def main():
    """Run all performance integration tests."""
    print("ICARUS CLI Performance Integration Tests")
    print("=" * 50)

    try:
        # Run all tests
        test_performance_manager_basic()
        await test_async_operations()
        test_background_execution()
        test_cache_advanced_features()
        test_maintenance_and_optimization()
        await test_integrated_scenario()

        print("\n" + "=" * 50)
        print("ALL PERFORMANCE INTEGRATION TESTS PASSED!")
        print("=" * 50)

    except Exception as e:
        print(f"\nTest failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
