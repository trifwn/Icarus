"""
Tests for JAX airfoil performance optimization features.

This module tests all performance optimization components including:
- Compilation profiling and caching
- Memory-efficient buffer reuse
- Gradient computation optimization
- Performance benchmarking
"""

import jax
import jax.numpy as jnp
import pytest

from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil
from ICARUS.airfoils.jax_implementation.optimized_buffer_manager import (
    OptimizedAirfoilBufferManager,
)
from ICARUS.airfoils.jax_implementation.optimized_buffer_manager import (
    cleanup_buffer_resources,
)
from ICARUS.airfoils.jax_implementation.optimized_buffer_manager import (
    get_buffer_usage_stats,
)
from ICARUS.airfoils.jax_implementation.optimized_buffer_manager import (
    get_optimal_buffer_size,
)
from ICARUS.airfoils.jax_implementation.optimized_ops import OptimizedJaxAirfoilOps
from ICARUS.airfoils.jax_implementation.performance_benchmark import AirfoilBenchmark
from ICARUS.airfoils.jax_implementation.performance_benchmark import run_quick_benchmark
from ICARUS.airfoils.jax_implementation.performance_optimizer import BufferPool
from ICARUS.airfoils.jax_implementation.performance_optimizer import CompilationCache
from ICARUS.airfoils.jax_implementation.performance_optimizer import CompilationProfiler
from ICARUS.airfoils.jax_implementation.performance_optimizer import GradientOptimizer
from ICARUS.airfoils.jax_implementation.performance_optimizer import cleanup_memory
from ICARUS.airfoils.jax_implementation.performance_optimizer import (
    get_compilation_report,
)


class TestCompilationProfiler:
    """Test compilation profiling functionality."""

    def test_profiler_decorator(self):
        """Test that profiler decorator works correctly."""
        profiler = CompilationProfiler()

        @profiler.profile_function("test_function")
        @jax.jit
        def test_func(x):
            return x * 2

        # Call function multiple times
        x = jnp.array([1.0, 2.0, 3.0])
        result1 = test_func(x)
        result2 = test_func(x)

        # Check results are correct
        expected = jnp.array([2.0, 4.0, 6.0])
        assert jnp.allclose(result1, expected)
        assert jnp.allclose(result2, expected)

        # Check profiler recorded the function
        report = profiler.get_compilation_report()
        assert report["total_functions"] >= 1
        assert "test_function" in str(report)

    def test_compilation_report_generation(self):
        """Test compilation report generation."""
        # Use global profiler to get real compilation data
        report = get_compilation_report()

        assert isinstance(report, dict)
        assert "total_functions" in report
        assert "functions" in report
        assert isinstance(report["total_functions"], int)
        assert isinstance(report["functions"], dict)

    def test_static_args_extraction(self):
        """Test static argument extraction from JIT functions."""
        profiler = CompilationProfiler()

        @profiler.profile_function("static_args_test")
        @jax.jit
        def test_func_with_static(x, static_val):
            return x + static_val

        x = jnp.array([1.0, 2.0])
        result = test_func_with_static(x, 5.0)

        expected = jnp.array([6.0, 7.0])
        assert jnp.allclose(result, expected)


class TestCompilationCache:
    """Test compilation caching functionality."""

    def test_cache_get_or_compile(self):
        """Test cache get or compile functionality."""
        cache = CompilationCache(max_cache_size=5)

        def compile_fn():
            @jax.jit
            def func(x):
                return x**2

            return func

        # First call should compile
        cache_key = ("test_func", (2, 3))
        func1 = cache.get_or_compile(cache_key, compile_fn)

        # Second call should use cache
        func2 = cache.get_or_compile(cache_key, compile_fn)

        # Should be the same function object
        assert func1 is func2

        # Test function works
        x = jnp.array([2.0, 3.0])
        result = func1(x)
        expected = jnp.array([4.0, 9.0])
        assert jnp.allclose(result, expected)

    def test_cache_eviction(self):
        """Test LRU cache eviction."""
        cache = CompilationCache(max_cache_size=2)

        def make_compile_fn(multiplier):
            def compile_fn():
                @jax.jit
                def func(x):
                    return x * multiplier

                return func

            return compile_fn

        # Fill cache
        func1 = cache.get_or_compile("key1", make_compile_fn(2))
        func2 = cache.get_or_compile("key2", make_compile_fn(3))

        # Add third item (should evict first)
        func3 = cache.get_or_compile("key3", make_compile_fn(4))

        # First item should be evicted, second and third should remain
        stats = cache.get_cache_stats()
        assert stats["cache_size"] == 2
        assert stats["total_requests"] == 3

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = CompilationCache()

        def compile_fn():
            return lambda x: x

        # Generate some cache activity
        cache.get_or_compile("key1", compile_fn)
        cache.get_or_compile("key1", compile_fn)  # Hit
        cache.get_or_compile("key2", compile_fn)  # Miss

        stats = cache.get_cache_stats()
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 2
        assert stats["total_requests"] == 3
        assert abs(stats["hit_rate"] - (1 / 3)) < 1e-6


class TestBufferPool:
    """Test buffer pool functionality."""

    def test_buffer_get_and_return(self):
        """Test buffer allocation and return."""
        pool = BufferPool()

        # Get buffer
        shape = (2, 100)
        buffer1 = pool.get_buffer(shape)
        assert buffer1.shape == shape

        # Return buffer
        pool.return_buffer(buffer1)

        # Get buffer again (should reuse)
        buffer2 = pool.get_buffer(shape)
        assert buffer2.shape == shape

        # Check memory stats
        stats = pool.get_memory_stats()
        assert stats.reused_buffers >= 1

    def test_buffer_pool_cleanup(self):
        """Test buffer pool cleanup."""
        pool = BufferPool()

        # Create many buffers
        buffers = []
        for i in range(15):
            buffer = pool.get_buffer((2, 50 + i))
            buffers.append(buffer)

        # Return all buffers
        for buffer in buffers:
            pool.return_buffer(buffer)

        # Cleanup with small max pool size
        pool.cleanup_unused_buffers(max_pool_size=5)

        # Check that cleanup worked
        total_pooled = sum(len(pool_list) for pool_list in pool._pools.values())
        assert total_pooled <= 5 * len(pool._pools)


class TestGradientOptimizer:
    """Test gradient optimization functionality."""

    def test_grad_mode_selection(self):
        """Test gradient mode selection logic."""
        # Few inputs, many outputs -> forward mode
        mode = GradientOptimizer.select_grad_mode(n_inputs=2, n_outputs=10)
        assert mode == "forward"

        # Many inputs, few outputs -> reverse mode
        mode = GradientOptimizer.select_grad_mode(n_inputs=10, n_outputs=2)
        assert mode == "reverse"

        # Balanced -> mixed/reverse mode
        mode = GradientOptimizer.select_grad_mode(n_inputs=5, n_outputs=5)
        assert mode in ["mixed", "reverse"]

    def test_efficient_grad_fn_creation(self):
        """Test creation of efficient gradient functions."""

        def test_func(x):
            return jnp.sum(x**2)

        # Create optimized gradient function
        grad_fn = GradientOptimizer.create_efficient_grad_fn(
            test_func,
            n_inputs=3,
            n_outputs=1,
        )

        # Test gradient computation
        x = jnp.array([1.0, 2.0, 3.0])
        grad_result = grad_fn(x)
        expected_grad = 2 * x  # Gradient of sum(x^2) is 2*x

        assert jnp.allclose(grad_result, expected_grad)

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing optimization."""

        def test_func(x):
            # Simple function for testing
            y = x**2
            z = y**2
            return jnp.sum(z)

        # Apply checkpointing
        checkpointed_func = GradientOptimizer.optimize_gradient_checkpointing(test_func)

        # Test that checkpointed function works
        x = jnp.array([1.0, 2.0])
        result = checkpointed_func(x)
        expected = jnp.sum((x**2) ** 2)

        assert jnp.allclose(result, expected)


class TestOptimizedJaxAirfoilOps:
    """Test optimized JAX airfoil operations."""

    def test_optimized_thickness_computation(self):
        """Test optimized thickness computation."""
        # Create test data
        buffer_size = 128
        n_upper = 50
        n_lower = 50

        upper_coords = jnp.zeros((2, buffer_size))
        lower_coords = jnp.zeros((2, buffer_size))
        query_x = jnp.linspace(0.0, 1.0, 10)

        # Set up some test coordinates
        x_vals = jnp.linspace(0, 1, n_upper)
        upper_coords = upper_coords.at[0, :n_upper].set(x_vals)
        upper_coords = upper_coords.at[1, :n_upper].set(0.1 * jnp.sin(jnp.pi * x_vals))

        lower_coords = lower_coords.at[0, :n_lower].set(x_vals)
        lower_coords = lower_coords.at[1, :n_lower].set(
            -0.05 * jnp.sin(jnp.pi * x_vals),
        )

        # Test optimized thickness computation
        thickness = OptimizedJaxAirfoilOps.compute_thickness_optimized(
            upper_coords,
            lower_coords,
            n_upper,
            n_lower,
            query_x,
        )

        assert thickness.shape == query_x.shape
        assert jnp.all(thickness >= 0)  # Thickness should be non-negative

    def test_batch_thickness_optimization(self):
        """Test optimized batch thickness computation."""
        batch_size = 3
        buffer_size = 64
        n_points = 50

        # Create batch data
        batch_coords = jnp.zeros((batch_size, 2, buffer_size))
        batch_upper_splits = jnp.full(batch_size, n_points // 2)
        batch_n_valid = jnp.full(batch_size, n_points)
        query_x = jnp.linspace(0.0, 1.0, 10)

        # Fill with test data
        for i in range(batch_size):
            x_vals = jnp.linspace(0, 1, n_points)
            y_upper = 0.1 * (i + 1) * jnp.sin(jnp.pi * x_vals)
            y_lower = -0.05 * (i + 1) * jnp.sin(jnp.pi * x_vals)

            # Upper surface (first half)
            batch_coords = batch_coords.at[i, 0, : n_points // 2].set(
                x_vals[: n_points // 2],
            )
            batch_coords = batch_coords.at[i, 1, : n_points // 2].set(
                y_upper[: n_points // 2],
            )

            # Lower surface (second half)
            batch_coords = batch_coords.at[i, 0, n_points // 2 : n_points].set(
                x_vals[n_points // 2 :],
            )
            batch_coords = batch_coords.at[i, 1, n_points // 2 : n_points].set(
                y_lower[n_points // 2 :],
            )

        # Test batch thickness computation
        batch_thickness = OptimizedJaxAirfoilOps.batch_thickness_optimized(
            batch_coords,
            batch_upper_splits,
            batch_n_valid,
            query_x,
            buffer_size,
        )

        assert batch_thickness.shape == (batch_size, len(query_x))
        assert jnp.all(batch_thickness >= 0)

    def test_optimized_morphing(self):
        """Test optimized airfoil morphing."""
        buffer_size = 64
        n_points = 50

        # Create test airfoils
        coords1 = jnp.zeros((2, buffer_size))
        coords2 = jnp.zeros((2, buffer_size))

        x_vals = jnp.linspace(0, 1, n_points)
        coords1 = coords1.at[0, :n_points].set(x_vals)
        coords1 = coords1.at[1, :n_points].set(0.1 * jnp.sin(jnp.pi * x_vals))

        coords2 = coords2.at[0, :n_points].set(x_vals)
        coords2 = coords2.at[1, :n_points].set(0.2 * jnp.sin(2 * jnp.pi * x_vals))

        mask1 = jnp.arange(buffer_size) < n_points
        mask2 = jnp.arange(buffer_size) < n_points

        # Test morphing
        morphed = OptimizedJaxAirfoilOps.morph_airfoils_optimized(
            coords1,
            coords2,
            mask1,
            mask2,
            0.5,
            n_points,
            n_points,
        )

        assert morphed.shape == (2, buffer_size)

        # Check that morphing is approximately the average
        expected_y = 0.5 * (coords1[1, :n_points] + coords2[1, :n_points])
        actual_y = morphed[1, :n_points]
        assert jnp.allclose(actual_y, expected_y, atol=1e-6)

    def test_precompilation(self):
        """Test precompilation of common operations."""
        # This should not raise any errors
        OptimizedJaxAirfoilOps.precompile_common_operations(
            buffer_sizes=[32, 64],
            n_points_list=[25, 50],
        )

        # Get optimization stats
        stats = OptimizedJaxAirfoilOps.get_optimization_stats()
        assert isinstance(stats, dict)
        assert "compilation" in stats
        assert "optimizations_applied" in stats


class TestOptimizedBufferManager:
    """Test optimized buffer manager functionality."""

    def test_optimal_buffer_size_determination(self):
        """Test optimal buffer size determination."""
        # Test different contexts
        size1 = get_optimal_buffer_size(50, "general")
        size2 = get_optimal_buffer_size(50, "batch")
        size3 = get_optimal_buffer_size(50, "morphing")

        assert all(isinstance(s, int) for s in [size1, size2, size3])
        assert all(s >= 50 for s in [size1, size2, size3])

        # Batch context might use larger buffer
        assert size2 >= size1

    def test_buffer_reuse_mechanism(self):
        """Test buffer reuse mechanism."""
        shape = (2, 64)

        # Get buffer
        buffer1 = OptimizedAirfoilBufferManager.get_reusable_buffer(shape)

        if buffer1 is not None:
            # Return buffer
            OptimizedAirfoilBufferManager.return_buffer_for_reuse(buffer1)

            # Get buffer again (should reuse)
            buffer2 = OptimizedAirfoilBufferManager.get_reusable_buffer(shape)
            assert buffer2 is not None
            assert buffer2.shape == shape

    def test_optimized_coordinate_padding(self):
        """Test optimized coordinate padding."""
        coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])
        target_size = 8

        padded = OptimizedAirfoilBufferManager.pad_coordinates_optimized(
            coords,
            target_size,
            use_buffer_pool=True,
        )

        assert padded.shape == (2, target_size)
        assert jnp.allclose(padded[:, :3], coords)
        assert jnp.all(jnp.isnan(padded[:, 3:]))

    def test_batch_buffer_creation(self):
        """Test optimized batch buffer creation."""
        # Create test coordinate arrays
        coord_list = [
            jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]]),
            jnp.array([[0.0, 0.3, 0.7, 1.0], [0.0, 0.05, 0.05, 0.0]]),
            jnp.array([[0.0, 1.0], [0.0, 0.0]]),
        ]

        batch_coords, batch_masks, n_valid_list = (
            OptimizedAirfoilBufferManager.create_batch_buffers_optimized(coord_list)
        )

        assert batch_coords.shape[0] == len(coord_list)
        assert batch_masks.shape[0] == len(coord_list)
        assert len(n_valid_list) == len(coord_list)

        # Check that valid points are correctly set
        for i, n_valid in enumerate(n_valid_list):
            assert jnp.sum(batch_masks[i]) == n_valid

    def test_buffer_usage_statistics(self):
        """Test buffer usage statistics collection."""
        # Generate some buffer usage
        for i in range(5):
            size = get_optimal_buffer_size(50 + i * 10, "test")

        stats = get_buffer_usage_stats()

        assert isinstance(stats, dict)
        assert "total_allocations" in stats
        assert "buffer_size_distribution" in stats
        assert "usage_patterns" in stats
        assert "optimization_recommendations" in stats

    def test_resource_cleanup(self):
        """Test resource cleanup functionality."""
        # Generate some usage to clean up
        for i in range(10):
            get_optimal_buffer_size(30 + i, f"context_{i}")

        # This should not raise any errors
        cleanup_buffer_resources()

        # Check that cleanup worked
        stats = get_buffer_usage_stats()
        assert isinstance(stats, dict)


class TestPerformanceBenchmark:
    """Test performance benchmarking functionality."""

    def test_benchmark_creation(self):
        """Test benchmark object creation."""
        benchmark = AirfoilBenchmark(warmup_iterations=1, benchmark_iterations=2)
        assert benchmark.warmup_iterations == 1
        assert benchmark.benchmark_iterations == 2
        assert len(benchmark.results) == 0

    def test_thickness_computation_benchmark(self):
        """Test thickness computation benchmarking."""
        benchmark = AirfoilBenchmark(warmup_iterations=1, benchmark_iterations=2)

        # Run benchmark with small parameters for speed
        results = benchmark.benchmark_thickness_computation(n_points_list=[50])

        assert len(results) == 1
        result = results[0]

        assert result.test_name.startswith("thickness_computation")
        assert result.jax_time > 0
        assert result.numpy_time > 0
        assert result.speedup > 0
        assert isinstance(result.error_metrics, dict)

    def test_naca_generation_benchmark(self):
        """Test NACA generation benchmarking."""
        benchmark = AirfoilBenchmark(warmup_iterations=1, benchmark_iterations=2)

        # Run benchmark with small parameters
        results = benchmark.benchmark_naca_generation(n_points_list=[50])

        assert len(results) >= 1  # Should have results for multiple NACA codes

        for result in results:
            assert result.test_name.startswith("naca_")
            assert result.jax_time > 0
            assert result.numpy_time > 0
            assert isinstance(result.error_metrics, dict)

    def test_quick_benchmark(self):
        """Test quick benchmark function."""
        # This should complete without errors
        results = run_quick_benchmark()

        assert isinstance(results, dict)
        assert "summary" in results
        assert isinstance(results["summary"], dict)

    def test_benchmark_memory_measurement(self):
        """Test memory usage measurement in benchmarks."""
        benchmark = AirfoilBenchmark(warmup_iterations=1, benchmark_iterations=1)

        # Test memory measurement method
        memory_usage = benchmark._measure_memory_usage()
        assert isinstance(memory_usage, (int, float))
        assert memory_usage >= 0


class TestIntegrationOptimizations:
    """Test integration of all optimization features."""

    def test_optimized_airfoil_creation(self):
        """Test that optimized features work with JaxAirfoil creation."""
        # Create airfoil (should use optimizations)
        airfoil = JaxAirfoil.naca4("2412", n_points=100)

        assert airfoil.n_points == 100
        assert airfoil.name == "NACA2412"

        # Test thickness computation (should use optimized ops)
        query_x = jnp.linspace(0.0, 1.0, 20)
        thickness = airfoil.thickness(query_x)

        assert thickness.shape == query_x.shape
        assert jnp.all(thickness >= 0)

    def test_optimized_batch_operations(self):
        """Test optimized batch operations integration."""
        # Create multiple airfoils
        airfoils = [
            JaxAirfoil.naca4("0012", n_points=50),
            JaxAirfoil.naca4("2412", n_points=50),
            JaxAirfoil.naca4("4412", n_points=50),
        ]

        # Create batch (should use optimized buffer management)
        batch_coords, batch_masks, upper_splits, n_valid = (
            JaxAirfoil.create_batch_from_list(airfoils)
        )

        assert batch_coords.shape[0] == len(airfoils)
        assert batch_masks.shape[0] == len(airfoils)
        assert len(upper_splits) == len(airfoils)
        assert len(n_valid) == len(airfoils)

    def test_memory_cleanup_integration(self):
        """Test that memory cleanup works across all components."""
        # Generate some activity across all components
        airfoil = JaxAirfoil.naca4("2412", n_points=100)
        _ = airfoil.thickness(jnp.linspace(0, 1, 10))

        # Create some buffer usage
        for i in range(5):
            get_optimal_buffer_size(50 + i * 10, "test")

        # Cleanup should work without errors
        cleanup_memory()
        cleanup_buffer_resources()

        # System should still work after cleanup
        airfoil2 = JaxAirfoil.naca4("0012", n_points=50)
        assert airfoil2.n_points == 50

    def test_performance_stats_collection(self):
        """Test that performance statistics are collected correctly."""
        # Generate some activity
        airfoil = JaxAirfoil.naca4("2412", n_points=100)
        _ = airfoil.thickness(jnp.linspace(0, 1, 10))

        # Get various statistics
        compilation_report = get_compilation_report()
        buffer_stats = get_buffer_usage_stats()
        optimization_stats = OptimizedJaxAirfoilOps.get_optimization_stats()

        assert isinstance(compilation_report, dict)
        assert isinstance(buffer_stats, dict)
        assert isinstance(optimization_stats, dict)

        # Check that stats contain expected keys
        assert "total_functions" in compilation_report
        assert "total_allocations" in buffer_stats
        assert "optimizations_applied" in optimization_stats


if __name__ == "__main__":
    pytest.main([__file__])
