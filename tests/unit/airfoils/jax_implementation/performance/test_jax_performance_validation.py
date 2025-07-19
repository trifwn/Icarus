"""
Performance validation and benchmarking for JAX airfoil implementation.

This module provides performance testing including:
- Memory usage profiling under various workloads
- JIT compilation time measurement
- Batch operation performance validation
- Gradient computation performance testing

Requirements covered: 4.1, 4.3
"""

import gc
import os
import time
from typing import Any
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import psutil
import pytest

from ICARUS.airfoils.jax_implementation.batch_processing import BatchAirfoilOps
from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


class PerformanceProfiler:
    """Utility class for performance profiling."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
        self.measurements = []

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()

    def start_measurement(self, label: str):
        """Start a performance measurement."""
        measurement = {
            "label": label,
            "start_time": time.time(),
            "start_memory": self.get_memory_usage(),
            "start_cpu": self.get_cpu_usage(),
        }
        self.measurements.append(measurement)
        return len(self.measurements) - 1

    def end_measurement(self, measurement_id: int):
        """End a performance measurement."""
        if measurement_id >= len(self.measurements):
            return None

        measurement = self.measurements[measurement_id]
        measurement.update(
            {
                "end_time": time.time(),
                "end_memory": self.get_memory_usage(),
                "end_cpu": self.get_cpu_usage(),
            },
        )

        # Calculate metrics
        measurement["duration"] = measurement["end_time"] - measurement["start_time"]
        measurement["memory_delta"] = (
            measurement["end_memory"] - measurement["start_memory"]
        )
        measurement["peak_memory"] = measurement["end_memory"]

        return measurement

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        completed_measurements = [m for m in self.measurements if "duration" in m]

        if not completed_measurements:
            return {}

        return {
            "total_measurements": len(completed_measurements),
            "total_duration": sum(m["duration"] for m in completed_measurements),
            "total_memory_delta": sum(
                m["memory_delta"] for m in completed_measurements
            ),
            "peak_memory": max(m["peak_memory"] for m in completed_measurements),
            "average_duration": np.mean(
                [m["duration"] for m in completed_measurements],
            ),
            "measurements": completed_measurements,
        }


class TestJITCompilationPerformance:
    """Test JIT compilation performance and caching (Requirement 4.1)."""

    def test_initial_compilation_time(self):
        """Test initial JIT compilation times are reasonable."""
        profiler = PerformanceProfiler()

        # Create test airfoil
        airfoil = JaxAirfoil.naca4("2412", n_points=200)
        query_x = jnp.linspace(0.0, 1.0, 50)

        # Test compilation times for different operations
        operations = [
            ("thickness", lambda: airfoil.thickness(query_x)),
            ("camber_line", lambda: airfoil.camber_line(query_x)),
            ("y_upper", lambda: airfoil.y_upper(query_x)),
            ("y_lower", lambda: airfoil.y_lower(query_x)),
        ]

        compilation_times = {}

        for op_name, op_func in operations:
            # First call triggers compilation
            start_id = profiler.start_measurement(f"compile_{op_name}")
            result1 = op_func()
            compile_measurement = profiler.end_measurement(start_id)

            # Second call should be much faster (cached)
            start_id = profiler.start_measurement(f"cached_{op_name}")
            result2 = op_func()
            cached_measurement = profiler.end_measurement(start_id)

            # Results should be identical
            assert jnp.allclose(result1, result2)

            # Store timing results
            compilation_times[op_name] = {
                "compile_time": compile_measurement["duration"],
                "cached_time": cached_measurement["duration"],
                "speedup": compile_measurement["duration"]
                / cached_measurement["duration"],
            }

        # Validate compilation times
        for op_name, times in compilation_times.items():
            # Initial compilation should be reasonable (< 5 seconds)
            assert (
                times["compile_time"] < 5.0
            ), f"{op_name} compilation too slow: {times['compile_time']:.3f}s"

            # Cached execution should be much faster
            assert (
                times["speedup"] > 10
            ), f"{op_name} caching not effective: {times['speedup']:.1f}x speedup"

            # Cached execution should be very fast (< 10ms)
            assert (
                times["cached_time"] < 0.01
            ), f"{op_name} cached execution too slow: {times['cached_time']:.6f}s"

    def test_buffer_size_compilation_scaling(self):
        """Test how compilation time scales with buffer size."""
        profiler = PerformanceProfiler()

        buffer_sizes = [64, 128, 256, 512, 1024]
        compilation_times = []

        for buffer_size in buffer_sizes:
            # Create airfoil with specific buffer size
            airfoil = JaxAirfoil.naca4(
                "0012",
                n_points=buffer_size // 2,
                buffer_size=buffer_size,
            )
            query_x = jnp.linspace(0.0, 1.0, 20)

            # Measure compilation time
            start_id = profiler.start_measurement(f"buffer_{buffer_size}")
            thickness = airfoil.thickness(query_x)
            measurement = profiler.end_measurement(start_id)

            compilation_times.append(
                {
                    "buffer_size": buffer_size,
                    "compile_time": measurement["duration"],
                    "memory_usage": measurement["memory_delta"],
                },
            )

            # Verify result is reasonable
            assert jnp.all(jnp.isfinite(thickness))
            assert jnp.all(thickness >= 0)

        # Check that compilation time doesn't grow too quickly with buffer size
        max_compile_time = max(ct["compile_time"] for ct in compilation_times)
        min_compile_time = min(ct["compile_time"] for ct in compilation_times)

        # Compilation time should not increase by more than 10x across buffer sizes
        time_ratio = max_compile_time / min_compile_time
        assert time_ratio < 10, f"Compilation time scaling too poor: {time_ratio:.1f}x"

    def test_recompilation_behavior(self):
        """Test controlled recompilation when buffer sizes change."""
        profiler = PerformanceProfiler()

        # Start with small airfoil
        small_airfoil = JaxAirfoil.naca4("0012", n_points=50)
        query_x = jnp.linspace(0.0, 1.0, 10)

        # Initial compilation
        start_id = profiler.start_measurement("initial_compile")
        result1 = small_airfoil.thickness(query_x)
        initial_measurement = profiler.end_measurement(start_id)

        # Create larger airfoil (should trigger recompilation)
        large_airfoil = JaxAirfoil.naca4("2412", n_points=500)

        start_id = profiler.start_measurement("recompile")
        result2 = large_airfoil.thickness(query_x)
        recompile_measurement = profiler.end_measurement(start_id)

        # Recompilation should be detected by longer execution time
        recompile_ratio = (
            recompile_measurement["duration"] / initial_measurement["duration"]
        )

        # If recompilation occurred, it should be slower than cached execution
        # but not necessarily slower than initial compilation
        assert (
            recompile_measurement["duration"] > 0.001
        ), "Recompilation should take measurable time"

        # Results should be valid
        assert jnp.all(jnp.isfinite(result1))
        assert jnp.all(jnp.isfinite(result2))


class TestMemoryUsageScaling:
    """Test memory usage scaling under various workloads (Requirement 4.3)."""

    def test_single_airfoil_memory_scaling(self):
        """Test memory usage scaling with airfoil size."""
        profiler = PerformanceProfiler()

        point_counts = [50, 100, 200, 500, 1000]
        memory_usage = []

        for n_points in point_counts:
            gc.collect()  # Clean up before measurement
            initial_memory = profiler.get_memory_usage()

            # Create airfoil
            airfoil = JaxAirfoil.naca4("2412", n_points=n_points)

            # Perform some operations to trigger compilation
            query_x = jnp.linspace(0.0, 1.0, 20)
            thickness = airfoil.thickness(query_x)
            camber = airfoil.camber_line(query_x)

            final_memory = profiler.get_memory_usage()
            memory_delta = final_memory - initial_memory

            memory_usage.append(
                {
                    "n_points": n_points,
                    "memory_mb": memory_delta,
                    "memory_per_point": memory_delta / n_points,
                },
            )

            # Clean up
            del airfoil, thickness, camber

        # Check memory scaling
        max_memory_per_point = max(mu["memory_per_point"] for mu in memory_usage)
        min_memory_per_point = min(mu["memory_per_point"] for mu in memory_usage)

        # Memory per point should be relatively consistent (within 5x)
        memory_ratio = max_memory_per_point / min_memory_per_point
        assert (
            memory_ratio < 5.0
        ), f"Memory scaling too poor: {memory_ratio:.1f}x variation"

        # Total memory usage should be reasonable
        max_total_memory = max(mu["memory_mb"] for mu in memory_usage)
        assert (
            max_total_memory < 100
        ), f"Memory usage too high: {max_total_memory:.1f} MB"

    def test_batch_operation_memory_efficiency(self):
        """Test memory efficiency of batch operations."""
        profiler = PerformanceProfiler()

        batch_sizes = [1, 5, 10, 25, 50]
        memory_results = []

        for batch_size in batch_sizes:
            gc.collect()
            initial_memory = profiler.get_memory_usage()

            # Create batch of airfoils
            airfoils = [
                JaxAirfoil.naca4(f"{i:04d}", n_points=100) for i in range(batch_size)
            ]
            query_x = jnp.linspace(0.0, 1.0, 20)

            # Perform batch operations
            start_id = profiler.start_measurement(f"batch_{batch_size}")
            batch_thickness = BatchAirfoilOps.batch_thickness(airfoils, query_x)
            batch_camber = BatchAirfoilOps.batch_camber_line(airfoils, query_x)
            measurement = profiler.end_measurement(start_id)

            memory_results.append(
                {
                    "batch_size": batch_size,
                    "memory_mb": measurement["memory_delta"],
                    "duration": measurement["duration"],
                    "memory_per_airfoil": measurement["memory_delta"] / batch_size,
                    "time_per_airfoil": measurement["duration"] / batch_size,
                },
            )

            # Verify results
            assert batch_thickness.shape == (batch_size, len(query_x))
            assert batch_camber.shape == (batch_size, len(query_x))

            # Clean up
            del airfoils, batch_thickness, batch_camber

        # Check batch efficiency
        # Memory per airfoil should decrease or stay constant with batch size
        memory_per_airfoil = [
            mr["memory_per_airfoil"] for mr in memory_results[1:]
        ]  # Skip single airfoil

        # Batch operations should be more memory efficient than individual operations
        single_memory_per_airfoil = memory_results[0]["memory_per_airfoil"]
        batch_memory_per_airfoil = np.mean(memory_per_airfoil)

        # Time per airfoil should decrease with batch size (vectorization benefit)
        time_per_airfoil = [mr["time_per_airfoil"] for mr in memory_results]
        time_improvement = time_per_airfoil[0] / time_per_airfoil[-1]  # First vs last

        assert (
            time_improvement > 2.0
        ), f"Batch time improvement insufficient: {time_improvement:.1f}x"

    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        profiler = PerformanceProfiler()

        # Baseline memory
        gc.collect()
        baseline_memory = profiler.get_memory_usage()

        # Perform repeated operations
        n_iterations = 50
        memory_samples = []

        for i in range(n_iterations):
            # Create and use airfoil
            airfoil = JaxAirfoil.naca4(f"{i % 10:04d}", n_points=100)
            query_x = jnp.linspace(0.0, 1.0, 10)

            # Perform operations
            thickness = airfoil.thickness(query_x)
            camber = airfoil.camber_line(query_x)
            upper = airfoil.y_upper(query_x)
            lower = airfoil.y_lower(query_x)

            # Sample memory every 10 iterations
            if i % 10 == 0:
                current_memory = profiler.get_memory_usage()
                memory_samples.append(current_memory - baseline_memory)

            # Clean up explicitly
            del airfoil, thickness, camber, upper, lower

        # Force garbage collection
        gc.collect()
        final_memory = profiler.get_memory_usage()

        # Check for memory leaks
        memory_growth = final_memory - baseline_memory

        # Memory growth should be minimal (< 50 MB)
        assert (
            memory_growth < 50
        ), f"Potential memory leak detected: {memory_growth:.1f} MB growth"

        # Memory should not grow continuously
        if len(memory_samples) > 3:
            # Check if memory is continuously increasing
            memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
            assert (
                memory_trend < 1.0
            ), f"Memory continuously increasing: {memory_trend:.2f} MB/iteration"


class TestGradientComputationPerformance:
    """Test performance of gradient computations (Requirement 2.1)."""

    def test_gradient_computation_overhead(self):
        """Test overhead of gradient computation vs forward pass."""
        profiler = PerformanceProfiler()

        # Create test airfoil
        airfoil = JaxAirfoil.naca4("2412", n_points=200)
        query_x = jnp.linspace(0.0, 1.0, 50)

        def objective_function(airfoil):
            thickness = airfoil.thickness(query_x)
            camber = airfoil.camber_line(query_x)
            return jnp.sum(thickness**2) + jnp.sum(camber**2)

        # Measure forward pass time
        n_forward_runs = 100
        start_id = profiler.start_measurement("forward_pass")
        for _ in range(n_forward_runs):
            result = objective_function(airfoil)
        forward_measurement = profiler.end_measurement(start_id)

        # Measure gradient computation time
        grad_fn = jax.grad(objective_function)
        n_grad_runs = 100
        start_id = profiler.start_measurement("gradient_pass")
        for _ in range(n_grad_runs):
            gradients = grad_fn(airfoil)
        grad_measurement = profiler.end_measurement(start_id)

        # Calculate overhead
        forward_time_per_run = forward_measurement["duration"] / n_forward_runs
        grad_time_per_run = grad_measurement["duration"] / n_grad_runs
        gradient_overhead = grad_time_per_run / forward_time_per_run

        # Gradient computation should not be more than 10x slower than forward pass
        assert (
            gradient_overhead < 10
        ), f"Gradient overhead too high: {gradient_overhead:.1f}x"

        # Both should be reasonably fast
        assert (
            forward_time_per_run < 0.01
        ), f"Forward pass too slow: {forward_time_per_run:.6f}s"
        assert (
            grad_time_per_run < 0.1
        ), f"Gradient computation too slow: {grad_time_per_run:.6f}s"

    def test_batch_gradient_efficiency(self):
        """Test efficiency of batch gradient computations."""
        profiler = PerformanceProfiler()

        # Create batch of airfoils
        batch_size = 10
        airfoils = [
            JaxAirfoil.naca4(f"{i:04d}", n_points=100) for i in range(batch_size)
        ]
        query_x = jnp.linspace(0.0, 1.0, 20)

        def batch_objective(airfoils_list):
            """Batch objective function."""
            total_objective = 0.0
            for airfoil in airfoils_list:
                thickness = airfoil.thickness(query_x)
                total_objective += jnp.sum(thickness**2)
            return total_objective

        # Individual gradient computations
        start_id = profiler.start_measurement("individual_gradients")
        individual_gradients = []
        for airfoil in airfoils:

            def single_objective(single_airfoil):
                thickness = single_airfoil.thickness(query_x)
                return jnp.sum(thickness**2)

            grad_fn = jax.grad(single_objective)
            grad = grad_fn(airfoil)
            individual_gradients.append(grad)
        individual_measurement = profiler.end_measurement(start_id)

        # Batch gradient computation
        start_id = profiler.start_measurement("batch_gradients")
        batch_grad_fn = jax.grad(batch_objective)
        batch_gradients = batch_grad_fn(airfoils)
        batch_measurement = profiler.end_measurement(start_id)

        # Compare efficiency
        individual_time = individual_measurement["duration"]
        batch_time = batch_measurement["duration"]
        efficiency_ratio = individual_time / batch_time

        # Batch gradients should be more efficient
        assert (
            efficiency_ratio > 1.5
        ), f"Batch gradient efficiency insufficient: {efficiency_ratio:.1f}x"

        # Verify gradient consistency
        assert len(batch_gradients) == len(individual_gradients)

    def test_higher_order_gradient_performance(self):
        """Test performance of higher-order gradients."""
        profiler = PerformanceProfiler()

        # Create test airfoil
        airfoil = JaxAirfoil.naca4("0012", n_points=100)
        query_x = jnp.array([0.5])  # Single point for simplicity

        def objective(airfoil):
            return jnp.sum(airfoil.thickness(query_x) ** 2)

        # First-order gradient
        start_id = profiler.start_measurement("first_order_grad")
        first_grad_fn = jax.grad(objective)
        first_grad = first_grad_fn(airfoil)
        first_measurement = profiler.end_measurement(start_id)

        # Second-order gradient (Hessian diagonal)
        start_id = profiler.start_measurement("second_order_grad")

        def grad_norm(airfoil):
            grad = first_grad_fn(airfoil)
            return jnp.sum(grad._coordinates**2)

        second_grad_fn = jax.grad(grad_norm)
        second_grad = second_grad_fn(airfoil)
        second_measurement = profiler.end_measurement(start_id)

        # Performance comparison
        first_order_time = first_measurement["duration"]
        second_order_time = second_measurement["duration"]
        hessian_overhead = second_order_time / first_order_time

        # Second-order gradients should be reasonable (< 100x overhead)
        assert (
            hessian_overhead < 100
        ), f"Hessian computation overhead too high: {hessian_overhead:.1f}x"

        # Both should complete in reasonable time
        assert (
            first_order_time < 1.0
        ), f"First-order gradient too slow: {first_order_time:.3f}s"
        assert (
            second_order_time < 10.0
        ), f"Second-order gradient too slow: {second_order_time:.3f}s"


class TestLargeScaleWorkloads:
    """Test performance under large-scale workloads."""

    def test_large_batch_processing(self):
        """Test processing of large batches of airfoils."""
        profiler = PerformanceProfiler()

        # Create large batch
        batch_size = 100
        start_id = profiler.start_measurement("large_batch_creation")

        airfoils = []
        for i in range(batch_size):
            naca_id = f"{i % 10:04d}"  # Cycle through different NACA airfoils
            airfoil = JaxAirfoil.naca4(naca_id, n_points=100)
            airfoils.append(airfoil)

        creation_measurement = profiler.end_measurement(start_id)

        # Process batch
        query_x = jnp.linspace(0.0, 1.0, 25)

        start_id = profiler.start_measurement("large_batch_processing")
        batch_thickness = BatchAirfoilOps.batch_thickness(airfoils, query_x)
        batch_camber = BatchAirfoilOps.batch_camber_line(airfoils, query_x)
        processing_measurement = profiler.end_measurement(start_id)

        # Verify results
        assert batch_thickness.shape == (batch_size, len(query_x))
        assert batch_camber.shape == (batch_size, len(query_x))
        assert jnp.all(jnp.isfinite(batch_thickness))
        assert jnp.all(jnp.isfinite(batch_camber))

        # Performance requirements
        creation_time_per_airfoil = creation_measurement["duration"] / batch_size
        processing_time_per_airfoil = processing_measurement["duration"] / batch_size

        assert (
            creation_time_per_airfoil < 0.1
        ), f"Airfoil creation too slow: {creation_time_per_airfoil:.4f}s per airfoil"
        assert (
            processing_time_per_airfoil < 0.01
        ), f"Batch processing too slow: {processing_time_per_airfoil:.6f}s per airfoil"

        # Memory usage should be reasonable
        total_memory = (
            creation_measurement["memory_delta"]
            + processing_measurement["memory_delta"]
        )
        memory_per_airfoil = total_memory / batch_size
        assert (
            memory_per_airfoil < 5.0
        ), f"Memory per airfoil too high: {memory_per_airfoil:.2f} MB"

    def test_high_resolution_airfoils(self):
        """Test performance with high-resolution airfoils."""
        profiler = PerformanceProfiler()

        # Test different resolutions
        resolutions = [500, 1000, 2000]
        performance_data = []

        for n_points in resolutions:
            start_id = profiler.start_measurement(f"high_res_{n_points}")

            # Create high-resolution airfoil
            airfoil = JaxAirfoil.naca4("2412", n_points=n_points)

            # Perform operations
            query_x = jnp.linspace(0.0, 1.0, 100)
            thickness = airfoil.thickness(query_x)
            camber = airfoil.camber_line(query_x)
            upper = airfoil.y_upper(query_x)
            lower = airfoil.y_lower(query_x)

            measurement = profiler.end_measurement(start_id)

            performance_data.append(
                {
                    "n_points": n_points,
                    "duration": measurement["duration"],
                    "memory_mb": measurement["memory_delta"],
                    "time_per_point": measurement["duration"] / n_points,
                    "memory_per_point": measurement["memory_delta"] / n_points,
                },
            )

            # Verify results quality
            assert jnp.all(jnp.isfinite(thickness))
            assert jnp.all(jnp.isfinite(camber))
            assert jnp.all(thickness >= 0)

            # Clean up
            del airfoil, thickness, camber, upper, lower

        # Check scaling behavior
        max_time_per_point = max(pd["time_per_point"] for pd in performance_data)
        min_time_per_point = min(pd["time_per_point"] for pd in performance_data)
        time_scaling_ratio = max_time_per_point / min_time_per_point

        # Time per point should not increase dramatically with resolution
        assert (
            time_scaling_ratio < 5.0
        ), f"Time scaling too poor: {time_scaling_ratio:.1f}x"

        # High-resolution operations should complete in reasonable time
        max_total_time = max(pd["duration"] for pd in performance_data)
        assert (
            max_total_time < 30.0
        ), f"High-resolution operations too slow: {max_total_time:.1f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
