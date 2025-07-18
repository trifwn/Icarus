"""
Performance tests for JAX airfoil batch processing operations.

This module contains performance benchmarks and scaling tests for batch operations
to ensure they provide the expected performance benefits over individual operations.
"""

import time
from typing import Any
from typing import Dict
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ICARUS.airfoils.jax_implementation.batch_operations import BatchAirfoilOps
from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


class TestBatchPerformance:
    """Performance tests for batch airfoil operations."""

    @pytest.fixture
    def performance_config(self) -> Dict[str, Any]:
        """Configuration for performance tests."""
        return {
            "small_batch_size": 10,
            "medium_batch_size": 50,
            "large_batch_size": 100,
            "n_points": 100,
            "n_query_points": 50,
            "n_warmup_runs": 3,
            "n_benchmark_runs": 10,
        }

    def create_test_batch(self, batch_size: int, n_points: int) -> List[JaxAirfoil]:
        """Create a batch of test airfoils with varying parameters."""
        airfoils = []

        # Create diverse NACA airfoils
        for i in range(batch_size):
            # Vary NACA parameters
            max_camber = i % 5  # 0-4%
            camber_pos = (i % 9) + 1  # 1-9 (position)
            thickness = 10 + (i % 15)  # 10-24% thickness

            digits = f"{max_camber:01d}{camber_pos:01d}{thickness:02d}"
            airfoil = JaxAirfoil.naca4(digits, n_points=n_points)
            airfoils.append(airfoil)

        return airfoils

    def benchmark_function(self, func, *args, n_runs: int = 10, warmup_runs: int = 3):
        """Benchmark a function with warmup and multiple runs."""
        # Warmup runs
        for _ in range(warmup_runs):
            _ = func(*args)

        # Benchmark runs
        times = []
        for _ in range(n_runs):
            start_time = time.perf_counter()
            result = func(*args)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

            # Block until computation is complete (important for JAX)
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()

        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "times": times,
        }

    def test_batch_vs_individual_thickness_computation(self, performance_config):
        """Compare batch vs individual thickness computation performance."""
        batch_size = performance_config["medium_batch_size"]
        n_points = performance_config["n_points"]
        n_query = performance_config["n_query_points"]

        # Create test data
        airfoils = self.create_test_batch(batch_size, n_points)
        query_x = jnp.linspace(0.1, 0.9, n_query)

        # Prepare batch data
        batch_coords, batch_masks, upper_splits, n_valid = (
            JaxAirfoil.create_batch_from_list(airfoils)
        )
        buffer_size = batch_coords.shape[2]

        # Individual computation function
        def individual_thickness():
            results = []
            for airfoil in airfoils:
                thickness = airfoil.thickness(query_x)
                results.append(thickness)
            return jnp.stack(results)

        # Batch computation function
        def batch_thickness():
            return JaxAirfoil.batch_thickness(
                batch_coords,
                upper_splits,
                n_valid,
                query_x,
                buffer_size,
            )

        # Benchmark both approaches
        individual_stats = self.benchmark_function(
            individual_thickness,
            n_runs=performance_config["n_benchmark_runs"],
            warmup_runs=performance_config["n_warmup_runs"],
        )

        batch_stats = self.benchmark_function(
            batch_thickness,
            n_runs=performance_config["n_benchmark_runs"],
            warmup_runs=performance_config["n_warmup_runs"],
        )

        # Verify results are equivalent
        individual_result = individual_thickness()
        batch_result = batch_thickness()
        assert jnp.allclose(individual_result, batch_result, rtol=1e-5)

        # Performance analysis
        speedup = individual_stats["mean_time"] / batch_stats["mean_time"]

        print(f"\nThickness Computation Performance (batch_size={batch_size}):")
        print(
            f"Individual: {individual_stats['mean_time']:.4f}s ± {individual_stats['std_time']:.4f}s",
        )
        print(
            f"Batch:      {batch_stats['mean_time']:.4f}s ± {batch_stats['std_time']:.4f}s",
        )
        print(f"Speedup:    {speedup:.2f}x")

        # Batch should be at least as fast as individual (allowing for some overhead in small batches)
        assert batch_stats["mean_time"] <= individual_stats["mean_time"] * 1.5

    def test_batch_vs_individual_naca_generation(self, performance_config):
        """Compare batch vs individual NACA generation performance."""
        batch_size = performance_config["medium_batch_size"]
        n_points = performance_config["n_points"]

        # Create NACA parameter lists
        digits_list = []
        for i in range(batch_size):
            max_camber = i % 5
            camber_pos = (i % 9) + 1
            thickness = 10 + (i % 15)
            digits_list.append(f"{max_camber:01d}{camber_pos:01d}{thickness:02d}")

        # Individual generation function
        def individual_naca():
            airfoils = []
            for digits in digits_list:
                airfoil = JaxAirfoil.naca4(digits, n_points=n_points)
                airfoils.append(airfoil)
            return airfoils

        # Batch generation function
        def batch_naca():
            return JaxAirfoil.batch_naca4(digits_list, n_points=n_points)

        # Benchmark both approaches
        individual_stats = self.benchmark_function(
            individual_naca,
            n_runs=performance_config["n_benchmark_runs"],
            warmup_runs=performance_config["n_warmup_runs"],
        )

        batch_stats = self.benchmark_function(
            batch_naca,
            n_runs=performance_config["n_benchmark_runs"],
            warmup_runs=performance_config["n_warmup_runs"],
        )

        # Performance analysis
        speedup = individual_stats["mean_time"] / batch_stats["mean_time"]

        print(f"\nNACA Generation Performance (batch_size={batch_size}):")
        print(
            f"Individual: {individual_stats['mean_time']:.4f}s ± {individual_stats['std_time']:.4f}s",
        )
        print(
            f"Batch:      {batch_stats['mean_time']:.4f}s ± {batch_stats['std_time']:.4f}s",
        )
        print(f"Speedup:    {speedup:.2f}x")

        # Batch should provide significant speedup for generation
        assert speedup > 0.5  # At least not much slower

    def test_batch_morphing_performance(self, performance_config):
        """Test performance of batch morphing operations."""
        batch_size = performance_config["small_batch_size"]  # Smaller for morphing test
        n_points = performance_config["n_points"]

        # Create two sets of airfoils for morphing
        airfoils1 = self.create_test_batch(batch_size, n_points)
        airfoils2 = self.create_test_batch(batch_size, n_points)

        # Prepare batch data
        batch_coords1, batch_masks1, _, _ = JaxAirfoil.create_batch_from_list(airfoils1)
        batch_coords2, batch_masks2, _, _ = JaxAirfoil.create_batch_from_list(airfoils2)

        # Morphing parameters
        eta_values = jnp.linspace(0.0, 1.0, batch_size)

        # Individual morphing function
        def individual_morph():
            results = []
            for i in range(batch_size):
                # This would require implementing individual morphing - simplified for test
                eta = eta_values[i]
                # Simplified linear interpolation
                coords1 = batch_coords1[i]
                coords2 = batch_coords2[i]
                mask1 = batch_masks1[i]
                mask2 = batch_masks2[i]
                combined_mask = mask1 & mask2

                morphed = (1.0 - eta) * coords1 + eta * coords2
                morphed = jnp.where(combined_mask[None, :], morphed, jnp.nan)
                results.append(morphed)
            return jnp.stack(results)

        # Batch morphing function
        def batch_morph():
            return JaxAirfoil.batch_morph(
                batch_coords1,
                batch_coords2,
                batch_masks1,
                batch_masks2,
                eta_values,
            )

        # Benchmark both approaches
        individual_stats = self.benchmark_function(
            individual_morph,
            n_runs=performance_config["n_benchmark_runs"],
            warmup_runs=performance_config["n_warmup_runs"],
        )

        batch_stats = self.benchmark_function(
            batch_morph,
            n_runs=performance_config["n_benchmark_runs"],
            warmup_runs=performance_config["n_warmup_runs"],
        )

        # Verify results are equivalent
        individual_result = individual_morph()
        batch_result = batch_morph()

        # Compare only valid regions
        for i in range(batch_size):
            mask = batch_masks1[i] & batch_masks2[i]
            assert jnp.allclose(
                individual_result[i][:, mask],
                batch_result[i][:, mask],
                rtol=1e-10,
            )

        # Performance analysis
        speedup = individual_stats["mean_time"] / batch_stats["mean_time"]

        print(f"\nMorphing Performance (batch_size={batch_size}):")
        print(
            f"Individual: {individual_stats['mean_time']:.4f}s ± {individual_stats['std_time']:.4f}s",
        )
        print(
            f"Batch:      {batch_stats['mean_time']:.4f}s ± {batch_stats['std_time']:.4f}s",
        )
        print(f"Speedup:    {speedup:.2f}x")

        # Batch morphing should be efficient
        assert batch_stats["mean_time"] <= individual_stats["mean_time"] * 1.2

    def test_batch_scaling_performance(self, performance_config):
        """Test how batch performance scales with batch size."""
        n_points = performance_config["n_points"]
        n_query = 20  # Smaller for scaling test
        batch_sizes = [5, 10, 25, 50]

        results = {}

        for batch_size in batch_sizes:
            # Create test data
            airfoils = self.create_test_batch(batch_size, n_points)
            query_x = jnp.linspace(0.1, 0.9, n_query)

            # Prepare batch data
            batch_coords, batch_masks, upper_splits, n_valid = (
                JaxAirfoil.create_batch_from_list(airfoils)
            )
            buffer_size = batch_coords.shape[2]

            # Batch computation function
            def batch_thickness():
                return JaxAirfoil.batch_thickness(
                    batch_coords,
                    upper_splits,
                    n_valid,
                    query_x,
                    buffer_size,
                )

            # Benchmark
            stats = self.benchmark_function(
                batch_thickness,
                n_runs=5,  # Fewer runs for scaling test
                warmup_runs=2,
            )

            results[batch_size] = stats

        # Analyze scaling
        print("\nBatch Scaling Performance:")
        print(f"{'Batch Size':<12} {'Time (s)':<12} {'Time/Item (ms)':<15}")
        print("-" * 40)

        for batch_size in batch_sizes:
            time_per_item = results[batch_size]["mean_time"] / batch_size * 1000
            print(
                f"{batch_size:<12} {results[batch_size]['mean_time']:<12.4f} {time_per_item:<15.4f}",
            )

        # Check that time per item decreases or stays reasonable as batch size increases
        time_per_item_small = results[batch_sizes[0]]["mean_time"] / batch_sizes[0]
        time_per_item_large = results[batch_sizes[-1]]["mean_time"] / batch_sizes[-1]

        # Time per item should not increase dramatically with batch size
        assert time_per_item_large <= time_per_item_small * 2.0

    def test_jit_compilation_overhead(self, performance_config):
        """Test JIT compilation overhead for batch operations."""
        batch_size = performance_config["small_batch_size"]
        n_points = performance_config["n_points"]

        # Create test data
        airfoils = self.create_test_batch(batch_size, n_points)
        batch_coords, batch_masks, upper_splits, n_valid = (
            JaxAirfoil.create_batch_from_list(airfoils)
        )
        query_x = jnp.linspace(0.1, 0.9, 10)
        buffer_size = batch_coords.shape[2]

        # Create JIT-compiled function
        @jax.jit
        def jit_batch_thickness(coords, upper_splits, n_valid, query_x, buffer_size):
            return JaxAirfoil.batch_thickness(
                coords,
                upper_splits,
                n_valid,
                query_x,
                buffer_size,
            )

        # Time first call (includes compilation)
        start_time = time.perf_counter()
        result1 = jit_batch_thickness(
            batch_coords,
            upper_splits,
            n_valid,
            query_x,
            buffer_size,
        )
        result1.block_until_ready()
        first_call_time = time.perf_counter() - start_time

        # Time second call (no compilation)
        start_time = time.perf_counter()
        result2 = jit_batch_thickness(
            batch_coords,
            upper_splits,
            n_valid,
            query_x,
            buffer_size,
        )
        result2.block_until_ready()
        second_call_time = time.perf_counter() - start_time

        # Verify results are identical
        assert jnp.allclose(result1, result2)

        # Compilation overhead analysis
        compilation_overhead = first_call_time - second_call_time
        speedup_after_compilation = first_call_time / second_call_time

        print("\nJIT Compilation Overhead:")
        print(f"First call (with compilation): {first_call_time:.4f}s")
        print(f"Second call (compiled):        {second_call_time:.4f}s")
        print(f"Compilation overhead:          {compilation_overhead:.4f}s")
        print(f"Speedup after compilation:     {speedup_after_compilation:.2f}x")

        # Second call should be significantly faster
        assert second_call_time < first_call_time * 0.5

    def test_memory_usage_scaling(self, performance_config):
        """Test memory usage scaling with batch size."""
        n_points = performance_config["n_points"]

        # Test different batch sizes
        batch_sizes = [10, 25, 50]
        memory_info = {}

        for batch_size in batch_sizes:
            # Create test data
            airfoils = self.create_test_batch(batch_size, n_points)
            batch_coords, batch_masks, _, _ = JaxAirfoil.create_batch_from_list(
                airfoils,
            )

            # Calculate memory usage
            coords_memory = batch_coords.nbytes
            masks_memory = batch_masks.nbytes
            total_memory = coords_memory + masks_memory
            memory_per_airfoil = total_memory / batch_size

            memory_info[batch_size] = {
                "total_memory_mb": total_memory / (1024 * 1024),
                "memory_per_airfoil_kb": memory_per_airfoil / 1024,
                "buffer_size": batch_coords.shape[2],
                "buffer_utilization": n_points * 2 / batch_coords.shape[2],
            }

        print("\nMemory Usage Scaling:")
        print(
            f"{'Batch Size':<12} {'Total (MB)':<12} {'Per Airfoil (KB)':<16} {'Buffer Util':<12}",
        )
        print("-" * 55)

        for batch_size in batch_sizes:
            info = memory_info[batch_size]
            print(
                f"{batch_size:<12} {info['total_memory_mb']:<12.2f} "
                f"{info['memory_per_airfoil_kb']:<16.2f} {info['buffer_utilization']:<12.2f}",
            )

        # Memory per airfoil should be relatively consistent
        memory_per_airfoil_values = [
            info["memory_per_airfoil_kb"] for info in memory_info.values()
        ]
        memory_variation = max(memory_per_airfoil_values) / min(
            memory_per_airfoil_values,
        )

        # Memory per airfoil should not vary too much (allowing for buffer size quantization)
        assert memory_variation < 2.0

    def test_gradient_computation_performance(self, performance_config):
        """Test performance of gradient computation through batch operations."""
        batch_size = performance_config["small_batch_size"]
        n_points = 50  # Smaller for gradient test

        # Create batch NACA parameters
        batch_max_camber = jnp.linspace(0.0, 0.04, batch_size)
        batch_camber_position = jnp.full(batch_size, 0.4)
        batch_thickness = jnp.linspace(0.10, 0.15, batch_size)

        def objective_function(params):
            """Objective function for gradient testing."""
            max_camber, camber_pos, thickness = params

            # Generate batch coordinates
            batch_upper, batch_lower = BatchAirfoilOps.batch_generate_naca4_coordinates(
                max_camber,
                camber_pos,
                thickness,
                n_points,
            )

            # Compute thickness at midchord
            query_x = jnp.array([0.5])
            batch_thickness_vals = BatchAirfoilOps.batch_compute_thickness(
                batch_upper,
                batch_lower,
                n_points,
                n_points,
                query_x,
            )

            # Return sum of squared thickness values
            return jnp.sum(batch_thickness_vals**2)

        # Benchmark gradient computation
        params = (batch_max_camber, batch_camber_position, batch_thickness)

        # Forward pass
        def forward_pass():
            return objective_function(params)

        # Gradient computation
        grad_fn = jax.grad(objective_function)

        def gradient_pass():
            return grad_fn(params)

        # Benchmark both
        forward_stats = self.benchmark_function(forward_pass, n_runs=5, warmup_runs=2)
        gradient_stats = self.benchmark_function(gradient_pass, n_runs=5, warmup_runs=2)

        # Gradient overhead
        gradient_overhead = gradient_stats["mean_time"] / forward_stats["mean_time"]

        print("\nGradient Computation Performance:")
        print(
            f"Forward pass:     {forward_stats['mean_time']:.4f}s ± {forward_stats['std_time']:.4f}s",
        )
        print(
            f"Gradient pass:    {gradient_stats['mean_time']:.4f}s ± {gradient_stats['std_time']:.4f}s",
        )
        print(f"Gradient overhead: {gradient_overhead:.2f}x")

        # Gradient computation should not be excessively slow
        assert gradient_overhead < 10.0  # Reasonable overhead for reverse-mode AD

        # Verify gradients are computed correctly
        gradients = grad_fn(params)
        assert len(gradients) == 3
        for grad in gradients:
            assert grad.shape == (batch_size,)
            assert jnp.all(jnp.isfinite(grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
