"""
Performance validation tests for JAX airfoil implementation.

This module tests JIT compilation, performance benchmarks, memory usage,
and optimization characteristics of JAX-based airfoils.
"""

import gc
import time

import jax.numpy as jnp
from jax import grad
from jax import jit
from jax import vmap

from ICARUS.airfoils.naca4 import NACA4


class TestJITCompilation:
    """Test JIT compilation performance and correctness."""

    def test_jit_surface_evaluation(self):
        """Test JIT compilation of surface evaluation methods."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # JIT compile surface methods
        jit_y_upper = jit(naca2412.y_upper)
        jit_y_lower = jit(naca2412.y_lower)
        jit_thickness = jit(naca2412.thickness)

        x_test = jnp.linspace(0, 1, 50)

        # Compare JIT and non-JIT results
        y_upper_normal = naca2412.y_upper(x_test)
        y_upper_jit = jit_y_upper(x_test)
        assert jnp.allclose(y_upper_normal, y_upper_jit, atol=1e-12)

        y_lower_normal = naca2412.y_lower(x_test)
        y_lower_jit = jit_y_lower(x_test)
        assert jnp.allclose(y_lower_normal, y_lower_jit, atol=1e-12)

        thickness_normal = naca2412.thickness(x_test)
        thickness_jit = jit_thickness(x_test)
        assert jnp.allclose(thickness_normal, thickness_jit, atol=1e-12)

    def test_jit_camber_operations(self):
        """Test JIT compilation of camber line operations."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # JIT compile camber methods
        jit_camber = jit(naca2412.camber_line)
        jit_camber_deriv = jit(naca2412.camber_line_derivative)
        jit_thickness_dist = jit(naca2412.thickness_distribution)

        x_test = jnp.linspace(0, 1, 50)

        # Compare results
        camber_normal = naca2412.camber_line(x_test)
        camber_jit = jit_camber(x_test)
        assert jnp.allclose(camber_normal, camber_jit, atol=1e-12)

        camber_deriv_normal = naca2412.camber_line_derivative(x_test)
        camber_deriv_jit = jit_camber_deriv(x_test)
        assert jnp.allclose(camber_deriv_normal, camber_deriv_jit, atol=1e-12)

        thickness_dist_normal = naca2412.thickness_distribution(x_test)
        thickness_dist_jit = jit_thickness_dist(x_test)
        assert jnp.allclose(thickness_dist_normal, thickness_dist_jit, atol=1e-12)

    def test_jit_compilation_timing(self):
        """Test JIT compilation timing and warm-up behavior."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)
        x_test = jnp.linspace(0, 1, 100)

        # Time non-JIT version
        start_time = time.time()
        for _ in range(10):
            result_normal = naca2412.y_upper(x_test)
        normal_time = time.time() - start_time

        # JIT compile and time first call (includes compilation)
        jit_y_upper = jit(naca2412.y_upper)
        start_time = time.time()
        result_jit_first = jit_y_upper(x_test)
        first_call_time = time.time() - start_time

        # Time subsequent JIT calls (no compilation)
        start_time = time.time()
        for _ in range(10):
            result_jit = jit_y_upper(x_test)
        jit_time = time.time() - start_time

        # Verify correctness
        assert jnp.allclose(result_normal, result_jit_first, atol=1e-12)
        assert jnp.allclose(result_normal, result_jit, atol=1e-12)

        # Performance characteristics
        print(f"Normal time (10 calls): {normal_time:.4f}s")
        print(f"JIT first call time: {first_call_time:.4f}s")
        print(f"JIT time (10 calls): {jit_time:.4f}s")

        # JIT should be faster for repeated calls
        if jit_time < normal_time:
            print(f"JIT speedup: {normal_time / jit_time:.2f}x")

    def test_jit_with_different_input_shapes(self):
        """Test JIT compilation with different input shapes."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)
        jit_y_upper = jit(naca2412.y_upper)

        # Test with different shapes
        x_1d = jnp.linspace(0, 1, 20)
        x_2d = jnp.reshape(jnp.linspace(0, 1, 20), (4, 5))
        x_3d = jnp.reshape(jnp.linspace(0, 1, 24), (2, 3, 4))

        # All should work with JIT
        result_1d = jit_y_upper(x_1d)
        result_2d = jit_y_upper(x_2d)
        result_3d = jit_y_upper(x_3d)

        assert result_1d.shape == x_1d.shape
        assert result_2d.shape == x_2d.shape
        assert result_3d.shape == x_3d.shape

    def test_jit_recompilation_behavior(self):
        """Test JIT recompilation with different static arguments."""

        def create_and_evaluate(m, p, xx, n_points, x_point):
            """Function that creates airfoil and evaluates at point."""
            naca = NACA4(M=m, P=p, XX=xx, n_points=n_points)
            return naca.y_upper(x_point)

        # JIT compile with static argument
        jit_func = jit(create_and_evaluate, static_argnums=(3,))

        # Test with same static argument (should not recompile)
        result1 = jit_func(0.02, 0.4, 0.12, 100, 0.5)
        result2 = jit_func(0.03, 0.5, 0.15, 100, 0.6)  # Same n_points

        # Test with different static argument (should recompile)
        result3 = jit_func(0.02, 0.4, 0.12, 200, 0.5)  # Different n_points

        assert isinstance(result1, (float, jnp.ndarray))
        assert isinstance(result2, (float, jnp.ndarray))
        assert isinstance(result3, (float, jnp.ndarray))


class TestPerformanceBenchmarks:
    """Test performance benchmarks and comparisons."""

    def test_surface_evaluation_performance(self):
        """Benchmark surface evaluation performance."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
        x_points = jnp.linspace(0, 1, 1000)

        # Benchmark different operations
        operations = {
            "y_upper": naca2412.y_upper,
            "y_lower": naca2412.y_lower,
            "thickness": naca2412.thickness,
            "camber_line": naca2412.camber_line,
            "thickness_distribution": naca2412.thickness_distribution,
        }

        results = {}
        for name, operation in operations.items():
            # Time the operation
            start_time = time.time()
            for _ in range(100):
                result = operation(x_points)
            elapsed_time = time.time() - start_time

            results[name] = {
                "time": elapsed_time,
                "result_shape": result.shape,
                "ops_per_second": 100 / elapsed_time,
            }

            print(
                f"{name}: {elapsed_time:.4f}s ({results[name]['ops_per_second']:.1f} ops/s)",
            )

        # All operations should complete in reasonable time
        for name, metrics in results.items():
            assert metrics["time"] < 10.0  # Should complete in under 10 seconds
            assert metrics["ops_per_second"] > 1.0  # At least 1 operation per second

    def test_jit_vs_normal_performance(self):
        """Compare JIT vs normal execution performance."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
        x_points = jnp.linspace(0, 1, 500)

        # Normal execution
        start_time = time.time()
        for _ in range(50):
            result_normal = naca2412.y_upper(x_points)
        normal_time = time.time() - start_time

        # JIT execution (with warm-up)
        jit_y_upper = jit(naca2412.y_upper)
        _ = jit_y_upper(x_points)  # Warm-up call

        start_time = time.time()
        for _ in range(50):
            result_jit = jit_y_upper(x_points)
        jit_time = time.time() - start_time

        # Verify correctness
        assert jnp.allclose(result_normal, result_jit, atol=1e-12)

        print(f"Normal execution: {normal_time:.4f}s")
        print(f"JIT execution: {jit_time:.4f}s")

        if jit_time < normal_time:
            speedup = normal_time / jit_time
            print(f"JIT speedup: {speedup:.2f}x")
        else:
            print("JIT overhead detected (expected for small operations)")

    def test_batch_operation_scaling(self):
        """Test performance scaling of batch operations."""

        def single_evaluation(params):
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            x_points = jnp.linspace(0, 1, 50)
            return naca.y_upper(x_points)

        batch_sizes = [1, 5, 10, 20, 50]
        times = []

        for batch_size in batch_sizes:
            # Create parameter batch
            params_batch = jnp.array(
                [
                    [0.02 * (i + 1) / batch_size, 0.4, 0.12 + 0.03 * i / batch_size]
                    for i in range(batch_size)
                ],
            )

            # Time batch operation
            vmap_eval = vmap(single_evaluation)

            start_time = time.time()
            for _ in range(10):
                results = vmap_eval(params_batch)
            elapsed_time = time.time() - start_time

            times.append(elapsed_time)
            time_per_item = elapsed_time / (10 * batch_size)

            print(
                f"Batch size {batch_size}: {elapsed_time:.4f}s ({time_per_item:.6f}s per item)",
            )

        # Performance should scale reasonably
        assert all(t > 0 for t in times)

        # Time per item should not increase dramatically with batch size
        time_per_item_small = times[0] / (10 * batch_sizes[0])
        time_per_item_large = times[-1] / (10 * batch_sizes[-1])
        scaling_factor = time_per_item_large / time_per_item_small

        print(f"Scaling factor: {scaling_factor:.2f}")
        assert scaling_factor < 5.0  # Should not be more than 5x slower per item

    def test_gradient_computation_performance(self):
        """Test performance of gradient computations."""

        def airfoil_objective(params):
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            x_points = jnp.linspace(0, 1, 50)
            y_upper = naca.y_upper(x_points)
            return jnp.sum(y_upper**2)

        params = jnp.array([0.02, 0.4, 0.12])

        # Time function evaluation
        start_time = time.time()
        for _ in range(100):
            value = airfoil_objective(params)
        function_time = time.time() - start_time

        # Time gradient computation
        grad_fn = grad(airfoil_objective)
        start_time = time.time()
        for _ in range(100):
            gradient = grad_fn(params)
        gradient_time = time.time() - start_time

        print(f"Function evaluation: {function_time:.4f}s")
        print(f"Gradient computation: {gradient_time:.4f}s")
        print(f"Gradient overhead: {gradient_time / function_time:.2f}x")

        # Gradient computation should be reasonable
        assert gradient_time < function_time * 10  # Less than 10x overhead
        assert jnp.all(jnp.isfinite(gradient))


class TestMemoryUsage:
    """Test memory usage characteristics."""

    def test_memory_efficiency_surface_evaluation(self):
        """Test memory efficiency of surface evaluation."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=1000)

        # Test with large arrays
        large_x = jnp.linspace(0, 1, 10000)

        # Force garbage collection before test
        gc.collect()

        # Evaluate surfaces (should not cause memory issues)
        y_upper = naca2412.y_upper(large_x)
        y_lower = naca2412.y_lower(large_x)
        thickness = naca2412.thickness(large_x)

        assert y_upper.shape == large_x.shape
        assert y_lower.shape == large_x.shape
        assert thickness.shape[0] <= large_x.shape[0]  # May filter NaN values

        # Clean up
        del y_upper, y_lower, thickness, large_x
        gc.collect()

    def test_memory_efficiency_batch_operations(self):
        """Test memory efficiency of batch operations."""

        def batch_operation(params_batch):
            def single_op(params):
                m, p, xx = params
                naca = NACA4(M=m, P=p, XX=xx, n_points=200)
                x_points = jnp.linspace(0, 1, 100)
                return naca.y_upper(x_points)

            return vmap(single_op)(params_batch)

        # Create large batch
        large_batch_size = 100
        params_batch = jnp.array(
            [
                [
                    0.02 * (i + 1) / large_batch_size,
                    0.4,
                    0.12 + 0.03 * i / large_batch_size,
                ]
                for i in range(large_batch_size)
            ],
        )

        # Force garbage collection
        gc.collect()

        # Execute batch operation (should not cause memory issues)
        results = batch_operation(params_batch)

        assert results.shape == (large_batch_size, 100)
        assert jnp.all(jnp.isfinite(results))

        # Clean up
        del results, params_batch
        gc.collect()

    def test_memory_reuse_jit_compilation(self):
        """Test memory reuse in JIT compiled functions."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        @jit
        def repeated_evaluation(x_points):
            """Function that performs multiple evaluations."""
            y1 = naca2412.y_upper(x_points)
            y2 = naca2412.y_lower(x_points)
            thickness = y1 - y2
            return jnp.sum(thickness)

        x_points = jnp.linspace(0, 1, 1000)

        # Multiple calls should reuse compiled code
        results = []
        for _ in range(10):
            result = repeated_evaluation(x_points)
            results.append(result)

        # All results should be consistent
        for result in results[1:]:
            assert jnp.allclose(result, results[0])

    def test_large_airfoil_handling(self):
        """Test handling of airfoils with many points."""
        # Create airfoil with many points
        large_naca = NACA4(M=0.02, P=0.4, XX=0.12, n_points=5000)

        # NACA4 divides points between upper and lower surfaces
        assert large_naca.n_points == 2500  # n_points // 2
        assert large_naca.max_thickness > 0

        # Test surface evaluation
        x_test = jnp.linspace(0, 1, 100)
        y_upper = large_naca.y_upper(x_test)
        y_lower = large_naca.y_lower(x_test)

        assert y_upper.shape == (100,)
        assert y_lower.shape == (100,)
        assert jnp.all(jnp.isfinite(y_upper))
        assert jnp.all(jnp.isfinite(y_lower))


class TestOptimizationCharacteristics:
    """Test optimization and numerical characteristics."""

    def test_gradient_accuracy(self):
        """Test accuracy of gradient computations."""

        def thickness_at_point(params, x_point):
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            return naca.thickness_distribution(x_point)

        params = jnp.array([0.02, 0.4, 0.12])
        x_point = 0.5

        # Compute analytical gradient
        grad_fn = grad(thickness_at_point, argnums=0)
        analytical_grad = grad_fn(params, x_point)

        # Compute numerical gradient
        eps = 1e-6
        numerical_grad = jnp.zeros_like(params)

        for i in range(len(params)):
            params_plus = params.at[i].add(eps)
            params_minus = params.at[i].add(-eps)

            f_plus = thickness_at_point(params_plus, x_point)
            f_minus = thickness_at_point(params_minus, x_point)

            numerical_grad = numerical_grad.at[i].set((f_plus - f_minus) / (2 * eps))

        # Compare gradients
        relative_error = jnp.abs(
            (analytical_grad - numerical_grad) / (numerical_grad + 1e-12),
        )

        print(f"Analytical gradient: {analytical_grad}")
        print(f"Numerical gradient: {numerical_grad}")
        print(f"Relative error: {relative_error}")

        # Gradients should match within reasonable tolerance
        assert jnp.all(relative_error < 1e-4)

    def test_optimization_convergence(self):
        """Test optimization convergence characteristics."""

        def objective(params):
            """Simple optimization objective."""
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)

            # Objective: minimize difference from target thickness
            target_thickness = 0.10
            actual_thickness = naca.max_thickness

            return (actual_thickness - target_thickness) ** 2

        # Initial guess
        initial_params = jnp.array([0.02, 0.4, 0.08])

        # Optimize (using simple gradient descent since scipy.optimize may not be available)
        learning_rate = 0.01
        params = initial_params

        grad_fn = grad(objective)

        for i in range(100):
            gradient = grad_fn(params)
            params = params - learning_rate * gradient

            # Ensure parameters stay in valid range
            params = jnp.clip(params, 0.001, 0.5)

            if i % 20 == 0:
                obj_value = objective(params)
                print(f"Iteration {i}: objective = {obj_value:.6f}, params = {params}")

        # Final objective should be small
        final_objective = objective(params)
        print(f"Final objective: {final_objective:.6f}")

        assert final_objective < 1e-4  # Should converge to target

    def test_numerical_stability(self):
        """Test numerical stability with extreme parameters."""
        # Test with very small parameters
        small_naca = NACA4(M=1e-6, P=0.1, XX=1e-6, n_points=100)

        x_test = jnp.linspace(0, 1, 50)
        y_upper_small = small_naca.y_upper(x_test)
        y_lower_small = small_naca.y_lower(x_test)

        assert jnp.all(jnp.isfinite(y_upper_small))
        assert jnp.all(jnp.isfinite(y_lower_small))

        # Test with parameters near boundaries
        boundary_naca = NACA4(M=0.09, P=0.9, XX=0.30, n_points=100)

        y_upper_boundary = boundary_naca.y_upper(x_test)
        y_lower_boundary = boundary_naca.y_lower(x_test)

        assert jnp.all(jnp.isfinite(y_upper_boundary))
        assert jnp.all(jnp.isfinite(y_lower_boundary))

    def test_derivative_continuity(self):
        """Test continuity of derivatives."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # Test camber line derivative continuity at p
        p = naca2412.p
        eps = 1e-6

        x_before = p - eps
        x_after = p + eps

        deriv_before = naca2412.camber_line_derivative(x_before)
        deriv_after = naca2412.camber_line_derivative(x_after)

        # Derivatives should be continuous (within numerical precision)
        continuity_error = jnp.abs(deriv_before - deriv_after)
        print(f"Derivative continuity error at p: {continuity_error}")

        # Allow for some numerical error
        assert continuity_error < 1e-3
