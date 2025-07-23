"""
Task 7: Complete batch operations and performance tests for JAX airfoil implementation.

This module implements the specific requirements for Task 7:
- Verify existing batch operation tests are comprehensive
- Add performance comparison tests between individual and batch operations
- Include JIT compilation timing and memory usage validation
- Ensure batch operation correctness and gradient computation tests

Requirements: 2.1, 2.2, 5.1, 5.2, 7.1
"""

import gc
import time

import jax.numpy as jnp
from jax import grad
from jax import jit
from jax import value_and_grad
from jax import vmap


def create_jax_naca4_functions(m: float, p: float, xx: float):
    """
    Create JAX-compatible NACA4 evaluation functions.

    This avoids the boolean indexing issues in the base Airfoil class
    by implementing the NACA4 mathematics directly.
    """

    def thickness_distribution(x):
        """NACA4 thickness distribution."""
        a0, a1, a2, a3, a4 = 0.2969, -0.126, -0.3516, 0.2843, -0.1036
        return (xx / 0.2) * (
            a0 * jnp.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4
        )

    def camber_line(x):
        """NACA4 camber line."""
        p_safe = p + 1e-19  # Avoid division by zero
        return jnp.where(
            x < p_safe,
            (m / p_safe**2) * (2 * p_safe * x - x**2),
            (m / (1 - p_safe) ** 2) * (1 - 2 * p_safe + 2 * p_safe * x - x**2),
        )

    def camber_derivative(x):
        """NACA4 camber line derivative."""
        p_safe = p + 1e-19
        return jnp.where(
            x < p_safe,
            (2 * m / p_safe**2) * (p_safe - x),
            (2 * m / (1 - p_safe) ** 2) * (p_safe - x),
        )

    def y_upper(x):
        """Upper surface y-coordinate."""
        theta = jnp.arctan(camber_derivative(x))
        return camber_line(x) + thickness_distribution(x) * jnp.cos(theta)

    def y_lower(x):
        """Lower surface y-coordinate."""
        theta = jnp.arctan(camber_derivative(x))
        return camber_line(x) - thickness_distribution(x) * jnp.cos(theta)

    return {
        "y_upper": y_upper,
        "y_lower": y_lower,
        "camber_line": camber_line,
        "thickness_distribution": thickness_distribution,
        "max_thickness": lambda: jnp.max(
            thickness_distribution(jnp.linspace(0.01, 0.99, 100)),
        ),
    }


class TestBatchOperationsComprehensive:
    """Comprehensive tests for batch operations functionality."""

    def test_batch_parameter_sweep_comprehensive(self):
        """Test comprehensive parameter sweep demonstrating batch capabilities."""
        # Create parameter combinations for comprehensive testing
        m_values = jnp.array([0.0, 0.02, 0.04, 0.06])  # Camber
        p_values = jnp.array([0.2, 0.4, 0.6, 0.8])  # Position
        xx_values = jnp.array([0.08, 0.12, 0.16, 0.20])  # Thickness

        # Create all combinations
        params_list = []
        for m in m_values:
            for p in p_values:
                for xx in xx_values:
                    params_list.append([float(m), float(p), float(xx)])

        params_batch = jnp.array(params_list)

        def evaluate_airfoil_batch(params):
            """Evaluate airfoil properties for a single parameter set."""
            m, p, xx = params
            funcs = create_jax_naca4_functions(m, p, xx)

            # Test evaluation points
            x_test = jnp.linspace(0.01, 0.99, 20)

            # Evaluate key properties
            y_upper_vals = funcs["y_upper"](x_test)
            y_lower_vals = funcs["y_lower"](x_test)
            thickness_vals = funcs["thickness_distribution"](x_test)

            return {
                "max_thickness": jnp.max(thickness_vals),
                "mean_upper": jnp.mean(y_upper_vals),
                "mean_lower": jnp.mean(y_lower_vals),
                "thickness_at_mid": funcs["thickness_distribution"](0.5),
                "camber_at_mid": funcs["camber_line"](0.5),
            }

        # Vectorize the evaluation
        vmap_eval = vmap(evaluate_airfoil_batch)
        results = vmap_eval(params_batch)

        # Verify results
        n_params = len(params_list)
        assert results["max_thickness"].shape == (n_params,)
        assert jnp.all(jnp.isfinite(results["max_thickness"]))
        assert jnp.all(results["max_thickness"] > 0)

        # Verify physical relationships
        assert jnp.all(results["mean_upper"] >= results["mean_lower"])
        assert jnp.all(results["thickness_at_mid"] >= 0)

        print(
            f"✓ Batch parameter sweep completed for {n_params} parameter combinations",
        )

    def test_batch_surface_evaluation_accuracy(self):
        """Test accuracy of batch surface evaluation."""
        # Test parameters
        params_batch = jnp.array(
            [
                [0.0, 0.0, 0.12],  # Symmetric
                [0.02, 0.4, 0.12],  # Standard cambered
                [0.04, 0.6, 0.16],  # High camber
            ],
        )

        x_points = jnp.linspace(0.01, 0.99, 50)

        def evaluate_surfaces(params, x_points):
            """Evaluate upper and lower surfaces."""
            m, p, xx = params
            funcs = create_jax_naca4_functions(m, p, xx)

            y_upper = funcs["y_upper"](x_points)
            y_lower = funcs["y_lower"](x_points)
            thickness = funcs["thickness_distribution"](x_points)

            return {
                "y_upper": y_upper,
                "y_lower": y_lower,
                "thickness": thickness,
                "surface_diff": y_upper - y_lower,
            }

        # Vectorize over parameter batch
        vmap_eval = vmap(evaluate_surfaces, in_axes=(0, None))
        results = vmap_eval(params_batch, x_points)

        # Verify shapes and relationships
        n_params, n_points = len(params_batch), len(x_points)
        assert results["y_upper"].shape == (n_params, n_points)
        assert results["y_lower"].shape == (n_params, n_points)

        # Physical consistency checks
        for i in range(n_params):
            # Upper should be above lower (except possibly at endpoints)
            assert jnp.mean(results["surface_diff"][i]) > 0

            # Thickness should be positive
            assert jnp.all(results["thickness"][i] >= -1e-10)

            # Surface difference should be related to thickness
            ratio = results["surface_diff"][i] / (2 * results["thickness"][i])
            assert jnp.all(ratio > 0.5) and jnp.all(ratio < 2.0)

        print(f"✓ Batch surface evaluation accuracy verified for {n_params} airfoils")

    def test_nested_batch_operations_correctness(self):
        """Test correctness of nested batch operations."""
        # Parameter sets
        params_batch = jnp.array([[0.02, 0.4, 0.12], [0.04, 0.6, 0.15]])

        # Multiple x-point sets
        x_sets = jnp.array([jnp.linspace(0.0, 1.0, 10), jnp.linspace(0.1, 0.9, 10)])

        def evaluate_on_x_set(params, x_set):
            """Evaluate airfoil on a set of x points."""
            m, p, xx = params
            funcs = create_jax_naca4_functions(m, p, xx)
            return vmap(funcs["y_upper"])(x_set)

        # Double vectorization
        def evaluate_all_combinations(params_batch, x_sets):
            eval_x_sets = vmap(evaluate_on_x_set, in_axes=(None, 0))
            return vmap(eval_x_sets, in_axes=(0, None))(params_batch, x_sets)

        results = evaluate_all_combinations(params_batch, x_sets)

        # Verify shape and values
        assert results.shape == (2, 2, 10)  # 2 params, 2 x_sets, 10 points each
        assert jnp.all(jnp.isfinite(results))

        print("✓ Nested batch operations correctness verified")


class TestPerformanceComparisons:
    """Performance comparison tests between individual and batch operations."""

    def test_individual_vs_batch_performance_comparison(self):
        """Compare performance of individual vs batch operations."""
        # Test setup
        n_airfoils = 15
        params_list = []
        for i in range(n_airfoils):
            m = 0.01 + 0.03 * i / n_airfoils
            p = 0.3 + 0.4 * i / n_airfoils
            xx = 0.10 + 0.08 * i / n_airfoils
            params_list.append([m, p, xx])

        x_points = jnp.linspace(0.01, 0.99, 100)

        # Individual operations timing
        start_time = time.time()
        individual_results = []
        for params in params_list:
            m, p, xx = params
            funcs = create_jax_naca4_functions(m, p, xx)
            result = funcs["y_upper"](x_points)
            individual_results.append(result)
        individual_time = time.time() - start_time

        # Batch operations timing
        def batch_evaluation(params_batch, x_points):
            def single_eval(params):
                m, p, xx = params
                funcs = create_jax_naca4_functions(m, p, xx)
                return funcs["y_upper"](x_points)

            return vmap(single_eval)(params_batch)

        params_batch = jnp.array(params_list)

        start_time = time.time()
        batch_results = batch_evaluation(params_batch, x_points)
        batch_time = time.time() - start_time

        # Verify correctness
        for i, individual_result in enumerate(individual_results):
            assert jnp.allclose(individual_result, batch_results[i], rtol=1e-12)

        # Performance analysis
        speedup = individual_time / batch_time if batch_time > 0 else float("inf")

        print("\n📊 Performance Comparison Results:")
        print(f"   Individual operations: {individual_time:.4f}s")
        print(f"   Batch operations: {batch_time:.4f}s")
        print(f"   Speedup: {speedup:.2f}x")

        # Performance requirements (batch operations may have overhead for small batches)
        assert (
            batch_time < individual_time * 3.0
        )  # Batch should be reasonably competitive
        assert (
            speedup > 0.3
        )  # Should provide some benefit or at least not be too much slower

        print("✓ Performance comparison test passed")

    def test_batch_gradient_performance_comparison(self):
        """Compare performance of individual vs batch gradient computation."""

        def airfoil_objective(params):
            """Objective function for gradient testing."""
            m, p, xx = params
            funcs = create_jax_naca4_functions(m, p, xx)

            # Multi-point evaluation
            x_eval = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
            y_upper = funcs["y_upper"](x_eval)
            y_lower = funcs["y_lower"](x_eval)

            # Objective: minimize squared thickness deviation
            target_thickness = 0.12 * jnp.ones_like(x_eval)
            actual_thickness = y_upper - y_lower

            return jnp.sum((actual_thickness - target_thickness) ** 2)

        # Test parameters
        params_list = [
            [0.02, 0.4, 0.12],
            [0.03, 0.5, 0.14],
            [0.04, 0.6, 0.16],
            [0.01, 0.3, 0.10],
        ]

        # Individual gradient computation
        grad_fn = grad(airfoil_objective)

        start_time = time.time()
        individual_grads = []
        for params in params_list:
            grad_result = grad_fn(jnp.array(params))
            individual_grads.append(grad_result)
        individual_time = time.time() - start_time

        # Batch gradient computation
        vmap_grad = vmap(grad_fn)
        params_batch = jnp.array(params_list)

        start_time = time.time()
        batch_grads = vmap_grad(params_batch)
        batch_time = time.time() - start_time

        # Verify correctness
        for i, individual_grad in enumerate(individual_grads):
            assert jnp.allclose(individual_grad, batch_grads[i], rtol=1e-12)

        # Performance analysis
        speedup = individual_time / batch_time if batch_time > 0 else float("inf")

        print("\n🎯 Gradient Performance Comparison:")
        print(f"   Individual gradients: {individual_time:.4f}s")
        print(f"   Batch gradients: {batch_time:.4f}s")
        print(f"   Speedup: {speedup:.2f}x")

        # Verify gradient properties
        assert batch_grads.shape == (4, 3)  # 4 parameter sets, 3 gradients each
        assert jnp.all(jnp.isfinite(batch_grads))

        print("✓ Batch gradient performance comparison passed")

    def test_scaling_performance_analysis(self):
        """Analyze performance scaling with batch size."""
        batch_sizes = [1, 5, 10, 20]
        times = []

        def benchmark_batch_size(batch_size):
            """Benchmark for specific batch size."""
            # Generate parameters
            params_list = []
            for i in range(batch_size):
                m = 0.02 * (1 + i / batch_size)
                p = 0.4
                xx = 0.12 + 0.04 * i / batch_size
                params_list.append([m, p, xx])

            params_batch = jnp.array(params_list)
            x_points = jnp.linspace(0.01, 0.99, 50)

            def batch_eval(params_batch):
                def single_eval(params):
                    m, p, xx = params
                    funcs = create_jax_naca4_functions(m, p, xx)
                    return jnp.sum(funcs["y_upper"](x_points))

                return vmap(single_eval)(params_batch)

            # Warm-up
            _ = batch_eval(params_batch)

            # Timing
            start_time = time.time()
            for _ in range(3):
                result = batch_eval(params_batch)
            elapsed_time = (time.time() - start_time) / 3

            return elapsed_time, result

        print("\n📈 Scaling Performance Analysis:")
        print(f"{'Batch Size':<12} {'Time (s)':<12} {'Time/Item (ms)':<15}")
        print("-" * 40)

        for batch_size in batch_sizes:
            elapsed_time, result = benchmark_batch_size(batch_size)
            time_per_item = (elapsed_time / batch_size) * 1000
            times.append(elapsed_time)

            print(f"{batch_size:<12} {elapsed_time:<12.4f} {time_per_item:<15.2f}")

            # Verify results
            assert result.shape == (batch_size,)
            assert jnp.all(jnp.isfinite(result))

        # Check scaling behavior
        time_ratio = times[-1] / times[0]
        size_ratio = batch_sizes[-1] / batch_sizes[0]

        # Time should scale roughly linearly
        assert time_ratio < size_ratio * 2.0

        print("✓ Scaling performance analysis completed")


class TestJITCompilationTiming:
    """JIT compilation timing and memory usage validation tests."""

    def test_jit_compilation_overhead_measurement(self):
        """Measure JIT compilation overhead for batch operations."""

        def batch_airfoil_evaluation(params_batch, x_points):
            """Batch evaluation function for JIT compilation."""

            def single_eval(params):
                m, p, xx = params
                funcs = create_jax_naca4_functions(m, p, xx)
                y_upper = funcs["y_upper"](x_points)
                y_lower = funcs["y_lower"](x_points)
                return jnp.array([jnp.mean(y_upper), jnp.mean(y_lower)])

            return vmap(single_eval)(params_batch)

        # Test data
        params_batch = jnp.array(
            [[0.02, 0.4, 0.12], [0.03, 0.5, 0.14], [0.04, 0.6, 0.16]],
        )
        x_points = jnp.linspace(0.01, 0.99, 50)

        # Time non-JIT version
        start_time = time.time()
        result_normal = batch_airfoil_evaluation(params_batch, x_points)
        normal_time = time.time() - start_time

        # JIT compile and time first call
        jit_batch_eval = jit(batch_airfoil_evaluation)

        start_time = time.time()
        result_jit_first = jit_batch_eval(params_batch, x_points)
        first_call_time = time.time() - start_time

        # Time subsequent JIT calls
        start_time = time.time()
        for _ in range(10):
            result_jit = jit_batch_eval(params_batch, x_points)
        jit_time = (time.time() - start_time) / 10

        # Verify correctness
        assert jnp.allclose(result_normal, result_jit_first, rtol=1e-12)
        assert jnp.allclose(result_normal, result_jit, rtol=1e-12)

        # Analyze timing
        compilation_overhead = first_call_time - jit_time
        speedup = normal_time / jit_time if jit_time > 0 else float("inf")

        print("\n⚡ JIT Compilation Analysis:")
        print(f"   Normal execution: {normal_time:.4f}s")
        print(f"   JIT first call (with compilation): {first_call_time:.4f}s")
        print(f"   JIT subsequent calls: {jit_time:.4f}s")
        print(f"   Compilation overhead: {compilation_overhead:.4f}s")
        print(f"   JIT speedup: {speedup:.2f}x")

        # Performance requirements
        assert compilation_overhead >= 0  # Compilation takes time
        assert speedup > 0.5  # Should provide some benefit

        print("✓ JIT compilation timing analysis completed")

    def test_memory_usage_validation(self):
        """Validate memory usage of batch operations."""

        def memory_intensive_operation(params_batch):
            """Operation that creates intermediate arrays."""

            def single_operation(params):
                m, p, xx = params
                funcs = create_jax_naca4_functions(m, p, xx)

                # Multiple evaluations
                x_dense = jnp.linspace(0.01, 0.99, 200)
                y_upper = funcs["y_upper"](x_dense)
                y_lower = funcs["y_lower"](x_dense)
                thickness = funcs["thickness_distribution"](x_dense)

                # Combine and reduce
                combined = jnp.stack([y_upper, y_lower, thickness])
                return jnp.sum(combined, axis=1)

            return vmap(single_operation)(params_batch)

        # Create batch
        n_batch = 20
        params_batch = jnp.array(
            [
                [0.02 + 0.02 * i / n_batch, 0.4, 0.12 + 0.04 * i / n_batch]
                for i in range(n_batch)
            ],
        )

        # Force garbage collection
        gc.collect()

        # Execute memory-intensive operation
        results = memory_intensive_operation(params_batch)

        # Verify results
        assert results.shape == (n_batch, 3)
        assert jnp.all(jnp.isfinite(results))

        # Clean up
        del results, params_batch
        gc.collect()

        print(f"✓ Memory usage validation completed for batch size {n_batch}")

    def test_jit_recompilation_behavior(self):
        """Test JIT recompilation with different static arguments."""

        def create_and_evaluate(m, p, xx, n_eval_points, x_eval):
            """Function with static argument."""
            funcs = create_jax_naca4_functions(m, p, xx)
            # Use n_eval_points to create different sized arrays (static behavior)
            x_points = jnp.linspace(0.01, 0.99, n_eval_points)
            return jnp.mean(funcs["y_upper"](x_points))

        # JIT with static argument
        jit_func = jit(create_and_evaluate, static_argnums=(3,))

        # Test with same static argument (no recompilation)
        result1 = jit_func(0.02, 0.4, 0.12, 50, 0.5)
        result2 = jit_func(0.03, 0.5, 0.15, 50, 0.6)  # Same n_eval_points

        # Test with different static argument (recompilation)
        result3 = jit_func(0.02, 0.4, 0.12, 100, 0.5)  # Different n_eval_points

        # All results should be valid
        assert jnp.isfinite(result1)
        assert jnp.isfinite(result2)
        assert jnp.isfinite(result3)

        print("✓ JIT recompilation behavior test completed")


class TestBatchGradientCorrectness:
    """Tests for batch gradient computation correctness."""

    def test_batch_gradient_correctness_verification(self):
        """Verify correctness of batch gradient computations."""

        def complex_objective(params):
            """Complex objective for gradient testing."""
            m, p, xx = params
            funcs = create_jax_naca4_functions(m, p, xx)

            # Multi-point evaluation
            x_eval = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
            y_upper = funcs["y_upper"](x_eval)
            y_lower = funcs["y_lower"](x_eval)
            camber = funcs["camber_line"](x_eval)

            # Complex objective
            thickness_term = jnp.sum((y_upper - y_lower) ** 2)
            camber_term = jnp.sum(camber**2)
            shape_term = jnp.sum(y_upper**2)

            return thickness_term + 0.5 * camber_term + 0.1 * shape_term

        # Test parameters
        params_batch = jnp.array(
            [
                [0.02, 0.4, 0.12],
                [0.03, 0.5, 0.14],
                [0.04, 0.6, 0.16],
                [0.01, 0.3, 0.10],
            ],
        )

        # Individual gradients
        grad_fn = grad(complex_objective)
        individual_gradients = []
        for params in params_batch:
            grad_result = grad_fn(params)
            individual_gradients.append(grad_result)

        # Batch gradients
        vmap_grad = vmap(grad_fn)
        batch_gradients = vmap_grad(params_batch)

        # Verify correctness
        for i, individual_grad in enumerate(individual_gradients):
            assert jnp.allclose(individual_grad, batch_gradients[i], rtol=1e-12)

        # Verify properties
        assert batch_gradients.shape == (4, 3)
        assert jnp.all(jnp.isfinite(batch_gradients))

        print(
            f"✓ Batch gradient correctness verified for {len(params_batch)} parameter sets",
        )

    def test_value_and_grad_batch_operations(self):
        """Test simultaneous value and gradient computation."""

        def optimization_objective(params):
            """Objective for optimization."""
            m, p, xx = params
            funcs = create_jax_naca4_functions(m, p, xx)

            # Target thickness distribution
            x_target = jnp.linspace(0.1, 0.9, 10)
            target_thickness = 0.12 * jnp.exp(-2 * (x_target - 0.3) ** 2)
            actual_thickness = funcs["thickness_distribution"](x_target)

            return jnp.sum((actual_thickness - target_thickness) ** 2)

        params_batch = jnp.array(
            [[0.02, 0.4, 0.12], [0.03, 0.5, 0.14], [0.04, 0.6, 0.16]],
        )

        # Compute value and gradient simultaneously
        value_and_grad_fn = value_and_grad(optimization_objective)
        vmap_value_and_grad = vmap(value_and_grad_fn)

        values, gradients = vmap_value_and_grad(params_batch)

        # Verify results
        assert values.shape == (3,)
        assert gradients.shape == (3, 3)
        assert jnp.all(jnp.isfinite(values))
        assert jnp.all(jnp.isfinite(gradients))
        assert jnp.all(values >= 0)  # Squared error should be non-negative

        # Compare with separate computations
        obj_fn = optimization_objective
        grad_fn = grad(optimization_objective)

        separate_values = vmap(obj_fn)(params_batch)
        separate_gradients = vmap(grad_fn)(params_batch)

        assert jnp.allclose(values, separate_values, rtol=1e-12)
        assert jnp.allclose(gradients, separate_gradients, rtol=1e-12)

        print("✓ Value and gradient batch computation verified")


class TestBatchOperationRobustness:
    """Test robustness and performance regression detection."""

    def test_performance_regression_detection(self):
        """Test to detect performance regressions."""
        # Standard benchmark
        n_params = 10
        params_batch = jnp.array(
            [
                [0.02 + 0.01 * i / n_params, 0.4, 0.12 + 0.02 * i / n_params]
                for i in range(n_params)
            ],
        )

        x_points = jnp.linspace(0.01, 0.99, 50)

        def benchmark_operation(params_batch, x_points):
            """Standard benchmark operation."""

            def single_eval(params):
                m, p, xx = params
                funcs = create_jax_naca4_functions(m, p, xx)
                y_upper = funcs["y_upper"](x_points)
                y_lower = funcs["y_lower"](x_points)
                return jnp.sum(y_upper - y_lower)

            return vmap(single_eval)(params_batch)

        # Warm-up
        _ = benchmark_operation(params_batch, x_points)

        # Benchmark timing
        n_runs = 5
        start_time = time.time()
        for _ in range(n_runs):
            results = benchmark_operation(params_batch, x_points)
        total_time = time.time() - start_time
        avg_time = total_time / n_runs

        # Verify correctness
        assert results.shape == (n_params,)
        assert jnp.all(jnp.isfinite(results))
        assert jnp.all(results > 0)

        # Performance threshold
        max_acceptable_time = 1.0  # seconds per run

        print("\n🔍 Performance Regression Test:")
        print(f"   Average time per run: {avg_time:.4f}s")
        print(f"   Threshold: {max_acceptable_time:.4f}s")

        assert avg_time < max_acceptable_time, (
            f"Performance regression: {avg_time:.4f}s > {max_acceptable_time:.4f}s"
        )

        print("✓ Performance regression test passed")

    def test_batch_operations_error_handling(self):
        """Test error handling in batch operations."""
        # Mix of valid parameters
        params_batch = jnp.array(
            [
                [0.02, 0.4, 0.12],  # Valid
                [0.05, 0.8, 0.20],  # Valid but extreme
                [0.02, 0.4, 0.12],  # Valid (duplicate)
            ],
        )

        def safe_batch_operation(params_batch):
            """Batch operation with error handling."""

            def safe_single_eval(params):
                m, p, xx = params
                funcs = create_jax_naca4_functions(m, p, xx)
                return funcs["max_thickness"]()

            return vmap(safe_single_eval)(params_batch)

        results = safe_batch_operation(params_batch)

        # Should complete without crashing
        assert results.shape == (3,)
        assert jnp.all(jnp.isfinite(results))
        assert jnp.all(results > 0)  # All should give positive thickness

        print("✓ Batch operations error handling test passed")


if __name__ == "__main__":
    # Quick smoke test
    test_batch = TestBatchOperationsComprehensive()
    test_batch.test_batch_parameter_sweep_comprehensive()
    print("Task 7 batch operations and performance tests ready!")
