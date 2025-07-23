"""
Comprehensive additional edge cases and error handling tests for JAX airfoil implementation.

This module provides additional edge case testing beyond the basic test suite,
focusing on comprehensive boundary conditions, numerical stability, and
production-ready error handling scenarios.
"""

import jax.numpy as jnp
from jax import grad

from ICARUS.airfoils.naca4 import NACA4


class TestNumericalStabilityEdgeCases:
    """Test numerical stability in extreme conditions."""

    def test_extreme_parameter_combinations(self):
        """Test combinations of extreme parameters."""
        extreme_combinations = [
            # (M, P, XX) - various extreme combinations
            (0.0, 0.1, 0.01),  # Zero camber, forward position, thin
            (0.0, 0.9, 0.01),  # Zero camber, aft position, thin
            (0.1, 0.1, 0.01),  # High camber, forward position, thin
            (0.1, 0.9, 0.01),  # High camber, aft position, thin
            (0.0, 0.1, 0.5),  # Zero camber, forward position, thick
            (0.0, 0.9, 0.5),  # Zero camber, aft position, thick
            (0.1, 0.1, 0.5),  # High camber, forward position, thick
            (0.1, 0.9, 0.5),  # High camber, aft position, thick
        ]

        for m, p, xx in extreme_combinations:
            try:
                naca = NACA4(M=m, P=p, XX=xx, n_points=50)

                # Test basic properties
                assert jnp.isfinite(naca.max_thickness)

                # Test surface evaluation
                x_test = jnp.linspace(0.01, 0.99, 20)
                y_upper = naca.y_upper(x_test)
                y_lower = naca.y_lower(x_test)

                assert jnp.all(jnp.isfinite(y_upper))
                assert jnp.all(jnp.isfinite(y_lower))

                # Test thickness is non-negative (allowing small numerical errors)
                thickness = y_upper - y_lower
                assert jnp.all(thickness >= -1e-12)

            except (ValueError, TypeError) as e:
                # Some extreme combinations might be invalid
                print(f"Extreme combination ({m}, {p}, {xx}) failed: {e}")

    def test_precision_at_boundaries(self):
        """Test numerical precision at coordinate boundaries."""
        naca = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # Test very close to boundaries
        boundary_coords = [
            jnp.array([1e-15, 1e-10, 1e-5]),  # Very close to leading edge
            jnp.array([1 - 1e-15, 1 - 1e-10, 1 - 1e-5]),  # Very close to trailing edge
            jnp.array([0.5 - 1e-15, 0.5, 0.5 + 1e-15]),  # Very close to midpoint
        ]

        for coords in boundary_coords:
            y_upper = naca.y_upper(coords)
            y_lower = naca.y_lower(coords)

            # Should handle precision gracefully
            assert jnp.all(jnp.isfinite(y_upper))
            assert jnp.all(jnp.isfinite(y_lower))

            # Thickness should be reasonable
            thickness = y_upper - y_lower
            assert jnp.all(thickness >= -1e-12)

    def test_gradient_precision_stability(self):
        """Test gradient computation precision under various conditions."""

        def thickness_at_point(params, x_eval):
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=30)
            y_upper = naca.y_upper(jnp.array([x_eval]))
            y_lower = naca.y_lower(jnp.array([x_eval]))
            return y_upper[0] - y_lower[0]

        base_params = jnp.array([0.02, 0.4, 0.12])

        # Test gradient at various x positions
        x_positions = [0.1, 0.25, 0.5, 0.75, 0.9]

        for x_pos in x_positions:
            gradient = grad(lambda p: thickness_at_point(p, x_pos))(base_params)

            # Gradient should be finite and reasonable
            assert jnp.all(jnp.isfinite(gradient))
            assert jnp.all(jnp.abs(gradient) < 100)  # Reasonable magnitude

    def test_interpolation_edge_cases(self):
        """Test interpolation behavior at edge cases."""
        naca = NACA4(M=0.02, P=0.4, XX=0.12, n_points=50)

        # Test with single point
        single_point = jnp.array([0.5])
        y_upper_single = naca.y_upper(single_point)
        y_lower_single = naca.y_lower(single_point)

        assert jnp.isfinite(y_upper_single)
        assert jnp.isfinite(y_lower_single)

        # Test with repeated points
        repeated_points = jnp.array([0.3, 0.3, 0.3, 0.7, 0.7])
        y_upper_repeated = naca.y_upper(repeated_points)
        y_lower_repeated = naca.y_lower(repeated_points)

        assert jnp.all(jnp.isfinite(y_upper_repeated))
        assert jnp.all(jnp.isfinite(y_lower_repeated))

        # Repeated evaluations should give same results
        assert jnp.allclose(y_upper_repeated[0], y_upper_repeated[1])
        assert jnp.allclose(y_upper_repeated[1], y_upper_repeated[2])


class TestErrorRecoveryAndRobustness:
    """Test error recovery and system robustness."""

    def test_sequential_error_recovery(self):
        """Test system recovery after sequential errors."""
        # Cause multiple errors in sequence
        error_scenarios = [
            lambda: NACA4.from_digits("abcd"),  # Invalid digits
            lambda: NACA4.from_digits("12345"),  # Too many digits
            lambda: NACA4(M=0.02, P=0.4, XX=0.12, n_points=0),  # Invalid n_points
        ]

        for error_func in error_scenarios:
            try:
                error_func()
            except (ValueError, TypeError, IndexError):
                pass  # Expected errors

        # System should still work normally after errors
        naca_good = NACA4(M=0.02, P=0.4, XX=0.12, n_points=50)
        x_test = jnp.array([0.25, 0.5, 0.75])
        y_upper = naca_good.y_upper(x_test)

        assert jnp.all(jnp.isfinite(y_upper))

    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure conditions."""
        # Test with progressively larger arrays
        sizes = [10, 100, 1000]

        naca = NACA4(M=0.02, P=0.4, XX=0.12, n_points=50)

        for size in sizes:
            try:
                x_large = jnp.linspace(0, 1, size)
                y_upper = naca.y_upper(x_large)
                y_lower = naca.y_lower(x_large)

                assert len(y_upper) == size
                assert len(y_lower) == size
                assert jnp.all(jnp.isfinite(y_upper))
                assert jnp.all(jnp.isfinite(y_lower))

            except MemoryError:
                # Acceptable to fail with memory error for very large arrays
                break

    def test_concurrent_operations_stability(self):
        """Test stability under concurrent operations."""
        naca = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # Simulate concurrent operations
        operations = []

        # Different evaluation points
        for i in range(10):
            x_eval = jnp.linspace(i * 0.1, (i + 1) * 0.1, 10)
            y_upper = naca.y_upper(x_eval)
            y_lower = naca.y_lower(x_eval)
            operations.append((y_upper, y_lower))

        # All operations should succeed
        for y_upper, y_lower in operations:
            assert jnp.all(jnp.isfinite(y_upper))
            assert jnp.all(jnp.isfinite(y_lower))

    def test_parameter_validation_edge_cases(self):
        """Test parameter validation at edge cases."""
        # Test boundary values
        boundary_tests = [
            # Test very small positive values
            (1e-10, 0.5, 0.12),
            (0.02, 1e-10, 0.12),
            (0.02, 0.5, 1e-10),
            # Test values very close to 1
            (0.99, 0.5, 0.12),
            (0.02, 0.99, 0.12),
            (0.02, 0.5, 0.99),
        ]

        for m, p, xx in boundary_tests:
            try:
                naca = NACA4(M=m, P=p, XX=xx, n_points=50)

                # Basic functionality should work
                x_test = jnp.array([0.5])
                y_upper = naca.y_upper(x_test)
                y_lower = naca.y_lower(x_test)

                assert jnp.isfinite(y_upper)
                assert jnp.isfinite(y_lower)

            except (ValueError, TypeError):
                # Some boundary values might be invalid
                pass


class TestAdvancedGradientSafety:
    """Test advanced gradient computation safety."""

    def test_gradient_through_complex_operations(self):
        """Test gradients through complex operation chains."""

        def complex_objective(params):
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=20)

            # Complex operation chain
            x_eval = jnp.linspace(0.1, 0.9, 10)
            y_upper = naca.y_upper(x_eval)
            y_lower = naca.y_lower(x_eval)

            # Combine multiple operations
            thickness = y_upper - y_lower
            area = jnp.trapezoid(thickness, x_eval)
            moment = jnp.trapezoid(thickness * x_eval, x_eval)

            return area + 0.1 * moment

        params = jnp.array([0.02, 0.4, 0.12])

        # Test gradient computation
        gradient = grad(complex_objective)(params)
        assert jnp.all(jnp.isfinite(gradient))
        assert gradient.shape == (3,)

    def test_gradient_consistency_across_evaluations(self):
        """Test gradient consistency across multiple evaluations."""

        def consistent_objective(params):
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=30)

            # Evaluate at multiple points
            x_points = jnp.array([0.2, 0.5, 0.8])
            y_upper = naca.y_upper(x_points)

            return jnp.sum(y_upper**2)

        params = jnp.array([0.02, 0.4, 0.12])

        # Compute gradient multiple times
        gradients = []
        for _ in range(5):
            grad_val = grad(consistent_objective)(params)
            gradients.append(grad_val)

        # All gradients should be identical
        for i in range(1, len(gradients)):
            assert jnp.allclose(gradients[0], gradients[i], rtol=1e-12)

    def test_higher_order_gradient_stability(self):
        """Test stability of higher-order gradients."""

        def simple_thickness_objective(params):
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=20)
            return naca.max_thickness

        params = jnp.array([0.02, 0.4, 0.12])

        # First-order gradient
        first_grad = grad(simple_thickness_objective)(params)
        assert jnp.all(jnp.isfinite(first_grad))

        # Test gradient of gradient (second-order)
        def grad_norm(p):
            g = grad(simple_thickness_objective)(p)
            return jnp.sum(g**2)

        second_order = grad(grad_norm)(params)
        assert jnp.all(jnp.isfinite(second_order))


class TestProductionScenarios:
    """Test realistic production usage scenarios."""

    def test_batch_processing_scenarios(self):
        """Test realistic batch processing scenarios."""
        # Create multiple airfoils with different parameters
        param_sets = [
            (0.01, 0.3, 0.10),
            (0.02, 0.4, 0.12),
            (0.03, 0.5, 0.14),
            (0.04, 0.6, 0.16),
            (0.05, 0.7, 0.18),
        ]

        airfoils = []
        for m, p, xx in param_sets:
            airfoil = NACA4(M=m, P=p, XX=xx, n_points=50)
            airfoils.append(airfoil)

        # Test batch evaluation
        x_eval = jnp.linspace(0, 1, 20)

        results = []
        for airfoil in airfoils:
            y_upper = airfoil.y_upper(x_eval)
            y_lower = airfoil.y_lower(x_eval)
            thickness = y_upper - y_lower
            results.append(thickness)

        # All results should be valid
        for thickness in results:
            assert jnp.all(jnp.isfinite(thickness))
            assert jnp.all(thickness >= -1e-12)

    def test_optimization_workflow_simulation(self):
        """Test simulation of optimization workflow."""

        def airfoil_performance_metric(params):
            """Simulate a performance metric for optimization."""
            m, p, xx = params

            # Clamp parameters to valid ranges
            m = jnp.clip(m, 0.0, 0.1)
            p = jnp.clip(p, 0.1, 0.9)
            xx = jnp.clip(xx, 0.01, 0.3)

            naca = NACA4(M=m, P=p, XX=xx, n_points=30)

            # Simulate performance calculation
            x_eval = jnp.linspace(0.1, 0.9, 15)
            y_upper = naca.y_upper(x_eval)
            y_lower = naca.y_lower(x_eval)

            # Simple performance metric (maximize area, minimize drag proxy)
            area = jnp.trapezoid(y_upper - y_lower, x_eval)
            drag_proxy = jnp.sum((y_upper[1:] - y_upper[:-1]) ** 2)

            return area - 0.1 * drag_proxy

        # Test optimization-like parameter sweep
        initial_params = jnp.array([0.02, 0.4, 0.12])

        # Test gradient computation (needed for optimization)
        gradient = grad(airfoil_performance_metric)(initial_params)
        assert jnp.all(jnp.isfinite(gradient))

        # Test parameter perturbations
        perturbations = [
            jnp.array([0.001, 0.0, 0.0]),
            jnp.array([0.0, 0.01, 0.0]),
            jnp.array([0.0, 0.0, 0.001]),
        ]

        base_performance = airfoil_performance_metric(initial_params)

        for perturbation in perturbations:
            perturbed_params = initial_params + perturbation
            perturbed_performance = airfoil_performance_metric(perturbed_params)

            assert jnp.isfinite(base_performance)
            assert jnp.isfinite(perturbed_performance)

    def test_real_world_coordinate_patterns(self):
        """Test with realistic coordinate evaluation patterns."""
        naca = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # Common evaluation patterns in real applications
        evaluation_patterns = [
            # Dense near leading edge
            jnp.concatenate([jnp.linspace(0, 0.1, 20), jnp.linspace(0.1, 1.0, 30)]),
            # Dense near trailing edge
            jnp.concatenate([jnp.linspace(0, 0.9, 30), jnp.linspace(0.9, 1.0, 20)]),
            # Cosine distribution (common in CFD)
            0.5 * (1 - jnp.cos(jnp.linspace(0, jnp.pi, 50))),
            # Random sampling (Monte Carlo applications)
            jnp.sort(
                jnp.array([0.1, 0.23, 0.45, 0.67, 0.89, 0.34, 0.78, 0.12, 0.56, 0.91]),
            ),
        ]

        for x_pattern in evaluation_patterns:
            y_upper = naca.y_upper(x_pattern)
            y_lower = naca.y_lower(x_pattern)

            assert len(y_upper) == len(x_pattern)
            assert len(y_lower) == len(x_pattern)
            assert jnp.all(jnp.isfinite(y_upper))
            assert jnp.all(jnp.isfinite(y_lower))

            # Thickness should be reasonable
            thickness = y_upper - y_lower
            assert jnp.all(thickness >= -1e-12)

    def test_long_running_computation_stability(self):
        """Test stability over long-running computations."""
        naca = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # Simulate long-running computation
        x_base = jnp.linspace(0, 1, 50)

        results = []
        for i in range(100):  # Many iterations
            # Slight variations in evaluation points
            x_eval = x_base + 1e-6 * jnp.sin(i * x_base)
            x_eval = jnp.clip(x_eval, 0, 1)  # Keep in valid range

            y_upper = naca.y_upper(x_eval)
            y_lower = naca.y_lower(x_eval)

            # Store some metric
            max_thickness = jnp.max(y_upper - y_lower)
            results.append(max_thickness)

        # Results should be stable and finite
        results = jnp.array(results)
        assert jnp.all(jnp.isfinite(results))

        # Should have reasonable consistency (allowing for small variations)
        mean_result = jnp.mean(results)
        std_result = jnp.std(results)
        assert std_result < 0.1 * mean_result  # Less than 10% variation
