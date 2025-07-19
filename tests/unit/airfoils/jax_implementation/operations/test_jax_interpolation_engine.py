"""
Tests for JAX-compatible interpolation engine.

This module tests the JaxInterpolationEngine class to ensure correct
interpolation behavior, JIT compatibility, and gradient preservation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ICARUS.airfoils.jax_implementation.interpolation import JaxInterpolationEngine


class TestJaxInterpolationEngine:
    """Test suite for JaxInterpolationEngine."""

    def test_linear_interpolate_1d_basic(self):
        """Test basic linear interpolation functionality."""
        # Simple linear function: y = 2x + 1
        x_coords = jnp.array([0.0, 1.0, 2.0, 3.0, 0.0, 0.0])  # Padded with zeros
        y_coords = jnp.array([1.0, 3.0, 5.0, 7.0, 0.0, 0.0])  # Padded with zeros
        n_valid = 4
        query_x = jnp.array([0.5, 1.5, 2.5])

        result = JaxInterpolationEngine.linear_interpolate_1d(
            x_coords,
            y_coords,
            n_valid,
            query_x,
        )

        expected = jnp.array([2.0, 4.0, 6.0])  # y = 2x + 1
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_linear_interpolate_1d_extrapolation(self):
        """Test extrapolation behavior."""
        x_coords = jnp.array([1.0, 2.0, 3.0, 0.0])  # Padded
        y_coords = jnp.array([2.0, 4.0, 6.0, 0.0])  # y = 2x, padded
        n_valid = 3
        query_x = jnp.array([0.0, 4.0])  # Before and after data range

        result = JaxInterpolationEngine.linear_interpolate_1d(
            x_coords,
            y_coords,
            n_valid,
            query_x,
        )

        # Linear extrapolation: y = 2x
        expected = jnp.array([0.0, 8.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_linear_interpolate_1d_jit_compilation(self):
        """Test that the function compiles with JIT."""
        x_coords = jnp.array([0.0, 1.0, 2.0, 0.0])
        y_coords = jnp.array([0.0, 1.0, 4.0, 0.0])
        n_valid = 3
        query_x = jnp.array([0.5, 1.5])

        # This should compile without errors
        jit_fn = jax.jit(
            JaxInterpolationEngine.linear_interpolate_1d,
            static_argnums=(2,),
        )

        result = jit_fn(x_coords, y_coords, n_valid, query_x)
        assert result.shape == (2,)

    def test_linear_interpolate_1d_gradients(self):
        """Test gradient computation through interpolation."""

        def interpolate_and_sum(coords, query_x):
            x_coords, y_coords = coords[0], coords[1]
            n_valid = 3
            result = JaxInterpolationEngine.linear_interpolate_1d(
                x_coords,
                y_coords,
                n_valid,
                query_x,
            )
            return jnp.sum(result)

        coords = jnp.array(
            [
                [0.0, 1.0, 2.0, 0.0],  # x coordinates
                [0.0, 2.0, 4.0, 0.0],  # y coordinates (y = 2x)
            ],
        )
        query_x = jnp.array([0.5, 1.5])

        # Compute gradients
        grad_fn = jax.grad(interpolate_and_sum, argnums=0)
        gradients = grad_fn(coords, query_x)

        # Gradients should be finite and non-zero for y coordinates
        assert jnp.all(jnp.isfinite(gradients))
        assert jnp.any(gradients[1] != 0)  # y-coordinate gradients should be non-zero

    def test_interpolate_surface_masked(self):
        """Test surface interpolation with masking."""
        # Create airfoil-like coordinates
        coords = jnp.array(
            [
                [0.0, 0.5, 1.0, 0.0, 0.0],  # x coordinates
                [0.0, 0.1, 0.0, 0.0, 0.0],  # y coordinates (simple arch)
            ],
        )
        mask = jnp.array([True, True, True, False, False])
        query_points = jnp.array([0.25, 0.75])

        result = JaxInterpolationEngine.interpolate_surface_masked(
            coords,
            mask,
            query_points,
            n_valid=3,
        )

        # Should interpolate the arch shape
        assert result.shape == (2,)
        assert jnp.all(result >= 0)  # All y values should be non-negative

    def test_query_airfoil_surface(self):
        """Test querying both upper and lower airfoil surfaces."""
        # Simple symmetric airfoil
        upper_coords = jnp.array(
            [
                [0.0, 0.5, 1.0, 0.0],  # x coordinates
                [0.0, 0.1, 0.0, 0.0],  # y coordinates (upper surface)
            ],
        )
        lower_coords = jnp.array(
            [
                [0.0, 0.5, 1.0, 0.0],  # x coordinates
                [0.0, -0.1, 0.0, 0.0],  # y coordinates (lower surface)
            ],
        )

        query_x = jnp.array([0.25, 0.75])

        upper_y, lower_y = JaxInterpolationEngine.query_airfoil_surface(
            upper_coords,
            lower_coords,
            3,
            3,
            query_x,
        )

        # Upper surface should be positive, lower should be negative
        assert jnp.all(upper_y >= 0)
        assert jnp.all(lower_y <= 0)
        # Should be symmetric
        np.testing.assert_allclose(upper_y, -lower_y, rtol=1e-10)

    def test_compute_thickness_distribution(self):
        """Test thickness computation."""
        # Simple symmetric airfoil
        upper_coords = jnp.array(
            [
                [0.0, 0.5, 1.0, 0.0],  # x coordinates
                [0.0, 0.1, 0.0, 0.0],  # y coordinates
            ],
        )
        lower_coords = jnp.array(
            [
                [0.0, 0.5, 1.0, 0.0],  # x coordinates
                [0.0, -0.1, 0.0, 0.0],  # y coordinates
            ],
        )

        query_x = jnp.array([0.0, 0.5, 1.0])

        thickness = JaxInterpolationEngine.compute_thickness_distribution(
            upper_coords,
            lower_coords,
            3,
            query_x,
        )

        # Thickness should be positive
        assert jnp.all(thickness >= 0)
        # Maximum thickness should be at x=0.5
        assert thickness[1] == jnp.max(thickness)
        # Thickness at leading/trailing edge should be zero
        np.testing.assert_allclose(thickness[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(thickness[2], 0.0, atol=1e-10)

    def test_handle_extrapolation_linear(self):
        """Test linear extrapolation."""
        x_coords = jnp.array([1.0, 2.0, 3.0])
        y_coords = jnp.array([2.0, 4.0, 6.0])  # y = 2x

        # Test extrapolation before data range
        result_before = JaxInterpolationEngine.handle_extrapolation(
            x_coords,
            y_coords,
            0.0,
            3,
            "linear",
        )
        np.testing.assert_allclose(result_before, 0.0, rtol=1e-10)

        # Test extrapolation after data range
        result_after = JaxInterpolationEngine.handle_extrapolation(
            x_coords,
            y_coords,
            4.0,
            3,
            "linear",
        )
        np.testing.assert_allclose(result_after, 8.0, rtol=1e-10)

    def test_handle_extrapolation_constant(self):
        """Test constant extrapolation."""
        x_coords = jnp.array([1.0, 2.0, 3.0])
        y_coords = jnp.array([2.0, 4.0, 6.0])

        # Test extrapolation before data range
        result_before = JaxInterpolationEngine.handle_extrapolation(
            x_coords,
            y_coords,
            0.0,
            3,
            "constant",
        )
        np.testing.assert_allclose(result_before, 2.0, rtol=1e-10)

        # Test extrapolation after data range
        result_after = JaxInterpolationEngine.handle_extrapolation(
            x_coords,
            y_coords,
            4.0,
            3,
            "constant",
        )
        np.testing.assert_allclose(result_after, 6.0, rtol=1e-10)

    def test_handle_extrapolation_zero(self):
        """Test zero extrapolation."""
        x_coords = jnp.array([1.0, 2.0, 3.0])
        y_coords = jnp.array([2.0, 4.0, 6.0])

        result = JaxInterpolationEngine.handle_extrapolation(
            x_coords,
            y_coords,
            0.0,
            3,
            "zero",
        )
        np.testing.assert_allclose(result, 0.0, rtol=1e-10)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with single point (should handle gracefully)
        x_coords = jnp.array([1.0, 0.0, 0.0])
        y_coords = jnp.array([2.0, 0.0, 0.0])
        n_valid = 1
        query_x = jnp.array([0.5, 1.5])

        # Should not crash, though results may not be meaningful
        result = JaxInterpolationEngine.linear_interpolate_1d(
            x_coords,
            y_coords,
            n_valid,
            query_x,
        )
        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))

    def test_unsorted_coordinates(self):
        """Test interpolation with unsorted x coordinates."""
        # The interpolate_surface_masked should handle sorting internally
        coords = jnp.array(
            [
                [1.0, 0.0, 0.5, 0.0],  # Unsorted x coordinates
                [2.0, 0.0, 1.0, 0.0],  # Corresponding y coordinates
            ],
        )
        mask = jnp.array([True, True, True, False])
        query_points = jnp.array([0.25, 0.75])

        result = JaxInterpolationEngine.interpolate_surface_masked(
            coords,
            mask,
            query_points,
            n_valid=3,
        )

        # Should handle unsorted data correctly
        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))

    def test_performance_with_large_arrays(self):
        """Test performance with larger arrays."""
        n_points = 1000
        n_valid = 500

        # Create large coordinate arrays
        x_coords = jnp.concatenate(
            [jnp.linspace(0, 1, n_valid), jnp.zeros(n_points - n_valid)],
        )
        y_coords = jnp.concatenate(
            [
                jnp.sin(jnp.linspace(0, 2 * jnp.pi, n_valid)),
                jnp.zeros(n_points - n_valid),
            ],
        )

        query_x = jnp.linspace(0.1, 0.9, 100)

        # Should handle large arrays efficiently
        result = JaxInterpolationEngine.linear_interpolate_1d(
            x_coords,
            y_coords,
            n_valid,
            query_x,
        )

        assert result.shape == (100,)
        assert jnp.all(jnp.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__])
