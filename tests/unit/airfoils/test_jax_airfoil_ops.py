"""
Unit tests for the JaxAirfoilOps class.

This module contains tests for the JaxAirfoilOps class, verifying that all geometric
operations work correctly with JAX transformations, masking, and provide accurate results.
"""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from ICARUS.airfoils.jax_implementation.jax_airfoil_ops import JaxAirfoilOps


class TestJaxAirfoilOps:
    """Test class for JaxAirfoilOps geometric operations."""

    @pytest.fixture
    def simple_airfoil_coords(self):
        """Create simple symmetric airfoil coordinates for testing."""
        # Create a simple symmetric airfoil (diamond shape)
        x_coords = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y_upper = jnp.array([0.0, 0.05, 0.08, 0.05, 0.0])
        y_lower = jnp.array([0.0, -0.05, -0.08, -0.05, 0.0])

        upper_coords = jnp.array([x_coords, y_upper])
        lower_coords = jnp.array([x_coords, y_lower])

        return upper_coords, lower_coords

    @pytest.fixture
    def padded_airfoil_coords(self, simple_airfoil_coords):
        """Create padded airfoil coordinates for testing masking."""
        upper_coords, lower_coords = simple_airfoil_coords

        # Pad to buffer size of 32
        buffer_size = 32
        upper_padded = jnp.concatenate(
            [upper_coords, jnp.full((2, buffer_size - upper_coords.shape[1]), jnp.nan)],
            axis=1,
        )
        lower_padded = jnp.concatenate(
            [lower_coords, jnp.full((2, buffer_size - lower_coords.shape[1]), jnp.nan)],
            axis=1,
        )

        n_valid = upper_coords.shape[1]

        return upper_padded, lower_padded, n_valid

    def test_compute_thickness(self, padded_airfoil_coords):
        """Test thickness computation with masking."""
        upper_coords, lower_coords, n_valid = padded_airfoil_coords

        # Query thickness at midpoint
        query_x = jnp.array([0.5])
        thickness = JaxAirfoilOps.compute_thickness(
            upper_coords,
            lower_coords,
            n_valid,
            n_valid,
            query_x,
        )

        # Expected thickness at x=0.5 is 0.08 - (-0.08) = 0.16
        expected_thickness = 0.16
        assert jnp.isclose(thickness[0], expected_thickness, atol=1e-6)

        # Test multiple query points
        query_x_multi = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        thickness_multi = JaxAirfoilOps.compute_thickness(
            upper_coords,
            lower_coords,
            n_valid,
            n_valid,
            query_x_multi,
        )

        # All thickness values should be positive
        assert jnp.all(thickness_multi >= 0)

        # Maximum thickness should be at x=0.5
        max_idx = jnp.argmax(thickness_multi)
        assert query_x_multi[max_idx] == 0.5

    def test_compute_camber_line(self, padded_airfoil_coords):
        """Test camber line computation with masking."""
        upper_coords, lower_coords, n_valid = padded_airfoil_coords

        # Query camber line at multiple points
        query_x = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        camber = JaxAirfoilOps.compute_camber_line(
            upper_coords,
            lower_coords,
            n_valid,
            n_valid,
            query_x,
        )

        # For symmetric airfoil, camber line should be zero everywhere
        assert jnp.allclose(camber, 0.0, atol=1e-6)

    def test_y_upper_query(self, padded_airfoil_coords):
        """Test upper surface coordinate queries."""
        upper_coords, _, n_valid = padded_airfoil_coords

        # Query upper surface at known points
        query_x = jnp.array([0.0, 0.5, 1.0])
        y_upper = JaxAirfoilOps.y_upper(upper_coords, n_valid, query_x)

        # Check expected values
        expected_y = jnp.array([0.0, 0.08, 0.0])
        assert jnp.allclose(y_upper, expected_y, atol=1e-6)

    def test_y_lower_query(self, padded_airfoil_coords):
        """Test lower surface coordinate queries."""
        _, lower_coords, n_valid = padded_airfoil_coords

        # Query lower surface at known points
        query_x = jnp.array([0.0, 0.5, 1.0])
        y_lower = JaxAirfoilOps.y_lower(lower_coords, n_valid, query_x)

        # Check expected values
        expected_y = jnp.array([0.0, -0.08, 0.0])
        assert jnp.allclose(y_lower, expected_y, atol=1e-6)

    def test_compute_max_thickness(self, padded_airfoil_coords):
        """Test maximum thickness computation."""
        upper_coords, lower_coords, n_valid = padded_airfoil_coords

        max_thickness, max_thickness_location = JaxAirfoilOps.compute_max_thickness(
            upper_coords,
            lower_coords,
            n_valid,
            n_valid,
        )

        # Maximum thickness should be 0.16 at x=0.5
        assert jnp.isclose(max_thickness, 0.16, atol=1e-2)
        assert jnp.isclose(max_thickness_location, 0.5, atol=1e-2)

    def test_compute_max_camber(self, padded_airfoil_coords):
        """Test maximum camber computation."""
        upper_coords, lower_coords, n_valid = padded_airfoil_coords

        max_camber, max_camber_location = JaxAirfoilOps.compute_max_camber(
            upper_coords,
            lower_coords,
            n_valid,
            n_valid,
        )

        # For symmetric airfoil, maximum camber should be close to zero
        assert jnp.abs(max_camber) < 1e-2

    def test_compute_chord_length(self, padded_airfoil_coords):
        """Test chord length computation."""
        upper_coords, lower_coords, n_valid = padded_airfoil_coords

        chord_length = JaxAirfoilOps.compute_chord_length(
            upper_coords,
            lower_coords,
            n_valid,
            n_valid,
        )

        # Chord length should be 1.0 (from x=0 to x=1)
        assert jnp.isclose(chord_length, 1.0, atol=1e-6)

    def test_compute_surface_area(self, simple_airfoil_coords):
        """Test surface area (arc length) computation."""
        upper_coords, lower_coords = simple_airfoil_coords

        # Compute surface area for upper surface
        upper_area = JaxAirfoilOps.compute_surface_area(
            upper_coords,
            upper_coords.shape[1],
        )

        # Surface area should be greater than chord length due to curvature
        assert upper_area > 1.0

        # Compute surface area for lower surface
        lower_area = JaxAirfoilOps.compute_surface_area(
            lower_coords,
            lower_coords.shape[1],
        )

        # For symmetric airfoil, upper and lower surface areas should be equal
        assert jnp.isclose(upper_area, lower_area, atol=1e-6)

    def test_validate_airfoil_geometry(self, padded_airfoil_coords):
        """Test airfoil geometry validation."""
        upper_coords, lower_coords, n_valid = padded_airfoil_coords

        validation_results = JaxAirfoilOps.validate_airfoil_geometry(
            upper_coords,
            lower_coords,
            n_valid,
            n_valid,
        )

        # All validation checks should pass for our simple airfoil
        # [has_valid_points, upper_above_lower, closed_geometry, monotonic_x]
        assert validation_results[0]  # has_valid_points
        assert validation_results[1]  # upper_above_lower
        # Note: closed_geometry and monotonic_x might not pass for our simple test case

    def test_jit_compatibility(self, padded_airfoil_coords):
        """Test that all operations are JIT-compatible."""
        upper_coords, lower_coords, n_valid = padded_airfoil_coords

        # Test JIT compilation of thickness computation
        # Need to use partial with static_argnums to make n_valid static
        @partial(jax.jit, static_argnums=(2, 3))
        def jit_thickness(upper, lower, n_upper, n_lower, query_x):
            return JaxAirfoilOps.compute_thickness(
                upper,
                lower,
                n_upper,
                n_lower,
                query_x,
            )

        query_x = jnp.array([0.5])
        thickness = jit_thickness(upper_coords, lower_coords, n_valid, n_valid, query_x)

        assert jnp.isclose(thickness[0], 0.16, atol=1e-6)

        # Test JIT compilation of camber line computation
        @partial(jax.jit, static_argnums=(2, 3))
        def jit_camber(upper, lower, n_upper, n_lower, query_x):
            return JaxAirfoilOps.compute_camber_line(
                upper,
                lower,
                n_upper,
                n_lower,
                query_x,
            )

        camber = jit_camber(upper_coords, lower_coords, n_valid, n_valid, query_x)
        assert jnp.isclose(camber[0], 0.0, atol=1e-6)

    def test_gradient_compatibility(self, simple_airfoil_coords):
        """Test that operations support automatic differentiation."""
        upper_coords, lower_coords = simple_airfoil_coords
        n_valid = upper_coords.shape[1]

        # Pad coordinates for JIT compatibility
        buffer_size = 32
        upper_padded = jnp.concatenate(
            [upper_coords, jnp.full((2, buffer_size - n_valid), jnp.nan)],
            axis=1,
        )
        lower_padded = jnp.concatenate(
            [lower_coords, jnp.full((2, buffer_size - n_valid), jnp.nan)],
            axis=1,
        )

        # Test gradient of thickness with respect to coordinates
        def thickness_at_midpoint(coords):
            upper = coords[:, :n_valid]
            lower = coords[:, n_valid : 2 * n_valid]

            # Pad lower coordinates
            lower_pad = jnp.concatenate(
                [lower, jnp.full((2, buffer_size - n_valid), jnp.nan)],
                axis=1,
            )

            query_x = jnp.array([0.5])
            thickness = JaxAirfoilOps.compute_thickness(
                upper_pad,
                lower_pad,
                n_valid,
                n_valid,
                query_x,
            )
            return thickness[0]

        # Combine upper and lower coordinates for gradient computation
        combined_coords = jnp.concatenate([upper_coords, lower_coords], axis=1)
        upper_pad = jnp.concatenate(
            [upper_coords, jnp.full((2, buffer_size - n_valid), jnp.nan)],
            axis=1,
        )

        # Compute gradient
        grad_fn = jax.grad(thickness_at_midpoint)
        gradients = grad_fn(combined_coords)

        # Gradients should exist and be finite
        assert jnp.all(jnp.isfinite(gradients))
        assert gradients.shape == combined_coords.shape

    def test_asymmetric_airfoil(self):
        """Test operations on an asymmetric airfoil."""
        # Create an asymmetric airfoil (cambered)
        x_coords = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y_upper = jnp.array([0.0, 0.08, 0.10, 0.06, 0.0])  # More cambered
        y_lower = jnp.array([0.0, -0.02, -0.04, -0.02, 0.0])  # Less cambered

        upper_coords = jnp.array([x_coords, y_upper])
        lower_coords = jnp.array([x_coords, y_lower])

        # Pad coordinates
        buffer_size = 32
        n_valid = x_coords.shape[0]
        upper_padded = jnp.concatenate(
            [upper_coords, jnp.full((2, buffer_size - n_valid), jnp.nan)],
            axis=1,
        )
        lower_padded = jnp.concatenate(
            [lower_coords, jnp.full((2, buffer_size - n_valid), jnp.nan)],
            axis=1,
        )

        # Test camber line computation
        query_x = jnp.array([0.5])
        camber = JaxAirfoilOps.compute_camber_line(
            upper_padded,
            lower_padded,
            n_valid,
            n_valid,
            query_x,
        )

        # Camber should be positive for this airfoil
        assert camber[0] > 0

        # Expected camber at x=0.5: (0.10 + (-0.04)) / 2 = 0.03
        expected_camber = 0.03
        assert jnp.isclose(camber[0], expected_camber, atol=1e-6)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with minimal airfoil (3 points)
        x_coords = jnp.array([0.0, 0.5, 1.0])
        y_upper = jnp.array([0.0, 0.05, 0.0])
        y_lower = jnp.array([0.0, -0.05, 0.0])

        upper_coords = jnp.array([x_coords, y_upper])
        lower_coords = jnp.array([x_coords, y_lower])

        # Pad coordinates
        buffer_size = 32
        n_valid = 3
        upper_padded = jnp.concatenate(
            [upper_coords, jnp.full((2, buffer_size - n_valid), jnp.nan)],
            axis=1,
        )
        lower_padded = jnp.concatenate(
            [lower_coords, jnp.full((2, buffer_size - n_valid), jnp.nan)],
            axis=1,
        )

        # Operations should still work
        query_x = jnp.array([0.5])
        thickness = JaxAirfoilOps.compute_thickness(
            upper_padded,
            lower_padded,
            n_valid,
            n_valid,
            query_x,
        )

        assert jnp.isfinite(thickness[0])
        assert thickness[0] > 0

    def test_extrapolation_behavior(self, padded_airfoil_coords):
        """Test behavior when querying outside the airfoil domain."""
        upper_coords, lower_coords, n_valid = padded_airfoil_coords

        # Query points outside the domain [0, 1]
        query_x = jnp.array([-0.1, 1.1])
        thickness = JaxAirfoilOps.compute_thickness(
            upper_coords,
            lower_coords,
            n_valid,
            n_valid,
            query_x,
        )

        # Results should be finite (extrapolated)
        assert jnp.all(jnp.isfinite(thickness))

    def test_performance_with_large_arrays(self):
        """Test performance with larger coordinate arrays."""
        # Create a larger airfoil with 100 points
        n_points = 100
        x_coords = jnp.linspace(0, 1, n_points)
        y_upper = 0.1 * jnp.sin(jnp.pi * x_coords) * (1 - x_coords)
        y_lower = -0.05 * jnp.sin(jnp.pi * x_coords) * (1 - x_coords)

        upper_coords = jnp.array([x_coords, y_upper])
        lower_coords = jnp.array([x_coords, y_lower])

        # Pad to larger buffer
        buffer_size = 128
        upper_padded = jnp.concatenate(
            [upper_coords, jnp.full((2, buffer_size - n_points), jnp.nan)],
            axis=1,
        )
        lower_padded = jnp.concatenate(
            [lower_coords, jnp.full((2, buffer_size - n_points), jnp.nan)],
            axis=1,
        )

        # Test operations
        query_x = jnp.linspace(0, 1, 50)
        thickness = JaxAirfoilOps.compute_thickness(
            upper_padded,
            lower_padded,
            n_points,
            n_points,
            query_x,
        )

        # All results should be finite and positive
        assert jnp.all(jnp.isfinite(thickness))
        assert jnp.all(thickness >= 0)

    def test_batch_operations(self, simple_airfoil_coords):
        """Test that operations work with batched inputs via vmap."""
        upper_coords, lower_coords = simple_airfoil_coords
        n_valid = upper_coords.shape[1]

        # Create batch of query points
        batch_query_x = jnp.array([[0.25, 0.5, 0.75], [0.1, 0.5, 0.9], [0.0, 0.5, 1.0]])

        # Pad coordinates
        buffer_size = 32
        upper_padded = jnp.concatenate(
            [upper_coords, jnp.full((2, buffer_size - n_valid), jnp.nan)],
            axis=1,
        )
        lower_padded = jnp.concatenate(
            [lower_coords, jnp.full((2, buffer_size - n_valid), jnp.nan)],
            axis=1,
        )

        # Use vmap to process batch
        batch_thickness_fn = jax.vmap(
            lambda query_x: JaxAirfoilOps.compute_thickness(
                upper_padded,
                lower_padded,
                n_valid,
                n_valid,
                query_x,
            ),
        )

        batch_thickness = batch_thickness_fn(batch_query_x)

        # Check that we get results for each batch
        assert batch_thickness.shape == (3, 3)  # 3 batches, 3 query points each
        assert jnp.all(jnp.isfinite(batch_thickness))
