"""
Unit tests for JAX airfoil repaneling operations.

This module contains tests for the repaneling functionality in the JAX airfoil
implementation, including cosine and uniform point distributions, arc-length
based repaneling, and gradient preservation.
"""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil
from ICARUS.airfoils.jax_implementation.operations import JaxAirfoilOps


class TestJaxRepanelOperations:
    """Test class for JAX airfoil repaneling operations."""

    @pytest.fixture
    def simple_airfoil(self):
        """Create a simple symmetric airfoil for testing."""
        # Create a simple symmetric airfoil (diamond shape)
        x_coords = jnp.array([1.0, 0.75, 0.5, 0.25, 0.0])  # TE to LE for upper
        y_upper = jnp.array([0.0, 0.05, 0.08, 0.05, 0.0])

        x_coords_lower = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])  # LE to TE for lower
        y_lower = jnp.array([0.0, -0.05, -0.08, -0.05, 0.0])

        upper_coords = jnp.array([x_coords, y_upper])
        lower_coords = jnp.array([x_coords_lower, y_lower])

        return JaxAirfoil.from_upper_lower(
            upper_coords,
            lower_coords,
            name="TestAirfoil",
        )

    @pytest.fixture
    def naca_airfoil(self):
        """Create a NACA 2412 airfoil for testing."""
        return JaxAirfoil.naca4("2412", n_points=100)

    def test_generate_cosine_distribution(self):
        """Test cosine distribution generation."""
        n_points = 10
        distribution = JaxAirfoilOps.generate_cosine_distribution(n_points)

        # Check shape
        assert distribution.shape == (n_points,)

        # Check range [0, 1]
        assert jnp.all(distribution >= 0.0)
        assert jnp.all(distribution <= 1.0)

        # Check endpoints
        assert jnp.isclose(distribution[0], 0.0)
        assert jnp.isclose(distribution[-1], 1.0)

        # Check monotonicity
        assert jnp.all(jnp.diff(distribution) >= 0)

        # Check that it's not uniform (should have more points near edges)
        uniform_dist = jnp.linspace(0, 1, n_points)
        assert not jnp.allclose(distribution, uniform_dist)

    def test_generate_uniform_distribution(self):
        """Test uniform distribution generation."""
        n_points = 10
        distribution = JaxAirfoilOps.generate_uniform_distribution(n_points)

        # Check shape
        assert distribution.shape == (n_points,)

        # Check range [0, 1]
        assert jnp.all(distribution >= 0.0)
        assert jnp.all(distribution <= 1.0)

        # Check endpoints
        assert jnp.isclose(distribution[0], 0.0)
        assert jnp.isclose(distribution[-1], 1.0)

        # Check uniformity
        expected_uniform = jnp.linspace(0, 1, n_points)
        assert jnp.allclose(distribution, expected_uniform)

    def test_compute_arc_length_parametrization(self, simple_airfoil):
        """Test arc length parametrization computation."""
        # Get upper and lower surface coordinates
        upper_coords = simple_airfoil._coordinates[:, : simple_airfoil._upper_split_idx]
        lower_coords = simple_airfoil._coordinates[
            :,
            simple_airfoil._upper_split_idx : simple_airfoil._n_valid_points,
        ]

        # Pad both coordinates to match buffer size for JIT compatibility
        upper_padded = jnp.concatenate(
            [
                upper_coords,
                jnp.full(
                    (2, simple_airfoil._max_buffer_size - upper_coords.shape[1]),
                    jnp.nan,
                ),
            ],
            axis=1,
        )
        lower_padded = jnp.concatenate(
            [
                lower_coords,
                jnp.full(
                    (2, simple_airfoil._max_buffer_size - lower_coords.shape[1]),
                    jnp.nan,
                ),
            ],
            axis=1,
        )

        # Compute arc length parametrization
        upper_arc_lengths, lower_arc_lengths = (
            JaxAirfoilOps.compute_arc_length_parametrization(
                upper_padded,
                lower_padded,
                simple_airfoil._upper_split_idx,
                simple_airfoil._n_valid_points - simple_airfoil._upper_split_idx,
            )
        )

        # Check shapes
        assert upper_arc_lengths.shape == (simple_airfoil._max_buffer_size,)
        assert lower_arc_lengths.shape == (simple_airfoil._max_buffer_size,)

        # Check valid portions
        upper_valid = upper_arc_lengths[: simple_airfoil._upper_split_idx]
        lower_valid = lower_arc_lengths[
            : simple_airfoil._n_valid_points - simple_airfoil._upper_split_idx
        ]

        # Check range [0, 1]
        assert jnp.all(upper_valid >= 0.0)
        assert jnp.all(upper_valid <= 1.0)
        assert jnp.all(lower_valid >= 0.0)
        assert jnp.all(lower_valid <= 1.0)

        # Check endpoints
        assert jnp.isclose(upper_valid[0], 0.0)
        assert jnp.isclose(upper_valid[-1], 1.0)
        assert jnp.isclose(lower_valid[0], 0.0)
        assert jnp.isclose(lower_valid[-1], 1.0)

        # Check monotonicity
        assert jnp.all(jnp.diff(upper_valid) >= 0)
        assert jnp.all(jnp.diff(lower_valid) >= 0)

    def test_repanel_surface_uniform(self, simple_airfoil):
        """Test surface repaneling with uniform distribution."""
        # Get upper surface coordinates
        upper_coords = simple_airfoil._coordinates[:, : simple_airfoil._upper_split_idx]

        # Pad upper coordinates to match buffer size for JIT compatibility
        upper_padded = jnp.concatenate(
            [
                upper_coords,
                jnp.full(
                    (2, simple_airfoil._max_buffer_size - upper_coords.shape[1]),
                    jnp.nan,
                ),
            ],
            axis=1,
        )

        # Compute arc length parametrization
        upper_arc_lengths, _ = JaxAirfoilOps.compute_arc_length_parametrization(
            upper_padded,
            jnp.full((2, simple_airfoil._max_buffer_size), jnp.nan),  # dummy lower
            simple_airfoil._upper_split_idx,
            1,  # dummy n_lower
        )

        # Repanel with different number of points
        n_new_points = 8
        new_coords = JaxAirfoilOps.repanel_surface_uniform(
            upper_padded,
            upper_arc_lengths,
            simple_airfoil._upper_split_idx,
            n_new_points,
            "uniform",
        )

        # Check shape
        assert new_coords.shape == (2, n_new_points)

        # Check that coordinates are finite
        assert jnp.all(jnp.isfinite(new_coords))

        # Check x-coordinate range is preserved
        original_x_min = jnp.min(upper_coords[0, : simple_airfoil._upper_split_idx])
        original_x_max = jnp.max(upper_coords[0, : simple_airfoil._upper_split_idx])
        new_x_min = jnp.min(new_coords[0, :])
        new_x_max = jnp.max(new_coords[0, :])

        assert jnp.isclose(new_x_min, original_x_min, atol=1e-6)
        assert jnp.isclose(new_x_max, original_x_max, atol=1e-6)

    def test_repanel_surface_cosine(self, simple_airfoil):
        """Test surface repaneling with cosine distribution."""
        # Get upper surface coordinates
        upper_coords = simple_airfoil._coordinates[:, : simple_airfoil._upper_split_idx]

        # Pad upper coordinates to match buffer size for JIT compatibility
        upper_padded = jnp.concatenate(
            [
                upper_coords,
                jnp.full(
                    (2, simple_airfoil._max_buffer_size - upper_coords.shape[1]),
                    jnp.nan,
                ),
            ],
            axis=1,
        )

        # Compute arc length parametrization
        upper_arc_lengths, _ = JaxAirfoilOps.compute_arc_length_parametrization(
            upper_padded,
            jnp.full((2, simple_airfoil._max_buffer_size), jnp.nan),  # dummy lower
            simple_airfoil._upper_split_idx,
            1,  # dummy n_lower
        )

        # Repanel with cosine distribution
        n_new_points = 8
        new_coords = JaxAirfoilOps.repanel_surface_uniform(
            upper_padded,
            upper_arc_lengths,
            simple_airfoil._upper_split_idx,
            n_new_points,
            "cosine",
        )

        # Check shape
        assert new_coords.shape == (2, n_new_points)

        # Check that coordinates are finite
        assert jnp.all(jnp.isfinite(new_coords))

        # Check x-coordinate range is preserved
        original_x_min = jnp.min(upper_coords[0, : simple_airfoil._upper_split_idx])
        original_x_max = jnp.max(upper_coords[0, : simple_airfoil._upper_split_idx])
        new_x_min = jnp.min(new_coords[0, :])
        new_x_max = jnp.max(new_coords[0, :])

        assert jnp.isclose(new_x_min, original_x_min, atol=1e-6)
        assert jnp.isclose(new_x_max, original_x_max, atol=1e-6)

    def test_repanel_airfoil_coordinates(self, simple_airfoil):
        """Test complete airfoil repaneling."""
        # Get upper and lower surface coordinates
        upper_coords = simple_airfoil._coordinates[:, : simple_airfoil._upper_split_idx]
        lower_coords = simple_airfoil._coordinates[
            :,
            simple_airfoil._upper_split_idx : simple_airfoil._n_valid_points,
        ]

        # Pad both coordinates to match buffer size for JIT compatibility
        upper_padded = jnp.concatenate(
            [
                upper_coords,
                jnp.full(
                    (2, simple_airfoil._max_buffer_size - upper_coords.shape[1]),
                    jnp.nan,
                ),
            ],
            axis=1,
        )
        lower_padded = jnp.concatenate(
            [
                lower_coords,
                jnp.full(
                    (2, simple_airfoil._max_buffer_size - lower_coords.shape[1]),
                    jnp.nan,
                ),
            ],
            axis=1,
        )

        # Repanel airfoil
        n_new_points = 20
        new_upper_coords, new_lower_coords = JaxAirfoilOps.repanel_airfoil_coordinates(
            upper_padded,
            lower_padded,
            simple_airfoil._upper_split_idx,
            simple_airfoil._n_valid_points - simple_airfoil._upper_split_idx,
            n_new_points,
            "cosine",
        )

        # Check shapes
        expected_points_per_surface = n_new_points // 2
        assert new_upper_coords.shape == (2, expected_points_per_surface)
        assert new_lower_coords.shape == (2, expected_points_per_surface)

        # Check that coordinates are finite
        assert jnp.all(jnp.isfinite(new_upper_coords))
        assert jnp.all(jnp.isfinite(new_lower_coords))

        # Check x-coordinate ranges are preserved
        original_x_min = jnp.min(
            simple_airfoil._coordinates[0, : simple_airfoil._n_valid_points],
        )
        original_x_max = jnp.max(
            simple_airfoil._coordinates[0, : simple_airfoil._n_valid_points],
        )

        new_x_min = jnp.min(
            jnp.concatenate([new_upper_coords[0, :], new_lower_coords[0, :]]),
        )
        new_x_max = jnp.max(
            jnp.concatenate([new_upper_coords[0, :], new_lower_coords[0, :]]),
        )

        assert jnp.isclose(new_x_min, original_x_min, atol=1e-6)
        assert jnp.isclose(new_x_max, original_x_max, atol=1e-6)

    def test_repanel_method_basic(self, simple_airfoil):
        """Test basic repanel method functionality."""
        original_n_points = simple_airfoil.n_points
        n_new_points = 16

        # Repanel the airfoil
        repaneled = simple_airfoil.repanel(n_new_points, distribution="cosine")

        # Should create a new airfoil
        assert repaneled is not simple_airfoil
        assert isinstance(repaneled, JaxAirfoil)

        # Check point count
        assert repaneled.n_points == n_new_points

        # Check that the name reflects repaneling
        assert "repaneled" in repaneled.name
        assert f"{n_new_points}pts" in repaneled.name
        assert "cosine" in repaneled.name

        # Check metadata
        assert repaneled._metadata.get("repanel_n_points") == n_new_points
        assert repaneled._metadata.get("repanel_distribution") == "cosine"
        assert repaneled._metadata.get("original_n_points") == original_n_points

    def test_repanel_method_uniform_distribution(self, simple_airfoil):
        """Test repanel method with uniform distribution."""
        n_new_points = 12

        # Repanel with uniform distribution
        repaneled = simple_airfoil.repanel(n_new_points, distribution="uniform")

        # Check basic properties
        assert isinstance(repaneled, JaxAirfoil)
        assert repaneled.n_points == n_new_points
        assert "uniform" in repaneled.name

        # Check metadata
        assert repaneled._metadata.get("repanel_distribution") == "uniform"

    def test_repanel_method_different_methods(self, simple_airfoil):
        """Test repanel method with different repaneling methods."""
        n_new_points = 14

        # Test arc_length method
        repaneled_arc = simple_airfoil.repanel(n_new_points, method="arc_length")
        assert isinstance(repaneled_arc, JaxAirfoil)
        assert repaneled_arc._metadata.get("repanel_method") == "arc_length"

        # Test chord_based method
        repaneled_chord = simple_airfoil.repanel(n_new_points, method="chord_based")
        assert isinstance(repaneled_chord, JaxAirfoil)
        assert repaneled_chord._metadata.get("repanel_method") == "chord_based"

    def test_repanel_parameter_validation(self, simple_airfoil):
        """Test parameter validation for repanel method."""
        # Test invalid n_points
        with pytest.raises(ValueError, match="n_points must be at least 4"):
            simple_airfoil.repanel(3)

        # Test invalid distribution
        with pytest.raises(ValueError, match="distribution must be"):
            simple_airfoil.repanel(10, distribution="invalid")

        # Test invalid method
        with pytest.raises(ValueError, match="method must be"):
            simple_airfoil.repanel(10, method="invalid")

    def test_repanel_preserves_airfoil_shape(self, naca_airfoil):
        """Test that repaneling preserves the overall airfoil shape."""
        original_max_thickness = naca_airfoil.max_thickness
        original_max_camber = naca_airfoil.max_camber
        original_chord_length = naca_airfoil.chord_length

        # Repanel with different point counts
        for n_points in [50, 100, 200]:
            repaneled = naca_airfoil.repanel(n_points)

            # Check that basic geometric properties are preserved
            assert jnp.isclose(
                repaneled.max_thickness,
                original_max_thickness,
                rtol=0.05,
            )
            assert jnp.isclose(repaneled.max_camber, original_max_camber, rtol=0.05)
            assert jnp.isclose(repaneled.chord_length, original_chord_length, rtol=0.01)

    def test_repanel_different_point_counts(self, naca_airfoil):
        """Test repaneling with various point counts."""
        point_counts = [20, 50, 100, 150, 300]

        for n_points in point_counts:
            repaneled = naca_airfoil.repanel(n_points)

            # Should create valid airfoil
            assert isinstance(repaneled, JaxAirfoil)
            assert repaneled.n_points == n_points

            # Coordinates should be finite
            x_coords, y_coords = repaneled.get_coordinates()
            assert jnp.all(jnp.isfinite(x_coords))
            assert jnp.all(jnp.isfinite(y_coords))

    def test_repanel_jit_compatibility(self, simple_airfoil):
        """Test that core repanel operations are JIT-compatible."""
        # Get upper and lower surface coordinates
        upper_coords = simple_airfoil._coordinates[:, : simple_airfoil._upper_split_idx]
        lower_coords = simple_airfoil._coordinates[
            :,
            simple_airfoil._upper_split_idx : simple_airfoil._n_valid_points,
        ]

        # Pad both coordinates to match buffer size for JIT compatibility
        upper_padded = jnp.concatenate(
            [
                upper_coords,
                jnp.full(
                    (2, simple_airfoil._max_buffer_size - upper_coords.shape[1]),
                    jnp.nan,
                ),
            ],
            axis=1,
        )
        lower_padded = jnp.concatenate(
            [
                lower_coords,
                jnp.full(
                    (2, simple_airfoil._max_buffer_size - lower_coords.shape[1]),
                    jnp.nan,
                ),
            ],
            axis=1,
        )

        # Test JIT compilation of the core repanel function
        @partial(jax.jit, static_argnums=(2, 3, 4, 5))
        def test_repanel_jit(upper, lower, n_upper, n_lower, n_new, dist_type):
            return JaxAirfoilOps.repanel_airfoil_coordinates(
                upper,
                lower,
                n_upper,
                n_lower,
                n_new,
                dist_type,
            )

        # This should compile and run without error
        result = test_repanel_jit(
            upper_padded,
            lower_padded,
            simple_airfoil._upper_split_idx,
            simple_airfoil._n_valid_points - simple_airfoil._upper_split_idx,
            20,
            "cosine",
        )

        # Should return valid results
        assert len(result) == 2
        new_upper_coords, new_lower_coords = result
        assert isinstance(new_upper_coords, jnp.ndarray)
        assert isinstance(new_lower_coords, jnp.ndarray)

    def test_repanel_gradient_compatibility(self, simple_airfoil):
        """Test that repanel operations support automatic differentiation."""

        # For now, we'll test gradient compatibility at the operation level
        # rather than through the full airfoil constructor which has eager operations

        # Get padded coordinates
        upper_coords = simple_airfoil._coordinates[:, : simple_airfoil._upper_split_idx]
        lower_coords = simple_airfoil._coordinates[
            :,
            simple_airfoil._upper_split_idx : simple_airfoil._n_valid_points,
        ]

        # Pad both coordinates to match buffer size for JIT compatibility
        upper_padded = jnp.concatenate(
            [
                upper_coords,
                jnp.full(
                    (2, simple_airfoil._max_buffer_size - upper_coords.shape[1]),
                    jnp.nan,
                ),
            ],
            axis=1,
        )
        lower_padded = jnp.concatenate(
            [
                lower_coords,
                jnp.full(
                    (2, simple_airfoil._max_buffer_size - lower_coords.shape[1]),
                    jnp.nan,
                ),
            ],
            axis=1,
        )

        def repanel_output_function(upper_coords_input):
            """Function to compute some output from repaneled coordinates."""
            result = JaxAirfoilOps.repanel_airfoil_coordinates(
                upper_coords_input,
                lower_padded,
                simple_airfoil._upper_split_idx,
                simple_airfoil._n_valid_points - simple_airfoil._upper_split_idx,
                16,
                "cosine",
            )
            new_upper_coords, new_lower_coords = result
            # Return sum of coordinates as a simple differentiable output
            return jnp.sum(new_upper_coords) + jnp.sum(new_lower_coords)

        # Compute gradient with respect to upper coordinates
        grad_fn = jax.grad(repanel_output_function)
        gradient = grad_fn(upper_padded)

        # Gradient computation should not crash
        assert isinstance(gradient, jnp.ndarray)
        assert gradient.shape == upper_padded.shape

    def test_repanel_accuracy_comparison(self, naca_airfoil):
        """Test repaneling accuracy by comparing interpolated values."""
        # Query points for comparison
        query_x = jnp.linspace(0.1, 0.9, 10)

        # Get original thickness and camber
        original_thickness = naca_airfoil.thickness(query_x)
        original_camber = naca_airfoil.camber_line(query_x)

        # Repanel with different point counts
        for n_points in [50, 100, 200]:
            repaneled = naca_airfoil.repanel(n_points)

            # Get repaneled thickness and camber
            repaneled_thickness = repaneled.thickness(query_x)
            repaneled_camber = repaneled.camber_line(query_x)

            # Should be reasonably close (allowing for interpolation differences)
            assert jnp.allclose(repaneled_thickness, original_thickness, rtol=0.05)
            assert jnp.allclose(repaneled_camber, original_camber, rtol=0.05)

    def test_repanel_cosine_vs_uniform_distribution(self, naca_airfoil):
        """Test differences between cosine and uniform distributions."""
        n_points = 100

        # Repanel with both distributions
        repaneled_cosine = naca_airfoil.repanel(n_points, distribution="cosine")
        repaneled_uniform = naca_airfoil.repanel(n_points, distribution="uniform")

        # Both should be valid
        assert isinstance(repaneled_cosine, JaxAirfoil)
        assert isinstance(repaneled_uniform, JaxAirfoil)
        assert repaneled_cosine.n_points == n_points
        assert repaneled_uniform.n_points == n_points

        # Get coordinates
        x_cos, y_cos = repaneled_cosine.get_coordinates()
        x_uni, y_uni = repaneled_uniform.get_coordinates()

        # Coordinates should be different (different distributions)
        assert not jnp.allclose(x_cos, x_uni)

        # But both should preserve the overall shape reasonably well
        query_x = jnp.linspace(0.1, 0.9, 20)
        thickness_cos = repaneled_cosine.thickness(query_x)
        thickness_uni = repaneled_uniform.thickness(query_x)
        original_thickness = naca_airfoil.thickness(query_x)

        # Both should be close to original
        assert jnp.allclose(thickness_cos, original_thickness, rtol=0.1)
        assert jnp.allclose(thickness_uni, original_thickness, rtol=0.1)

    def test_repanel_multiple_applications(self, naca_airfoil):
        """Test applying repaneling multiple times."""
        # First repanel
        repaneled1 = naca_airfoil.repanel(80, distribution="cosine")

        # Second repanel
        repaneled2 = repaneled1.repanel(60, distribution="uniform")

        # Should create valid airfoil
        assert isinstance(repaneled2, JaxAirfoil)
        assert repaneled2.n_points == 60

        # Name should reflect multiple repanelings
        assert "repaneled" in repaneled2.name

    def test_repanel_edge_cases(self, simple_airfoil):
        """Test repanel operation edge cases."""
        # Very small number of points (minimum allowed)
        repaneled_small = simple_airfoil.repanel(4)
        assert isinstance(repaneled_small, JaxAirfoil)
        assert repaneled_small.n_points == 4

        # Large number of points
        repaneled_large = simple_airfoil.repanel(500)
        assert isinstance(repaneled_large, JaxAirfoil)
        assert repaneled_large.n_points == 500

        # Same number of points as original
        original_n_points = simple_airfoil.n_points
        repaneled_same = simple_airfoil.repanel(original_n_points)
        assert isinstance(repaneled_same, JaxAirfoil)
        assert repaneled_same.n_points == original_n_points

    def test_repanel_buffer_size_handling(self, simple_airfoil):
        """Test that repanel operation handles buffer sizes correctly."""
        original_buffer_size = simple_airfoil.buffer_size

        # Repanel with more points than original buffer
        large_n_points = original_buffer_size + 50
        repaneled = simple_airfoil.repanel(large_n_points)

        # Buffer size should be appropriately managed
        assert repaneled.buffer_size >= large_n_points
        assert repaneled.n_points == large_n_points

    def test_repanel_performance_with_large_airfoil(self):
        """Test repanel operation performance with larger airfoils."""
        # Create a larger NACA airfoil
        large_airfoil = JaxAirfoil.naca4("0012", n_points=1000)

        # Apply repanel operation
        repaneled = large_airfoil.repanel(500, distribution="cosine")

        # Should handle large airfoils efficiently
        assert isinstance(repaneled, JaxAirfoil)
        assert repaneled.n_points == 500

        # All coordinates should be finite
        x_coords, y_coords = repaneled.get_coordinates()
        assert jnp.all(jnp.isfinite(x_coords))
        assert jnp.all(jnp.isfinite(y_coords))

    def test_repanel_metadata_preservation(self, simple_airfoil):
        """Test that repanel operation preserves and updates metadata correctly."""
        # Add some custom metadata
        simple_airfoil._metadata["custom_field"] = "test_value"

        repaneled = simple_airfoil.repanel(
            20,
            distribution="cosine",
            method="arc_length",
        )

        # Should preserve original metadata
        assert repaneled._metadata.get("custom_field") == "test_value"

        # Should add repanel-specific metadata
        assert repaneled._metadata.get("repanel_n_points") == 20
        assert repaneled._metadata.get("repanel_distribution") == "cosine"
        assert repaneled._metadata.get("repanel_method") == "arc_length"
        assert repaneled._metadata.get("original_n_points") == simple_airfoil.n_points

    def test_repanel_coordinate_ordering_preservation(self, naca_airfoil):
        """Test that repaneling preserves coordinate ordering conventions."""
        repaneled = naca_airfoil.repanel(100)

        # Get upper and lower surface points
        x_upper, y_upper = repaneled.upper_surface_points
        x_lower, y_lower = repaneled.lower_surface_points

        # Upper surface should be ordered from TE to LE (decreasing x for typical airfoils)
        # Lower surface should be ordered from LE to TE (increasing x for typical airfoils)

        # Check that we have reasonable coordinate ranges
        assert jnp.all(jnp.isfinite(x_upper))
        assert jnp.all(jnp.isfinite(y_upper))
        assert jnp.all(jnp.isfinite(x_lower))
        assert jnp.all(jnp.isfinite(y_lower))

        # Check that upper surface y-coordinates are generally above lower surface
        # (at least at some points)
        x_mid = 0.5  # Check at mid-chord
        y_upper_mid = repaneled.y_upper(x_mid)
        y_lower_mid = repaneled.y_lower(x_mid)
        assert y_upper_mid > y_lower_mid  # Upper should be above lower

    def test_repanel_interpolation_consistency(self, naca_airfoil):
        """Test that repaneled airfoils maintain interpolation consistency."""
        # Create repaneled versions with different point counts
        repaneled_50 = naca_airfoil.repanel(50)
        repaneled_100 = naca_airfoil.repanel(100)
        repaneled_200 = naca_airfoil.repanel(200)

        # Query the same points on all versions
        query_x = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Get thickness values
        thickness_50 = repaneled_50.thickness(query_x)
        thickness_100 = repaneled_100.thickness(query_x)
        thickness_200 = repaneled_200.thickness(query_x)
        thickness_original = naca_airfoil.thickness(query_x)

        # All should be reasonably close to the original
        assert jnp.allclose(thickness_50, thickness_original, rtol=0.1)
        assert jnp.allclose(thickness_100, thickness_original, rtol=0.05)
        assert jnp.allclose(thickness_200, thickness_original, rtol=0.02)

        # Higher resolution should be more accurate
        error_50 = jnp.mean(jnp.abs(thickness_50 - thickness_original))
        error_100 = jnp.mean(jnp.abs(thickness_100 - thickness_original))
        error_200 = jnp.mean(jnp.abs(thickness_200 - thickness_original))

        # Generally, higher resolution should have lower error (allowing some tolerance)
        assert (
            error_200 <= error_50 * 1.5
        )  # Allow some flexibility due to interpolation
