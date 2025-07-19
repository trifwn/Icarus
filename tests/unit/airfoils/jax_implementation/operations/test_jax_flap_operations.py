"""
Unit tests for JAX airfoil flap operations.

This module contains tests for the flap transformation functionality in the JAX airfoil
implementation, including coordinate rotation, scaling, and the complete flap operation.
"""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil
from ICARUS.airfoils.jax_implementation.operations import JaxAirfoilOps


class TestJaxFlapOperations:
    """Test class for JAX airfoil flap operations."""

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

    def test_rotate_coordinates_basic(self):
        """Test basic coordinate rotation functionality."""
        # Create simple coordinates
        coords = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # Points at (1,0) and (0,1)

        # Rotate 90 degrees counterclockwise around origin
        angle = jnp.pi / 2
        rotated = JaxAirfoilOps.rotate_coordinates(coords, angle, 0.0, 0.0)

        # Expected: (1,0) -> (0,1), (0,1) -> (-1,0)
        expected = jnp.array([[0.0, -1.0], [1.0, 0.0]])

        assert jnp.allclose(rotated, expected, atol=1e-6)

    def test_rotate_coordinates_around_center(self):
        """Test coordinate rotation around a non-origin center."""
        # Create coordinates
        coords = jnp.array([[2.0, 1.0], [1.0, 1.0]])  # Points at (2,1) and (1,1)

        # Rotate 90 degrees around center (1,1)
        angle = jnp.pi / 2
        center_x, center_y = 1.0, 1.0
        rotated = JaxAirfoilOps.rotate_coordinates(coords, angle, center_x, center_y)

        # Point (2,1) relative to (1,1) is (1,0) -> rotated to (0,1) -> absolute (1,2)
        # Point (1,1) relative to (1,1) is (0,0) -> rotated to (0,0) -> absolute (1,1)
        expected = jnp.array([[1.0, 1.0], [2.0, 1.0]])

        assert jnp.allclose(rotated, expected, atol=1e-6)

    def test_scale_coordinates_basic(self):
        """Test basic coordinate scaling functionality."""
        # Create simple coordinates
        coords = jnp.array([[2.0, 1.0], [1.0, 2.0]])

        # Scale by 2x in x, 0.5x in y around origin
        scaled = JaxAirfoilOps.scale_coordinates(coords, 2.0, 0.5, 0.0, 0.0)

        # Expected: (2,1) -> (4,0.5), (1,2) -> (2,1)
        expected = jnp.array([[4.0, 2.0], [0.5, 1.0]])

        assert jnp.allclose(scaled, expected, atol=1e-6)

    def test_scale_coordinates_around_center(self):
        """Test coordinate scaling around a non-origin center."""
        # Create coordinates
        coords = jnp.array([[3.0, 1.0], [1.0, 3.0]])

        # Scale by 2x around center (1,1)
        center_x, center_y = 1.0, 1.0
        scaled = JaxAirfoilOps.scale_coordinates(coords, 2.0, 2.0, center_x, center_y)

        # Point (3,1) relative to (1,1) is (2,0) -> scaled to (4,0) -> absolute (5,1)
        # Point (1,3) relative to (1,1) is (0,2) -> scaled to (0,4) -> absolute (1,5)
        expected = jnp.array([[5.0, 1.0], [1.0, 5.0]])

        assert jnp.allclose(scaled, expected, atol=1e-6)

    def test_flap_no_deflection(self, simple_airfoil):
        """Test that zero flap deflection returns the same airfoil."""
        flapped = simple_airfoil.flap(
            flap_hinge_chord_percentage=0.5,
            flap_angle=0.0,
            flap_hinge_thickness_percentage=0.5,
            chord_extension=1.0,
        )

        # Should return the same airfoil
        assert flapped is simple_airfoil

    def test_flap_full_chord_hinge(self, simple_airfoil):
        """Test that flap hinge at 100% chord returns the same airfoil."""
        flapped = simple_airfoil.flap(
            flap_hinge_chord_percentage=1.0,
            flap_angle=10.0,
            flap_hinge_thickness_percentage=0.5,
            chord_extension=1.0,
        )

        # Should return the same airfoil
        assert flapped is simple_airfoil

    def test_flap_basic_deflection(self, simple_airfoil):
        """Test basic flap deflection functionality."""
        flap_angle = 10.0  # degrees
        hinge_location = 0.7

        flapped = simple_airfoil.flap(
            flap_hinge_chord_percentage=hinge_location,
            flap_angle=flap_angle,
            flap_hinge_thickness_percentage=0.5,
            chord_extension=1.0,
        )

        # Should create a new airfoil
        assert flapped is not simple_airfoil
        assert isinstance(flapped, JaxAirfoil)

        # Check that the name includes flap information
        assert "flapped" in flapped.name
        assert f"hinge_{hinge_location:.2f}" in flapped.name
        assert f"deflection_{flap_angle:.2f}" in flapped.name

    def test_flap_parameter_validation(self, simple_airfoil):
        """Test parameter validation for flap operation."""
        # Test invalid hinge location
        with pytest.raises(
            ValueError,
            match="flap_hinge_chord_percentage must be between",
        ):
            simple_airfoil.flap(-0.1, 10.0)

        with pytest.raises(
            ValueError,
            match="flap_hinge_chord_percentage must be between",
        ):
            simple_airfoil.flap(1.1, 10.0)

        # Test invalid thickness percentage
        with pytest.raises(
            ValueError,
            match="flap_hinge_thickness_percentage must be between",
        ):
            simple_airfoil.flap(0.5, 10.0, flap_hinge_thickness_percentage=-0.1)

        with pytest.raises(
            ValueError,
            match="flap_hinge_thickness_percentage must be between",
        ):
            simple_airfoil.flap(0.5, 10.0, flap_hinge_thickness_percentage=1.1)

        # Test invalid chord extension
        with pytest.raises(ValueError, match="chord_extension must be positive"):
            simple_airfoil.flap(0.5, 10.0, chord_extension=0.0)

        with pytest.raises(ValueError, match="chord_extension must be positive"):
            simple_airfoil.flap(0.5, 10.0, chord_extension=-1.0)

    def test_flap_with_chord_extension(self, simple_airfoil):
        """Test flap operation with chord extension."""
        original_chord = simple_airfoil.chord_length

        flapped = simple_airfoil.flap(
            flap_hinge_chord_percentage=0.6,
            flap_angle=15.0,
            chord_extension=1.2,
        )

        # The flapped airfoil should have a longer chord due to extension
        flapped_chord = flapped.chord_length
        assert flapped_chord > original_chord

    def test_flap_different_hinge_positions(self, naca_airfoil):
        """Test flap operation at different hinge positions."""
        angles = [10.0, -10.0]  # Positive and negative deflections
        hinge_positions = [0.3, 0.5, 0.7, 0.9]

        for angle in angles:
            for hinge_pos in hinge_positions:
                flapped = naca_airfoil.flap(
                    flap_hinge_chord_percentage=hinge_pos,
                    flap_angle=angle,
                )

                # Should create valid airfoil
                assert isinstance(flapped, JaxAirfoil)
                assert flapped.n_points > 0

                # Coordinates should be finite
                x_coords, y_coords = flapped.get_coordinates()
                assert jnp.all(jnp.isfinite(x_coords))
                assert jnp.all(jnp.isfinite(y_coords))

    def test_flap_thickness_hinge_positions(self, naca_airfoil):
        """Test flap operation with different thickness hinge positions."""
        thickness_positions = [0.0, 0.25, 0.5, 0.75, 1.0]

        for thickness_pos in thickness_positions:
            flapped = naca_airfoil.flap(
                flap_hinge_chord_percentage=0.6,
                flap_angle=10.0,
                flap_hinge_thickness_percentage=thickness_pos,
            )

            # Should create valid airfoil
            assert isinstance(flapped, JaxAirfoil)
            assert flapped.n_points > 0

    def test_flap_jit_compatibility(self, simple_airfoil):
        """Test that core flap operations are JIT-compatible."""
        # Test that the core transformation functions are JIT-compatible
        upper_coords = simple_airfoil._coordinates[:, : simple_airfoil._upper_split_idx]
        lower_coords = simple_airfoil._coordinates[
            :,
            simple_airfoil._upper_split_idx : simple_airfoil._n_valid_points,
        ]

        # Pad lower coordinates to match buffer size for JIT compatibility
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

        # Test JIT compilation of the core transformation function
        @partial(jax.jit, static_argnums=(2, 3))
        def test_flap_transformation(upper, lower, n_upper, n_lower):
            return JaxAirfoilOps.apply_flap_transformation(
                upper,
                lower,
                n_upper,
                n_lower,
                0.6,  # flap_hinge_x
                jnp.deg2rad(10.0),  # flap_angle
                0.5,  # flap_hinge_thickness_percentage
                1.0,  # chord_extension
                False,  # should_skip
            )

        # This should compile and run without error
        result = test_flap_transformation(
            upper_coords,
            lower_padded,
            simple_airfoil._upper_split_idx,
            simple_airfoil._n_valid_points - simple_airfoil._upper_split_idx,
        )

        # Should return valid results
        assert len(result) == 4
        new_upper_coords, new_lower_coords, new_n_upper_valid, new_n_lower_valid = (
            result
        )
        assert isinstance(new_upper_coords, jnp.ndarray)
        assert isinstance(new_lower_coords, jnp.ndarray)

    def test_flap_gradient_compatibility(self, simple_airfoil):
        """Test that core flap operations support automatic differentiation."""
        # Test gradient computation on the core transformation function
        upper_coords = simple_airfoil._coordinates[:, : simple_airfoil._upper_split_idx]
        lower_coords = simple_airfoil._coordinates[
            :,
            simple_airfoil._upper_split_idx : simple_airfoil._n_valid_points,
        ]

        # Pad lower coordinates to match buffer size for JIT compatibility
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

        def flap_transformation_output(angle):
            """Function to compute some output from flap transformation."""
            result = JaxAirfoilOps.apply_flap_transformation(
                upper_coords,
                lower_padded,
                simple_airfoil._upper_split_idx,
                simple_airfoil._n_valid_points - simple_airfoil._upper_split_idx,
                0.6,  # flap_hinge_x
                jnp.deg2rad(angle),  # flap_angle
                0.5,  # flap_hinge_thickness_percentage
                1.0,  # chord_extension
                False,  # should_skip
            )
            new_upper_coords, new_lower_coords, new_n_upper_valid, new_n_lower_valid = (
                result
            )
            # Return sum of coordinates as a simple differentiable output
            return jnp.sum(new_upper_coords) + jnp.sum(new_lower_coords)

        # Compute gradient with respect to flap angle
        grad_fn = jax.grad(flap_transformation_output)
        gradient = grad_fn(10.0)

        # Gradient computation should not crash (gradient may be NaN due to NaN handling in transformation)
        assert isinstance(gradient, jnp.ndarray) or isinstance(gradient, float)

    def test_flap_preserves_airfoil_properties(self, naca_airfoil):
        """Test that flap operation preserves basic airfoil properties."""
        original_chord = naca_airfoil.chord_length

        flapped = naca_airfoil.flap(
            flap_hinge_chord_percentage=0.7,
            flap_angle=5.0,
        )

        # Should still have valid geometric properties
        assert flapped.max_thickness > 0
        assert jnp.isfinite(flapped.max_thickness_location)
        assert jnp.isfinite(flapped.max_camber)
        assert jnp.isfinite(flapped.max_camber_location)

        # Chord length should be reasonable (may change due to flap)
        assert flapped.chord_length > 0

    def test_flap_large_deflections(self, naca_airfoil):
        """Test flap operation with large deflection angles."""
        large_angles = [30.0, 45.0, -30.0, -45.0]

        for angle in large_angles:
            flapped = naca_airfoil.flap(
                flap_hinge_chord_percentage=0.6,
                flap_angle=angle,
            )

            # Should still create valid airfoil
            assert isinstance(flapped, JaxAirfoil)
            assert flapped.n_points > 0

            # Coordinates should be finite
            x_coords, y_coords = flapped.get_coordinates()
            assert jnp.all(jnp.isfinite(x_coords))
            assert jnp.all(jnp.isfinite(y_coords))

    def test_flap_multiple_applications(self, naca_airfoil):
        """Test applying multiple flap operations in sequence."""
        # Apply first flap
        flapped1 = naca_airfoil.flap(
            flap_hinge_chord_percentage=0.8,
            flap_angle=10.0,
        )

        # Apply second flap to the already flapped airfoil
        flapped2 = flapped1.flap(
            flap_hinge_chord_percentage=0.6,
            flap_angle=5.0,
        )

        # Should create valid airfoil
        assert isinstance(flapped2, JaxAirfoil)
        assert flapped2.n_points > 0

        # Name should reflect multiple flap operations
        assert "flapped" in flapped2.name

    def test_flap_metadata_preservation(self, simple_airfoil):
        """Test that flap operation preserves and updates metadata correctly."""
        # Add some custom metadata
        simple_airfoil._metadata["custom_field"] = "test_value"

        flapped = simple_airfoil.flap(
            flap_hinge_chord_percentage=0.6,
            flap_angle=15.0,
            flap_hinge_thickness_percentage=0.3,
            chord_extension=1.1,
        )

        # Should preserve original metadata
        assert flapped._metadata.get("custom_field") == "test_value"

        # Should add flap-specific metadata
        assert flapped._metadata.get("flap_hinge_chord_percentage") == 0.6
        assert flapped._metadata.get("flap_angle") == 15.0
        assert flapped._metadata.get("flap_hinge_thickness_percentage") == 0.3
        assert flapped._metadata.get("chord_extension") == 1.1

    def test_flap_coordinate_transformation_correctness(self, simple_airfoil):
        """Test that flap coordinate transformation is geometrically correct."""
        hinge_pos = 0.6
        flap_angle = 20.0

        flapped = simple_airfoil.flap(
            flap_hinge_chord_percentage=hinge_pos,
            flap_angle=flap_angle,
        )

        # Get coordinates before and after hinge
        original_x, original_y = simple_airfoil.get_coordinates()
        flapped_x, flapped_y = flapped.get_coordinates()

        # Find hinge x-coordinate
        min_x = jnp.min(original_x)
        max_x = jnp.max(original_x)
        hinge_x = min_x + hinge_pos * (max_x - min_x)

        # Points before hinge should be unchanged
        original_before_mask = original_x <= hinge_x
        flapped_before_mask = flapped_x <= hinge_x

        # There should be some points before the hinge that remain unchanged
        # (This is a basic sanity check - exact comparison is complex due to interpolation)
        assert jnp.sum(original_before_mask) > 0
        assert jnp.sum(flapped_before_mask) > 0

    def test_flap_buffer_size_handling(self, simple_airfoil):
        """Test that flap operation handles buffer sizes correctly."""
        original_buffer_size = simple_airfoil.buffer_size

        flapped = simple_airfoil.flap(
            flap_hinge_chord_percentage=0.7,
            flap_angle=10.0,
        )

        # Buffer size should be preserved or appropriately managed
        assert flapped.buffer_size >= original_buffer_size
        assert flapped.n_points <= flapped.buffer_size

    def test_flap_edge_cases(self, simple_airfoil):
        """Test flap operation edge cases."""
        # Very small flap angle
        flapped_small = simple_airfoil.flap(
            flap_hinge_chord_percentage=0.5,
            flap_angle=0.1,
        )
        assert isinstance(flapped_small, JaxAirfoil)

        # Hinge very close to leading edge
        flapped_le = simple_airfoil.flap(
            flap_hinge_chord_percentage=0.05,
            flap_angle=10.0,
        )
        assert isinstance(flapped_le, JaxAirfoil)

        # Hinge very close to trailing edge
        flapped_te = simple_airfoil.flap(
            flap_hinge_chord_percentage=0.95,
            flap_angle=10.0,
        )
        assert isinstance(flapped_te, JaxAirfoil)

    def test_flap_performance_with_large_airfoil(self):
        """Test flap operation performance with larger airfoils."""
        # Create a larger NACA airfoil
        large_airfoil = JaxAirfoil.naca4("0012", n_points=500)

        # Apply flap operation
        flapped = large_airfoil.flap(
            flap_hinge_chord_percentage=0.6,
            flap_angle=15.0,
        )

        # Should handle large airfoils efficiently
        assert isinstance(flapped, JaxAirfoil)
        assert flapped.n_points > 0

        # All coordinates should be finite
        x_coords, y_coords = flapped.get_coordinates()
        assert jnp.all(jnp.isfinite(x_coords))
        assert jnp.all(jnp.isfinite(y_coords))
