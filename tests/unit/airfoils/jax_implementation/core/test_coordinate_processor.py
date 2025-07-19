"""
Unit tests for the CoordinateProcessor class.

Tests cover NaN filtering, coordinate validation, ordering, closure,
selig format conversion, and the complete preprocessing pipeline.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from ICARUS.airfoils.jax_implementation.buffer_management import AirfoilBufferManager
from ICARUS.airfoils.jax_implementation.coordinate_processor import CoordinateProcessor


class TestCoordinateProcessor:
    """Test suite for CoordinateProcessor class."""

    def test_filter_nan_coordinates_basic(self):
        """Test basic NaN filtering functionality."""
        # Create coordinates with some NaN values
        coords = jnp.array(
            [[0.0, 0.5, jnp.nan, 1.0, 0.8], [0.0, 0.1, 0.05, jnp.nan, -0.1]],
        )

        filtered = CoordinateProcessor.filter_nan_coordinates(coords)

        # Should remove columns with any NaN values (columns 2 and 3 have NaN)
        expected = jnp.array([[0.0, 0.5, 0.8], [0.0, 0.1, -0.1]])

        assert filtered.shape == expected.shape
        np.testing.assert_array_equal(filtered, expected)

    def test_filter_nan_coordinates_all_valid(self):
        """Test NaN filtering with no NaN values."""
        coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])

        filtered = CoordinateProcessor.filter_nan_coordinates(coords)

        # Should return unchanged
        np.testing.assert_array_equal(filtered, coords)

    def test_filter_nan_coordinates_all_nan(self):
        """Test NaN filtering with all NaN values."""
        coords = jnp.array([[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]])

        filtered = CoordinateProcessor.filter_nan_coordinates(coords)

        # Should return empty array
        assert filtered.shape == (2, 0)

    def test_filter_nan_coordinates_invalid_shape(self):
        """Test NaN filtering with invalid input shape."""
        coords = jnp.array([0.0, 0.5, 1.0])  # 1D array

        with pytest.raises(ValueError, match="Coordinates must have shape"):
            CoordinateProcessor.filter_nan_coordinates(coords)

    def test_validate_coordinates_valid(self):
        """Test coordinate validation with valid coordinates."""
        coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])

        # Should not raise any exception
        CoordinateProcessor.validate_coordinates(coords)

    def test_validate_coordinates_empty(self):
        """Test coordinate validation with empty array."""
        coords = jnp.empty((2, 0))

        with pytest.raises(ValueError, match="Coordinate array cannot be empty"):
            CoordinateProcessor.validate_coordinates(coords)

    def test_validate_coordinates_infinite(self):
        """Test coordinate validation with infinite values."""
        coords = jnp.array([[0.0, jnp.inf, 1.0], [0.0, 0.1, 0.0]])

        with pytest.raises(ValueError, match="Coordinates contain infinite values"):
            CoordinateProcessor.validate_coordinates(coords)

    def test_validate_coordinates_too_large(self):
        """Test coordinate validation with extremely large values."""
        coords = jnp.array(
            [
                [0.0, 0.5, 1.0],
                [0.0, 15.0, 0.0],  # Too large
            ],
        )

        with pytest.raises(ValueError, match="Coordinates contain values larger than"):
            CoordinateProcessor.validate_coordinates(coords)

    def test_validate_coordinates_invalid_shape(self):
        """Test coordinate validation with invalid shape."""
        coords = jnp.array([0.0, 0.5, 1.0])  # 1D array

        with pytest.raises(ValueError, match="Coordinates must have shape"):
            CoordinateProcessor.validate_coordinates(coords)

    def test_order_surface_points_needs_reversal(self):
        """Test point ordering when reversal is needed."""
        # Points running from trailing edge to leading edge (needs reversal)
        coords = jnp.array(
            [
                [1.0, 0.5, 0.0],  # x: TE to LE
                [0.0, 0.1, 0.0],  # y values
            ],
        )

        ordered = CoordinateProcessor.order_surface_points(coords)

        # Should be reversed to run from LE to TE
        expected = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])

        np.testing.assert_array_equal(ordered, expected)

    def test_order_surface_points_no_reversal(self):
        """Test point ordering when no reversal is needed."""
        # Points already running from leading edge to trailing edge
        coords = jnp.array(
            [
                [0.0, 0.5, 1.0],  # x: LE to TE
                [0.0, 0.1, 0.0],  # y values
            ],
        )

        ordered = CoordinateProcessor.order_surface_points(coords)

        # Should remain unchanged
        np.testing.assert_array_equal(ordered, coords)

    def test_order_surface_points_single_point(self):
        """Test point ordering with single point."""
        coords = jnp.array([[0.5], [0.1]])

        ordered = CoordinateProcessor.order_surface_points(coords)

        # Should remain unchanged
        np.testing.assert_array_equal(ordered, coords)

    def test_close_airfoil_surfaces_basic(self):
        """Test basic airfoil closure functionality."""
        # Upper surface: LE to TE
        upper = jnp.array(
            [
                [0.1, 0.5, 0.9],  # x coordinates
                [0.05, 0.1, 0.02],  # y coordinates (upper)
            ],
        )

        # Lower surface: LE to TE
        lower = jnp.array(
            [
                [0.0, 0.5, 1.0],  # x coordinates
                [0.0, -0.05, 0.0],  # y coordinates (lower)
            ],
        )

        lower_closed, upper_closed = CoordinateProcessor.close_airfoil_surfaces(
            upper,
            lower,
        )

        # Upper should get lower's LE and TE points
        assert upper_closed.shape[1] == upper.shape[1] + 2  # Added LE and TE
        assert lower_closed.shape[1] == lower.shape[1]  # No additions needed

        # Check that lower's LE was added to upper's beginning
        np.testing.assert_array_equal(upper_closed[:, 0:1], lower[:, 0:1])
        # Check that lower's TE was added to upper's end
        np.testing.assert_array_equal(upper_closed[:, -1:], lower[:, -1:])

    def test_close_airfoil_surfaces_no_closure_needed(self):
        """Test airfoil closure when no closure is needed."""
        # Surfaces with matching leading and trailing edges
        upper = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])

        lower = jnp.array([[0.0, 0.5, 1.0], [0.0, -0.1, 0.0]])

        lower_closed, upper_closed = CoordinateProcessor.close_airfoil_surfaces(
            upper,
            lower,
        )

        # Should remain unchanged
        np.testing.assert_array_equal(upper_closed, upper)
        np.testing.assert_array_equal(lower_closed, lower)

    def test_close_airfoil_surfaces_empty(self):
        """Test airfoil closure with empty surfaces."""
        upper = jnp.empty((2, 0))
        lower = jnp.array([[0.0, 1.0], [0.0, 0.0]])

        lower_closed, upper_closed = CoordinateProcessor.close_airfoil_surfaces(
            upper,
            lower,
        )

        # Should return original arrays
        assert upper_closed.shape == upper.shape
        np.testing.assert_array_equal(lower_closed, lower)

    def test_to_selig_format(self):
        """Test conversion to selig format."""
        upper = jnp.array(
            [
                [0.0, 0.5, 1.0],  # LE to TE
                [0.0, 0.1, 0.0],
            ],
        )

        lower = jnp.array(
            [
                [0.0, 0.5, 1.0],  # LE to TE
                [0.0, -0.1, 0.0],
            ],
        )

        selig = CoordinateProcessor.to_selig_format(upper, lower)

        # Should be: upper reversed + lower
        expected = jnp.array(
            [
                [1.0, 0.5, 0.0, 0.0, 0.5, 1.0],  # TE to LE (upper) + LE to TE (lower)
                [0.0, 0.1, 0.0, 0.0, -0.1, 0.0],
            ],
        )

        np.testing.assert_array_equal(selig, expected)

    def test_split_selig_format(self):
        """Test splitting selig format coordinates."""
        # Selig format: TE to LE (upper) + LE to TE (lower)
        selig = jnp.array(
            [[1.0, 0.5, 0.0, 0.0, 0.5, 1.0], [0.0, 0.1, 0.0, 0.0, -0.1, 0.0]],
        )

        upper, lower, split_idx = CoordinateProcessor.split_selig_format(selig)

        # Upper should be LE to TE (reversed from first part)
        expected_upper = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])

        # Lower should be LE to TE (starting from last LE point)
        expected_lower = jnp.array([[0.0, 0.5, 1.0], [0.0, -0.1, 0.0]])

        np.testing.assert_array_equal(upper, expected_upper)
        np.testing.assert_array_equal(lower, expected_lower)
        assert split_idx == 2  # Index of leading edge (minimum x)

    def test_split_selig_format_with_mask(self):
        """Test splitting selig format with validity mask."""
        # Padded selig format
        selig = jnp.array(
            [
                [1.0, 0.5, 0.0, 0.0, 0.5, 1.0, jnp.nan, jnp.nan],
                [0.0, 0.1, 0.0, 0.0, -0.1, 0.0, jnp.nan, jnp.nan],
            ],
        )

        mask = jnp.array([True, True, True, True, True, True, False, False])

        upper, lower, split_idx = CoordinateProcessor.split_selig_format(selig, mask)

        # Should work the same as without padding
        expected_upper = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])

        # Lower should start from the last LE point
        expected_lower = jnp.array([[0.0, 0.5, 1.0], [0.0, -0.1, 0.0]])

        np.testing.assert_array_equal(upper, expected_upper)
        np.testing.assert_array_equal(lower, expected_lower)

    def test_split_selig_format_empty(self):
        """Test splitting empty selig format."""
        selig = jnp.empty((2, 0))

        upper, lower, split_idx = CoordinateProcessor.split_selig_format(selig)

        assert upper.shape == (2, 0)
        assert lower.shape == (2, 0)
        assert split_idx == 0

    def test_remove_duplicate_points(self):
        """Test removal of duplicate consecutive points."""
        # Coordinates with duplicates
        coords = jnp.array(
            [
                [0.0, 0.0, 0.5, 0.5, 1.0],  # Duplicates at positions 0-1 and 2-3
                [0.0, 0.0, 0.1, 0.1, 0.0],
            ],
        )

        unique = CoordinateProcessor.remove_duplicate_points(coords)

        # Should remove duplicates
        expected = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])

        np.testing.assert_array_equal(unique, expected)

    def test_remove_duplicate_points_no_duplicates(self):
        """Test duplicate removal with no duplicates."""
        coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])

        unique = CoordinateProcessor.remove_duplicate_points(coords)

        # Should remain unchanged
        np.testing.assert_array_equal(unique, coords)

    def test_remove_duplicate_points_single_point(self):
        """Test duplicate removal with single point."""
        coords = jnp.array([[0.5], [0.1]])

        unique = CoordinateProcessor.remove_duplicate_points(coords)

        # Should remain unchanged
        np.testing.assert_array_equal(unique, coords)

    def test_preprocess_coordinates_complete_pipeline(self):
        """Test the complete preprocessing pipeline."""
        # Create test data with various issues
        upper = jnp.array(
            [
                [1.0, 0.5, jnp.nan, 0.1, 0.1, 0.0],  # Reversed, NaN, duplicate
                [0.0, 0.1, 0.05, 0.05, 0.05, 0.0],
            ],
        )

        lower = jnp.array(
            [
                [1.0, 0.5, 0.2, 0.0],  # Reversed
                [0.0, -0.1, -0.05, 0.0],
            ],
        )

        upper_proc, lower_proc = CoordinateProcessor.preprocess_coordinates(
            upper,
            lower,
            remove_duplicates=True,
            validate=True,
        )

        # Check that processing worked
        assert upper_proc.shape[0] == 2
        assert lower_proc.shape[0] == 2
        assert upper_proc.shape[1] > 0
        assert lower_proc.shape[1] > 0

        # Check that points are ordered (LE to TE)
        assert upper_proc[0, 0] <= upper_proc[0, -1]  # x increases
        assert lower_proc[0, 0] <= lower_proc[0, -1]  # x increases

        # Check that no NaN values remain
        assert not jnp.any(jnp.isnan(upper_proc))
        assert not jnp.any(jnp.isnan(lower_proc))

    def test_preprocess_selig_coordinates(self):
        """Test preprocessing of selig format coordinates."""
        # Selig format with issues
        selig = jnp.array(
            [
                [1.0, 0.5, jnp.nan, 0.0, 0.0, 0.5, 1.0],  # NaN and duplicate
                [0.0, 0.1, 0.05, 0.0, 0.0, -0.1, 0.0],
            ],
        )

        upper_proc, lower_proc = CoordinateProcessor.preprocess_selig_coordinates(
            selig,
            remove_duplicates=True,
            validate=True,
        )

        # Check that processing worked
        assert upper_proc.shape[0] == 2
        assert lower_proc.shape[0] == 2
        assert upper_proc.shape[1] > 0
        assert lower_proc.shape[1] > 0

        # Check that no NaN values remain
        assert not jnp.any(jnp.isnan(upper_proc))
        assert not jnp.any(jnp.isnan(lower_proc))

    def test_prepare_for_jit(self):
        """Test preparation for JIT compilation."""
        upper = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])

        lower = jnp.array([[0.0, 0.5, 1.0], [0.0, -0.1, 0.0]])

        padded_coords, validity_mask, n_valid, n_upper, buffer_size = (
            CoordinateProcessor.prepare_for_jit(upper, lower)
        )

        # Check output shapes and values
        assert padded_coords.shape[0] == 2
        assert padded_coords.shape[1] == buffer_size
        assert validity_mask.shape[0] == buffer_size
        assert n_valid == 6  # 3 upper + 3 lower
        assert n_upper == 3
        assert buffer_size >= n_valid

        # Check that valid points are correct
        valid_coords = padded_coords[:, validity_mask]
        assert valid_coords.shape[1] == n_valid

        # Check that invalid points are NaN
        invalid_coords = padded_coords[:, ~validity_mask]
        if invalid_coords.shape[1] > 0:
            assert jnp.all(jnp.isnan(invalid_coords))

    def test_prepare_for_jit_custom_buffer_size(self):
        """Test JIT preparation with custom buffer size."""
        upper = jnp.array([[0.0, 1.0], [0.0, 0.0]])

        lower = jnp.array([[0.0, 1.0], [0.0, 0.0]])

        custom_buffer_size = 64

        padded_coords, validity_mask, n_valid, n_upper, buffer_size = (
            CoordinateProcessor.prepare_for_jit(upper, lower, custom_buffer_size)
        )

        assert buffer_size == custom_buffer_size
        assert padded_coords.shape[1] == custom_buffer_size
        assert validity_mask.shape[0] == custom_buffer_size

    def test_preprocessing_preserves_airfoil_shape(self):
        """Test that preprocessing preserves the overall airfoil shape."""
        # Create a simple symmetric airfoil
        x = jnp.linspace(0, 1, 10)
        y_upper = 0.1 * jnp.sin(jnp.pi * x)
        y_lower = -0.1 * jnp.sin(jnp.pi * x)

        upper = jnp.array([x, y_upper])
        lower = jnp.array([x, y_lower])

        upper_proc, lower_proc = CoordinateProcessor.preprocess_coordinates(
            upper,
            lower,
        )

        # Check that the airfoil is still symmetric (approximately)
        # Leading edge should be at x=0
        assert jnp.abs(upper_proc[0, 0]) < 1e-10
        assert jnp.abs(lower_proc[0, 0]) < 1e-10

        # Trailing edge should be at x=1
        assert jnp.abs(upper_proc[0, -1] - 1.0) < 1e-10
        assert jnp.abs(lower_proc[0, -1] - 1.0) < 1e-10

        # Upper surface should be above lower surface
        # (at least at some points)
        mid_idx = upper_proc.shape[1] // 2
        assert upper_proc[1, mid_idx] > lower_proc[1, mid_idx]


class TestCoordinateProcessorIntegration:
    """Integration tests for CoordinateProcessor with other components."""

    def test_integration_with_buffer_manager(self):
        """Test integration between CoordinateProcessor and AirfoilBufferManager."""
        # Create test airfoil
        upper = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])

        lower = jnp.array([[0.0, 0.5, 1.0], [0.0, -0.1, 0.0]])

        # Preprocess coordinates
        upper_proc, lower_proc = CoordinateProcessor.preprocess_coordinates(
            upper,
            lower,
        )

        # Convert to selig format
        selig_coords = CoordinateProcessor.to_selig_format(upper_proc, lower_proc)

        # Use buffer manager to pad
        buffer_size = AirfoilBufferManager.determine_buffer_size(selig_coords.shape[1])
        padded_coords, validity_mask, n_valid = AirfoilBufferManager.pad_and_mask(
            selig_coords,
            buffer_size,
        )

        # Verify integration
        assert padded_coords.shape == (2, buffer_size)
        assert validity_mask.shape == (buffer_size,)
        assert n_valid == selig_coords.shape[1]

        # Extract valid data and verify it matches original
        valid_coords = AirfoilBufferManager.extract_valid_data(
            padded_coords,
            validity_mask,
        )
        np.testing.assert_array_equal(valid_coords, selig_coords)

    def test_round_trip_selig_conversion(self):
        """Test that selig format conversion is reversible."""
        # Original surfaces
        upper = jnp.array([[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.1, 0.15, 0.1, 0.0]])

        lower = jnp.array([[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, -0.05, -0.1, -0.05, 0.0]])

        # Convert to selig and back
        selig = CoordinateProcessor.to_selig_format(upper, lower)
        upper_recovered, lower_recovered, _ = CoordinateProcessor.split_selig_format(
            selig,
        )

        # Should recover original surfaces (within numerical precision)
        # Note: The recovered lower surface includes the leading edge point
        np.testing.assert_array_almost_equal(upper_recovered, upper, decimal=10)
        np.testing.assert_array_almost_equal(lower_recovered, lower, decimal=10)

    def test_preprocessing_with_real_airfoil_data(self):
        """Test preprocessing with realistic airfoil data patterns."""
        # Simulate NACA 0012-like airfoil data with some realistic issues
        n_points = 50
        x = jnp.linspace(0, 1, n_points)

        # NACA 0012 thickness distribution (simplified)
        thickness = 0.12 * (
            0.2969 * jnp.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )

        # Create upper and lower surfaces
        y_upper = thickness / 2
        y_lower = -thickness / 2

        # Add some realistic issues
        upper = jnp.array([x, y_upper])
        lower = jnp.array([x, y_lower])

        # Add a NaN in the middle
        upper = upper.at[1, 25].set(jnp.nan)

        # Add a duplicate point
        lower = jnp.concatenate([lower, lower[:, 10:11]], axis=1)

        # Preprocess
        upper_proc, lower_proc = CoordinateProcessor.preprocess_coordinates(
            upper,
            lower,
            remove_duplicates=True,
            validate=True,
        )

        # Verify results
        assert not jnp.any(jnp.isnan(upper_proc))
        assert not jnp.any(jnp.isnan(lower_proc))
        assert upper_proc.shape[1] == n_points - 1  # One NaN removed
        assert (
            lower_proc.shape[1] == n_points
        )  # Duplicate removed, back to original size

        # Verify airfoil properties are preserved
        assert upper_proc[0, 0] == 0.0  # Leading edge at x=0
        assert upper_proc[0, -1] == 1.0  # Trailing edge at x=1

        # For comparison, we need to handle the fact that surfaces might have different lengths
        # due to preprocessing. Let's check at common x positions
        min_len = min(upper_proc.shape[1], lower_proc.shape[1])
        if min_len > 0:
            # Check that upper surface is above lower surface at some points
            mid_idx = min_len // 2
            assert upper_proc[1, mid_idx] >= lower_proc[1, mid_idx]  # Upper above lower


if __name__ == "__main__":
    pytest.main([__file__])
