"""
Unit tests for JAX airfoil buffer management system.

Tests the AirfoilBufferManager class functionality including:
- Buffer size determination
- Coordinate padding and masking
- Buffer overflow handling
- Data extraction utilities
"""

import jax.numpy as jnp
import numpy as np
import pytest

from ICARUS.airfoils.jax.buffer_manager import AirfoilBufferManager


class TestAirfoilBufferManager:
    """Test suite for AirfoilBufferManager class."""

    def test_determine_buffer_size_basic(self):
        """Test basic buffer size determination."""
        # Test cases: (n_points, expected_buffer_size)
        test_cases = [
            (1, 32),  # Minimum buffer size
            (16, 32),  # Within minimum
            (32, 32),  # Exact match
            (33, 64),  # Next power of 2
            (64, 64),  # Exact match
            (65, 128),  # Next power of 2
            (200, 256),  # Typical airfoil size
            (1000, 1024),  # Large airfoil
            (4096, 4096),  # Maximum size
        ]

        for n_points, expected in test_cases:
            result = AirfoilBufferManager.determine_buffer_size(n_points)
            assert (
                result == expected
            ), f"For {n_points} points, expected {expected}, got {result}"

    def test_determine_buffer_size_edge_cases(self):
        """Test edge cases for buffer size determination."""
        # Test zero points
        with pytest.raises(ValueError, match="Number of points must be positive"):
            AirfoilBufferManager.determine_buffer_size(0)

        # Test negative points
        with pytest.raises(ValueError, match="Number of points must be positive"):
            AirfoilBufferManager.determine_buffer_size(-1)

        # Test exceeding maximum buffer size
        with pytest.raises(ValueError, match="exceeds maximum buffer size"):
            AirfoilBufferManager.determine_buffer_size(5000)

    def test_next_power_of_2(self):
        """Test next power of 2 calculation."""
        test_cases = [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 4),
            (4, 4),
            (5, 8),
            (8, 8),
            (9, 16),
            (16, 16),
            (17, 32),
            (100, 128),
            (1024, 1024),
            (1025, 2048),
        ]

        for n, expected in test_cases:
            result = AirfoilBufferManager.next_power_of_2(n)
            assert result == expected, f"For {n}, expected {expected}, got {result}"

    def test_pad_coordinates_basic(self):
        """Test basic coordinate padding functionality."""
        # Create test coordinates
        coords = jnp.array([[0.0, 0.5, 1.0], [0.1, 0.2, 0.1]], dtype=jnp.float32)
        target_size = 5

        padded = AirfoilBufferManager.pad_coordinates(coords, target_size)

        # Check shape
        assert padded.shape == (2, 5)

        # Check that original data is preserved
        np.testing.assert_array_equal(padded[:, :3], coords)

        # Check that padding is NaN
        assert jnp.isnan(padded[:, 3:]).all()

    def test_pad_coordinates_no_padding_needed(self):
        """Test padding when target size equals current size."""
        coords = jnp.array([[0.0, 0.5, 1.0], [0.1, 0.2, 0.1]], dtype=jnp.float32)
        target_size = 3

        padded = AirfoilBufferManager.pad_coordinates(coords, target_size)

        # Should return identical array
        np.testing.assert_array_equal(padded, coords)

    def test_pad_coordinates_invalid_input(self):
        """Test padding with invalid inputs."""
        coords = jnp.array([[0.0, 0.5, 1.0], [0.1, 0.2, 0.1]], dtype=jnp.float32)

        # Test target size smaller than current
        with pytest.raises(ValueError, match="Target size .* must be >= current size"):
            AirfoilBufferManager.pad_coordinates(coords, 2)

        # Test invalid coordinate shape
        invalid_coords = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)  # 1D array
        with pytest.raises(ValueError, match="Coordinates must have shape"):
            AirfoilBufferManager.pad_coordinates(invalid_coords, 5)

        # Test wrong number of coordinate dimensions
        invalid_coords = jnp.array(
            [[[0.0, 0.5]], [[0.1, 0.2]]],
            dtype=jnp.float32,
        )  # 3D array
        with pytest.raises(ValueError, match="Coordinates must have shape"):
            AirfoilBufferManager.pad_coordinates(invalid_coords, 5)

    def test_create_validity_mask(self):
        """Test validity mask creation."""
        # Test basic mask creation
        mask = AirfoilBufferManager.create_validity_mask(3, 5)
        expected = jnp.array([True, True, True, False, False])
        np.testing.assert_array_equal(mask, expected)

        # Test full buffer
        mask = AirfoilBufferManager.create_validity_mask(5, 5)
        expected = jnp.array([True, True, True, True, True])
        np.testing.assert_array_equal(mask, expected)

        # Test empty buffer
        mask = AirfoilBufferManager.create_validity_mask(0, 5)
        expected = jnp.array([False, False, False, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_create_validity_mask_invalid_input(self):
        """Test validity mask creation with invalid inputs."""
        # Test negative n_valid
        with pytest.raises(
            ValueError,
            match="Number of valid points must be non-negative",
        ):
            AirfoilBufferManager.create_validity_mask(-1, 5)

        # Test n_valid exceeding buffer size
        with pytest.raises(
            ValueError,
            match="Number of valid points .* exceeds buffer size",
        ):
            AirfoilBufferManager.create_validity_mask(6, 5)

    def test_pad_and_mask(self):
        """Test combined padding and masking operation."""
        coords = jnp.array([[0.0, 0.5, 1.0], [0.1, 0.2, 0.1]], dtype=jnp.float32)
        target_size = 5

        padded_coords, validity_mask, n_valid = AirfoilBufferManager.pad_and_mask(
            coords,
            target_size,
        )

        # Check shapes
        assert padded_coords.shape == (2, 5)
        assert validity_mask.shape == (5,)
        assert n_valid == 3

        # Check that original data is preserved
        np.testing.assert_array_equal(padded_coords[:, :3], coords)

        # Check mask
        expected_mask = jnp.array([True, True, True, False, False])
        np.testing.assert_array_equal(validity_mask, expected_mask)

        # Check padding is NaN
        assert jnp.isnan(padded_coords[:, 3:]).all()

    def test_handle_buffer_overflow(self):
        """Test buffer overflow handling."""
        # Test normal overflow
        new_size = AirfoilBufferManager.handle_buffer_overflow(64, 100)
        assert new_size == 128

        # Test when required size equals current size (shouldn't happen in practice)
        new_size = AirfoilBufferManager.handle_buffer_overflow(64, 64)
        assert new_size >= 64

        # Test exceeding maximum buffer size
        with pytest.raises(ValueError, match="exceeds maximum buffer size"):
            AirfoilBufferManager.handle_buffer_overflow(1024, 5000)

    def test_extract_valid_data(self):
        """Test extraction of valid data from padded arrays."""
        # Create padded coordinates with some NaN padding
        padded_coords = jnp.array(
            [[0.0, 0.5, 1.0, jnp.nan, jnp.nan], [0.1, 0.2, 0.1, jnp.nan, jnp.nan]],
            dtype=jnp.float32,
        )

        validity_mask = jnp.array([True, True, True, False, False])

        valid_coords = AirfoilBufferManager.extract_valid_data(
            padded_coords,
            validity_mask,
        )

        # Check shape
        assert valid_coords.shape == (2, 3)

        # Check that we got the valid data
        expected = jnp.array([[0.0, 0.5, 1.0], [0.1, 0.2, 0.1]], dtype=jnp.float32)
        np.testing.assert_array_equal(valid_coords, expected)

    def test_get_buffer_info(self):
        """Test buffer information utility."""
        coords = jnp.array([[0.0, 0.5, 1.0], [0.1, 0.2, 0.1]], dtype=jnp.float32)

        info = AirfoilBufferManager.get_buffer_info(coords)

        # Check that all expected keys are present
        expected_keys = {
            "n_points",
            "recommended_buffer_size",
            "padding_needed",
            "memory_efficiency",
            "buffer_utilization",
        }
        assert set(info.keys()) == expected_keys

        # Check values
        assert info["n_points"] == 3
        assert info["recommended_buffer_size"] == 32
        assert info["padding_needed"] == 29
        assert info["memory_efficiency"] == 3 / 32
        assert info["buffer_utilization"] == "9.4%"

    def test_buffer_sizes_are_powers_of_2(self):
        """Test that all default buffer sizes are powers of 2."""
        for size in AirfoilBufferManager.DEFAULT_BUFFER_SIZES:
            # Check if size is a power of 2
            assert size > 0 and (size & (size - 1)) == 0, f"{size} is not a power of 2"

    def test_buffer_sizes_are_sorted(self):
        """Test that default buffer sizes are in ascending order."""
        sizes = AirfoilBufferManager.DEFAULT_BUFFER_SIZES
        assert sizes == sorted(sizes), "Buffer sizes should be in ascending order"

    def test_min_max_buffer_sizes(self):
        """Test that min and max buffer sizes are consistent with defaults."""
        sizes = AirfoilBufferManager.DEFAULT_BUFFER_SIZES
        assert AirfoilBufferManager.MIN_BUFFER_SIZE == sizes[0]
        assert AirfoilBufferManager.MAX_BUFFER_SIZE == sizes[-1]


class TestBufferManagerIntegration:
    """Integration tests for buffer manager with realistic airfoil data."""

    def test_typical_airfoil_workflow(self):
        """Test typical workflow with realistic airfoil coordinates."""
        # Create realistic airfoil coordinates (NACA-like)
        n_points = 100
        x = jnp.linspace(0, 1, n_points)
        y_upper = 0.05 * jnp.sqrt(x) * (1 - x)  # Simple airfoil shape
        y_lower = -0.03 * jnp.sqrt(x) * (1 - x)

        coords = jnp.array([x, y_upper], dtype=jnp.float32)

        # Determine buffer size
        buffer_size = AirfoilBufferManager.determine_buffer_size(n_points)
        assert buffer_size == 128  # Next power of 2 after 100

        # Pad and mask
        padded_coords, validity_mask, n_valid = AirfoilBufferManager.pad_and_mask(
            coords,
            buffer_size,
        )

        # Verify results
        assert padded_coords.shape == (2, 128)
        assert validity_mask.shape == (128,)
        assert n_valid == 100
        assert validity_mask.sum() == 100

        # Extract valid data and verify it matches original
        extracted = AirfoilBufferManager.extract_valid_data(
            padded_coords,
            validity_mask,
        )
        np.testing.assert_array_equal(extracted, coords)

    def test_batch_processing_different_sizes(self):
        """Test buffer management for batch processing with different airfoil sizes."""
        # Create airfoils with different point counts
        sizes = [50, 75, 120, 200]
        airfoils = []

        for size in sizes:
            x = jnp.linspace(0, 1, size)
            y = 0.05 * jnp.sin(jnp.pi * x)  # Simple shape
            coords = jnp.array([x, y], dtype=jnp.float32)
            airfoils.append(coords)

        # Determine buffer size for batch (should accommodate largest)
        max_size = max(sizes)
        buffer_size = AirfoilBufferManager.determine_buffer_size(max_size)
        assert buffer_size == 256  # Next power of 2 after 200

        # Process each airfoil with the same buffer size
        processed_airfoils = []
        for coords in airfoils:
            padded, mask, n_valid = AirfoilBufferManager.pad_and_mask(
                coords,
                buffer_size,
            )
            processed_airfoils.append((padded, mask, n_valid))

        # Verify all have same buffer size but different valid counts
        for i, (padded, mask, n_valid) in enumerate(processed_airfoils):
            assert padded.shape == (2, buffer_size)
            assert mask.shape == (buffer_size,)
            assert n_valid == sizes[i]
            assert mask.sum() == sizes[i]

    def test_memory_efficiency_analysis(self):
        """Test memory efficiency analysis for different airfoil sizes."""
        test_sizes = [10, 50, 100, 200, 500, 1000]

        for size in test_sizes:
            # Create dummy coordinates
            coords = jnp.zeros((2, size), dtype=jnp.float32)

            # Get buffer info
            info = AirfoilBufferManager.get_buffer_info(coords)

            # Verify efficiency calculations
            expected_efficiency = size / info["recommended_buffer_size"]
            assert abs(info["memory_efficiency"] - expected_efficiency) < 1e-6

            # Efficiency should be reasonable (at least 25% for our power-of-2 strategy)
            assert (
                info["memory_efficiency"] >= 0.25
            ), f"Poor efficiency for size {size}: {info['memory_efficiency']}"


if __name__ == "__main__":
    pytest.main([__file__])
