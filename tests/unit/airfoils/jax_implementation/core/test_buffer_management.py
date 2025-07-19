"""
Buffer management tests for JAX airfoil implementation.

This module tests the buffer management functionality including:
- Buffer allocation and resizing
- Padding and masking operations
- Memory efficiency
- JIT compatibility of buffer operations

Requirements covered: 2.1, 2.2, 5.1, 5.2
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ICARUS.airfoils.jax_implementation.buffer_management import AirfoilBufferManager
from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


class TestBufferAllocation:
    """Test buffer allocation and management."""

    def test_buffer_size_determination(self):
        """Test automatic buffer size determination."""
        # Test with various input sizes
        for n_points in [10, 50, 100, 200, 500]:
            buffer_size = AirfoilBufferManager.determine_buffer_size(n_points)

            # Buffer should be power of 2 and >= n_points
            assert buffer_size >= n_points
            assert (buffer_size & (buffer_size - 1)) == 0  # Check if power of 2

            # Buffer shouldn't be excessively large
            assert buffer_size <= 2 * n_points

    def test_minimum_buffer_size(self):
        """Test minimum buffer size enforcement."""
        # Test with small input
        buffer_size = AirfoilBufferManager.determine_buffer_size(5)
        assert buffer_size >= AirfoilBufferManager.MIN_BUFFER_SIZE

    def test_pad_and_mask(self):
        """Test padding and masking operations."""
        # Create test data
        coords = jnp.array(
            [[1.0, 0.8, 0.6, 0.4, 0.2, 0.0], [0.0, 0.1, 0.2, 0.1, 0.05, 0.0]],
        )

        # Test padding to various sizes
        for buffer_size in [8, 16, 32]:
            padded, mask, n_valid = AirfoilBufferManager.pad_and_mask(
                coords,
                buffer_size,
            )

            # Check dimensions
            assert padded.shape == (2, buffer_size)
            assert mask.shape == (buffer_size,)
            assert n_valid == coords.shape[1]

            # Check mask correctness
            assert jnp.sum(mask) == coords.shape[1]
            assert jnp.all(mask[: coords.shape[1]])
            assert not jnp.any(mask[coords.shape[1] :])

            # Check padded values
            np.testing.assert_array_equal(padded[:, : coords.shape[1]], coords)
            assert jnp.all(jnp.isnan(padded[:, coords.shape[1] :]))

    def test_extract_valid_data(self):
        """Test extraction of valid data from padded buffer."""
        # Create test data with padding
        valid_data = jnp.array(
            [[1.0, 0.8, 0.6, 0.4, 0.2, 0.0], [0.0, 0.1, 0.2, 0.1, 0.05, 0.0]],
        )
        buffer_size = 16
        padded = jnp.pad(
            valid_data,
            ((0, 0), (0, buffer_size - valid_data.shape[1])),
            mode="constant",
            constant_values=jnp.nan,
        )
        mask = jnp.concatenate(
            [
                jnp.ones(valid_data.shape[1], dtype=bool),
                jnp.zeros(buffer_size - valid_data.shape[1], dtype=bool),
            ],
        )

        # Extract valid data
        extracted = AirfoilBufferManager.extract_valid_data(padded, mask)

        # Check result
        assert extracted.shape == valid_data.shape
        np.testing.assert_array_equal(extracted, valid_data)

    @pytest.mark.parametrize("jit_compile", [False, True])
    def test_jit_compatibility(self, jit_compile):
        """Test JIT compatibility of buffer operations."""
        # Create test data
        coords = jnp.array(
            [[1.0, 0.8, 0.6, 0.4, 0.2, 0.0], [0.0, 0.1, 0.2, 0.1, 0.05, 0.0]],
        )
        buffer_size = 16

        # Define function to test
        def pad_and_extract(coords, buffer_size):
            padded, mask, n_valid = AirfoilBufferManager.pad_and_mask(
                coords,
                buffer_size,
            )
            return AirfoilBufferManager.extract_valid_data(padded, mask)

        # JIT compile if requested
        if jit_compile:
            pad_and_extract = jax.jit(pad_and_extract)

        # Run function
        result = pad_and_extract(coords, buffer_size)

        # Check result
        assert result.shape == coords.shape
        np.testing.assert_array_equal(result, coords)


class TestBufferResizing:
    """Test buffer resizing operations."""

    def test_resize_buffer(self):
        """Test buffer resizing."""
        # Create airfoil with small buffer
        small_coords = jnp.array(
            [[1.0, 0.5, 0.0, 0.5, 1.0], [0.0, 0.1, 0.0, -0.1, 0.0]],
        )
        airfoil = JaxAirfoil(small_coords, buffer_size=8)

        # Resize to larger buffer
        larger_airfoil = airfoil.resize_buffer(32)

        # Check properties
        assert larger_airfoil.buffer_size == 32
        assert larger_airfoil.n_points == airfoil.n_points

        # Check coordinates preserved
        orig_x, orig_y = airfoil.get_coordinates()
        new_x, new_y = larger_airfoil.get_coordinates()
        np.testing.assert_array_equal(orig_x, new_x)
        np.testing.assert_array_equal(orig_y, new_y)

    def test_auto_resize_on_operations(self):
        """Test automatic buffer resizing during operations."""
        # Create airfoil with minimal buffer
        small_coords = jnp.array(
            [[1.0, 0.5, 0.0, 0.5, 1.0], [0.0, 0.1, 0.0, -0.1, 0.0]],
        )
        airfoil = JaxAirfoil(small_coords, buffer_size=8)

        # Create another airfoil with more points
        larger_coords = jnp.array(
            [
                jnp.linspace(1.0, 0.0, 20),
                jnp.concatenate(
                    [jnp.linspace(0.0, 0.1, 10), jnp.linspace(0.1, 0.0, 10)],
                ),
            ],
        )
        larger_airfoil = JaxAirfoil(larger_coords)

        # Combine airfoils (should trigger resize)
        combined = airfoil.combine_with(larger_airfoil)

        # Check buffer size increased
        assert combined.buffer_size >= larger_airfoil.n_points + airfoil.n_points
        assert combined.n_points > airfoil.n_points

    def test_optimize_buffer_size(self):
        """Test buffer size optimization."""
        # Create airfoil with oversized buffer
        small_coords = jnp.array(
            [[1.0, 0.5, 0.0, 0.5, 1.0], [0.0, 0.1, 0.0, -0.1, 0.0]],
        )
        airfoil = JaxAirfoil(small_coords, buffer_size=128)

        # Optimize buffer size
        optimized = airfoil.optimize_buffer()

        # Check buffer size reduced but still sufficient
        assert optimized.buffer_size < airfoil.buffer_size
        assert optimized.buffer_size >= optimized.n_points
        assert optimized.n_points == airfoil.n_points

        # Check coordinates preserved
        orig_x, orig_y = airfoil.get_coordinates()
        new_x, new_y = optimized.get_coordinates()
        np.testing.assert_array_equal(orig_x, new_x)
        np.testing.assert_array_equal(orig_y, new_y)
