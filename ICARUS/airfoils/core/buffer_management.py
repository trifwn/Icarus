"""
Buffer management system for JAX airfoil implementation.

This module provides static memory allocation utilities to enable efficient JIT compilation
while handling variable-sized airfoil data through padding and masking.
"""

from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

from .error_handling import AirfoilErrorHandler
from .error_handling import BufferOverflowError


class AirfoilBufferManager:
    """
    Manages static buffer allocation for JAX airfoil operations.

    This class provides utilities for:
    - Determining appropriate buffer sizes for variable-sized airfoil data
    - Padding coordinate arrays to fixed sizes for JIT compatibility
    - Creating validity masks for padded data
    - Managing buffer reallocation when capacity is exceeded
    """

    # Buffer sizes follow powers of 2 for efficient memory usage
    DEFAULT_BUFFER_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    MIN_BUFFER_SIZE = 32
    MAX_BUFFER_SIZE = 16384  # Increased to handle larger airfoils

    @staticmethod
    def determine_buffer_size(n_points: int) -> int:
        """
        Determine the appropriate buffer size for a given number of points.

        Uses the next power of 2 that can accommodate the required points,
        with a minimum buffer size to avoid frequent recompilation.

        Args:
            n_points: Number of actual data points needed

        Returns:
            Buffer size (power of 2) that can accommodate n_points

        Raises:
            BufferOverflowError: If n_points exceeds maximum supported buffer size
        """
        if n_points <= 0:
            raise BufferOverflowError(
                f"Number of points must be positive, got {n_points}. "
                f"Check that coordinate data is not empty.",
            )

        # Use the error handler to check buffer capacity
        needs_reallocation, suggested_size = AirfoilErrorHandler.check_buffer_capacity(
            n_points,
            0,
            AirfoilBufferManager.MAX_BUFFER_SIZE,
        )

        if needs_reallocation and suggested_size is not None:
            return suggested_size

        # Find the smallest buffer size that can accommodate n_points
        for buffer_size in AirfoilBufferManager.DEFAULT_BUFFER_SIZES:
            if n_points <= buffer_size:
                return buffer_size

        # If we get here, n_points is larger than our largest predefined size
        # but still within MAX_BUFFER_SIZE, so return MAX_BUFFER_SIZE
        return AirfoilBufferManager.MAX_BUFFER_SIZE

    @staticmethod
    def next_power_of_2(n: int) -> int:
        """
        Find the next power of 2 greater than or equal to n.

        Args:
            n: Input number

        Returns:
            Next power of 2 >= n
        """
        if n <= 0:
            return 1

        # Handle the case where n is already a power of 2
        if n & (n - 1) == 0:
            return n

        # Find the next power of 2
        power = 1
        while power < n:
            power <<= 1

        return power

    @staticmethod
    def pad_coordinates(
        coords: Float[Array, "2 n_points"],
        target_size: int,
    ) -> Float[Array, "2 target_size"]:
        """
        Pad coordinate array to target size with NaN values.

        Args:
            coords: Input coordinate array of shape (2, n_points) where first row is x, second is y
            target_size: Target buffer size

        Returns:
            Padded coordinate array of shape (2, target_size)

        Raises:
            ValueError: If target_size is smaller than current array size
        """
        if coords.ndim != 2 or coords.shape[0] != 2:
            raise ValueError(
                f"Coordinates must have shape (2, n_points), got {coords.shape}",
            )

        current_size = coords.shape[1]

        if target_size < current_size:
            raise ValueError(
                f"Target size ({target_size}) must be >= current size ({current_size})",
            )

        if target_size == current_size:
            return coords

        # Create padding with NaN values
        padding_size = target_size - current_size
        padding = jnp.full((2, padding_size), jnp.nan, dtype=coords.dtype)

        # Concatenate original coordinates with padding
        padded_coords = jnp.concatenate([coords, padding], axis=1)

        return padded_coords

    @staticmethod
    def create_validity_mask(
        n_valid: int,
        buffer_size: int,
    ) -> Bool[Array, " buffer_size"]:
        """
        Create boolean mask indicating which points in the buffer are valid.

        Args:
            n_valid: Number of valid points
            buffer_size: Total buffer size

        Returns:
            Boolean mask of shape (buffer_size,) where True indicates valid points

        Raises:
            ValueError: If n_valid exceeds buffer_size
        """
        if n_valid < 0:
            raise ValueError(
                f"Number of valid points must be non-negative, got {n_valid}",
            )

        if n_valid > buffer_size:
            raise ValueError(
                f"Number of valid points ({n_valid}) exceeds buffer size ({buffer_size})",
            )

        # Create mask: True for valid indices, False for padded indices
        mask = jnp.arange(buffer_size) < n_valid

        return mask

    @staticmethod
    def pad_and_mask(
        coords: Float[Array, "2 n_points"],
        target_size: int,
    ) -> Tuple[Float[Array, "2 target_size"], Bool[Array, " target_size"], int]:
        """
        Convenience function to pad coordinates and create validity mask in one call.

        Args:
            coords: Input coordinate array of shape (2, n_points)
            target_size: Target buffer size

        Returns:
            Tuple of (padded_coords, validity_mask, n_valid_points)
        """
        n_valid = coords.shape[1]
        padded_coords = AirfoilBufferManager.pad_coordinates(coords, target_size)
        validity_mask = AirfoilBufferManager.create_validity_mask(n_valid, target_size)

        return padded_coords, validity_mask, n_valid

    @staticmethod
    def handle_buffer_overflow(current_size: int, required_size: int) -> int:
        """
        Determine new buffer size when current buffer is too small.

        This triggers controlled recompilation with a larger buffer size.

        Args:
            current_size: Current buffer size
            required_size: Required number of points

        Returns:
            New buffer size that can accommodate required_size

        Raises:
            ValueError: If required_size exceeds maximum supported buffer size
        """
        if required_size > AirfoilBufferManager.MAX_BUFFER_SIZE:
            raise ValueError(
                f"Required size ({required_size}) exceeds maximum buffer size "
                f"({AirfoilBufferManager.MAX_BUFFER_SIZE})",
            )

        # Find the next appropriate buffer size
        new_size = AirfoilBufferManager.determine_buffer_size(required_size)

        # Ensure we're actually increasing the buffer size
        if new_size <= current_size:
            # This shouldn't happen with our power-of-2 strategy, but just in case
            new_size = AirfoilBufferManager.next_power_of_2(required_size)

            # Clamp to maximum size
            new_size = min(new_size, AirfoilBufferManager.MAX_BUFFER_SIZE)

        return new_size

    @staticmethod
    def extract_valid_data(
        padded_coords: Float[Array, "2 buffer_size"],
        validity_mask: Bool[Array, " buffer_size"],
    ) -> Float[Array, "2 n_valid"]:
        """
        Extract only the valid (non-padded) data from a padded coordinate array.

        Args:
            padded_coords: Padded coordinate array of shape (2, buffer_size)
            validity_mask: Boolean mask indicating valid points

        Returns:
            Coordinate array containing only valid points
        """
        # Use the mask to select only valid columns
        valid_coords = padded_coords[:, validity_mask]

        return valid_coords

    @staticmethod
    def get_buffer_info(coords: Float[Array, "2 n_points"]) -> dict:
        """
        Get buffer allocation information for given coordinates.

        Args:
            coords: Input coordinate array

        Returns:
            Dictionary containing buffer allocation information
        """
        n_points = coords.shape[1]
        recommended_buffer_size = AirfoilBufferManager.determine_buffer_size(n_points)

        return {
            "n_points": n_points,
            "recommended_buffer_size": recommended_buffer_size,
            "padding_needed": recommended_buffer_size - n_points,
            "memory_efficiency": n_points / recommended_buffer_size,
            "buffer_utilization": f"{100 * n_points / recommended_buffer_size:.1f}%",
        }
