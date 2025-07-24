"""
Coordinate preprocessing pipeline for JAX airfoil implementation.

This module handles the transition from eager (Python-only) operations to JIT-compatible
operations by preprocessing variable-sized airfoil coordinate data. It includes NaN filtering,
coordinate validation, ordering, closure, and selig format conversion utilities.
"""

from typing import Optional
from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import ArrayLike
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int

from .buffer_management import AirfoilBufferManager
from .error_handling import AirfoilErrorHandler
from .error_handling import AirfoilValidationError


class CoordinateProcessor:
    """
    Handles eager preprocessing of airfoil coordinates before JIT compilation.

    This class provides utilities for:
    - NaN filtering and coordinate validation
    - Coordinate ordering (leading edge to trailing edge)
    - Airfoil closure (ensuring proper leading/trailing edge connection)
    - Selig format conversion and splitting
    - Preprocessing pipeline for variable-sized input data
    """

    @staticmethod
    def filter_nan_coordinates(
        coords: Float[Array, "2 n_points"],
    ) -> Float[Array, "2 n_valid"]:
        """
        Remove NaN values from coordinate arrays.

        Args:
            coords: Input coordinate array of shape (2, n_points) where first row is x, second is y

        Returns:
            Filtered coordinate array with NaN values removed

        Raises:
            AirfoilValidationError: If coordinates have invalid shape
        """
        try:
            AirfoilErrorHandler.validate_coordinate_shape(coords, "input coordinates")
        except AirfoilValidationError as e:
            raise AirfoilValidationError(
                f"Cannot filter NaN coordinates: {e}. "
                f"{AirfoilErrorHandler.suggest_fixes('invalid_shape')}",
            )

        # Create mask for valid (non-NaN) points - both x and y must be valid
        valid_mask = ~jnp.isnan(coords[0, :]) & ~jnp.isnan(coords[1, :])

        # Filter coordinates using the mask
        filtered_coords = coords[:, valid_mask]

        # Check if any points remain after filtering
        if filtered_coords.shape[1] == 0:
            raise AirfoilValidationError(
                "All coordinate points contain NaN values. "
                f"{AirfoilErrorHandler.suggest_fixes('nan_coordinates')}",
            )

        return filtered_coords

    @staticmethod
    def validate_coordinates(coords: Float[Array, "2 {n_points}"]) -> None:
        """
        Validate coordinate arrays for common issues.

        Args:
            coords: Coordinate array to validate

        Raises:
            AirfoilValidationError: If coordinates contain invalid values or have invalid shape
        """
        # Use the comprehensive error handler for validation
        AirfoilErrorHandler.validate_coordinate_shape(coords, "coordinates")
        AirfoilErrorHandler.validate_coordinate_values(coords, "coordinates")
        AirfoilErrorHandler.validate_surface_ordering(coords, "coordinate surface")

    @staticmethod
    def order_surface_points(
        coords: Float[Array, "2 n_points"],
    ) -> Float[Array, "2 n_points"]:
        """
        Order surface points from leading edge to trailing edge.

        Args:
            coords: Surface coordinate array of shape (2, n_points)

        Returns:
            Ordered coordinate array with points running from leading edge to trailing edge
        """
        if coords.shape[1] <= 1:
            return coords

        x_coords = coords[0, :]

        # Determine if points need to be reversed
        # If first x > last x, then points run from trailing edge to leading edge
        needs_reversal = x_coords[0] > x_coords[-1]

        if needs_reversal:
            # Reverse the order of points
            ordered_coords = coords[:, ::-1]
        else:
            ordered_coords = coords

        return ordered_coords

    @staticmethod
    def close_airfoil_surfaces(
        upper: Float[Array, "2 n_upper"],
        lower: Float[Array, "2 n_lower"],
    ) -> Tuple[
        Float[Array, "2 n_upper_closed"],
        Float[Array, "2 n_lower_closed"],
    ]:
        """
        Close airfoil by ensuring proper leading/trailing edge connections.

        This function adds points at leading/trailing edges if needed to ensure
        the upper and lower surfaces connect properly.

        Args:
            upper: Upper surface coordinates
            lower: Lower surface coordinates

        Returns:
            Tuple of (closed_lower, closed_upper) surfaces
        """
        if upper.shape[1] == 0 or lower.shape[1] == 0:
            return lower, upper

        # Extract edge points
        upper_le = upper[:, 0:1]  # Leading edge of upper surface
        upper_te = upper[:, -1:]  # Trailing edge of upper surface
        lower_le = lower[:, 0:1]  # Leading edge of lower surface
        lower_te = lower[:, -1:]  # Trailing edge of lower surface

        # Get x-coordinates for comparison
        upper_le_x = upper[0, 0]
        lower_le_x = lower[0, 0]
        upper_te_x = upper[0, -1]
        lower_te_x = lower[0, -1]

        # Determine what needs to be added
        add_lower_le_to_upper = lower_le_x < upper_le_x
        add_upper_le_to_lower = upper_le_x < lower_le_x
        add_lower_te_to_upper = upper_te_x < lower_te_x
        add_upper_te_to_lower = lower_te_x < upper_te_x

        # Build closed upper surface
        upper_parts = []
        if add_lower_le_to_upper:
            upper_parts.append(lower_le)
        upper_parts.append(upper)
        if add_lower_te_to_upper:
            upper_parts.append(lower_te)

        closed_upper = jnp.concatenate(upper_parts, axis=1)

        # Build closed lower surface
        lower_parts = []
        if add_upper_le_to_lower:
            lower_parts.append(upper_le)
        lower_parts.append(lower)
        if add_upper_te_to_lower:
            lower_parts.append(upper_te)

        closed_lower = jnp.concatenate(lower_parts, axis=1)

        return closed_lower, closed_upper

    @staticmethod
    def to_selig_format(
        upper: Float[Array, "2 n_upper"],
        lower: Float[Array, "2 n_lower"],
    ) -> Float[Array, "2 n_total"]:
        """
        Convert upper and lower surface coordinates to selig format.

        Selig format runs from trailing edge, around the leading edge,
        back to the trailing edge. Upper surface is reversed, then lower surface is appended.

        Args:
            upper: Upper surface coordinates (leading edge to trailing edge)
            lower: Lower surface coordinates (leading edge to trailing edge)

        Returns:
            Coordinates in selig format
        """
        # Reverse upper surface (trailing edge to leading edge)
        upper_reversed = upper[:, ::-1]

        # Concatenate reversed upper surface with lower surface
        selig_coords = jnp.concatenate([upper_reversed, lower], axis=1)

        return selig_coords

    @staticmethod
    def split_selig_format(
        selig_coords: Float[Array, "2 n_total"],
        validity_mask: Optional[Bool[Array, " n_total"]] = None,
    ) -> Tuple[
        Float[Array, "2 n_upper"],
        Float[Array, "2 n_lower"],
        Int[ArrayLike, ""],
    ]:
        """
        Split selig format coordinates into upper and lower surfaces.

        Args:
            selig_coords: Coordinates in selig format
            validity_mask: Optional mask indicating valid points (for padded arrays)

        Returns:
            Tuple of (upper_surface, lower_surface, split_index)
            where split_index indicates where upper surface ends
        """
        # For JIT compatibility, we need to avoid dynamic slicing
        # We'll use a fixed approach that works with the selig format

        # In selig format, the coordinates start at the trailing edge,
        # go around the upper surface to the leading edge, and then
        # continue along the lower surface back to the trailing edge

        # For simplicity and JIT compatibility, we'll assume the middle point
        # is approximately the leading edge (minimum x-coordinate)

        if validity_mask is not None:
            # Count valid points
            n_valid = jnp.sum(validity_mask).astype(jnp.int32)
        else:
            n_valid = selig_coords.shape[1]

        # Approximate the split point as half the valid points
        # This is a simplification that works for most airfoils
        split_idx = n_valid // 2

        # For JIT compatibility, we'll use a fixed-size approach
        # We'll create upper and lower surfaces with fixed maximum sizes
        max_size = selig_coords.shape[1]

        # Use static slicing to avoid boolean indexing issues in JIT
        # Upper surface: first split_idx points (reversed to go from LE to TE)
        upper_coords = selig_coords[:, :split_idx][:, ::-1]

        # Lower surface: remaining points (from split_idx to n_valid)
        lower_coords = selig_coords[:, split_idx:n_valid]

        return upper_coords, lower_coords, split_idx

    @staticmethod
    def remove_duplicate_points(
        coords: Float[Array, "2 n_points"],
    ) -> Float[Array, "2 n_unique"]:
        """
        Remove duplicate consecutive points from coordinate array.

        Args:
            coords: Input coordinate array

        Returns:
            Coordinate array with duplicate consecutive points removed
        """
        if coords.shape[1] <= 1:
            return coords

        # JAX-compatible duplicate detection
        # Create complex numbers for easy duplicate detection
        complex_coords = coords[0, :] + 1j * coords[1, :]

        # For JAX compatibility, we'll use a simpler approach
        # Check consecutive points for duplicates
        if coords.shape[1] <= 1:
            return coords

        # Calculate differences between consecutive points
        diffs = jnp.diff(complex_coords)

        # Points are duplicates if the difference is very small
        is_duplicate = jnp.abs(diffs) < 1e-12

        # Create mask for points to keep (first point + non-duplicates)
        keep_mask = jnp.concatenate([jnp.array([True]), ~is_duplicate])

        # Apply mask to coordinates
        return coords[:, keep_mask]

    @staticmethod
    def preprocess_coordinates(
        upper: Float[Array, "2 n_upper"],
        lower: Float[Array, "2 n_lower"],
        remove_duplicates: bool = True,
        validate: bool = True,
    ) -> Tuple[Float[Array, "2 n_upper_proc"], Float[Array, "2 n_lower_proc"]]:
        """
        Complete preprocessing pipeline for airfoil coordinates.

        This function performs all necessary eager preprocessing operations:
        1. NaN filtering
        2. Coordinate validation (optional)
        3. Duplicate point removal (optional)
        4. Point ordering (leading edge to trailing edge)
        5. Airfoil closure

        Args:
            upper: Upper surface coordinates
            lower: Lower surface coordinates
            remove_duplicates: Whether to remove duplicate consecutive points
            validate: Whether to perform coordinate validation

        Returns:
            Tuple of (processed_upper, processed_lower) coordinates
        """
        # Step 1: Filter NaN values
        upper_filtered = CoordinateProcessor.filter_nan_coordinates(upper)
        lower_filtered = CoordinateProcessor.filter_nan_coordinates(lower)

        # Step 2: Validate coordinates (optional)
        if validate:
            if upper_filtered.shape[1] > 0:
                CoordinateProcessor.validate_coordinates(upper_filtered)
            if lower_filtered.shape[1] > 0:
                CoordinateProcessor.validate_coordinates(lower_filtered)

        # Step 3: Remove duplicate points (optional)
        if remove_duplicates:
            upper_filtered = CoordinateProcessor.remove_duplicate_points(upper_filtered)
            lower_filtered = CoordinateProcessor.remove_duplicate_points(lower_filtered)

        # Step 4: Order points from leading edge to trailing edge
        upper_ordered = CoordinateProcessor.order_surface_points(upper_filtered)
        lower_ordered = CoordinateProcessor.order_surface_points(lower_filtered)

        # Step 5: Close airfoil
        lower_closed, upper_closed = CoordinateProcessor.close_airfoil_surfaces(
            upper_ordered,
            lower_ordered,
        )

        return upper_closed, lower_closed

    @staticmethod
    def preprocess_selig_coordinates(
        selig_coords: Float[Array, "2 n_total"],
        remove_duplicates: bool = True,
        validate: bool = True,
    ) -> Tuple[Float[Array, "2 n_upper"], Float[Array, "2 n_lower"]]:
        """
        Preprocessing pipeline for coordinates already in selig format.

        Args:
            selig_coords: Coordinates in selig format
            remove_duplicates: Whether to remove duplicate consecutive points
            validate: Whether to perform coordinate validation

        Returns:
            Tuple of (processed_upper, processed_lower) coordinates
        """
        # Step 1: Filter NaN values
        filtered_coords = CoordinateProcessor.filter_nan_coordinates(selig_coords)

        # Step 2: Validate coordinates (optional)
        if validate and filtered_coords.shape[1] > 0:
            CoordinateProcessor.validate_coordinates(filtered_coords)

        # Step 3: Remove duplicate points (optional)
        if remove_duplicates:
            filtered_coords = CoordinateProcessor.remove_duplicate_points(
                filtered_coords,
            )

        # Step 4: Split into upper and lower surfaces
        upper, lower, _ = CoordinateProcessor.split_selig_format(filtered_coords)

        # Step 5: Close airfoil
        lower_closed, upper_closed = CoordinateProcessor.close_airfoil_surfaces(
            upper,
            lower,
        )

        return upper_closed, lower_closed

    @staticmethod
    def prepare_for_jit(
        upper: Float[Array, "2 n_upper"],
        lower: Float[Array, "2 n_lower"],
        target_buffer_size: Optional[int] = None,
    ) -> Tuple[
        Float[Array, "2 buffer_size"],
        Bool[Array, " buffer_size"],
        int,
        int,
        int,
    ]:
        """
        Prepare preprocessed coordinates for JIT compilation.

        This function converts preprocessed coordinates to selig format,
        pads them to a fixed buffer size, and creates validity masks.

        Args:
            upper: Preprocessed upper surface coordinates
            lower: Preprocessed lower surface coordinates
            target_buffer_size: Target buffer size (auto-determined if None)

        Returns:
            Tuple of (padded_selig_coords, validity_mask, n_valid_points,
                     n_upper_points, buffer_size)
        """
        # Convert to selig format
        selig_coords = CoordinateProcessor.to_selig_format(upper, lower)

        # Determine buffer size if not provided
        if target_buffer_size is None:
            target_buffer_size = AirfoilBufferManager.determine_buffer_size(
                selig_coords.shape[1],
            )

        # Pad coordinates and create mask
        padded_coords, validity_mask, n_valid = AirfoilBufferManager.pad_and_mask(
            selig_coords,
            target_buffer_size,
        )

        return padded_coords, validity_mask, n_valid, upper.shape[1], target_buffer_size
