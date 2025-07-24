"""
JAX-compatible airfoil class with JIT compilation and automatic differentiation support.

This module provides the JaxAirfoil class, which is a fully JAX-compatible implementation
of an airfoil with support for JIT compilation and automatic differentiation. It uses
static buffer allocation with padding and masking to handle variable-sized airfoil data
efficiently while maintaining API compatibility with the original implementation.
"""

import os
import re
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import requests
from jaxtyping import Array
from matplotlib.axes import Axes

from .buffer_management import AirfoilBufferManager
from .coordinate_processor import CoordinateProcessor
from .error_handling import AirfoilErrorHandler
from .error_handling import AirfoilValidationError
from .operations import JaxAirfoilOps


@jax.tree_util.register_pytree_node_class
class JaxAirfoil:
    """
    JAX-compatible airfoil class with JIT compilation and automatic differentiation support.

    This class provides a fully JAX-compatible implementation of an airfoil that supports
    JIT compilation and automatic differentiation. It uses static buffer allocation with
    padding and masking to handle variable-sized airfoil data efficiently.

    The class follows a functional programming approach with immutable data structures
    and pure functions for all geometric operations. It maintains API compatibility with
    the original implementation while enabling efficient JAX transformations.

    Attributes:
        _coordinates: JAX array of shape (2, max_buffer_size) containing [x, y] coordinates
                     in selig format (padded with NaN values)
        _validity_mask: Boolean mask indicating which points in the buffer are valid
        _n_valid_points: Number of actual valid points (static for JIT)
        _upper_split_idx: Index where upper surface ends in selig format
        _max_buffer_size: Current buffer capacity (static for JIT)
        _metadata: Non-differentiable metadata (name, etc.)
    """

    def __init__(
        self,
        coordinates: Optional[Union[Array, np.ndarray]] = None,
        name: str = "JaxAirfoil",
        buffer_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a JaxAirfoil instance with the given coordinates.

        Args:
            coordinates: Airfoil coordinates in selig format (2, n_points) where first row is x, second is y.
                        If None, creates an empty airfoil.
            name: Name of the airfoil
            buffer_size: Optional buffer size to use (auto-determined if None)
            metadata: Optional additional metadata

        Raises:
            ValueError: If coordinates have invalid shape or contain invalid values
        """
        # Initialize metadata
        self._metadata = metadata or {}
        self._metadata["name"] = name

        if coordinates is None:
            # Create empty airfoil with minimum buffer size
            buffer_size = buffer_size or AirfoilBufferManager.MIN_BUFFER_SIZE
            self._coordinates = jnp.full((2, buffer_size), jnp.nan)
            self._validity_mask = jnp.zeros(buffer_size, dtype=bool)
            self._n_valid_points = 0
            self._upper_split_idx = 0
            self._max_buffer_size = buffer_size
            return

        # Convert to JAX array if needed
        if isinstance(coordinates, np.ndarray):
            coordinates = jnp.array(coordinates)

        # Validate input using comprehensive error handling
        try:
            AirfoilErrorHandler.validate_coordinate_shape(
                coordinates,
                "input coordinates",
            )
        except AirfoilValidationError as e:
            raise AirfoilValidationError(
                f"Invalid coordinates for JaxAirfoil '{name}': {e}. "
                f"{AirfoilErrorHandler.suggest_fixes('invalid_shape')}",
            )

        # Preprocess coordinates (filter NaN, validate, etc.)
        upper, lower, split_idx = CoordinateProcessor.split_selig_format(coordinates)
        upper_processed, lower_processed = CoordinateProcessor.preprocess_coordinates(
            upper,
            lower,
        )

        # Determine buffer size if not provided
        n_points = upper_processed.shape[1] + lower_processed.shape[1]
        if buffer_size is None:
            buffer_size = AirfoilBufferManager.determine_buffer_size(n_points)

        # Convert back to selig format and prepare for JIT
        selig_coords = CoordinateProcessor.to_selig_format(
            upper_processed,
            lower_processed,
        )
        padded_coords, validity_mask, n_valid = AirfoilBufferManager.pad_and_mask(
            selig_coords,
            buffer_size,
        )

        # Store the processed data
        self._coordinates = padded_coords
        self._validity_mask = validity_mask
        self._n_valid_points = n_valid
        self._upper_split_idx = upper_processed.shape[1]
        self._max_buffer_size = buffer_size

    @classmethod
    def from_upper_lower(
        cls,
        upper: Union[Array, np.ndarray],
        lower: Union[Array, np.ndarray],
        name: str = "JaxAirfoil",
        buffer_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "JaxAirfoil":
        """
        Create a JaxAirfoil from separate upper and lower surface coordinates.

        Args:
            upper: Upper surface coordinates (2, n_upper) where first row is x, second is y
            lower: Lower surface coordinates (2, n_lower) where first row is x, second is y
            name: Name of the airfoil
            buffer_size: Optional buffer size to use (auto-determined if None)
            metadata: Optional additional metadata

        Returns:
            JaxAirfoil instance
        """
        # Convert to JAX arrays if needed
        if isinstance(upper, np.ndarray):
            upper = jnp.array(upper)
        if isinstance(lower, np.ndarray):
            lower = jnp.array(lower)

        # Preprocess coordinates
        upper_processed, lower_processed = CoordinateProcessor.preprocess_coordinates(
            upper,
            lower,
        )

        # Convert to selig format
        selig_coords = CoordinateProcessor.to_selig_format(
            upper_processed,
            lower_processed,
        )

        # Create airfoil instance
        return cls(selig_coords, name=name, buffer_size=buffer_size, metadata=metadata)

    def tree_flatten(self) -> Tuple[Tuple[Array, ...], Dict[str, Any]]:
        """
        Flatten the JaxAirfoil instance for JAX pytree registration.

        This method is required for JAX pytree registration and enables the class
        to work with JAX transformations like jit, grad, and vmap.

        Returns:
            Tuple of (children, aux_data) where:
                - children: Tuple of arrays that are differentiable
                - aux_data: Dictionary of auxiliary data that is not differentiable
        """
        # Children are the arrays that should be differentiable
        children = (self._coordinates,)

        # Auxiliary data is everything else (not differentiable)
        aux_data = {
            "validity_mask": self._validity_mask,
            "n_valid_points": self._n_valid_points,
            "upper_split_idx": self._upper_split_idx,
            "max_buffer_size": self._max_buffer_size,
            "metadata": self._metadata,
        }

        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Dict[str, Any],
        children: Tuple[Array, ...],
    ) -> "JaxAirfoil":
        """
        Unflatten a JaxAirfoil instance from flattened data.

        This method is required for JAX pytree registration and enables the class
        to work with JAX transformations like jit, grad, and vmap.

        Args:
            aux_data: Dictionary of auxiliary data (not differentiable)
            children: Tuple of arrays that are differentiable

        Returns:
            Reconstructed JaxAirfoil instance
        """
        # Create a new instance
        airfoil = cls.__new__(cls)

        # Restore differentiable arrays
        airfoil._coordinates = children[0]

        # Restore auxiliary data
        airfoil._validity_mask = aux_data["validity_mask"]
        airfoil._n_valid_points = aux_data["n_valid_points"]
        airfoil._upper_split_idx = aux_data["upper_split_idx"]
        airfoil._max_buffer_size = aux_data["max_buffer_size"]
        airfoil._metadata = aux_data["metadata"]

        return airfoil

    @property
    def name(self) -> str:
        """Get the airfoil name."""
        return self._metadata.get("name", "JaxAirfoil")

    @property
    def n_points(self) -> int:
        """Get the number of valid points in the airfoil."""
        return self._n_valid_points

    @property
    def buffer_size(self) -> int:
        """Get the current buffer size."""
        return self._max_buffer_size

    @property
    def upper_surface_points(self) -> Tuple[Array, Array]:
        """
        Get the upper surface points (x, y) of the airfoil.

        Returns:
            Tuple of (x_upper, y_upper) arrays
        """
        # Use static slicing based on the stored split index
        # Upper surface: first _upper_split_idx points (already in TE to LE order in selig format)
        # Keep them in TE to LE order (decreasing x) as expected by the test
        upper_coords = self._coordinates[:, : self._upper_split_idx]

        return upper_coords[0, :], upper_coords[1, :]

    @property
    def lower_surface_points(self) -> Tuple[Array, Array]:
        """
        Get the lower surface points (x, y) of the airfoil.

        Returns:
            Tuple of (x_lower, y_lower) arrays
        """
        # Use static slicing based on the stored split index
        # Lower surface: from _upper_split_idx to n_valid_points
        lower_coords = self._coordinates[
            :,
            self._upper_split_idx : self._n_valid_points,
        ]

        return lower_coords[0, :], lower_coords[1, :]

    def get_upper_surface_points(
        self,
        n_points: Optional[int] = None,
        distribution: str = "cosine",
    ) -> Tuple[Array, Array]:
        """
        Get the upper surface points (x, y) of the airfoil, optionally resampled.

        Args:
            n_points: Optional number of points to resample to. If None, returns original points.
            distribution: Point distribution method ("cosine", "uniform", or "original")

        Returns:
            Tuple of (x_upper, y_upper) arrays
        """
        if n_points is None:
            return self.upper_surface_points

        # Get original upper surface coordinates
        x_upper_orig, y_upper_orig = self.upper_surface_points

        # Generate new x-coordinates based on distribution
        if distribution == "cosine":
            # Cosine spacing provides better resolution near leading/trailing edges
            beta = jnp.linspace(0, jnp.pi, n_points)
            x_new = 0.5 * (1 - jnp.cos(beta))
        elif distribution == "uniform":
            x_new = jnp.linspace(0, 1, n_points)
        else:  # "original" - use original distribution
            if n_points >= len(x_upper_orig):
                return self.upper_surface_points
            # Subsample original points
            indices = jnp.linspace(0, len(x_upper_orig) - 1, n_points).astype(int)
            return x_upper_orig[indices], y_upper_orig[indices]

        # Scale x_new to actual coordinate range
        x_min = jnp.min(x_upper_orig)
        x_max = jnp.max(x_upper_orig)
        x_new_scaled = x_min + (x_max - x_min) * x_new

        # Interpolate y-coordinates at new x positions
        y_new = self.y_upper(x_new_scaled)

        return x_new_scaled, y_new

    def get_lower_surface_points(
        self,
        n_points: Optional[int] = None,
        distribution: str = "cosine",
    ) -> Tuple[Array, Array]:
        """
        Get the lower surface points (x, y) of the airfoil, optionally resampled.

        Args:
            n_points: Optional number of points to resample to. If None, returns original points.
            distribution: Point distribution method ("cosine", "uniform", or "original")

        Returns:
            Tuple of (x_lower, y_lower) arrays
        """
        if n_points is None:
            return self.lower_surface_points

        # Get original lower surface coordinates
        x_lower_orig, y_lower_orig = self.lower_surface_points

        # Generate new x-coordinates based on distribution
        if distribution == "cosine":
            # Cosine spacing provides better resolution near leading/trailing edges
            beta = jnp.linspace(0, jnp.pi, n_points)
            x_new = 0.5 * (1 - jnp.cos(beta))
        elif distribution == "uniform":
            x_new = jnp.linspace(0, 1, n_points)
        else:  # "original" - use original distribution
            if n_points >= len(x_lower_orig):
                return self.lower_surface_points
            # Subsample original points
            indices = jnp.linspace(0, len(x_lower_orig) - 1, n_points).astype(int)
            return x_lower_orig[indices], y_lower_orig[indices]

        # Scale x_new to actual coordinate range
        x_min = jnp.min(x_lower_orig)
        x_max = jnp.max(x_lower_orig)
        x_new_scaled = x_min + (x_max - x_min) * x_new

        # Interpolate y-coordinates at new x positions
        y_new = self.y_lower(x_new_scaled)

        return x_new_scaled, y_new

    def resample_airfoil(
        self,
        n_points: int,
        distribution: str = "cosine",
        method: str = "interpolation",
    ) -> "JaxAirfoil":
        """
        Create a new airfoil with resampled surface points.

        Args:
            n_points: Total number of points for the resampled airfoil
            distribution: Point distribution method ("cosine", "uniform", "arc_length")
            method: Resampling method ("interpolation" or "adaptive")

        Returns:
            New JaxAirfoil instance with resampled points
        """
        # Use the get_*_surface_points methods which handle resampling correctly
        n_per_surface = n_points // 2

        # Get resampled surface points using the existing methods
        x_upper_new, y_upper_new = self.get_upper_surface_points(
            n_points=n_per_surface,
            distribution=distribution,
        )
        x_lower_new, y_lower_new = self.get_lower_surface_points(
            n_points=n_per_surface,
            distribution=distribution,
        )

        # Stack into coordinate arrays
        upper_resampled = jnp.stack([x_upper_new, y_upper_new])
        lower_resampled = jnp.stack([x_lower_new, y_lower_new])

        # Create new airfoil from resampled surfaces
        return self.from_upper_lower(
            upper_resampled,
            lower_resampled,
            name=f"{self.name}_resampled_{n_points}pts",
            metadata=self._metadata.copy(),
        )

    def get_surface_points_with_spacing(
        self,
        spacing_type: str = "uniform",
        n_points: Optional[int] = None,
    ) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
        """
        Get surface points with specific spacing characteristics.

        Args:
            spacing_type: Type of spacing ("uniform", "cosine", "arc_length", "adaptive")
            n_points: Number of points per surface (if None, uses current count)

        Returns:
            Tuple of ((x_upper, y_upper), (x_lower, y_lower))
        """
        if n_points is None:
            n_upper = self._upper_split_idx
            n_lower = self._n_valid_points - self._upper_split_idx
        else:
            n_upper = n_lower = n_points

        # Get resampled points
        x_upper, y_upper = self.get_upper_surface_points(n_upper, spacing_type)
        x_lower, y_lower = self.get_lower_surface_points(n_lower, spacing_type)

        return (x_upper, y_upper), (x_lower, y_lower)

    def get_coordinates(self) -> Tuple[Array, Array]:
        """
        Get all airfoil coordinates (x, y) in selig format.

        Returns:
            Tuple of (x_coords, y_coords) arrays
        """
        # Extract valid coordinates
        valid_coords = AirfoilBufferManager.extract_valid_data(
            self._coordinates,
            self._validity_mask,
        )

        return valid_coords[0, :], valid_coords[1, :]

    def thickness(self, query_x: Union[Array, np.ndarray, float]) -> Array:
        """
        Compute airfoil thickness at given x-coordinates.

        Args:
            query_x: X coordinates to query thickness at

        Returns:
            Thickness values at query points
        """
        # Convert to JAX array if needed
        if isinstance(query_x, (float, int)):
            query_x = jnp.array([query_x])
        elif isinstance(query_x, np.ndarray):
            query_x = jnp.array(query_x)

        # Get upper and lower surface coordinates
        upper_coords = self._coordinates[:, : self._upper_split_idx]
        lower_coords = self._coordinates[
            :,
            self._upper_split_idx : self._n_valid_points,
        ]

        # Pad lower coordinates to match buffer size for JIT compatibility
        lower_padded = jnp.concatenate(
            [
                lower_coords,
                jnp.full((2, self._max_buffer_size - lower_coords.shape[1]), jnp.nan),
            ],
            axis=1,
        )

        return JaxAirfoilOps.compute_thickness(
            upper_coords,
            lower_padded,
            self._upper_split_idx,
            self._n_valid_points - self._upper_split_idx,
            query_x,
        )

    def camber_line(self, query_x: Union[Array, np.ndarray, float]) -> Array:
        """
        Compute airfoil camber line at given x-coordinates.

        Args:
            query_x: X coordinates to query camber line at

        Returns:
            Camber line y-coordinates at query points
        """
        # Convert to JAX array if needed
        if isinstance(query_x, (float, int)):
            query_x = jnp.array([query_x])
        elif isinstance(query_x, np.ndarray):
            query_x = jnp.array(query_x)

        # Get upper and lower surface coordinates
        upper_coords = self._coordinates[:, : self._upper_split_idx]
        lower_coords = self._coordinates[
            :,
            self._upper_split_idx : self._n_valid_points,
        ]

        # Pad lower coordinates to match buffer size for JIT compatibility
        lower_padded = jnp.concatenate(
            [
                lower_coords,
                jnp.full((2, self._max_buffer_size - lower_coords.shape[1]), jnp.nan),
            ],
            axis=1,
        )

        return JaxAirfoilOps.compute_camber_line(
            upper_coords,
            lower_padded,
            self._upper_split_idx,
            self._n_valid_points - self._upper_split_idx,
            query_x,
        )

    def y_upper(self, query_x: Union[Array, np.ndarray, float]) -> Array:
        """
        Query upper surface y-coordinates at given x-coordinates.

        Args:
            query_x: X coordinates to query

        Returns:
            Upper surface y-coordinates at query points
        """
        # Convert to JAX array if needed
        if isinstance(query_x, (float, int)):
            query_x = jnp.array([query_x])
        elif isinstance(query_x, np.ndarray):
            query_x = jnp.array(query_x)

        # Get upper surface coordinates
        upper_coords = self._coordinates[:, : self._upper_split_idx]

        return JaxAirfoilOps.y_upper(upper_coords, self._upper_split_idx, query_x)

    def y_lower(self, query_x: Union[Array, np.ndarray, float]) -> Array:
        """
        Query lower surface y-coordinates at given x-coordinates.

        Args:
            query_x: X coordinates to query

        Returns:
            Lower surface y-coordinates at query points
        """
        # Convert to JAX array if needed
        if isinstance(query_x, (float, int)):
            query_x = jnp.array([query_x])
        elif isinstance(query_x, np.ndarray):
            query_x = jnp.array(query_x)

        # Get lower surface coordinates
        lower_coords = self._coordinates[
            :,
            self._upper_split_idx : self._n_valid_points,
        ]

        # Pad lower coordinates to match buffer size for JIT compatibility
        lower_padded = jnp.concatenate(
            [
                lower_coords,
                jnp.full((2, self._max_buffer_size - lower_coords.shape[1]), jnp.nan),
            ],
            axis=1,
        )

        return JaxAirfoilOps.y_lower(
            lower_padded,
            self._n_valid_points - self._upper_split_idx,
            query_x,
        )

    @property
    def max_thickness(self) -> float:
        """
        Get the maximum thickness of the airfoil.

        Returns:
            Maximum thickness value
        """
        # Get upper and lower surface coordinates
        upper_coords = self._coordinates[:, : self._upper_split_idx]
        lower_coords = self._coordinates[
            :,
            self._upper_split_idx : self._n_valid_points,
        ]

        # Pad lower coordinates to match buffer size for JIT compatibility
        lower_padded = jnp.concatenate(
            [
                lower_coords,
                jnp.full((2, self._max_buffer_size - lower_coords.shape[1]), jnp.nan),
            ],
            axis=1,
        )

        max_thickness, _ = JaxAirfoilOps.compute_max_thickness(
            upper_coords,
            lower_padded,
            self._upper_split_idx,
            self._n_valid_points - self._upper_split_idx,
        )

        return float(max_thickness)

    @property
    def max_thickness_location(self) -> float:
        """
        Get the x-coordinate location of maximum thickness.

        Returns:
            X-coordinate of maximum thickness
        """
        # Get upper and lower surface coordinates
        upper_coords = self._coordinates[:, : self._upper_split_idx]
        lower_coords = self._coordinates[
            :,
            self._upper_split_idx : self._n_valid_points,
        ]

        # Pad lower coordinates to match buffer size for JIT compatibility
        lower_padded = jnp.concatenate(
            [
                lower_coords,
                jnp.full((2, self._max_buffer_size - lower_coords.shape[1]), jnp.nan),
            ],
            axis=1,
        )

        _, max_thickness_location = JaxAirfoilOps.compute_max_thickness(
            upper_coords,
            lower_padded,
            self._upper_split_idx,
            self._n_valid_points - self._upper_split_idx,
        )

        return float(max_thickness_location)

    @property
    def max_camber(self) -> float:
        """
        Get the maximum camber of the airfoil.

        Returns:
            Maximum camber value
        """
        # Get upper and lower surface coordinates
        upper_coords = self._coordinates[:, : self._upper_split_idx]
        lower_coords = self._coordinates[
            :,
            self._upper_split_idx : self._n_valid_points,
        ]

        # Pad lower coordinates to match buffer size for JIT compatibility
        lower_padded = jnp.concatenate(
            [
                lower_coords,
                jnp.full((2, self._max_buffer_size - lower_coords.shape[1]), jnp.nan),
            ],
            axis=1,
        )

        max_camber, _ = JaxAirfoilOps.compute_max_camber(
            upper_coords,
            lower_padded,
            self._upper_split_idx,
            self._n_valid_points - self._upper_split_idx,
        )

        return float(max_camber)

    @property
    def max_camber_location(self) -> float:
        """
        Get the x-coordinate location of maximum camber.

        Returns:
            X-coordinate of maximum camber
        """
        # Get upper and lower surface coordinates
        upper_coords = self._coordinates[:, : self._upper_split_idx]
        lower_coords = self._coordinates[
            :,
            self._upper_split_idx : self._n_valid_points,
        ]

        # Pad lower coordinates to match buffer size for JIT compatibility
        lower_padded = jnp.concatenate(
            [
                lower_coords,
                jnp.full((2, self._max_buffer_size - lower_coords.shape[1]), jnp.nan),
            ],
            axis=1,
        )

        _, max_camber_location = JaxAirfoilOps.compute_max_camber(
            upper_coords,
            lower_padded,
            self._upper_split_idx,
            self._n_valid_points - self._upper_split_idx,
        )

        return float(max_camber_location)

    @property
    def chord_length(self) -> float:
        """
        Get the chord length of the airfoil.

        Returns:
            Chord length (distance from leading edge to trailing edge)
        """
        # Get upper and lower surface coordinates
        upper_coords = self._coordinates[:, : self._upper_split_idx]
        lower_coords = self._coordinates[
            :,
            self._upper_split_idx : self._n_valid_points,
        ]

        # Pad lower coordinates to match buffer size for JIT compatibility
        lower_padded = jnp.concatenate(
            [
                lower_coords,
                jnp.full((2, self._max_buffer_size - lower_coords.shape[1]), jnp.nan),
            ],
            axis=1,
        )

        chord_length = JaxAirfoilOps.compute_chord_length(
            upper_coords,
            lower_padded,
            self._upper_split_idx,
            self._n_valid_points - self._upper_split_idx,
        )

        return float(chord_length)

    @classmethod
    def naca4(
        cls,
        digits: str,
        n_points: int = 200,
        buffer_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "JaxAirfoil":
        """
        Create a NACA 4-digit airfoil.

        Args:
            digits: NACA 4-digit designation (e.g., "2412")
            n_points: Number of points for each surface
            buffer_size: Optional buffer size to use (auto-determined if None)
            metadata: Optional additional metadata

        Returns:
            JaxAirfoil instance with NACA 4-digit airfoil

        Raises:
            ValueError: If digits are invalid
        """
        # Validate input using comprehensive error handling
        try:
            AirfoilErrorHandler.validate_naca_parameters(digits, "4-digit")
        except AirfoilValidationError as e:
            raise AirfoilValidationError(
                f"Invalid NACA 4-digit parameters: {e}. "
                f"{AirfoilErrorHandler.suggest_fixes('naca_invalid')}",
            )

        # Parse NACA parameters
        M = int(digits[0]) / 100.0  # Maximum camber
        P = int(digits[1]) / 10.0  # Position of maximum camber
        XX = int(digits[2:4]) / 100.0  # Maximum thickness

        # Generate coordinates using JAX operations
        upper_coords, lower_coords = JaxAirfoilOps.generate_naca4_coordinates(
            M,
            P,
            XX,
            n_points,
        )

        # Create airfoil name
        name = f"NACA{digits}"

        # Create airfoil instance
        return cls.from_upper_lower(
            upper_coords,
            lower_coords,
            name=name,
            buffer_size=buffer_size,
            metadata=metadata,
        )

    @classmethod
    def naca5(
        cls,
        digits: str,
        n_points: int = 200,
        buffer_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "JaxAirfoil":
        """
        Create a NACA 5-digit airfoil.

        Args:
            digits: NACA 5-digit designation (e.g., "23012")
            n_points: Number of points for each surface
            buffer_size: Optional buffer size to use (auto-determined if None)
            metadata: Optional additional metadata

        Returns:
            JaxAirfoil instance with NACA 5-digit airfoil

        Raises:
            ValueError: If digits are invalid
        """
        # Validate input using comprehensive error handling
        try:
            AirfoilErrorHandler.validate_naca_parameters(digits, "5-digit")
        except AirfoilValidationError as e:
            raise AirfoilValidationError(
                f"Invalid NACA 5-digit parameters: {e}. "
                f"{AirfoilErrorHandler.suggest_fixes('naca_invalid')}",
            )

        # Parse NACA parameters
        L = int(digits[0])  # Design coefficient of lift parameter
        P = int(digits[1])  # Position of maximum camber parameter
        Q = int(digits[2])  # Reflex parameter (0=standard, 1=reflex)
        XX = int(digits[3:5])  # Maximum thickness

        # Convert to physical parameters
        design_cl = L * (3.0 / 2.0) / 10.0  # Design coefficient of lift
        max_camber_pos = 0.5 * P / 10.0  # Position of maximum camber
        reflex = Q == 1  # Reflex camber line flag
        thickness = XX / 100.0  # Maximum thickness

        # Generate coordinates using JAX operations
        upper_coords, lower_coords = JaxAirfoilOps.generate_naca5_coordinates(
            design_cl,
            max_camber_pos,
            reflex,
            thickness,
            n_points,
        )

        # Create airfoil name
        name = f"NACA{digits}"

        # Create airfoil instance
        return cls.from_upper_lower(
            upper_coords,
            lower_coords,
            name=name,
            buffer_size=buffer_size,
            metadata=metadata,
        )

    @classmethod
    def naca(
        cls,
        designation: str,
        n_points: int = 200,
        buffer_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "JaxAirfoil":
        """
        Create a NACA airfoil from a designation string.

        This method automatically detects whether the designation is for a 4-digit
        or 5-digit NACA airfoil and calls the appropriate generation method.

        Args:
            designation: NACA designation string (e.g., "2412", "23012")
            n_points: Number of points for each surface
            buffer_size: Optional buffer size to use (auto-determined if None)
            metadata: Optional additional metadata

        Returns:
            JaxAirfoil instance with the specified NACA airfoil

        Raises:
            ValueError: If designation is invalid or unsupported
        """
        # Clean the designation string
        if isinstance(designation, str):
            # Remove "NACA" prefix if present and keep only digits
            clean_digits = "".join(filter(str.isdigit, designation.upper()))
        else:
            raise ValueError("NACA designation must be a string")

        # Determine airfoil type based on number of digits
        if len(clean_digits) == 4:
            return cls.naca4(clean_digits, n_points, buffer_size, metadata)
        elif len(clean_digits) == 5:
            return cls.naca5(clean_digits, n_points, buffer_size, metadata)
        else:
            raise ValueError(
                f"Unsupported NACA designation: {designation}. "
                "Only 4-digit and 5-digit NACA airfoils are supported.",
            )

    def flap(
        self,
        flap_hinge_chord_percentage: float,
        flap_angle: float,
        flap_hinge_thickness_percentage: float = 0.5,
        chord_extension: float = 1.0,
    ) -> "JaxAirfoil":
        """
        Apply flap transformation to the airfoil.

        This method creates a new JaxAirfoil with a flap applied at the specified
        hinge location and angle. The flap transformation includes rotation and
        optional chord extension.

        Args:
            flap_hinge_chord_percentage: Chordwise location of the flap hinge (0.0 to 1.0)
            flap_angle: Flap deflection angle in degrees (positive = downward)
            flap_hinge_thickness_percentage: Position of hinge through thickness
                                           (0.0 = lower surface, 1.0 = upper surface)
            chord_extension: Scaling factor for flap chord length (1.0 = no extension)

        Returns:
            New JaxAirfoil instance with flap applied

        Raises:
            ValueError: If parameters are out of valid ranges
        """
        # Validate input parameters using comprehensive error handling
        try:
            AirfoilErrorHandler.validate_flap_parameters(
                flap_hinge_chord_percentage,
                flap_angle,
                flap_hinge_thickness_percentage,
                chord_extension,
            )
        except AirfoilValidationError as e:
            raise AirfoilValidationError(
                f"Invalid flap parameters for airfoil '{self.name}': {e}. "
                f"{AirfoilErrorHandler.suggest_fixes('flap_invalid')}",
            )

        # Check if we should skip flap operation
        should_skip_flap = (flap_angle == 0.0) | (flap_hinge_chord_percentage >= 1.0)

        # Early return for cases where no transformation is needed
        if should_skip_flap:
            return self

        # Convert angle to radians
        flap_angle_rad = jnp.deg2rad(flap_angle)

        # Determine flap hinge x-coordinate in actual coordinates
        min_x = jnp.min(self._coordinates[0, : self._n_valid_points])
        max_x = jnp.max(self._coordinates[0, : self._n_valid_points])
        flap_hinge_x = min_x + flap_hinge_chord_percentage * (max_x - min_x)

        # Get upper and lower surface coordinates
        upper_coords = self._coordinates[:, : self._upper_split_idx]
        lower_coords = self._coordinates[
            :,
            self._upper_split_idx : self._n_valid_points,
        ]

        # Pad lower coordinates to match buffer size for JIT compatibility
        lower_padded = jnp.concatenate(
            [
                lower_coords,
                jnp.full((2, self._max_buffer_size - lower_coords.shape[1]), jnp.nan),
            ],
            axis=1,
        )

        # Apply flap transformation
        (
            new_upper_coords,
            new_lower_coords,
            new_n_upper_valid,
            new_n_lower_valid,
        ) = JaxAirfoilOps.apply_flap_transformation(
            upper_coords,
            lower_padded,
            self._upper_split_idx,
            self._n_valid_points - self._upper_split_idx,
            flap_hinge_x,
            flap_angle_rad,
            flap_hinge_thickness_percentage,
            chord_extension,
            should_skip_flap,
        )

        # Convert back to selig format
        # Extract valid coordinates using dynamic slicing for JAX compatibility
        upper_valid = jax.lax.dynamic_slice(
            new_upper_coords,
            (0, 0),
            (2, new_n_upper_valid),
        )
        lower_valid = jax.lax.dynamic_slice(
            new_lower_coords,
            (0, 0),
            (2, new_n_lower_valid),
        )

        # Convert to selig format
        new_selig_coords = CoordinateProcessor.to_selig_format(upper_valid, lower_valid)

        # Create new airfoil instance
        new_name = f"{self.name}_flapped_hinge_{flap_hinge_chord_percentage:.2f}_deflection_{jnp.rad2deg(flap_angle):.2f}"
        return JaxAirfoil(
            new_selig_coords,
            name=new_name,
            buffer_size=self._max_buffer_size,
            metadata=self._metadata.copy(),
        )

    # ============================================================================
    # API COMPATIBILITY LAYER
    # ============================================================================

    # Property getters/setters for backward compatibility
    @property
    def upper_surface(self) -> Array:
        """Returns the upper surface coordinates of the airfoil (API compatibility)."""
        x_upper, y_upper = self.upper_surface_points
        return jnp.array([x_upper, y_upper])

    @property
    def lower_surface(self) -> Array:
        """Returns the lower surface coordinates of the airfoil (API compatibility)."""
        x_lower, y_lower = self.lower_surface_points
        return jnp.array([x_lower, y_lower])

    @property
    def file_name(self) -> str:
        """Returns the file name of the airfoil (API compatibility)."""
        if self.name is not None and self.name.strip():
            return f"{self.name}.dat"
        return "Airfoil.dat"

    @name.setter
    def name(self, value: str) -> None:
        """Sets the name of the airfoil (API compatibility)."""
        if not isinstance(value, str):
            raise TypeError("Name must be a string")
        self._metadata["name"] = value.replace(" ", "")

    # Wrapper methods to maintain existing API
    def to_selig(self) -> Array:
        """
        Returns the airfoil in the selig format (API compatibility).

        Selig format runs from the trailing edge, around the leading edge,
        back to the trailing edge.

        Returns:
            JAX array of shape (2, n_points) with x and y coordinates
        """
        # Extract valid coordinates
        valid_coords = AirfoilBufferManager.extract_valid_data(
            self._coordinates,
            self._validity_mask,
        )
        return valid_coords

    def repanel_from_internal(
        self,
        n_points: int,
        distribution: str = "cosine",
    ) -> None:
        """
        Repanels the airfoil to have n_points (API compatibility - modifies in place).

        Note: This method modifies the airfoil in place for API compatibility,
        but creates a new instance internally due to JAX immutability.

        Args:
            n_points: Number of points to generate
            distribution: Distribution type ("cosine", "uniform", or "arc_length")
        """
        # Create repaneled airfoil
        repaneled = self.repanel(n_points, distribution)

        # Update this instance with the repaneled data
        self._coordinates = repaneled._coordinates
        self._validity_mask = repaneled._validity_mask
        self._n_valid_points = repaneled._n_valid_points
        self._upper_split_idx = repaneled._upper_split_idx
        self._max_buffer_size = repaneled._max_buffer_size

    # Conversion utilities between old and new formats
    @classmethod
    def from_legacy_airfoil(cls, legacy_airfoil) -> "JaxAirfoil":
        """
        Create a JaxAirfoil from a legacy Airfoil instance.

        Args:
            legacy_airfoil: Original ICARUS Airfoil instance

        Returns:
            JaxAirfoil instance with equivalent data
        """
        # Get upper and lower surfaces from legacy airfoil
        upper_surface = legacy_airfoil.upper_surface
        lower_surface = legacy_airfoil.lower_surface

        return cls.from_upper_lower(
            upper_surface,
            lower_surface,
            name=legacy_airfoil.name,
            metadata={"source": "legacy_conversion"},
        )

    def to_legacy_format(self) -> Dict[str, Any]:
        """
        Convert JaxAirfoil to legacy format data.

        Returns:
            Dictionary with data compatible with legacy Airfoil class
        """
        x_upper, y_upper = self.upper_surface_points
        x_lower, y_lower = self.lower_surface_points

        return {
            "upper": jnp.array([x_upper, y_upper]),
            "lower": jnp.array([x_lower, y_lower]),
            "name": self.name,
            "n_points": self.n_points,
        }

    # File I/O methods
    @classmethod
    def from_file(cls, filename: str) -> "JaxAirfoil":
        """
        Initialize the JaxAirfoil class from a file (API compatibility).

        Args:
            filename: Name of the file to load the airfoil from

        Returns:
            JaxAirfoil instance loaded from file

        Raises:
            ValueError: If file cannot be parsed or contains invalid data
            FileNotFoundError: If file does not exist
        """
        import logging
        import os

        x = []
        y = []

        logging.info(f"Loading airfoil from {filename}")

        try:
            with open(filename) as file:
                for line in file:
                    line = line.strip()

                    if line == "\n" or line == "":
                        continue

                    # Check if it contains two numbers
                    if len(line.split()) != 2:
                        continue

                    try:
                        x_i = float(line.split()[0])
                        y_i = float(line.split()[1])
                        if jnp.abs(x_i) > 2.0 or jnp.abs(y_i) > 2.0:
                            continue
                        x.append(x_i)
                        y.append(y_i)
                    except (ValueError, IndexError):
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")

        if len(x) == 0 or len(y) == 0:
            raise ValueError(f"No valid coordinate data found in {filename}")

        # Convert to JAX arrays
        x_arr = jnp.array(x)
        y_arr = jnp.array(y)

        # Split into upper and lower surfaces using the same logic as original
        lower, upper = cls._split_sides(x_arr, y_arr)

        try:
            airfoil_name = os.path.split(filename)[-1].replace(".dat", "")
            return cls.from_upper_lower(upper, lower, name=airfoil_name)
        except ValueError as e:
            raise ValueError(f"Error loading airfoil from {filename}: {e}")

    @staticmethod
    def _split_sides(x: Array, y: Array) -> Tuple[Array, Array]:
        """
        Split airfoil coordinates into upper and lower surfaces (API compatibility).

        Args:
            x: X coordinates
            y: Y coordinates

        Returns:
            Tuple of (lower_surface, upper_surface) arrays
        """
        # Remove duplicate points from the array
        x_arr = jnp.asarray(x)
        y_arr = jnp.asarray(y)

        # Combine x and y coordinates into a single array of complex numbers
        complex_coords = x_arr + 1j * y_arr
        # Find unique complex coordinates
        unique_indices = jnp.sort(jnp.unique(complex_coords, return_index=True)[1])

        # Use the unique indices to get the unique x and y coordinates
        x_clean = x_arr[unique_indices]
        y_clean = y_arr[unique_indices]

        # Locate the leading edge (minimum x value)
        idxs = jnp.where(x_arr == jnp.min(x_arr))[0].flatten()
        if len(idxs) == 0:
            # Find where the x_arr is closest to minimum
            idx = jnp.argmin(x_arr)
        elif len(idxs) == 1:
            idx = idxs[0] + 1
        else:
            idx = idxs[-1]

        # Calibrate idx to account for removed duplicates
        idx_int = int(jnp.sum(unique_indices < idx))

        lower = jnp.array([x_clean[idx_int:], y_clean[idx_int:]])
        upper = jnp.array([x_clean[:idx_int], y_clean[:idx_int]])

        return lower, upper

    def save_selig(
        self,
        directory: Optional[str] = None,
        header: bool = False,
        inverse: bool = False,
    ) -> None:
        """
        Saves the airfoil in the selig format (API compatibility).

        Args:
            directory: Directory to save the airfoil. Defaults to None.
            header: Whether to include the header. Defaults to False.
            inverse: Whether to save the airfoil in the reverse selig format. Defaults to False.
        """
        import os

        if directory is not None:
            # If directory does not exist, create it
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_name = os.path.join(directory, self.file_name)
        else:
            file_name = self.file_name

        with open(file_name, "w") as file:
            if header:
                file.write(f"{self.name} with {self.n_points}\n")

            pts = self.to_selig()
            if inverse:
                pts = pts[:, ::-1]

            x = pts[0, :]
            y = pts[1, :]

            # Remove NaN values and duplicates
            x_arr = jnp.array(x)
            y_arr = jnp.array(y)

            # Combine x and y coordinates into a single array of complex numbers
            complex_coords = x_arr + 1j * y_arr
            # Round all the values to 6 decimals
            complex_coords = jnp.round(complex_coords, 5)

            # Find unique complex coordinates
            unique_indices = jnp.sort(jnp.unique(complex_coords, return_index=True)[1])

            # Use the unique indices to get the unique x and y coordinates
            x_clean = x_arr[unique_indices]
            y_clean = y_arr[unique_indices]

            # Remove NaN values
            idx_nan = jnp.isnan(x_clean) | jnp.isnan(y_clean)
            x_clean = x_clean[~idx_nan]
            y_clean = y_clean[~idx_nan]

            for x_val, y_val in zip(x_clean, y_clean):
                file.write(f"{x_val:.6f} {y_val:.6f}\n")

    def save_le(
        self,
        directory: Optional[str] = None,
        header: bool = False,
    ) -> None:
        """
        Saves the airfoil in the reverse selig format (API compatibility).

        Args:
            directory: Directory to save the airfoil. Defaults to None.
            header: Whether to include the header. Defaults to False.
        """
        import os

        # Get coordinates in reverse order (LE format)
        x_upper, y_upper = self.upper_surface_points
        x_lower, y_lower = self.lower_surface_points

        x = jnp.hstack((x_lower, x_upper[::-1]))
        y = jnp.hstack((y_lower, y_upper[::-1]))

        pts = jnp.vstack((x, y))

        if directory is not None:
            # If directory does not exist, create it
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_name = os.path.join(directory, self.file_name)
        else:
            file_name = self.file_name

        with open(file_name, "w") as file:
            if header:
                file.write(f"{self.name} with {self.n_points}\n")
            for x_val, y_val in pts.T:
                file.write(f" {x_val:.6f} {y_val:.6f}\n")

    # Additional API compatibility methods
    @property
    def selig_original(self) -> Array:
        """
        Returns the original airfoil coordinates in selig format (API compatibility).

        This property provides access to the airfoil coordinates in the same format
        as they were originally loaded or created, maintaining compatibility with
        the original Airfoil class.

        Returns:
            JAX array of shape (2, n_points) with x and y coordinates in selig format
        """
        return self.to_selig()

    def repanel_spl(self, n_points: int = 200) -> None:
        """
        Repanel the airfoil using spline-based interpolation (API compatibility).

        This method provides spline-based repaneling functionality similar to the
        original implementation, but uses JAX-compatible operations instead of scipy.

        Args:
            n_points: Number of points for the repaneled airfoil

        Note:
            This method modifies the airfoil in place for API compatibility.
            The JAX implementation uses linear interpolation instead of cubic splines
            to maintain JIT compatibility, but provides similar results.
        """
        # Use the existing repanel method which handles coordinate validation properly
        repaneled = self.repanel(n_points, distribution="cosine")

        # Update this instance with the repaneled data (for in-place compatibility)
        self._coordinates = repaneled._coordinates
        self._validity_mask = repaneled._validity_mask
        self._n_valid_points = repaneled._n_valid_points
        self._upper_split_idx = repaneled._upper_split_idx
        self._max_buffer_size = repaneled._max_buffer_size

    @classmethod
    def load_from_web(cls, name: str) -> "JaxAirfoil":
        """
        Fetch airfoil data from the UIUC airfoil database (API compatibility).

        This method downloads airfoil coordinate data from the University of Illinois
        at Urbana-Champaign airfoil database and creates a JaxAirfoil instance.

        Args:
            name: Name of the airfoil to fetch (case-insensitive)

        Returns:
            JaxAirfoil instance with data from the web database

        Raises:
            FileNotFoundError: If the airfoil cannot be found or downloaded
            requests.exceptions.RequestException: If there are network issues
        """
        db_url = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
        base_url = "https://m-selig.ae.illinois.edu/ads/"

        try:
            response = requests.get(db_url, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise FileNotFoundError(f"Error fetching airfoil database: {e}")

        # Find all lines containing .dat filenames
        lines = response.text.split("\n")
        filenames = []
        for line in lines:
            match = re.search(r'href="(.*?)\.dat"', line)
            if match:
                filenames.append(f"{match.group(1)}.dat")

        # Search for the requested airfoil
        for filename in filenames:
            # Get the airfoil name from the filename
            airfoil_name = filename.split(".")[0].split("/")[-1]
            if airfoil_name.upper() != name.upper():
                continue

            # Found matching airfoil, download it
            download_url = base_url + filename

            try:
                response = requests.get(download_url, timeout=30)
                response.raise_for_status()

                # Parse the downloaded data
                lines = response.text.strip().split("\n")
                x_coords = []
                y_coords = []

                for line in lines:
                    line = line.strip()
                    if not line or len(line.split()) != 2:
                        continue

                    try:
                        x_val = float(line.split()[0])
                        y_val = float(line.split()[1])

                        # Basic validation
                        if abs(x_val) > 2.0 or abs(y_val) > 2.0:
                            continue

                        x_coords.append(x_val)
                        y_coords.append(y_val)
                    except (ValueError, IndexError):
                        continue

                if not x_coords or not y_coords:
                    raise ValueError(
                        "No valid coordinate data found in downloaded file",
                    )

                # Convert to JAX arrays and split into surfaces
                x_arr = jnp.array(x_coords)
                y_arr = jnp.array(y_coords)
                lower, upper = cls._split_sides(x_arr, y_arr)

                # Save to local database if possible
                try:
                    from ICARUS.database import Database

                    DB = Database.get_instance()
                    DB2D = DB.DB2D

                    dirname = airfoil_name.upper()
                    os.makedirs(os.path.join(DB2D, dirname), exist_ok=True)
                    local_filename = os.path.join(
                        DB2D,
                        dirname,
                        f"{airfoil_name.lower()}.dat",
                    )

                    with open(local_filename, "w") as f:
                        for x_val, y_val in zip(x_coords, y_coords):
                            f.write(f"{x_val:.6f} {y_val:.6f}\n")

                    print(f"Downloaded and saved: {local_filename}")
                except Exception as e:
                    print(f"Warning: Could not save to local database: {e}")

                # Create and return the airfoil
                return cls.from_upper_lower(
                    upper,
                    lower,
                    name=airfoil_name.upper(),
                    metadata={"source": "UIUC_database", "url": download_url},
                )

            except requests.exceptions.RequestException as e:
                raise FileNotFoundError(f"Error downloading {filename}: {e}")

        # If we get here, the airfoil was not found
        raise FileNotFoundError(f"Airfoil '{name}' not found in UIUC database")

    # Plotting support
    def plot(
        self,
        camber: bool = False,
        scatter: bool = False,
        max_thickness: bool = False,
        ax: Optional[Axes] = None,
        overide_color: Optional[str] = None,
        linewidth: float = 1.5,
        markersize: float = 2.0,
        alpha: float = 1.0,
        show_legend: bool = False,
    ) -> Optional[Axes]:
        """
        Plots the airfoil in the selig format (API compatibility).

        Args:
            camber: Whether to plot the camber line. Defaults to False.
            scatter: Whether to plot the airfoil as a scatter plot. Defaults to False.
            max_thickness: Whether to plot the max thickness. Defaults to False.
            ax: Matplotlib axes object. If None, creates new figure.
            overide_color: Override color for the plot.
            linewidth: Line width for the plot. Defaults to 1.5.
            markersize: Marker size for scatter plots. Defaults to 2.0.
            alpha: Transparency level. Defaults to 1.0.
            show_legend: Whether to show legend. Defaults to False.

        Returns:
            Matplotlib axes object if ax was None, otherwise None.

        Raises:
            ImportError: If matplotlib is not available.
            ValueError: If airfoil has no valid points.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting functionality")

        # Check if airfoil has valid points
        if self.n_points == 0:
            raise ValueError("Cannot plot airfoil with no valid points")

        pts = self.to_selig()
        x, y = pts

        # Create axes if not provided
        return_ax = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot airfoil surfaces
        if scatter:
            # Scatter plot mode
            scatter_size = markersize * 10  # Convert to scatter size
            ax.scatter(
                x[: self._upper_split_idx],
                y[: self._upper_split_idx],
                s=scatter_size,
                alpha=alpha,
                color=overide_color if overide_color else "red",
                label="Upper Surface" if show_legend else None,
            )
            ax.scatter(
                x[self._upper_split_idx :],
                y[self._upper_split_idx :],
                s=scatter_size,
                alpha=alpha,
                color=overide_color if overide_color else "blue",
                marker="s",
                label="Lower Surface" if show_legend else None,
            )
        else:
            # Line plot mode
            if overide_color is not None:
                col_up = overide_color
                col_lo = overide_color
            else:
                col_up = "red"
                col_lo = "blue"

            ax.plot(
                x[: self._upper_split_idx],
                y[: self._upper_split_idx],
                color=col_up,
                linewidth=linewidth,
                alpha=alpha,
                label="Upper Surface" if show_legend else None,
            )
            ax.plot(
                x[self._upper_split_idx :],
                y[self._upper_split_idx :],
                color=col_lo,
                linewidth=linewidth,
                alpha=alpha,
                linestyle="--" if overide_color is None else "-",
                label="Lower Surface" if show_legend else None,
            )

        # Add camber line if requested
        if camber:
            try:
                x_min = float(jnp.min(x))
                x_max = float(jnp.max(x))
                x_camber = jnp.linspace(x_min, x_max, 100)
                y_camber = self.camber_line(x_camber)
                ax.plot(
                    x_camber,
                    y_camber,
                    "k--",
                    linewidth=linewidth,
                    alpha=alpha * 0.8,
                    label="Camber Line" if show_legend else None,
                )
            except Exception as e:
                print(f"Warning: Could not plot camber line: {e}")

        # Add maximum thickness indicator if requested
        if max_thickness:
            try:
                x_max_thick = self.max_thickness_location
                thick = self.max_thickness
                y_up = float(self.y_upper(jnp.array([x_max_thick]))[0])
                y_lo = float(self.y_lower(jnp.array([x_max_thick]))[0])

                # Plot a line from the upper to the lower surface
                ax.plot(
                    [x_max_thick, x_max_thick],
                    [y_up, y_lo],
                    "k-",
                    linewidth=linewidth * 1.5,
                    alpha=alpha * 0.8,
                    label=f"Max Thickness: {thick:.3f}" if show_legend else None,
                )

                # Add a text with the thickness
                ax.text(
                    x_max_thick,
                    y_lo - 0.02,
                    f"t_max = {thick:.3f}\nat x = {x_max_thick:.3f}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )
            except Exception as e:
                print(f"Warning: Could not plot max thickness: {e}")

        # Configure plot appearance
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.set_title(f"Airfoil: {self.name} ({self.n_points} points)")

        # Add legend if requested
        if show_legend:
            ax.legend()

        # Return axes if we created them
        return ax if return_ax else None

    def __str__(self) -> str:
        """Returns the string representation of the airfoil."""
        return f"JaxAirfoil: {self.name} with {self.n_points} points (buffer: {self.buffer_size})"

    # Morphing compatibility
    @classmethod
    def morph_new_from_two_foils(
        cls,
        airfoil1: "JaxAirfoil",
        airfoil2: "JaxAirfoil",
        eta: float,
        n_points: int,
    ) -> "JaxAirfoil":
        """
        Returns a new airfoil morphed between two airfoils (API compatibility).

        Args:
            airfoil1: First airfoil
            airfoil2: Second airfoil
            eta: Morphing parameter (0.0 = airfoil1, 1.0 = airfoil2)
            n_points: Number of points to generate

        Returns:
            New JaxAirfoil morphed between the two airfoils

        Raises:
            ValueError: If eta is not in range [0,1]
        """
        # Validate morphing parameters using comprehensive error handling
        try:
            AirfoilErrorHandler.validate_morphing_parameters(
                eta,
                airfoil1.name,
                airfoil2.name,
            )
        except AirfoilValidationError as e:
            raise AirfoilValidationError(
                f"Invalid morphing parameters: {e}. "
                f"{AirfoilErrorHandler.suggest_fixes('morphing_invalid')}",
            )

        # Round to 2 decimals
        eta = round(eta, 2)
        if eta == 0.0:
            return airfoil1
        if eta == 1.0:
            return airfoil2

        # Create cosine distribution for morphing
        ksi = jnp.linspace(0, jnp.pi, n_points // 2)
        x = 0.5 * (1.0 - jnp.cos(ksi))

        # Get surface values from both airfoils
        y_upper_af1 = airfoil1.y_upper(x)
        y_lower_af1 = airfoil1.y_lower(x)
        y_upper_af2 = airfoil2.y_upper(x)
        y_lower_af2 = airfoil2.y_lower(x)

        # Morph between the surfaces
        y_upper_new = y_upper_af1 * (1 - eta) + y_upper_af2 * eta
        y_lower_new = y_lower_af1 * (1 - eta) + y_lower_af2 * eta

        # Create new surfaces
        upper = jnp.array([x, y_upper_new])
        lower = jnp.array([x, y_lower_new])

        # Convert to selig format
        morphed = CoordinateProcessor.to_selig_format(upper, lower)

        # Create morphed name
        eta_perc = int(eta * 100)
        name = f"morphed_{airfoil1.name}_{airfoil2.name}_at_{eta_perc}%"

        return cls(
            morphed,
            name=name,
            metadata={
                "morphing_eta": eta,
                "parent1": airfoil1.name,
                "parent2": airfoil2.name,
            },
        )

    def repanel(
        self,
        n_points: int,
        distribution: str = "cosine",
        method: str = "arc_length",
    ) -> "JaxAirfoil":
        """
        Repanel the airfoil with a new point distribution.

        This method redistributes the airfoil points using either uniform or cosine
        spacing, with optional arc-length based parametrization for better point
        distribution along curved surfaces.

        Args:
            n_points: Total number of points for the repaneled airfoil
            distribution: Point distribution type - "cosine" or "uniform"
            method: Repaneling method - "arc_length" or "chord_based"

        Returns:
            New JaxAirfoil instance with repaneled coordinates

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate input parameters using comprehensive error handling
        try:
            AirfoilErrorHandler.validate_repanel_parameters(
                n_points,
                distribution,
                method,
            )
        except AirfoilValidationError as e:
            raise AirfoilValidationError(
                f"Invalid repanel parameters for airfoil '{self.name}': {e}",
            )

        # Get upper and lower surface coordinates
        upper_coords = self._coordinates[:, : self._upper_split_idx]
        lower_coords = self._coordinates[
            :,
            self._upper_split_idx : self._n_valid_points,
        ]

        # Pad both coordinates to match buffer size for JIT compatibility
        upper_padded = jnp.concatenate(
            [
                upper_coords,
                jnp.full((2, self._max_buffer_size - upper_coords.shape[1]), jnp.nan),
            ],
            axis=1,
        )
        lower_padded = jnp.concatenate(
            [
                lower_coords,
                jnp.full((2, self._max_buffer_size - lower_coords.shape[1]), jnp.nan),
            ],
            axis=1,
        )

        # Repanel the airfoil coordinates
        new_upper_coords, new_lower_coords = JaxAirfoilOps.repanel_airfoil_coordinates(
            upper_padded,
            lower_padded,
            self._upper_split_idx,
            self._n_valid_points - self._upper_split_idx,
            n_points,
            distribution,
        )

        # Create new airfoil name
        new_name = f"{self.name}_repaneled_{n_points}pts_{distribution}"

        # Create new metadata
        new_metadata = self._metadata.copy()
        try:
            new_metadata["repanel_n_points"] = n_points
            new_metadata["repanel_distribution"] = distribution
            new_metadata["repanel_method"] = method
            new_metadata["original_n_points"] = self._n_valid_points
        except (TypeError, AttributeError):
            # Skip metadata updates during JAX transformations
            pass

        # Create new JaxAirfoil instance from the repaneled coordinates
        return JaxAirfoil.from_upper_lower(
            new_upper_coords,
            new_lower_coords,
            name=new_name,
            metadata=new_metadata,
        )

    def __repr__(self) -> str:
        """String representation of the airfoil."""
        return f"JaxAirfoil(name='{self.name}', n_points={self.n_points}, buffer_size={self.buffer_size})"
