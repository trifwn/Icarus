"""
Comprehensive error handling system for JAX airfoil implementation.

This module provides gradient-safe error handling, validation functions, and meaningful
error messages for common issues in the JAX airfoil implementation. It ensures that
error conditions are handled in the eager preprocessing phase to avoid breaking
gradient flow during JIT compilation.
"""

import warnings
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float


class AirfoilValidationError(ValueError):
    """Custom exception for airfoil validation errors."""

    pass


class BufferOverflowError(RuntimeError):
    """Custom exception for buffer overflow conditions."""

    pass


class GeometryError(ValueError):
    """Custom exception for airfoil geometry issues."""

    pass


class AirfoilErrorHandler:
    """
    Comprehensive error handling and validation system for JAX airfoils.

    This class provides gradient-safe error handling by performing all validation
    in the eager preprocessing phase, before JIT compilation. It includes:
    - Coordinate validation and sanitization
    - Buffer overflow detection and handling
    - Geometry validation for degenerate cases
    - Meaningful error messages with context
    """

    # Validation thresholds
    MIN_POINTS = 3
    MAX_COORDINATE_VALUE = 100.0
    MIN_COORDINATE_VALUE = -100.0
    MIN_POINT_DISTANCE = 1e-12
    MAX_ASPECT_RATIO = 1000.0
    MIN_CHORD_LENGTH = 1e-6
    MAX_THICKNESS_RATIO = 2.0

    @staticmethod
    def validate_coordinate_shape(
        coords: Union[jax.Array, np.ndarray],
        name: str = "coordinates",
    ) -> None:
        """
        Validate that coordinates have the correct shape.

        Args:
            coords: Coordinate array to validate
            name: Name of the coordinate array for error messages

        Raises:
            AirfoilValidationError: If coordinates have invalid shape
        """
        if coords.ndim != 2:
            raise AirfoilValidationError(
                f"{name} must be a 2D array, got {coords.ndim}D array with shape {coords.shape}",
            )

        if coords.shape[0] != 2:
            raise AirfoilValidationError(
                f"{name} must have shape (2, n_points) where first row is x and second is y, "
                f"got shape {coords.shape}",
            )

        if coords.shape[1] < AirfoilErrorHandler.MIN_POINTS:
            raise AirfoilValidationError(
                f"{name} must have at least {AirfoilErrorHandler.MIN_POINTS} points, "
                f"got {coords.shape[1]} points",
            )

    @staticmethod
    def validate_coordinate_values(
        coords: Union[jax.Array, np.ndarray],
        name: str = "coordinates",
    ) -> None:
        """
        Validate that coordinate values are within reasonable bounds.

        Args:
            coords: Coordinate array to validate
            name: Name of the coordinate array for error messages

        Raises:
            AirfoilValidationError: If coordinates contain invalid values
        """
        # Check for NaN values
        if jnp.any(jnp.isnan(coords)):
            nan_count = jnp.sum(jnp.isnan(coords))
            raise AirfoilValidationError(
                f"{name} contain {nan_count} NaN values. "
                "Use coordinate preprocessing to filter NaN values before validation.",
            )

        # Check for infinite values
        if jnp.any(jnp.isinf(coords)):
            inf_count = jnp.sum(jnp.isinf(coords))
            raise AirfoilValidationError(
                f"{name} contain {inf_count} infinite values. "
                "Coordinate values must be finite.",
            )

        # Check for extremely large values
        max_val = jnp.max(jnp.abs(coords))
        if max_val > AirfoilErrorHandler.MAX_COORDINATE_VALUE:
            raise AirfoilValidationError(
                f"{name} contain values larger than {AirfoilErrorHandler.MAX_COORDINATE_VALUE} "
                f"(maximum absolute value: {float(max_val):.6f}). "
                "This may indicate incorrect coordinate scaling or data corruption.",
            )

        # Check for extremely small values that might indicate precision issues
        if max_val < AirfoilErrorHandler.MIN_COORDINATE_VALUE:
            raise AirfoilValidationError(
                f"{name} have maximum absolute value {float(max_val):.6e} which is too small. "
                "This may indicate precision issues or incorrect scaling.",
            )

    @staticmethod
    def validate_airfoil_geometry(
        upper: Float[jax.Array, "2 n_upper"],
        lower: Float[jax.Array, "2 n_lower"],
        name: str = "airfoil",
    ) -> None:
        """
        Validate airfoil geometry for common issues.

        Args:
            upper: Upper surface coordinates
            lower: Lower surface coordinates
            name: Name of the airfoil for error messages

        Raises:
            GeometryError: If airfoil geometry is invalid
        """
        # Check that surfaces have points
        if upper.shape[1] == 0:
            raise GeometryError(f"{name} upper surface has no points")
        if lower.shape[1] == 0:
            raise GeometryError(f"{name} lower surface has no points")

        # Check chord length
        upper_x_range = jnp.max(upper[0, :]) - jnp.min(upper[0, :])
        lower_x_range = jnp.max(lower[0, :]) - jnp.min(lower[0, :])
        chord_length = max(float(upper_x_range), float(lower_x_range))

        if chord_length < AirfoilErrorHandler.MIN_CHORD_LENGTH:
            raise GeometryError(
                f"{name} has chord length {chord_length:.6e} which is too small. "
                f"Minimum chord length is {AirfoilErrorHandler.MIN_CHORD_LENGTH}.",
            )

        # Check for reasonable thickness
        # Estimate thickness at midchord
        midchord_x = (jnp.max(upper[0, :]) + jnp.min(upper[0, :])) / 2

        # Find closest points to midchord
        upper_distances = jnp.abs(upper[0, :] - midchord_x)
        lower_distances = jnp.abs(lower[0, :] - midchord_x)

        upper_idx = jnp.argmin(upper_distances)
        lower_idx = jnp.argmin(lower_distances)

        thickness = float(upper[1, upper_idx] - lower[1, lower_idx])
        thickness_ratio = thickness / chord_length

        if thickness_ratio > AirfoilErrorHandler.MAX_THICKNESS_RATIO:
            raise GeometryError(
                f"{name} has thickness ratio {thickness_ratio:.3f} which is unrealistic. "
                f"Maximum thickness ratio is {AirfoilErrorHandler.MAX_THICKNESS_RATIO}. "
                "This may indicate swapped upper/lower surfaces or incorrect data.",
            )

        # Check that upper surface is generally above lower surface
        if thickness < 0:
            warnings.warn(
                f"{name} appears to have upper surface below lower surface at midchord. "
                "This may indicate swapped surfaces or incorrect coordinate ordering.",
                UserWarning,
            )

    @staticmethod
    def validate_surface_ordering(
        coords: Float[jax.Array, "2 n_points"],
        surface_name: str = "surface",
    ) -> None:
        """
        Validate that surface points are properly ordered.

        Args:
            coords: Surface coordinates
            surface_name: Name of the surface for error messages

        Raises:
            GeometryError: If surface ordering is problematic
        """
        if coords.shape[1] < 2:
            return  # Can't validate ordering with less than 2 points

        x_coords = coords[0, :]

        # Check for duplicate consecutive points
        diffs = jnp.diff(x_coords)
        zero_diffs = jnp.abs(diffs) < AirfoilErrorHandler.MIN_POINT_DISTANCE

        if jnp.any(zero_diffs):
            duplicate_count = jnp.sum(zero_diffs)
            raise GeometryError(
                f"{surface_name} has {duplicate_count} duplicate consecutive x-coordinates. "
                "Use coordinate preprocessing to remove duplicates.",
            )

        # Check for reasonable monotonicity (allowing some variation)
        increasing = jnp.sum(diffs > 0)
        decreasing = jnp.sum(diffs < 0)
        total = len(diffs)

        # If more than 80% of points go in one direction, consider it monotonic
        if increasing / total < 0.2 and decreasing / total < 0.2:
            warnings.warn(
                f"{surface_name} has highly irregular x-coordinate ordering. "
                "This may cause interpolation issues.",
                UserWarning,
            )

    @staticmethod
    def check_buffer_capacity(
        required_size: int,
        current_buffer_size: int,
        max_buffer_size: int = 4096,
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if buffer capacity is sufficient and suggest new size if needed.

        Args:
            required_size: Number of points needed
            current_buffer_size: Current buffer capacity
            max_buffer_size: Maximum allowed buffer size

        Returns:
            Tuple of (needs_reallocation, suggested_new_size)

        Raises:
            BufferOverflowError: If required size exceeds maximum buffer size
        """
        if required_size <= current_buffer_size:
            return False, None

        if required_size > max_buffer_size:
            raise BufferOverflowError(
                f"Required buffer size ({required_size}) exceeds maximum allowed size "
                f"({max_buffer_size}). Consider reducing the number of airfoil points or "
                f"increasing MAX_BUFFER_SIZE if more memory is available.",
            )

        # Find next power of 2 that can accommodate required size
        new_size = 1
        while new_size < required_size:
            new_size *= 2

        # Clamp to maximum size
        new_size = min(new_size, max_buffer_size)

        return True, new_size

    @staticmethod
    def validate_morphing_parameters(
        eta: float,
        airfoil1_name: str = "airfoil1",
        airfoil2_name: str = "airfoil2",
    ) -> None:
        """
        Validate morphing parameters.

        Args:
            eta: Morphing parameter (0.0 to 1.0)
            airfoil1_name: Name of first airfoil for error messages
            airfoil2_name: Name of second airfoil for error messages

        Raises:
            AirfoilValidationError: If morphing parameters are invalid
        """
        if not isinstance(eta, (int, float)):
            raise AirfoilValidationError(
                f"Morphing parameter eta must be a number, got {type(eta)}",
            )

        if not (0.0 <= eta <= 1.0):
            raise AirfoilValidationError(
                f"Morphing parameter eta must be between 0.0 and 1.0, got {eta}. "
                f"eta=0.0 gives {airfoil1_name}, eta=1.0 gives {airfoil2_name}.",
            )

        if jnp.isnan(eta) or jnp.isinf(eta):
            raise AirfoilValidationError(
                f"Morphing parameter eta must be finite, got {eta}",
            )

    @staticmethod
    def validate_flap_parameters(
        flap_hinge_chord_percentage: float,
        flap_angle: float,
        flap_hinge_thickness_percentage: float = 0.5,
        chord_extension: float = 1.0,
    ) -> None:
        """
        Validate flap operation parameters.

        Args:
            flap_hinge_chord_percentage: Chordwise location of flap hinge (0.0 to 1.0)
            flap_angle: Flap deflection angle in degrees
            flap_hinge_thickness_percentage: Position through thickness (0.0 to 1.0)
            chord_extension: Chord extension factor (must be positive)

        Raises:
            AirfoilValidationError: If flap parameters are invalid
        """
        if not (0.0 <= flap_hinge_chord_percentage <= 1.0):
            raise AirfoilValidationError(
                f"flap_hinge_chord_percentage must be between 0.0 and 1.0, "
                f"got {flap_hinge_chord_percentage}. "
                "0.0 = leading edge, 1.0 = trailing edge.",
            )

        if not (0.0 <= flap_hinge_thickness_percentage <= 1.0):
            raise AirfoilValidationError(
                f"flap_hinge_thickness_percentage must be between 0.0 and 1.0, "
                f"got {flap_hinge_thickness_percentage}. "
                "0.0 = lower surface, 1.0 = upper surface.",
            )

        if chord_extension <= 0.0:
            raise AirfoilValidationError(
                f"chord_extension must be positive, got {chord_extension}. "
                "Values > 1.0 extend the flap, values < 1.0 shorten it.",
            )

        if not (-180.0 <= flap_angle <= 180.0):
            warnings.warn(
                f"Flap angle {flap_angle}° is outside typical range [-180°, 180°]. "
                "Large flap angles may produce unrealistic geometries.",
                UserWarning,
            )

        # Check for problematic combinations
        if flap_hinge_chord_percentage > 0.95 and abs(flap_angle) > 1.0:
            warnings.warn(
                f"Flap hinge very close to trailing edge ({flap_hinge_chord_percentage:.3f}) "
                f"with significant deflection ({flap_angle}°) may produce poor geometry.",
                UserWarning,
            )

    @staticmethod
    def validate_naca_parameters(digits: str, airfoil_type: str = "4-digit") -> None:
        """
        Validate NACA airfoil parameters.

        Args:
            digits: NACA digit string
            airfoil_type: Type of NACA airfoil ("4-digit" or "5-digit")

        Raises:
            AirfoilValidationError: If NACA parameters are invalid
        """
        if not isinstance(digits, str):
            raise AirfoilValidationError(
                f"NACA digits must be a string, got {type(digits)}",
            )

        if not digits.isdigit():
            raise AirfoilValidationError(
                f"NACA digits must contain only numeric characters, got '{digits}'",
            )

        if airfoil_type == "4-digit":
            if len(digits) != 4:
                raise AirfoilValidationError(
                    f"NACA 4-digit designation must be exactly 4 digits, got '{digits}' "
                    f"with {len(digits)} digits",
                )

            # Parse and validate individual parameters
            M = int(digits[0])  # Maximum camber
            P = int(digits[1])  # Position of maximum camber
            XX = int(digits[2:4])  # Maximum thickness

            if not (0 <= M <= 9):
                raise AirfoilValidationError(
                    f"NACA 4-digit maximum camber parameter must be 0-9, got {M} "
                    f"from digits '{digits}'",
                )

            if not (0 <= P <= 9):
                raise AirfoilValidationError(
                    f"NACA 4-digit camber position parameter must be 0-9, got {P} "
                    f"from digits '{digits}'",
                )

            if not (1 <= XX <= 99):
                raise AirfoilValidationError(
                    f"NACA 4-digit thickness parameter must be 01-99, got {XX:02d} "
                    f"from digits '{digits}'",
                )

        elif airfoil_type == "5-digit":
            if len(digits) != 5:
                raise AirfoilValidationError(
                    f"NACA 5-digit designation must be exactly 5 digits, got '{digits}' "
                    f"with {len(digits)} digits",
                )

            # Parse and validate individual parameters
            L = int(digits[0])  # Design coefficient of lift parameter
            P = int(digits[1])  # Position of maximum camber parameter
            Q = int(digits[2])  # Reflex parameter
            XX = int(digits[3:5])  # Maximum thickness

            if not (0 <= L <= 9):
                raise AirfoilValidationError(
                    f"NACA 5-digit design CL parameter must be 0-9, got {L} "
                    f"from digits '{digits}'",
                )

            if not (0 <= P <= 9):
                raise AirfoilValidationError(
                    f"NACA 5-digit camber position parameter must be 0-9, got {P} "
                    f"from digits '{digits}'",
                )

            if Q not in [0, 1]:
                raise AirfoilValidationError(
                    f"NACA 5-digit reflex parameter must be 0 or 1, got {Q} "
                    f"from digits '{digits}'",
                )

            if not (1 <= XX <= 99):
                raise AirfoilValidationError(
                    f"NACA 5-digit thickness parameter must be 01-99, got {XX:02d} "
                    f"from digits '{digits}'",
                )

        else:
            raise AirfoilValidationError(
                f"Unsupported NACA airfoil type: {airfoil_type}. "
                "Supported types are '4-digit' and '5-digit'.",
            )

    @staticmethod
    def validate_repanel_parameters(
        n_points: int,
        distribution: str = "cosine",
        method: str = "chord_based",
    ) -> None:
        """
        Validate repaneling parameters.

        Args:
            n_points: Number of points for repaneling
            distribution: Point distribution method
            method: Repaneling method

        Raises:
            AirfoilValidationError: If repaneling parameters are invalid
        """
        if not isinstance(n_points, int) or n_points < 4:
            raise AirfoilValidationError(
                f"n_points must be an integer >= 4, got {n_points}",
            )

        valid_distributions = ["cosine", "uniform"]
        if distribution not in valid_distributions:
            raise AirfoilValidationError(
                f"distribution must be one of {valid_distributions}, got '{distribution}'",
            )

        valid_methods = ["chord_based", "arc_length"]
        if method not in valid_methods:
            raise AirfoilValidationError(
                f"method must be one of {valid_methods}, got '{method}'",
            )

    @staticmethod
    def create_error_context(
        operation: str,
        airfoil_name: str = "airfoil",
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create contextual error message for debugging.

        Args:
            operation: Name of the operation that failed
            airfoil_name: Name of the airfoil involved
            additional_info: Additional context information

        Returns:
            Formatted error context string
        """
        context_parts = [f"Error in {operation} for {airfoil_name}"]

        if additional_info:
            for key, value in additional_info.items():
                context_parts.append(f"{key}: {value}")

        return " | ".join(context_parts)

    @staticmethod
    def handle_gradient_safe_error(
        error_condition: bool,
        error_message: str,
        fallback_value: Any = None,
    ) -> Any:
        """
        Handle errors in a gradient-safe manner during JIT compilation.

        This function should be used sparingly and only for cases where
        we need to handle errors within JIT-compiled functions without
        breaking gradient flow.

        Args:
            error_condition: Boolean indicating if error occurred
            error_message: Error message for logging
            fallback_value: Value to return if error occurred

        Returns:
            fallback_value if error_condition is True, otherwise None
        """
        # In JIT context, we can't raise exceptions, so we use conditional logic
        # The actual error handling should happen in eager preprocessing
        return jax.lax.cond(error_condition, lambda: fallback_value, lambda: None)

    @staticmethod
    def suggest_fixes(error_type: str, **kwargs) -> str:
        """
        Suggest fixes for common error conditions.

        Args:
            error_type: Type of error encountered
            **kwargs: Additional context for suggestions

        Returns:
            String with suggested fixes
        """
        suggestions = {
            "nan_coordinates": (
                "Try using CoordinateProcessor.filter_nan_coordinates() to remove NaN values, "
                "or check your data source for corruption."
            ),
            "invalid_shape": (
                "Ensure coordinates are provided as (2, n_points) arrays where the first row "
                "contains x-coordinates and the second row contains y-coordinates."
            ),
            "buffer_overflow": (
                "Consider reducing the number of airfoil points, increasing MAX_BUFFER_SIZE, "
                "or using batch processing for multiple airfoils."
            ),
            "geometry_invalid": (
                "Check that upper and lower surfaces are correctly oriented, "
                "coordinates are properly scaled, and the airfoil is closed."
            ),
            "naca_invalid": (
                "Verify NACA designation format (e.g., '2412' for 4-digit, '23012' for 5-digit) "
                "and ensure parameters are within valid ranges."
            ),
            "morphing_invalid": (
                "Ensure morphing parameter eta is between 0.0 and 1.0, and both airfoils "
                "have compatible geometries."
            ),
            "flap_invalid": (
                "Check that flap parameters are within valid ranges and the combination "
                "produces realistic geometry."
            ),
        }

        base_suggestion = suggestions.get(
            error_type,
            "Check input parameters and data quality.",
        )

        # Add context-specific suggestions
        if error_type == "buffer_overflow" and "required_size" in kwargs:
            base_suggestion += f" Required size: {kwargs['required_size']} points."

        return base_suggestion
