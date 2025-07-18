"""
JAX-compatible geometric operations for airfoil analysis.

This module provides JIT-compiled functions for core airfoil geometric operations
including thickness computation, camber line calculation, and surface coordinate queries.
All functions are designed to work with static buffer allocation and masking.
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from .interpolation_engine import JaxInterpolationEngine


class JaxAirfoilOps:
    """
    JAX-compatible geometric operations for airfoil analysis.

    This class provides static methods for all core geometric computations on airfoils.
    All methods are JIT-compiled and support automatic differentiation. They work with
    padded arrays and masking to handle variable-sized airfoil data efficiently.

    The operations include:
    - Thickness computation with masking
    - Camber line calculation
    - Surface coordinate queries (y_upper, y_lower)
    - Geometric property calculations
    - Flap operations
    """

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def compute_thickness(
        upper_coords: Float[Array, "2 n_buffer"],
        lower_coords: Float[Array, "2 n_buffer"],
        n_upper_valid: int,
        n_lower_valid: int,
        query_x: Float[Array, "n_query"],
    ) -> Float[Array, "n_query"]:
        """
        Compute airfoil thickness distribution at query points with masking.

        Args:
            upper_coords: Upper surface coordinates [x, y] with shape (2, n_buffer)
            lower_coords: Lower surface coordinates [x, y] with shape (2, n_buffer)
            n_upper_valid: Number of valid upper surface points (static for JIT)
            n_lower_valid: Number of valid lower surface points (static for JIT)
            query_x: X coordinates to query thickness at

        Returns:
            Thickness values at query points (upper_y - lower_y)
        """
        # Create validity masks
        upper_mask = jnp.arange(upper_coords.shape[1]) < n_upper_valid
        lower_mask = jnp.arange(lower_coords.shape[1]) < n_lower_valid

        # Interpolate both surfaces at query points
        upper_y = JaxInterpolationEngine.interpolate_surface_masked(
            upper_coords,
            upper_mask,
            query_x,
            n_valid=n_upper_valid,
        )

        lower_y = JaxInterpolationEngine.interpolate_surface_masked(
            lower_coords,
            lower_mask,
            query_x,
            n_valid=n_lower_valid,
        )

        # Thickness is the difference between upper and lower surfaces
        thickness = upper_y - lower_y

        return thickness

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def compute_camber_line(
        upper_coords: Float[Array, "2 n_buffer"],
        lower_coords: Float[Array, "2 n_buffer"],
        n_upper_valid: int,
        n_lower_valid: int,
        query_x: Float[Array, "n_query"],
    ) -> Float[Array, "n_query"]:
        """
        Compute airfoil camber line at query points with masking.

        The camber line is the mean line between the upper and lower surfaces.

        Args:
            upper_coords: Upper surface coordinates [x, y] with shape (2, n_buffer)
            lower_coords: Lower surface coordinates [x, y] with shape (2, n_buffer)
            n_upper_valid: Number of valid upper surface points (static for JIT)
            n_lower_valid: Number of valid lower surface points (static for JIT)
            query_x: X coordinates to query camber line at

        Returns:
            Camber line y-coordinates at query points
        """
        # Create validity masks
        upper_mask = jnp.arange(upper_coords.shape[1]) < n_upper_valid
        lower_mask = jnp.arange(lower_coords.shape[1]) < n_lower_valid

        # Interpolate both surfaces at query points
        upper_y = JaxInterpolationEngine.interpolate_surface_masked(
            upper_coords,
            upper_mask,
            query_x,
            n_valid=n_upper_valid,
        )

        lower_y = JaxInterpolationEngine.interpolate_surface_masked(
            lower_coords,
            lower_mask,
            query_x,
            n_valid=n_lower_valid,
        )

        # Camber line is the average of upper and lower surfaces
        camber_y = 0.5 * (upper_y + lower_y)

        return camber_y

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def y_upper(
        upper_coords: Float[Array, "2 n_buffer"],
        n_valid: int,
        query_x: Float[Array, "n_query"],
    ) -> Float[Array, "n_query"]:
        """
        Query upper surface y-coordinates at given x-coordinates.

        Args:
            upper_coords: Upper surface coordinates [x, y] with shape (2, n_buffer)
            n_valid: Number of valid points (static for JIT)
            query_x: X coordinates to query

        Returns:
            Upper surface y-coordinates at query points
        """
        # Create validity mask
        mask = jnp.arange(upper_coords.shape[1]) < n_valid

        # Interpolate upper surface
        upper_y = JaxInterpolationEngine.interpolate_surface_masked(
            upper_coords,
            mask,
            query_x,
            n_valid=n_valid,
        )

        return upper_y

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def y_lower(
        lower_coords: Float[Array, "2 n_buffer"],
        n_valid: int,
        query_x: Float[Array, "n_query"],
    ) -> Float[Array, "n_query"]:
        """
        Query lower surface y-coordinates at given x-coordinates.

        Args:
            lower_coords: Lower surface coordinates [x, y] with shape (2, n_buffer)
            n_valid: Number of valid points (static for JIT)
            query_x: X coordinates to query

        Returns:
            Lower surface y-coordinates at query points
        """
        # Create validity mask
        mask = jnp.arange(lower_coords.shape[1]) < n_valid

        # Interpolate lower surface
        lower_y = JaxInterpolationEngine.interpolate_surface_masked(
            lower_coords,
            mask,
            query_x,
            n_valid=n_valid,
        )

        return lower_y

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def compute_max_thickness(
        upper_coords: Float[Array, "2 n_buffer"],
        lower_coords: Float[Array, "2 n_buffer"],
        n_upper_valid: int,
        n_lower_valid: int,
        n_sample_points: int = 100,
    ) -> Tuple[float, float]:
        """
        Compute maximum thickness and its location.

        Args:
            upper_coords: Upper surface coordinates [x, y]
            lower_coords: Lower surface coordinates [x, y]
            n_upper_valid: Number of valid upper surface points
            n_lower_valid: Number of valid lower surface points
            n_sample_points: Number of points to sample for thickness computation

        Returns:
            Tuple of (max_thickness, max_thickness_location)
        """
        # Determine x-coordinate range for sampling
        # Use the valid portions of both surfaces
        upper_x_valid = upper_coords[0, :n_upper_valid]
        lower_x_valid = lower_coords[0, :n_lower_valid]

        # Find common x-coordinate range
        x_min = jnp.max(jnp.array([jnp.min(upper_x_valid), jnp.min(lower_x_valid)]))
        x_max = jnp.min(jnp.array([jnp.max(upper_x_valid), jnp.max(lower_x_valid)]))

        # Create sample points
        query_x = jnp.linspace(x_min, x_max, n_sample_points)

        # Compute thickness distribution
        thickness = JaxAirfoilOps.compute_thickness(
            upper_coords,
            lower_coords,
            n_upper_valid,
            n_lower_valid,
            query_x,
        )

        # Find maximum thickness and its location
        max_idx = jnp.argmax(thickness)
        max_thickness = thickness[max_idx]
        max_thickness_location = query_x[max_idx]

        return max_thickness, max_thickness_location

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def compute_max_camber(
        upper_coords: Float[Array, "2 n_buffer"],
        lower_coords: Float[Array, "2 n_buffer"],
        n_upper_valid: int,
        n_lower_valid: int,
        n_sample_points: int = 100,
    ) -> Tuple[float, float]:
        """
        Compute maximum camber and its location.

        Args:
            upper_coords: Upper surface coordinates [x, y]
            lower_coords: Lower surface coordinates [x, y]
            n_upper_valid: Number of valid upper surface points
            n_lower_valid: Number of valid lower surface points
            n_sample_points: Number of points to sample for camber computation

        Returns:
            Tuple of (max_camber, max_camber_location)
        """
        # Determine x-coordinate range for sampling
        upper_x_valid = upper_coords[0, :n_upper_valid]
        lower_x_valid = lower_coords[0, :n_lower_valid]

        # Find common x-coordinate range
        x_min = jnp.max(jnp.array([jnp.min(upper_x_valid), jnp.min(lower_x_valid)]))
        x_max = jnp.min(jnp.array([jnp.max(upper_x_valid), jnp.max(lower_x_valid)]))

        # Create sample points
        query_x = jnp.linspace(x_min, x_max, n_sample_points)

        # Compute camber line
        camber = JaxAirfoilOps.compute_camber_line(
            upper_coords,
            lower_coords,
            n_upper_valid,
            n_lower_valid,
            query_x,
        )

        # Find maximum camber and its location
        max_idx = jnp.argmax(jnp.abs(camber))
        max_camber = camber[max_idx]
        max_camber_location = query_x[max_idx]

        return max_camber, max_camber_location

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def compute_chord_length(
        upper_coords: Float[Array, "2 n_buffer"],
        lower_coords: Float[Array, "2 n_buffer"],
        n_upper_valid: int,
        n_lower_valid: int,
    ) -> float:
        """
        Compute the chord length of the airfoil.

        Args:
            upper_coords: Upper surface coordinates [x, y]
            lower_coords: Lower surface coordinates [x, y]
            n_upper_valid: Number of valid upper surface points
            n_lower_valid: Number of valid lower surface points

        Returns:
            Chord length (distance from leading edge to trailing edge)
        """
        # Get valid x-coordinates from both surfaces
        upper_x_valid = upper_coords[0, :n_upper_valid]
        lower_x_valid = lower_coords[0, :n_lower_valid]

        # Find leading and trailing edge x-coordinates
        all_x = jnp.concatenate([upper_x_valid, lower_x_valid])
        x_min = jnp.min(all_x)  # Leading edge
        x_max = jnp.max(all_x)  # Trailing edge

        # Chord length is the difference
        chord_length = x_max - x_min

        return chord_length

    @staticmethod
    @jax.jit
    def rotate_coordinates(
        coords: Float[Array, "2 n_points"],
        angle: float,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> Float[Array, "2 n_points"]:
        """
        Rotate coordinates around a center point.

        Args:
            coords: Coordinates to rotate [x, y] with shape (2, n_points)
            angle: Rotation angle in radians (positive = counterclockwise)
            center_x: X coordinate of rotation center
            center_y: Y coordinate of rotation center

        Returns:
            Rotated coordinates with same shape as input
        """
        # Translate to origin
        x_translated = coords[0, :] - center_x
        y_translated = coords[1, :] - center_y

        # Apply rotation matrix
        cos_angle = jnp.cos(angle)
        sin_angle = jnp.sin(angle)

        x_rotated = x_translated * cos_angle - y_translated * sin_angle
        y_rotated = x_translated * sin_angle + y_translated * cos_angle

        # Translate back
        x_final = x_rotated + center_x
        y_final = y_rotated + center_y

        return jnp.stack([x_final, y_final])

    @staticmethod
    @jax.jit
    def scale_coordinates(
        coords: Float[Array, "2 n_points"],
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> Float[Array, "2 n_points"]:
        """
        Scale coordinates around a center point.

        Args:
            coords: Coordinates to scale [x, y] with shape (2, n_points)
            scale_x: Scaling factor in x direction
            scale_y: Scaling factor in y direction
            center_x: X coordinate of scaling center
            center_y: Y coordinate of scaling center

        Returns:
            Scaled coordinates with same shape as input
        """
        # Translate to origin
        x_translated = coords[0, :] - center_x
        y_translated = coords[1, :] - center_y

        # Apply scaling
        x_scaled = x_translated * scale_x
        y_scaled = y_translated * scale_y

        # Translate back
        x_final = x_scaled + center_x
        y_final = y_scaled + center_y

        return jnp.stack([x_final, y_final])

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def apply_flap_transformation(
        upper_coords: Float[Array, "2 n_buffer"],
        lower_coords: Float[Array, "2 n_buffer"],
        n_upper_valid: int,
        n_lower_valid: int,
        flap_hinge_x: float,
        flap_angle: float,
        flap_hinge_thickness_percentage: float = 0.5,
        chord_extension: float = 1.0,
        should_skip: bool = False,
    ) -> Tuple[Float[Array, "2 n_buffer"], Float[Array, "2 n_buffer"], int, int]:
        """
        Apply flap transformation to airfoil coordinates.

        Args:
            upper_coords: Upper surface coordinates [x, y] with shape (2, n_buffer)
            lower_coords: Lower surface coordinates [x, y] with shape (2, n_buffer)
            n_upper_valid: Number of valid upper surface points
            n_lower_valid: Number of valid lower surface points
            flap_hinge_x: X-coordinate of the flap hinge
            flap_angle: Flap deflection angle in radians (positive = downward)
            flap_hinge_thickness_percentage: Position of hinge through thickness (0=lower, 1=upper)
            chord_extension: Scaling factor for flap chord length
            should_skip: Whether to skip the transformation and return original coordinates

        Returns:
            Tuple of (new_upper_coords, new_lower_coords, new_n_upper_valid, new_n_lower_valid)
        """

        # Define functions for conditional execution
        def skip_transformation():
            return upper_coords, lower_coords, n_upper_valid, n_lower_valid

        def apply_transformation():
            # Extract valid coordinates
            upper_valid = upper_coords[:, :n_upper_valid]
            lower_valid = lower_coords[:, :n_lower_valid]

            # Find hinge y-coordinate by interpolating between upper and lower surfaces
            upper_y_at_hinge = JaxInterpolationEngine.interpolate_surface_masked(
                upper_valid,
                jnp.ones(n_upper_valid, dtype=bool),
                jnp.array([flap_hinge_x]),
                n_valid=n_upper_valid,
            )[0]

            lower_y_at_hinge = JaxInterpolationEngine.interpolate_surface_masked(
                lower_valid,
                jnp.ones(n_lower_valid, dtype=bool),
                jnp.array([flap_hinge_x]),
                n_valid=n_lower_valid,
            )[0]

            # Calculate hinge y-coordinate based on thickness percentage
            hinge_y = (
                lower_y_at_hinge * flap_hinge_thickness_percentage
                + upper_y_at_hinge * (1.0 - flap_hinge_thickness_percentage)
            )

            # Separate coordinates before and after hinge
            # Upper surface
            upper_before_mask = upper_valid[0, :] <= flap_hinge_x
            upper_after_mask = upper_valid[0, :] > flap_hinge_x

            upper_before = jnp.where(upper_before_mask[None, :], upper_valid, jnp.nan)
            upper_after = jnp.where(upper_after_mask[None, :], upper_valid, jnp.nan)

            # Lower surface
            lower_before_mask = lower_valid[0, :] <= flap_hinge_x
            lower_after_mask = lower_valid[0, :] > flap_hinge_x

            lower_before = jnp.where(lower_before_mask[None, :], lower_valid, jnp.nan)
            lower_after = jnp.where(lower_after_mask[None, :], lower_valid, jnp.nan)

            # Transform the flap section (after hinge)
            # 1. Scale by chord extension
            # 2. Rotate by flap angle

            # Transform upper flap section
            upper_after_transformed = JaxAirfoilOps.scale_coordinates(
                upper_after,
                scale_x=chord_extension,
                scale_y=1.0,
                center_x=flap_hinge_x,
                center_y=hinge_y,
            )
            upper_after_transformed = JaxAirfoilOps.rotate_coordinates(
                upper_after_transformed,
                flap_angle,
                center_x=flap_hinge_x,
                center_y=hinge_y,
            )

            # Transform lower flap section
            lower_after_transformed = JaxAirfoilOps.scale_coordinates(
                lower_after,
                scale_x=chord_extension,
                scale_y=1.0,
                center_x=flap_hinge_x,
                center_y=hinge_y,
            )
            lower_after_transformed = JaxAirfoilOps.rotate_coordinates(
                lower_after_transformed,
                flap_angle,
                center_x=flap_hinge_x,
                center_y=hinge_y,
            )

            # Filter out points that moved backward (x < flap_hinge_x) due to rotation
            upper_after_valid_mask = (
                upper_after_transformed[0, :] >= flap_hinge_x
            ) & jnp.isfinite(upper_after_transformed[0, :])
            lower_after_valid_mask = (
                lower_after_transformed[0, :] >= flap_hinge_x
            ) & jnp.isfinite(lower_after_transformed[0, :])

            upper_after_filtered = jnp.where(
                upper_after_valid_mask[None, :],
                upper_after_transformed,
                jnp.nan,
            )
            lower_after_filtered = jnp.where(
                lower_after_valid_mask[None, :],
                lower_after_transformed,
                jnp.nan,
            )

            # Combine before and after sections
            # Use jnp.where to combine, prioritizing finite values
            upper_combined = jnp.where(
                jnp.isfinite(upper_before),
                upper_before,
                upper_after_filtered,
            )
            lower_combined = jnp.where(
                jnp.isfinite(lower_before),
                lower_before,
                lower_after_filtered,
            )

            # Pad back to buffer size
            upper_result = jnp.concatenate(
                [
                    upper_combined,
                    jnp.full(
                        (2, upper_coords.shape[1] - upper_combined.shape[1]),
                        jnp.nan,
                    ),
                ],
                axis=1,
            )

            lower_result = jnp.concatenate(
                [
                    lower_combined,
                    jnp.full(
                        (2, lower_coords.shape[1] - lower_combined.shape[1]),
                        jnp.nan,
                    ),
                ],
                axis=1,
            )

            # Count valid points in result
            new_n_upper_valid = jnp.sum(jnp.isfinite(upper_result[0, :]))
            new_n_lower_valid = jnp.sum(jnp.isfinite(lower_result[0, :]))

            return upper_result, lower_result, new_n_upper_valid, new_n_lower_valid

        # Use jax.lax.cond for conditional execution
        return jax.lax.cond(should_skip, skip_transformation, apply_transformation)

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def generate_cosine_spacing(n_points: int) -> Float[Array, "n_points"]:
        """
        Generate cosine-spaced points from 0 to 1.

        This spacing provides better resolution near the leading and trailing edges
        where gradients are typically higher.

        Args:
            n_points: Number of points to generate (must be static for JIT)

        Returns:
            Cosine-spaced x-coordinates from 0 to 1
        """
        beta = jnp.linspace(0, jnp.pi, n_points)
        x_coords = 0.5 * (1 - jnp.cos(beta))
        return x_coords

    @staticmethod
    @jax.jit
    def naca4_thickness_distribution(
        x: Float[Array, "n_points"],
        thickness: float,
    ) -> Float[Array, "n_points"]:
        """
        Compute NACA 4-digit thickness distribution.

        Args:
            x: X coordinates (normalized to [0, 1])
            thickness: Maximum thickness as fraction of chord (e.g., 0.12 for 12%)

        Returns:
            Thickness distribution at x coordinates
        """
        # NACA 4-digit thickness distribution coefficients
        a0 = 0.2969
        a1 = -0.126
        a2 = -0.3516
        a3 = 0.2843
        a4 = -0.1036  # For zero thickness trailing edge

        # Thickness distribution formula
        yt = (thickness / 0.2) * (
            a0 * jnp.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4
        )

        return yt

    @staticmethod
    @jax.jit
    def naca4_camber_line(
        x: Float[Array, "n_points"],
        max_camber: float,
        camber_position: float,
    ) -> Tuple[Float[Array, "n_points"], Float[Array, "n_points"]]:
        """
        Compute NACA 4-digit camber line and its derivative.

        Args:
            x: X coordinates (normalized to [0, 1])
            max_camber: Maximum camber as fraction of chord (e.g., 0.02 for 2%)
            camber_position: Position of maximum camber as fraction of chord (e.g., 0.4 for 40%)

        Returns:
            Tuple of (camber_line_y, camber_line_derivative)
        """
        # Avoid division by zero
        p = camber_position + 1e-19
        m = max_camber

        # Camber line using vectorized conditionals
        yc = jnp.where(
            x <= p,
            (m / p**2) * (2 * p * x - x**2),  # Forward portion
            (m / (1 - p) ** 2) * (1 - 2 * p + 2 * p * x - x**2),  # Aft portion
        )

        # Camber line derivative
        dyc_dx = jnp.where(
            x <= p,
            (2 * m / p**2) * (p - x),  # Forward portion
            (2 * m / (1 - p) ** 2) * (p - x),  # Aft portion
        )

        return yc, dyc_dx

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def generate_naca4_coordinates(
        max_camber: float,
        camber_position: float,
        thickness: float,
        n_points: int,
    ) -> Tuple[Float[Array, "2 n_points"], Float[Array, "2 n_points"]]:
        """
        Generate NACA 4-digit airfoil coordinates.

        Args:
            max_camber: Maximum camber as fraction of chord (M/100)
            camber_position: Position of maximum camber as fraction of chord (P/10)
            thickness: Maximum thickness as fraction of chord (XX/100)
            n_points: Number of points for each surface

        Returns:
            Tuple of (upper_coords, lower_coords) where each is shape (2, n_points)
        """
        # Generate cosine-spaced x coordinates
        x = JaxAirfoilOps.generate_cosine_spacing(n_points)

        # Compute thickness distribution
        yt = JaxAirfoilOps.naca4_thickness_distribution(x, thickness)

        # Compute camber line and its derivative
        yc, dyc_dx = JaxAirfoilOps.naca4_camber_line(x, max_camber, camber_position)

        # Compute angle of camber line
        theta = jnp.arctan(dyc_dx)

        # Compute upper and lower surface coordinates
        x_upper = x - yt * jnp.sin(theta)
        y_upper = yc + yt * jnp.cos(theta)
        x_lower = x + yt * jnp.sin(theta)
        y_lower = yc - yt * jnp.cos(theta)

        # Stack into coordinate arrays
        # Note: For proper selig format, upper surface should run from TE to LE
        # and lower surface should run from LE to TE
        upper_coords = jnp.stack(
            [x_upper[::-1], y_upper[::-1]],
        )  # Reverse upper surface
        lower_coords = jnp.stack([x_lower, y_lower])

        return upper_coords, lower_coords

    @staticmethod
    @jax.jit
    def naca5_camber_line_standard(
        x: Float[Array, "n_points"],
        design_cl: float,
        max_camber_pos: float,
    ) -> Tuple[Float[Array, "n_points"], Float[Array, "n_points"]]:
        """
        Compute NACA 5-digit standard camber line and its derivative.

        Args:
            x: X coordinates (normalized to [0, 1])
            design_cl: Design coefficient of lift (L * 3/20)
            max_camber_pos: Position of maximum camber (P/20)

        Returns:
            Tuple of (camber_line_y, camber_line_derivative)
        """
        # Standard NACA 5-digit parameters
        # These are lookup table values - simplified for JAX compatibility
        P_values = jnp.array([0.05, 0.10, 0.15, 0.20, 0.25])
        M_values = jnp.array([0.0580, 0.1260, 0.2025, 0.2900, 0.3910])
        K_values = jnp.array([361.4, 51.64, 15.957, 6.643, 3.230])

        # Interpolate to find m and k1 for the given position
        # Use linear interpolation (simplified)
        m = jnp.interp(max_camber_pos, P_values, M_values)
        k1 = jnp.interp(m, M_values, K_values)

        # Compute r (position parameter) - simplified calculation
        r = max_camber_pos + m * jnp.sqrt(m / 3)

        # Camber line calculation
        yc = jnp.where(
            x <= r,
            (design_cl / 0.3) * (k1 / 6) * (x**3 - 3 * r * x**2 + r**2 * (3 - r) * x),
            (design_cl / 0.3) * (k1 / 6) * r**3 * (1 - x),
        )

        # Camber line derivative
        dyc_dx = jnp.where(
            x <= r,
            (design_cl / 0.3) * (k1 / 6) * (3 * x**2 - 6 * r * x + r**2 * (3 - r)),
            -(design_cl / 0.3) * (k1 / 6) * r**3,
        )

        return yc, dyc_dx

    @staticmethod
    @partial(jax.jit, static_argnums=(4,))
    def generate_naca5_coordinates(
        design_cl: float,
        max_camber_pos: float,
        reflex: bool,
        thickness: float,
        n_points: int,
    ) -> Tuple[Float[Array, "2 n_points"], Float[Array, "2 n_points"]]:
        """
        Generate NACA 5-digit airfoil coordinates.

        Args:
            design_cl: Design coefficient of lift (L * 3/20)
            max_camber_pos: Position of maximum camber (P/20)
            reflex: Whether to use reflex camber line (Q=1) or standard (Q=0)
            thickness: Maximum thickness as fraction of chord (XX/100)
            n_points: Number of points for each surface

        Returns:
            Tuple of (upper_coords, lower_coords) where each is shape (2, n_points)
        """
        # Generate cosine-spaced x coordinates
        x = JaxAirfoilOps.generate_cosine_spacing(n_points)

        # Compute thickness distribution (same as NACA 4-digit)
        yt = JaxAirfoilOps.naca4_thickness_distribution(x, thickness)

        # Compute camber line based on type using jnp.where for JAX compatibility
        yc_standard, dyc_dx_standard = JaxAirfoilOps.naca5_camber_line_standard(
            x,
            design_cl,
            max_camber_pos,
        )
        yc_reflex = jnp.zeros_like(x)
        dyc_dx_reflex = jnp.zeros_like(x)

        # Use jnp.where instead of if statement for JAX compatibility
        yc = jnp.where(reflex, yc_reflex, yc_standard)
        dyc_dx = jnp.where(reflex, dyc_dx_reflex, dyc_dx_standard)

        # Compute angle of camber line
        theta = jnp.arctan(dyc_dx)

        # Compute upper and lower surface coordinates
        x_upper = x - yt * jnp.sin(theta)
        y_upper = yc + yt * jnp.cos(theta)
        x_lower = x + yt * jnp.sin(theta)
        y_lower = yc - yt * jnp.cos(theta)

        # Stack into coordinate arrays
        # Note: For proper selig format, upper surface should run from TE to LE
        # and lower surface should run from LE to TE
        upper_coords = jnp.stack(
            [x_upper[::-1], y_upper[::-1]],
        )  # Reverse upper surface
        lower_coords = jnp.stack([x_lower, y_lower])

        return upper_coords, lower_coords

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def generate_cosine_distribution(n_points: int) -> Float[Array, "n_points"]:
        """
        Generate cosine-spaced distribution of points from 0 to 1.

        This creates a distribution with more points near the leading and trailing edges,
        which is beneficial for airfoil analysis.

        Args:
            n_points: Number of points to generate

        Returns:
            Array of cosine-spaced points from 0 to 1
        """
        # Generate linearly spaced angles from 0 to pi
        beta = jnp.linspace(0, jnp.pi, n_points)

        # Apply cosine spacing transformation
        x = 0.5 * (1 - jnp.cos(beta))

        return x

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def generate_uniform_distribution(n_points: int) -> Float[Array, "n_points"]:
        """
        Generate uniformly spaced distribution of points from 0 to 1.

        Args:
            n_points: Number of points to generate

        Returns:
            Array of uniformly spaced points from 0 to 1
        """
        return jnp.linspace(0, 1, n_points)

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def compute_arc_length_parametrization(
        upper_coords: Float[Array, "2 n_buffer"],
        lower_coords: Float[Array, "2 n_buffer"],
        n_upper_valid: int,
        n_lower_valid: int,
    ) -> Tuple[Float[Array, "n_buffer"], Float[Array, "n_buffer"]]:
        """
        Compute arc length parametrization for upper and lower surfaces.

        This function computes the cumulative arc length along each surface,
        which can be used for arc-length based repaneling.

        Args:
            upper_coords: Upper surface coordinates [x, y] with shape (2, n_buffer)
            lower_coords: Lower surface coordinates [x, y] with shape (2, n_buffer)
            n_upper_valid: Number of valid upper surface points
            n_lower_valid: Number of valid lower surface points

        Returns:
            Tuple of (upper_arc_lengths, lower_arc_lengths) normalized to [0, 1]
        """
        # Extract valid coordinates
        upper_x = upper_coords[0, :n_upper_valid]
        upper_y = upper_coords[1, :n_upper_valid]
        lower_x = lower_coords[0, :n_lower_valid]
        lower_y = lower_coords[1, :n_lower_valid]

        # Compute arc lengths for upper surface
        upper_dx = jnp.diff(upper_x)
        upper_dy = jnp.diff(upper_y)
        upper_ds = jnp.sqrt(upper_dx**2 + upper_dy**2)
        upper_s = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(upper_ds)])

        # Compute arc lengths for lower surface
        lower_dx = jnp.diff(lower_x)
        lower_dy = jnp.diff(lower_y)
        lower_ds = jnp.sqrt(lower_dx**2 + lower_dy**2)
        lower_s = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(lower_ds)])

        # Normalize to [0, 1]
        upper_s_norm = upper_s / upper_s[-1]
        lower_s_norm = lower_s / lower_s[-1]

        # Pad to buffer size (assume both arrays have the same buffer size)
        # This is a requirement for the function to work correctly with JIT
        buffer_size = upper_coords.shape[1]

        # Ensure lower coords have the same buffer size
        if lower_coords.shape[1] != buffer_size:
            raise ValueError(
                "Upper and lower coordinate arrays must have the same buffer size",
            )

        upper_s_padded = jnp.concatenate(
            [upper_s_norm, jnp.full(buffer_size - n_upper_valid, jnp.nan)],
        )
        lower_s_padded = jnp.concatenate(
            [lower_s_norm, jnp.full(buffer_size - n_lower_valid, jnp.nan)],
        )

        return upper_s_padded, lower_s_padded

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4))
    def repanel_surface_uniform(
        coords: Float[Array, "2 n_buffer"],
        arc_lengths: Float[Array, "n_buffer"],
        n_valid: int,
        n_new_points: int,
        distribution_type: str = "uniform",
    ) -> Float[Array, "2 n_new_points"]:
        """
        Repanel a single surface with uniform or cosine distribution.

        Args:
            coords: Surface coordinates [x, y] with shape (2, n_buffer)
            arc_lengths: Normalized arc length parameters for the surface
            n_valid: Number of valid points in the original surface
            n_new_points: Number of points in the repaneled surface
            distribution_type: "uniform" or "cosine" distribution

        Returns:
            New surface coordinates with shape (2, n_new_points)
        """
        # Extract valid coordinates and arc lengths
        x_valid = coords[0, :n_valid]
        y_valid = coords[1, :n_valid]
        s_valid = arc_lengths[:n_valid]

        # Generate new parameter distribution
        if distribution_type == "cosine":
            s_new = JaxAirfoilOps.generate_cosine_distribution(n_new_points)
        else:  # uniform
            s_new = JaxAirfoilOps.generate_uniform_distribution(n_new_points)

        # Interpolate coordinates at new parameter values
        x_new = jnp.interp(s_new, s_valid, x_valid)
        y_new = jnp.interp(s_new, s_valid, y_valid)

        # Stack into coordinate array
        new_coords = jnp.stack([x_new, y_new])

        return new_coords

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4))
    def repanel_surface_arc_length(
        coords: Float[Array, "2 n_buffer"],
        arc_lengths: Float[Array, "n_buffer"],
        n_valid: int,
        n_new_points: int,
        distribution_type: str = "uniform",
    ) -> Float[Array, "2 n_new_points"]:
        """
        Repanel a single surface based on arc length parametrization.

        This method redistributes points along the surface based on arc length,
        which can provide better point distribution for curved surfaces.

        Args:
            coords: Surface coordinates [x, y] with shape (2, n_buffer)
            arc_lengths: Normalized arc length parameters for the surface
            n_valid: Number of valid points in the original surface
            n_new_points: Number of points in the repaneled surface
            distribution_type: "uniform" or "cosine" distribution along arc length

        Returns:
            New surface coordinates with shape (2, n_new_points)
        """
        # This is the same as repanel_surface_uniform since we're already using arc length
        # The difference would be in how we compute the arc_lengths parameter
        return JaxAirfoilOps.repanel_surface_uniform(
            coords,
            arc_lengths,
            n_valid,
            n_new_points,
            distribution_type,
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4, 5))
    def repanel_airfoil_coordinates(
        upper_coords: Float[Array, "2 n_buffer"],
        lower_coords: Float[Array, "2 n_buffer"],
        n_upper_valid: int,
        n_lower_valid: int,
        n_new_points: int,
        distribution_type: str = "cosine",
    ) -> Tuple[Float[Array, "2 n_new_half"], Float[Array, "2 n_new_half"]]:
        """
        Repanel both upper and lower surfaces of an airfoil.

        Args:
            upper_coords: Upper surface coordinates [x, y] with shape (2, n_buffer)
            lower_coords: Lower surface coordinates [x, y] with shape (2, n_buffer)
            n_upper_valid: Number of valid upper surface points
            n_lower_valid: Number of valid lower surface points
            n_new_points: Total number of points for the repaneled airfoil
            distribution_type: "uniform" or "cosine" distribution

        Returns:
            Tuple of (new_upper_coords, new_lower_coords) each with n_new_points//2 points
        """
        # Compute arc length parametrization
        upper_arc_lengths, lower_arc_lengths = (
            JaxAirfoilOps.compute_arc_length_parametrization(
                upper_coords,
                lower_coords,
                n_upper_valid,
                n_lower_valid,
            )
        )

        # Calculate points per surface (split evenly)
        n_points_per_surface = n_new_points // 2

        # Repanel upper surface
        new_upper_coords = JaxAirfoilOps.repanel_surface_arc_length(
            upper_coords,
            upper_arc_lengths,
            n_upper_valid,
            n_points_per_surface,
            distribution_type,
        )

        # Repanel lower surface
        new_lower_coords = JaxAirfoilOps.repanel_surface_arc_length(
            lower_coords,
            lower_arc_lengths,
            n_lower_valid,
            n_points_per_surface,
            distribution_type,
        )

        return new_upper_coords, new_lower_coords

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2, 3))
    def resample_surface_points(
        coords: Float[Array, "2 n_buffer"],
        n_valid: int,
        n_new_points: int,
        distribution: str = "cosine",
    ) -> Float[Array, "2 n_new_points"]:
        """
        Resample surface points with specified distribution.

        Args:
            coords: Surface coordinates [x, y] with shape (2, n_buffer)
            n_valid: Number of valid points (static for JIT)
            n_new_points: Number of points to resample to (static for JIT)
            distribution: Distribution type ("cosine", "uniform", or "arc_length")

        Returns:
            Resampled coordinates with shape (2, n_new_points)
        """
        # Extract valid coordinates using dynamic slice
        valid_coords = jax.lax.dynamic_slice(coords, (0, 0), (2, n_valid))
        x_coords = valid_coords[0, :]
        y_coords = valid_coords[1, :]

        # Find coordinate range
        x_min = jnp.min(x_coords)
        x_max = jnp.max(x_coords)

        # Generate new x-coordinates based on distribution
        if distribution == "cosine":
            # Cosine spacing for better resolution near edges
            beta = jnp.linspace(0, jnp.pi, n_new_points)
            x_new = x_min + (x_max - x_min) * 0.5 * (1 - jnp.cos(beta))
        elif distribution == "uniform":
            # Uniform spacing
            x_new = jnp.linspace(x_min, x_max, n_new_points)
        else:  # arc_length
            # Arc-length based spacing
            x_new = JaxAirfoilOps._generate_arc_length_distribution(
                valid_coords,
                n_valid,
                n_new_points,
            )

        # Interpolate y-coordinates at new x positions
        y_new = JaxInterpolationEngine.linear_interpolate_1d(
            x_coords,
            y_coords,
            n_valid,
            x_new,
        )

        return jnp.stack([x_new, y_new])

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2))
    def _generate_arc_length_distribution(
        coords: Float[Array, "2 n_points"],
        n_valid: int,
        n_new_points: int,
    ) -> Float[Array, "n_new_points"]:
        """
        Generate x-coordinates based on arc-length parametrization.

        Args:
            coords: Surface coordinates [x, y]
            n_valid: Number of valid points
            n_new_points: Number of new points to generate

        Returns:
            X-coordinates distributed according to arc-length
        """
        # Use dynamic slice to extract valid coordinates
        valid_coords = jax.lax.dynamic_slice(coords, (0, 0), (2, n_valid))
        x_coords = valid_coords[0, :]
        y_coords = valid_coords[1, :]

        # Compute cumulative arc length
        dx = jnp.diff(x_coords)
        dy = jnp.diff(y_coords)
        ds = jnp.sqrt(dx**2 + dy**2)
        s = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(ds)])

        # Normalize arc length
        s_total = s[-1]
        s_normalized = s / s_total

        # Generate uniformly spaced arc length parameters
        s_new = jnp.linspace(0, 1, n_new_points)

        # Interpolate x-coordinates at new arc length positions
        x_new = JaxInterpolationEngine.linear_interpolate_1d(
            s_normalized,
            x_coords,
            n_valid,
            s_new,
        )

        return x_new

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4, 5))
    def batch_resample_surfaces(
        upper_coords_batch: Float[Array, "batch_size 2 n_buffer"],
        lower_coords_batch: Float[Array, "batch_size 2 n_buffer"],
        n_upper_valid_batch: Float[Array, "batch_size"],
        n_lower_valid_batch: Float[Array, "batch_size"],
        n_new_points: int,
        distribution: str = "cosine",
    ) -> Tuple[
        Float[Array, "batch_size 2 n_new_points"],
        Float[Array, "batch_size 2 n_new_points"],
    ]:
        """
        Batch resample multiple airfoil surfaces.

        Args:
            upper_coords_batch: Batch of upper surface coordinates
            lower_coords_batch: Batch of lower surface coordinates
            n_upper_valid_batch: Number of valid points for each upper surface
            n_lower_valid_batch: Number of valid points for each lower surface
            n_new_points: Number of points to resample to
            distribution: Distribution type

        Returns:
            Tuple of (resampled_upper_batch, resampled_lower_batch)
        """

        def resample_single_airfoil(args):
            upper_coords, lower_coords, n_upper_valid, n_lower_valid = args

            # Resample upper surface
            upper_resampled = JaxAirfoilOps.resample_surface_points(
                upper_coords,
                n_upper_valid,
                n_new_points,
                distribution,
            )

            # Resample lower surface
            lower_resampled = JaxAirfoilOps.resample_surface_points(
                lower_coords,
                n_lower_valid,
                n_new_points,
                distribution,
            )

            return upper_resampled, lower_resampled

        # Use vmap for batch processing
        upper_resampled_batch, lower_resampled_batch = jax.vmap(
            resample_single_airfoil,
        )(
            (
                upper_coords_batch,
                lower_coords_batch,
                n_upper_valid_batch,
                n_lower_valid_batch,
            ),
        )

        return upper_resampled_batch, lower_resampled_batch

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def compute_resampling_error(
        original_coords: Float[Array, "2 n_buffer"],
        resampled_coords: Float[Array, "2 n_new_points"],
        n_original_valid: int,
        n_resampled_points: int,
    ) -> Tuple[float, float]:
        """
        Compute resampling error metrics.

        Args:
            original_coords: Original surface coordinates
            resampled_coords: Resampled surface coordinates
            n_original_valid: Number of valid original points
            n_resampled_points: Number of resampled points

        Returns:
            Tuple of (max_error, rms_error)
        """
        # Extract valid original coordinates using dynamic slice
        orig_coords_valid = jax.lax.dynamic_slice(
            original_coords,
            (0, 0),
            (2, n_original_valid),
        )
        orig_x = orig_coords_valid[0, :]
        orig_y = orig_coords_valid[1, :]

        # Get resampled coordinates using dynamic slice
        resamp_coords_valid = jax.lax.dynamic_slice(
            resampled_coords,
            (0, 0),
            (2, n_resampled_points),
        )
        resamp_x = resamp_coords_valid[0, :]
        resamp_y = resamp_coords_valid[1, :]

        # Interpolate original surface at resampled x positions
        orig_y_interp = JaxInterpolationEngine.linear_interpolate_1d(
            orig_x,
            orig_y,
            n_original_valid,
            resamp_x,
        )

        # Compute errors
        errors = jnp.abs(resamp_y - orig_y_interp)
        max_error = jnp.max(errors)
        rms_error = jnp.sqrt(jnp.mean(errors**2))

        return max_error, rms_error

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4))
    def adaptive_resample_surface(
        coords: Float[Array, "2 n_buffer"],
        n_valid: int,
        target_points: int,
        max_error: float = 1e-4,
        max_iterations: int = 10,
    ) -> Float[Array, "2 target_points"]:
        """
        Adaptively resample surface to meet error tolerance.

        Args:
            coords: Surface coordinates [x, y]
            n_valid: Number of valid points
            target_points: Target number of points
            max_error: Maximum allowable error
            max_iterations: Maximum refinement iterations

        Returns:
            Adaptively resampled coordinates
        """

        def refine_iteration(carry):
            current_coords, current_n_points, iteration = carry

            # Check if we've reached target or max iterations
            should_continue = (iteration < max_iterations) & (
                current_n_points < target_points
            )

            def continue_refinement():
                # Increase point count
                new_n_points = jnp.minimum(
                    current_n_points + target_points // 4,
                    target_points,
                )

                # Resample with new point count
                new_coords = JaxAirfoilOps.resample_surface_points(
                    coords,
                    n_valid,
                    new_n_points,
                    "cosine",
                )

                # Compute error
                max_err, _ = JaxAirfoilOps.compute_resampling_error(
                    coords,
                    new_coords,
                    n_valid,
                    new_n_points,
                )

                # If error is acceptable or we've reached target, stop
                final_coords = jnp.where(
                    (max_err <= max_error) | (new_n_points >= target_points),
                    new_coords,
                    current_coords,
                )

                final_n_points = jnp.where(
                    (max_err <= max_error) | (new_n_points >= target_points),
                    new_n_points,
                    current_n_points,
                )

                return final_coords, final_n_points, iteration + 1

            def stop_refinement():
                return current_coords, current_n_points, iteration + 1

            return jax.lax.cond(should_continue, continue_refinement, stop_refinement)

        # Initial resampling
        initial_coords = JaxAirfoilOps.resample_surface_points(
            coords,
            n_valid,
            target_points // 2,
            "cosine",
        )

        # Iterative refinement
        final_coords, _, _ = jax.lax.while_loop(
            lambda carry: carry[2] < max_iterations,
            refine_iteration,
            (initial_coords, target_points // 2, 0),
        )

        # Ensure we have exactly target_points
        if final_coords.shape[1] != target_points:
            final_coords = JaxAirfoilOps.resample_surface_points(
                coords,
                n_valid,
                target_points,
                "cosine",
            )

        return final_coords
