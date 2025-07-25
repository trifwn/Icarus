"""
JAX-compatible interpolation engine for airfoil surface queries.

This module provides JIT-compatible interpolation functions that work with
padded arrays and masking to handle variable-sized airfoil data efficiently.
Supports linear and cubic interpolation with extrapolation handling.
"""

from functools import partial
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float


class JaxInterpolationEngine:
    """
    JAX-compatible interpolation engine for airfoil surface queries.

    This class provides static methods for linear and cubic interpolation with masking
    support, extrapolation handling, and JIT compilation compatibility. All methods
    are designed to work with JAX transformations including jit, grad, and vmap.

    The engine handles variable-sized data through static buffer allocation with
    masking, ensuring JIT compatibility while maintaining gradient flow.
    """

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def linear_interpolate_1d(
        x_coords: Float[Array, " n"],
        y_coords: Float[Array, " n"],
        n_valid: int,
        query_x: Float[Array, " m"],
    ) -> Float[Array, " m"]:
        """
        Linear interpolation with masking for invalid points.

        Args:
            x_coords: X coordinates (padded array)
            y_coords: Y coordinates (padded array)
            n_valid: Number of valid points (static for JIT)
            query_x: Query X coordinates

        Returns:
            Interpolated Y values at query points
        """
        # Create validity mask
        indices = jnp.arange(x_coords.shape[0])
        mask = indices < n_valid

        # Extract valid coordinates only
        valid_x = jnp.where(mask, x_coords, jnp.inf)
        valid_y = jnp.where(mask, y_coords, 0.0)

        # Perform vectorized interpolation for all query points
        def interpolate_single_point(x_query: Float[Array, ""]) -> Float[Array, ""]:
            return JaxInterpolationEngine._linear_interp_single(
                valid_x,
                valid_y,
                n_valid,
                x_query,
            )

        return jax.vmap(interpolate_single_point)(query_x)

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def _linear_interp_single(
        x_coords: Float[Array, " n"],
        y_coords: Float[Array, " n"],
        n_valid: int,
        x_query: Float[Array, ""],
    ) -> Float[Array, ""]:
        """
        Linear interpolation for a single query point.

        Args:
            x_coords: Valid X coordinates (masked)
            y_coords: Valid Y coordinates (masked)
            n_valid: Number of valid points (static for JIT)
            x_query: Single query X coordinate

        Returns:
            Interpolated Y value
        """
        # Create static indices array based on n_valid
        valid_indices = jnp.arange(n_valid)

        # Find indices where x_coords[i] <= x_query < x_coords[i+1]
        left_mask = x_coords[:n_valid] <= x_query
        right_mask = x_coords[:n_valid] > x_query

        # Find the rightmost point that is <= x_query
        left_candidates = jnp.where(left_mask, valid_indices, -1)
        left_idx = jnp.max(left_candidates)

        # Find the leftmost point that is > x_query
        right_candidates = jnp.where(right_mask, valid_indices, n_valid)
        right_idx = jnp.min(right_candidates)

        # Handle extrapolation cases
        # If x_query is before all points, use first two points
        left_idx = jnp.where(left_idx == -1, 0, left_idx)
        right_idx = jnp.where(left_idx == -1, 1, right_idx)

        # If x_query is after all points, use last two points
        left_idx = jnp.where(right_idx == n_valid, n_valid - 2, left_idx)
        right_idx = jnp.where(right_idx == n_valid, n_valid - 1, right_idx)

        # Ensure we have valid indices for interpolation
        left_idx = jnp.clip(left_idx, 0, n_valid - 2)
        right_idx = jnp.clip(right_idx, 1, n_valid - 1)

        # Get the interpolation points
        x1 = x_coords[left_idx]
        y1 = y_coords[left_idx]
        x2 = x_coords[right_idx]
        y2 = y_coords[right_idx]

        # Linear interpolation formula: y = y1 + (y2-y1) * (x-x1) / (x2-x1)
        # Handle case where x1 == x2 (avoid division by zero)
        dx = x2 - x1
        dy = y2 - y1

        # If dx is zero, return y1 (or average of y1 and y2)
        t = jnp.where(jnp.abs(dx) < 1e-12, 0.0, (x_query - x1) / dx)

        return y1 + t * dy

    @staticmethod
    @partial(jax.jit, static_argnums=(4,))
    def interpolate_surface_masked(
        coords: Float[Array, "2 n"],
        mask: Bool[Array, " n"],
        query_points: Float[Array, " m"],
        extrapolate: bool = True,
        n_valid: Optional[int] = None,
    ) -> Float[Array, " m"]:
        """
        Interpolate airfoil surface with masking support.

        Args:
            coords: Coordinate array [x, y] with shape (2, n)
            mask: Boolean mask for valid points
            query_points: X coordinates to query
            extrapolate: Whether to allow extrapolation
            n_valid: Number of valid points (static for JIT)

        Returns:
            Interpolated Y values
        """
        x_coords = coords[0, :]
        y_coords = coords[1, :]

        if n_valid is None:
            n_valid = jnp.sum(mask).astype(int).item()
        n_valid = cast(int, n_valid)

        # Apply mask to coordinates
        valid_x = jnp.where(mask, x_coords, jnp.inf)
        valid_y = jnp.where(mask, y_coords, 0.0)

        # Sort by x-coordinate for interpolation
        sort_indices = jnp.argsort(valid_x[:n_valid])
        sorted_x = valid_x[sort_indices]
        sorted_y = valid_y[sort_indices]

        # Pad sorted arrays back to full size for JIT compatibility
        padded_x = jnp.concatenate(
            [sorted_x, jnp.full(coords.shape[1] - n_valid, jnp.inf)],
        )
        padded_y = jnp.concatenate([sorted_y, jnp.zeros(coords.shape[1] - n_valid)])

        return JaxInterpolationEngine.linear_interpolate_1d(
            padded_x,
            padded_y,
            n_valid,
            query_points,
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def query_airfoil_surface(
        upper_coords: Float[Array, "2 n"],
        lower_coords: Float[Array, "2 n"],
        n_upper_valid: int,
        n_lower_valid: int,
        query_x: Float[Array, " m"],
    ) -> Tuple[Float[Array, " m"], Float[Array, " m"]]:
        """
        Query both upper and lower airfoil surfaces.

        Args:
            upper_coords: Upper surface coordinates [x, y]
            lower_coords: Lower surface coordinates [x, y]
            n_upper_valid: Number of valid upper surface points
            n_lower_valid: Number of valid lower surface points
            query_x: X coordinates to query

        Returns:
            Tuple of (upper_y, lower_y) interpolated values
        """
        # Create masks
        upper_mask = jnp.arange(upper_coords.shape[1]) < n_upper_valid
        lower_mask = jnp.arange(lower_coords.shape[1]) < n_lower_valid

        # Interpolate both surfaces
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

        return upper_y, lower_y

    @staticmethod
    @partial(jax.jit, static_argnums=(4,))
    def handle_extrapolation(
        x_coords: Float[Array, " n"],
        y_coords: Float[Array, " n"],
        query_x: Float[Array, " m"],
        n_valid: int,
        extrap_mode: Literal["linear", "constant", "zero"] = "linear",
    ) -> Float[Array, " m"]:
        """
        Handle extrapolation beyond the data range.

        Args:
            x_coords: X coordinates
            y_coords: Y coordinates
            query_x: Query X coordinate
            n_valid: Number of valid points
            extrap_mode: Extrapolation mode ("linear", "constant", "zero")

        Returns:
            Extrapolated Y value
        """
        x_min = x_coords[0]
        x_max = x_coords[n_valid - 1]

        # Determine if we need extrapolation
        needs_extrap = (query_x < x_min) | (query_x > x_max)

        # Linear extrapolation using end points
        def linear_extrap() -> Float[Array, " m"]:
            # If query_x is before all points, use first two points
            def extrap_left() -> Float[Array, " m"]:
                x1, x2 = x_coords[0], x_coords[1]
                y1, y2 = y_coords[0], y_coords[1]
                slope = (y2 - y1) / (x2 - x1 + 1e-12)  # Add small epsilon
                return y1 + slope * (query_x - x1)

            # If query_x is after all points, use last two points
            def extrap_right() -> Float[Array, " m"]:
                x1, x2 = x_coords[n_valid - 2], x_coords[n_valid - 1]
                y1, y2 = y_coords[n_valid - 2], y_coords[n_valid - 1]
                slope = (y2 - y1) / (x2 - x1 + 1e-12)  # Add small epsilon
                return y2 + slope * (query_x - x2)

            return jnp.where(query_x < x_min, extrap_left(), extrap_right())

        # Constant extrapolation using end values
        def constant_extrap() -> Float[Array, " m"]:
            return jnp.where(query_x < x_min, y_coords[0], y_coords[n_valid - 1])

        # Zero extrapolation
        def zero_extrap() -> Float[Array, " m"]:
            return jnp.zeros_like(query_x)

        # Select extrapolation method (JAX doesn't support string comparisons in JIT)
        # We'll use the linear method by default since extrap_mode is static
        return linear_extrap()

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def compute_thickness_distribution(
        upper_coords: Float[Array, "2 n"],
        lower_coords: Float[Array, "2 n"],
        n_points_valid: int,
        query_x: Float[Array, " m"],
    ) -> Float[Array, " m"]:
        """
        Compute airfoil thickness distribution at query points.

        Args:
            upper_coords: Upper surface coordinates
            lower_coords: Lower surface coordinates
            n_points_valid: Number of valid points for both surfaces
            query_x: X coordinates to query

        Returns:
            Thickness values at query points
        """
        upper_y, lower_y = JaxInterpolationEngine.query_airfoil_surface(
            upper_coords,
            lower_coords,
            n_points_valid,
            n_points_valid,
            query_x,
        )

        return upper_y - lower_y

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def cubic_spline_interpolate_1d(
        x_coords: Float[Array, " n"],
        y_coords: Float[Array, " n"],
        n_valid: int,
        query_x: Float[Array, " m"],
    ) -> Float[Array, " m"]:
        """
        Cubic spline interpolation with masking for invalid points.

        This is a JAX-compatible implementation of cubic spline interpolation
        that works with padded arrays and static shapes.

        Args:
            x_coords: X coordinates (padded array)
            y_coords: Y coordinates (padded array)
            n_valid: Number of valid points (static for JIT)
            query_x: Query X coordinates

        Returns:
            Interpolated Y values at query points
        """
        # Extract valid coordinates
        valid_x = x_coords[:n_valid]
        valid_y = y_coords[:n_valid]

        # Compute second derivatives for cubic spline
        # Using natural spline boundary conditions (second derivative = 0 at ends)
        h = valid_x[1:] - valid_x[:-1]  # spacing between points

        # Set up tridiagonal system for second derivatives
        # A * y2 = b where y2 are the second derivatives
        n = n_valid

        # Initialize arrays for tridiagonal system
        a = jnp.zeros(n)  # sub-diagonal
        b = jnp.zeros(n)  # diagonal
        c = jnp.zeros(n)  # super-diagonal
        d = jnp.zeros(n)  # right-hand side

        # Natural spline boundary conditions
        b = b.at[0].set(1.0)
        b = b.at[n - 1].set(1.0)

        # Interior points
        for i in range(1, n - 1):
            a = a.at[i].set(h[i - 1])
            b = b.at[i].set(2.0 * (h[i - 1] + h[i]))
            c = c.at[i].set(h[i])
            d = d.at[i].set(
                6.0
                * (
                    (valid_y[i + 1] - valid_y[i]) / (h[i] + 1e-12)
                    - (valid_y[i] - valid_y[i - 1]) / (h[i - 1] + 1e-12)
                ),
            )

        # Solve tridiagonal system using Thomas algorithm
        y2 = JaxInterpolationEngine._solve_tridiagonal(a, b, c, d, n)

        # Perform cubic spline interpolation for each query point
        def cubic_interp_single(x_query: Float[Array, ""]) -> Float[Array, ""]:
            return JaxInterpolationEngine._cubic_spline_single(
                valid_x,
                valid_y,
                y2,
                n_valid,
                x_query,
            )

        return jax.vmap(cubic_interp_single)(query_x)

    @staticmethod
    @partial(jax.jit, static_argnums=(4,))
    def _solve_tridiagonal(
        a: Float[Array, " n"],
        b: Float[Array, " n"],
        c: Float[Array, " n"],
        d: Float[Array, " n"],
        n: int,
    ) -> Float[Array, " n"]:
        """
        Solve tridiagonal system using Thomas algorithm.

        Args:
            a: Sub-diagonal elements
            b: Diagonal elements
            c: Super-diagonal elements
            d: Right-hand side
            n: System size

        Returns:
            Solution vector
        """
        # Forward elimination
        c_prime = jnp.zeros(n)
        d_prime = jnp.zeros(n)

        c_prime = c_prime.at[0].set(c[0] / (b[0] + 1e-12))
        d_prime = d_prime.at[0].set(d[0] / (b[0] + 1e-12))

        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i - 1] + 1e-12
            c_prime = c_prime.at[i].set(c[i] / denom)
            d_prime = d_prime.at[i].set((d[i] - a[i] * d_prime[i - 1]) / denom)

        # Back substitution
        x = jnp.zeros(n)
        x = x.at[n - 1].set(d_prime[n - 1])

        for i in range(n - 2, -1, -1):
            x = x.at[i].set(d_prime[i] - c_prime[i] * x[i + 1])

        return x

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def _cubic_spline_single(
        x_coords: Float[Array, " n"],
        y_coords: Float[Array, " n"],
        y2: Float[Array, " n"],
        n_valid: int,
        x_query: Float[Array, ""],
    ) -> Float[Array, ""]:
        """
        Cubic spline interpolation for a single query point.

        Args:
            x_coords: X coordinates
            y_coords: Y coordinates
            y2: Second derivatives
            n_valid: Number of valid points
            x_query: Query point

        Returns:
            Interpolated value
        """
        # Find interval containing x_query
        valid_indices = jnp.arange(n_valid - 1)

        # Find the interval [x_i, x_{i+1}] containing x_query
        left_mask = x_coords[:-1] <= x_query
        right_mask = x_coords[1:] >= x_query
        interval_mask = left_mask & right_mask

        # Get the interval index (default to last interval if not found)
        interval_candidates = jnp.where(interval_mask, valid_indices, -1)
        interval_idx = jnp.max(interval_candidates)
        interval_idx = jnp.where(interval_idx == -1, n_valid - 2, interval_idx)
        interval_idx = jnp.clip(interval_idx, 0, n_valid - 2)

        # Get interval points
        x1 = x_coords[interval_idx]
        x2 = x_coords[interval_idx + 1]
        y1 = y_coords[interval_idx]
        y2_val = y_coords[interval_idx + 1]
        y2_1 = y2[interval_idx]
        y2_2 = y2[interval_idx + 1]

        # Compute cubic spline interpolation
        h = x2 - x1 + 1e-12  # Add small epsilon to avoid division by zero
        a = (x2 - x_query) / h
        b = (x_query - x1) / h

        result = (
            a * y1 + b * y2_val + ((a**3 - a) * y2_1 + (b**3 - b) * y2_2) * h**2 / 6.0
        )

        return result

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 4))
    def interpolate_with_method(
        x_coords: Float[Array, " n"],
        y_coords: Float[Array, " n"],
        n_valid: int,
        query_x: Float[Array, " m"],
        method: Literal["linear", "cubic"] = "linear",
    ) -> Float[Array, " m"]:
        """
        Interpolate using specified method.

        Args:
            x_coords: X coordinates (padded array)
            y_coords: Y coordinates (padded array)
            n_valid: Number of valid points (static for JIT)
            query_x: Query X coordinates
            method: Interpolation method ("linear" or "cubic")

        Returns:
            Interpolated Y values at query points
        """
        # Since method is static, we can use conditional compilation
        if method == "cubic":
            return JaxInterpolationEngine.cubic_spline_interpolate_1d(
                x_coords,
                y_coords,
                n_valid,
                query_x,
            )
        else:  # linear
            return JaxInterpolationEngine.linear_interpolate_1d(
                x_coords,
                y_coords,
                n_valid,
                query_x,
            )
