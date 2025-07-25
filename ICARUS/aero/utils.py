"""
Aerodynamic Utility Functions

This module contains utility functions for aerodynamic calculations
using JAX for high-performance computing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float


def panel_dimensions(panel: Float[Array, 2]) -> tuple[Float, Float, Float]:
    """Calculate the length and width of a panel."""
    dx = ((panel[3, 0] - panel[0, 0]) + (panel[2, 0] - panel[1, 0])) / 2.0

    dy = ((panel[3, 1] - panel[2, 1]) + (panel[0, 1] - panel[1, 1])) / 2.0

    dz = ((panel[3, 2] - panel[0, 2]) + (panel[2, 2] - panel[1, 2])) / 2.0
    return dx, dy, dz


def panel_cp(panel: Array) -> Array:
    """
    Calculate the control point of a panel using the quarter-chord rule.

    The control point is located at the quarter chord position (3/4 from leading edge)
    and at the mid-span position of the panel.

    Args:
        panel: Panel vertices as array of shape (4, 3) where:
               panel[0] = bottom-left (leading edge, tip)
               panel[1] = bottom-right (leading edge, root)
               panel[2] = top-right (trailing edge, root)
               panel[3] = top-left (trailing edge, tip)

    Returns:
        Array: Control point coordinates [x, y, z] of shape (3,)
    """
    # Calculate mid-span positions at leading and trailing edges
    leading_edge_mid = (panel[0] + panel[1]) / 2
    trailing_edge_mid = (panel[2] + panel[3]) / 2

    # Control point at quarter chord (3/4 from leading edge to trailing edge)
    control_point_x = leading_edge_mid[0] + 3 / 4 * (
        trailing_edge_mid[0] - leading_edge_mid[0]
    )
    control_point_y = leading_edge_mid[1] + 1 / 2 * (
        trailing_edge_mid[1] - leading_edge_mid[1]
    )
    control_point_z = leading_edge_mid[2] + 1 / 2 * (
        trailing_edge_mid[2] - leading_edge_mid[2]
    )

    control_point = jnp.array([control_point_x, control_point_y, control_point_z])
    return control_point


def panel_cp_normal(panel: Array) -> Array:
    """
    Calculate the unit normal vector of a panel.

    The normal vector is computed using the cross product of the panel diagonals
    and normalized to unit length. The direction follows the right-hand rule.

    Args:
        panel: Panel vertices as array of shape (4, 3) where:
               panel[0] = bottom-left (leading edge, tip)
               panel[1] = bottom-right (leading edge, root)
               panel[2] = top-right (trailing edge, root)
               panel[3] = top-left (trailing edge, tip)

    Returns:
        Array: Unit normal vector [nx, ny, nz] of shape (3,)
    """
    # Calculate diagonal vectors
    # Ak goes from bottom-left to top-right
    Ak = panel[0] - panel[2]
    # Bk goes from bottom-right to top-left
    Bk = panel[1] - panel[3]

    # Cross product gives normal vector
    cross_prod = jnp.cross(Ak, Bk)

    # Normalize to unit vector
    norm = jnp.linalg.norm(cross_prod)
    control_nj = cross_prod / norm

    return control_nj


def panel_area(panel: Array) -> Array:
    """
    Calculate the area of a panel.

    Args:
        panel: Panel vertices as array of shape (4, 3)

    Returns:
        Array: Panel area as scalar
    """
    # Calculate diagonal vectors
    Ak = panel[0] - panel[2]
    Bk = panel[1] - panel[3]

    # Area is half the magnitude of cross product
    cross_prod = jnp.cross(Ak, Bk)
    area = jnp.linalg.norm(cross_prod) / 2

    return area


def panel_center(panel: Array) -> Array:
    """
    Calculate the geometric center of a panel.

    Args:
        panel: Panel vertices as array of shape (4, 3)

    Returns:
        Array: Geometric center coordinates [x, y, z] of shape (3,)
    """
    return jnp.mean(panel, axis=0)


panel_cp_normal = jax.jit(panel_cp_normal)
panel_area = jax.jit(panel_area)
panel_center = jax.jit(panel_center)
panel_cp = jax.jit(panel_cp)
panel_dimensions = jax.jit(panel_dimensions)
