"""
Functional Differentiable Geometry Generation

This module provides JAX-differentiable functions for generating VLM panel
geometry from design parameters (chord, span, twist, offsets).

The functions here mirror the computation in vehicle/surface.py but use
jax.numpy throughout, enabling automatic differentiation with respect to
design parameters like wingspan, chord distribution, and twist angles.

Example usage:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from ICARUS.aero.vlm.geometry import generate_panels, PanelGeometry
    >>>
    >>> # Define design parameters as JAX arrays
    >>> chord_dist = jnp.array([1.0, 0.8, 0.6])  # 3 spanwise stations
    >>> span_dist = jnp.array([0.0, 2.5, 5.0])
    >>> twist = jnp.array([0.0, 0.0, 0.0])
    >>> x_offsets = jnp.zeros(3)
    >>> z_offsets = jnp.zeros(3)
    >>>
    >>> # Pre-compute airfoil camber at chordwise stations (fixed)
    >>> camber_z = ...  # shape (M, N) - camber line z-values
    >>>
    >>> geom = generate_panels(chord_dist, span_dist, twist, x_offsets, z_offsets,
    >>>                        camber_z, chord_eta, R_MAT, origin)
    >>>
    >>> # Differentiate panel positions w.r.t. wingspan!
    >>> dgeom_dspan = jax.jacobian(generate_panels, argnums=1)(...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

if TYPE_CHECKING:
    from ICARUS.vehicle import WingSurface


class PanelGeometry:
    """Container for the output of panel geometry generation.

    All arrays are JAX arrays suitable for differentiation.
    """

    def __init__(
        self,
        grid: Array,
        panels: Array,
        panel_cps: Array,
        panel_normals: Array,
        N: int,
        M: int,
    ):
        self.grid = grid  # (N*M, 3)
        self.panels = panels  # ((N-1)*(M-1), 4, 3)
        self.panel_cps = panel_cps  # ((N-1)*(M-1), 3)
        self.panel_normals = panel_normals  # ((N-1)*(M-1), 3)
        self.N = N
        self.M = M
        self.num_panels = (N - 1) * (M - 1)


def grid_to_panels_jax(
    grid: Float[Array, "NM 3"],
    N: int,
    M: int,
) -> tuple[Array, Array, Array]:
    """Convert a flat grid array to panels, control points, and normals.

    This is a JAX-differentiable version of WingSurface.grid_to_panels().

    Args:
        grid: Flattened grid points of shape (N*M, 3)
        N: Number of spanwise stations
        M: Number of chordwise stations

    Returns:
        Tuple of (panels, control_points, control_normals)
        - panels: shape ((N-1)*(M-1), 4, 3)
        - control_points: shape ((N-1)*(M-1), 3)
        - control_normals: shape ((N-1)*(M-1), 3)
    """
    grid_3d = grid.reshape(N, M, 3)

    # Panel corner assembly (same indexing as surface.py)
    p0 = grid_3d[1:, : M - 1].reshape(-1, 3)  # i+1, j
    p1 = grid_3d[: N - 1, : M - 1].reshape(-1, 3)  # i, j
    p2 = grid_3d[: N - 1, 1:M].reshape(-1, 3)  # i, j+1
    p3 = grid_3d[1:, 1:M].reshape(-1, 3)  # i+1, j+1

    panels = jnp.stack([p0, p1, p2, p3], axis=1)  # (num_panels, 4, 3)

    # Control points at 3/4 chord, 1/2 span of each panel
    le_mid = (p0 + p1) / 2  # Leading edge midpoint
    te_mid = (p2 + p3) / 2  # Trailing edge midpoint

    cp_x = le_mid[:, 0:1] + 3 / 4 * (te_mid[:, 0:1] - le_mid[:, 0:1])
    cp_y = le_mid[:, 1:2] + 1 / 2 * (te_mid[:, 1:2] - le_mid[:, 1:2])
    cp_z = le_mid[:, 2:3] + 1 / 2 * (te_mid[:, 2:3] - le_mid[:, 2:3])
    control_points = jnp.concatenate([cp_x, cp_y, cp_z], axis=1)

    # Normal vectors via cross product of diagonals
    Ak = p0 - p2
    Bk = p1 - p3
    cross_prod = jnp.cross(Ak, Bk)
    norm = jnp.linalg.norm(cross_prod, axis=1, keepdims=True)
    control_normals = cross_prod / norm

    return panels, control_points, control_normals


def generate_grid(
    chord_dist: Float[Array, "N"],
    span_dist: Float[Array, "N"],
    twist_angles: Float[Array, "N"],
    x_offsets: Float[Array, "N"],
    z_offsets: Float[Array, "N"],
    camber_z: Float[Array, "M N"],
    chord_eta: Float[Array, "M"],
    R_MAT: Float[Array, "3 3"],
    origin: Float[Array, "3"],
    norm_factors: Float[Array, "N"],
) -> Array:
    """Generate a wing surface grid from design parameters.

    This is a JAX-differentiable version of the grid generation in
    WingSurface.define_grid(). All inputs are JAX arrays.

    Args:
        chord_dist: Chord lengths at each spanwise station (N,)
        span_dist: Spanwise positions (N,)
        twist_angles: Twist angles in radians at each station (N,)
        x_offsets: Chordwise offsets at each station (N,)
        z_offsets: Vertical offsets at each station (N,)
        camber_z: Pre-computed normalized camber z-values at (chord_eta, station)
                  shape (M, N). These are the airfoil camber line y-values
                  evaluated at chord_eta positions, normalized by chord.
        chord_eta: Chordwise parametric positions 0..1 of shape (M,)
        R_MAT: 3x3 rotation matrix for wing orientation
        origin: Wing origin in 3D space (3,)
        norm_factors: Airfoil normalization factors per station (N,)

    Returns:
        Grid array of shape (N*M, 3) suitable for grid_to_panels_jax()
    """
    N = chord_dist.shape[0]
    M = chord_eta.shape[0]

    # Build x-coordinates: chord_eta * chord + x_offset
    # Adjust for airfoil norm factors
    effective_chords = chord_dist * norm_factors
    xs = jnp.outer(chord_eta, effective_chords) + x_offsets[None, :]  # (M, N)

    # Build y-coordinates: constant across chord
    ys = jnp.tile(span_dist, (M, 1))  # (M, N)

    # Build z-coordinates: camber_z * chord + z_offset
    zs = camber_z * effective_chords[None, :] + z_offsets[None, :]  # (M, N)

    # Apply global rotation (R_MAT)
    coords_flat = jnp.stack(
        [xs.flatten(), ys.flatten(), zs.flatten()],
        axis=0,
    )  # (3, M*N)
    coords_rotated = jnp.matmul(R_MAT, coords_flat).reshape(3, M, N)

    # Find quarter-chord points for twist rotation
    c_4 = coords_rotated[:, 0, :] + (coords_rotated[:, -1, :] - coords_rotated[:, 0, :]) / 4

    # Apply twist rotation per spanwise station
    # Twist is a rotation in the xz plane about the quarter-chord point
    # Build all twist rotation matrices at once using vectorized operations
    cos_t = jnp.cos(twist_angles)  # (N,)
    sin_t = jnp.sin(twist_angles)  # (N,)
    zeros = jnp.zeros_like(cos_t)
    ones = jnp.ones_like(cos_t)

    # R_twist[k] is the 3x3 rotation matrix for station k
    # Shape: (N, 3, 3)
    R_twist = jnp.stack([
        jnp.stack([cos_t, zeros, sin_t], axis=-1),
        jnp.stack([zeros, ones, zeros], axis=-1),
        jnp.stack([-sin_t, zeros, cos_t], axis=-1),
    ], axis=-2)

    # Apply twist: for each station k, rotate coords[:, :, k] around c_4[:, k]
    # coords_rotated shape: (3, M, N), c_4 shape: (3, N)
    centered = coords_rotated - c_4[:, None, :]  # (3, M, N)

    # Rearrange for batched matmul: (N, 3, M)
    centered_T = centered.transpose(2, 0, 1)  # (N, 3, M)
    twisted_T = jnp.matmul(R_twist, centered_T)  # (N, 3, M)
    coords_twisted = twisted_T.transpose(1, 2, 0) + c_4[:, None, :]  # (3, M, N)

    # Add origin
    coords_final = coords_twisted + origin[:, None, None]

    # Transpose to (N, M, 3) and flatten to (N*M, 3)
    grid = coords_final.transpose(2, 1, 0).reshape(-1, 3)
    return grid


def generate_panels(
    chord_dist: Float[Array, "N"],
    span_dist: Float[Array, "N"],
    twist_angles: Float[Array, "N"],
    x_offsets: Float[Array, "N"],
    z_offsets: Float[Array, "N"],
    camber_z: Float[Array, "M N"],
    chord_eta: Float[Array, "M"],
    R_MAT: Float[Array, "3 3"],
    origin: Float[Array, "3"],
    norm_factors: Float[Array, "N"],
    N: int,
    M: int,
) -> PanelGeometry:
    """Generate complete panel geometry from design parameters.

    This is the main entry point for differentiable geometry generation.
    Combines grid generation and panel extraction into a single function.

    Args:
        chord_dist: Chord lengths at each spanwise station (N,)
        span_dist: Spanwise positions (N,)
        twist_angles: Twist angles in radians (N,)
        x_offsets: Chordwise offsets (N,)
        z_offsets: Vertical offsets (N,)
        camber_z: Pre-computed normalized camber values (M, N)
        chord_eta: Chordwise parametric positions (M,)
        R_MAT: 3x3 rotation matrix
        origin: Wing origin (3,)
        norm_factors: Airfoil normalization factors (N,)
        N: Number of spanwise stations (static)
        M: Number of chordwise stations (static)

    Returns:
        PanelGeometry with all arrays as JAX arrays
    """
    grid = generate_grid(
        chord_dist, span_dist, twist_angles,
        x_offsets, z_offsets, camber_z,
        chord_eta, R_MAT, origin, norm_factors,
    )

    panels, panel_cps, panel_normals = grid_to_panels_jax(grid, N, M)

    return PanelGeometry(
        grid=grid,
        panels=panels,
        panel_cps=panel_cps,
        panel_normals=panel_normals,
        N=N,
        M=M,
    )


def extract_geometry_params(surface: WingSurface) -> dict:
    """Extract JAX-compatible geometry parameters from a WingSurface.

    This converts the NumPy arrays in a WingSurface to JAX arrays,
    and pre-computes the airfoil camber values needed for differentiable
    geometry generation.

    Args:
        surface: A fully initialized WingSurface

    Returns:
        Dictionary of JAX arrays suitable for generate_panels()
    """
    import numpy as np

    M = surface.M

    # Chordwise parametric positions
    chord_eta = np.array(
        [surface.chord_discretization_function(int(i)) for i in range(M)],
    )
    chord_eta[-1] -= 1e-7
    chord_eta[0] += 1e-7

    N = surface.N

    # Pre-compute camber z-values for each spanwise station
    camber_z = np.zeros((M, N))
    norm_factors = np.ones(N)
    for j in range(N):
        airf_j = surface.airfoils[j]
        norm_factors[j] = airf_j.norm_factor
        camber_z[:, j] = airf_j.camber_line(chord_eta)

    return {
        "chord_dist": jnp.asarray(surface._chord_dist),
        "span_dist": jnp.asarray(surface._span_dist),
        "twist_angles": jnp.asarray(surface.twist_angles),
        "x_offsets": jnp.asarray(surface._xoffset_dist),
        "z_offsets": jnp.asarray(surface._zoffset_dist),
        "camber_z": jnp.asarray(camber_z),
        "chord_eta": jnp.asarray(chord_eta),
        "R_MAT": jnp.asarray(surface.R_MAT),
        "origin": jnp.asarray(surface._origin),
        "norm_factors": jnp.asarray(norm_factors),
        "N": N,
        "M": M,
    }
