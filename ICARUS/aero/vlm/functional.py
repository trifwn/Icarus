"""
Functional Differentiable VLM API

This module provides pure functional interfaces to the VLM solver that are
fully compatible with JAX automatic differentiation (jax.grad, jax.jacobian).

The key idea: separate the static geometry (panels, normals, indices) from the
differentiable parameters (angle of attack, airspeed) so JAX can trace through
the entire computation graph.

Example usage:
    >>> from ICARUS.aero.vlm.functional import make_vlm_force_fn
    >>> import jax
    >>>
    >>> # Build geometry (not differentiable - done once)
    >>> lspt_plane = LSPT_Plane(airplane)
    >>>
    >>> # Create differentiable force function
    >>> force_fn = make_vlm_force_fn(lspt_plane, airspeed=20.0, density=1.225)
    >>>
    >>> # Evaluate forces
    >>> lift, drag, moment_y = force_fn(alpha=5.0)
    >>>
    >>> # Compute gradients!
    >>> dforces_dalpha = jax.jacobian(force_fn)(5.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float

from ICARUS.aero.utils import panel_cp
from ICARUS.aero.utils import panel_cp_normal
from ICARUS.aero.utils import panel_dimensions

from .geometry import PanelGeometry
from .geometry import extract_geometry_params
from .geometry import generate_panels
from .geometry import grid_to_panels_jax
from .matrices import get_LHS
from .matrices import get_RHS

if TYPE_CHECKING:
    from ICARUS.aero import LSPT_Plane
    from ICARUS.vehicle import WingSurface


@dataclass
class StripInfo:
    """Static geometry information for a single strip.

    All arrays are JAX arrays. This data is constant w.r.t. the
    differentiable parameters (alpha, airspeed, etc.).
    """

    panel_idxs: Array  # Global panel indices for this strip
    panels: Array  # Panel vertices (M, 4, 3)
    chord: float
    width: float
    mean_panel_width: Array  # (M,)


def extract_strip_info(lspt_plane: LSPT_Plane) -> list[StripInfo]:
    """Extract static strip geometry from an LSPT_Plane.

    This separates the static geometry data needed for force computation
    from the LSPT_Plane object, enabling pure functional computation.
    """
    strips: list[StripInfo] = []

    surf_panel_index = 0
    for surf in lspt_plane.surfaces:
        N = surf.N - 1  # Number of strips along span
        M = surf.M - 1  # Number of panels along chord

        for i in range(N):
            strip_idxs = surf_panel_index + (i * M) + jnp.arange(M)
            strip_panels = lspt_plane.panels[strip_idxs, :, :]
            chord = float((surf.chords[i] + surf.chords[i + 1]) / 2)
            width = float(surf.span_positions[i + 1] - surf.span_positions[i])

            _, mean_panel_width, _ = vmap(panel_dimensions)(strip_panels)

            strips.append(
                StripInfo(
                    panel_idxs=strip_idxs,
                    panels=strip_panels,
                    chord=chord,
                    width=width,
                    mean_panel_width=mean_panel_width,
                ),
            )
        surf_panel_index += surf.num_all_panels

    return strips


def compute_velocity_vector(alpha_deg: Array, airspeed: float) -> Array:
    """Compute freestream velocity vector from angle of attack.

    Args:
        alpha_deg: Angle of attack in degrees (JAX scalar)
        airspeed: Freestream velocity magnitude (m/s)

    Returns:
        Velocity vector [u, v, w] as JAX array
    """
    alpha_rad = jnp.deg2rad(alpha_deg)
    u = airspeed * jnp.cos(alpha_rad)
    v = jnp.array(0.0)
    w = airspeed * jnp.sin(alpha_rad)
    return jnp.array([u, v, w])


def compute_strip_potential_forces(
    gammas: Array,
    w_induced: Array,
    strip: StripInfo,
    density: float,
    airspeed: float,
    reference_point: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Compute potential lift, drag, and moments for a single strip.

    All operations use JAX and are fully differentiable.

    Returns:
        Tuple of (lift, drag, Mx, My, Mz) as JAX scalars
    """
    strip_gammas = gammas[strip.panel_idxs]
    strip_w = w_induced[strip.panel_idxs]

    # Compute delta-gamma (difference scheme)
    delta_gammas = strip_gammas.copy()
    delta_gammas = delta_gammas.at[1:].set(strip_gammas[1:] - strip_gammas[:-1])

    # Panel forces
    panel_L = density * airspeed * delta_gammas * strip.mean_panel_width
    panel_D = -density * strip_w * delta_gammas * strip.mean_panel_width

    # Total strip forces
    lift = jnp.sum(panel_L)
    drag = jnp.sum(panel_D)

    # Moments about reference point
    panel_cps = vmap(panel_cp)(strip.panels)
    panel_normals = vmap(panel_cp_normal)(strip.panels)
    lever_arms = panel_cps - reference_point

    M_lift = jnp.sum(
        panel_L[:, None] * jnp.cross(lever_arms, panel_normals),
        axis=0,
    )
    M_drag = jnp.sum(
        panel_D[:, None] * jnp.cross(lever_arms, panel_normals),
        axis=0,
    )

    M = M_lift + M_drag
    return lift, drag, M[0], M[1], M[2]


def vlm_forces_at_alpha(
    alpha_deg: Array,
    lspt_plane: LSPT_Plane,
    A_LU: Array,
    A_piv: Array,
    A_star: Array,
    strips: list[StripInfo],
    airspeed: float,
    density: float,
) -> tuple[Array, Array, Array]:
    """Compute VLM potential forces at a given angle of attack.

    This is a pure function of alpha_deg (given pre-computed geometry),
    and is fully differentiable via jax.grad or jax.jacobian.

    Args:
        alpha_deg: Angle of attack in degrees (JAX scalar or float)
        lspt_plane: Pre-computed LSPT_Plane geometry
        A_LU: Pre-factored LHS matrix (LU decomposition)
        A_piv: Pivot indices from LU decomposition
        A_star: Trailing-edge influence matrix
        strips: Pre-extracted strip geometry info
        airspeed: Freestream velocity (m/s)
        density: Air density (kg/m^3)

    Returns:
        Tuple of (total_lift, total_drag, total_My) as JAX scalars.
        Forces include the x2 symmetry factor.
    """
    # Compute velocity vector from alpha
    Q = compute_velocity_vector(alpha_deg, airspeed)

    # Solve VLM system
    RHS = get_RHS(lspt_plane, Q)
    gammas = jax.scipy.linalg.lu_solve((A_LU, A_piv), RHS)
    w_induced = jnp.matmul(A_star, gammas)

    # Compute forces per strip and accumulate
    reference_point = jnp.array(lspt_plane.CG)
    total_lift = jnp.array(0.0)
    total_drag = jnp.array(0.0)
    total_Mx = jnp.array(0.0)
    total_My = jnp.array(0.0)
    total_Mz = jnp.array(0.0)

    for strip in strips:
        lift, drag, Mx, My, Mz = compute_strip_potential_forces(
            gammas,
            w_induced,
            strip,
            density,
            airspeed,
            reference_point,
        )
        total_lift = total_lift + lift
        total_drag = total_drag + drag
        total_Mx = total_Mx + Mx
        total_My = total_My + My
        total_Mz = total_Mz + Mz

    # Apply symmetry factor (factor of 2 for symmetric wings)
    total_lift = total_lift * 2
    total_drag = total_drag * 2
    total_My = total_My * 2

    return total_lift, total_drag, total_My


def make_vlm_force_fn(
    lspt_plane: LSPT_Plane,
    airspeed: float,
    density: float,
):
    """Create a differentiable function: alpha_deg -> (lift, drag, moment_y).

    This function pre-computes and caches all geometry-dependent quantities
    (LHS factorization, strip info), returning a closure that is a pure
    function of alpha suitable for JAX differentiation.

    Args:
        lspt_plane: LSPT_Plane with fully assembled geometry
        airspeed: Freestream velocity (m/s)
        density: Air density (kg/m^3)

    Returns:
        A callable: alpha_deg -> (lift, drag, My)
        where alpha_deg is a JAX scalar (angle of attack in degrees),
        and the returned values are JAX scalars.

    Example:
        >>> force_fn = make_vlm_force_fn(lspt_plane, airspeed=20.0, density=1.225)
        >>> lift, drag, My = force_fn(5.0)
        >>> # Gradient of lift w.r.t. alpha
        >>> dlift_dalpha = jax.grad(lambda a: force_fn(a)[0])(5.0)
        >>> # Full Jacobian
        >>> J = jax.jacobian(force_fn)(5.0)
    """
    # Pre-compute geometry
    A, A_star = get_LHS(lspt_plane)
    A_LU, A_piv = jax.scipy.linalg.lu_factor(A)
    strips = extract_strip_info(lspt_plane)

    def force_fn(alpha_deg: Float) -> tuple[Array, Array, Array]:
        return vlm_forces_at_alpha(
            jnp.asarray(alpha_deg, dtype=jnp.float64),
            lspt_plane,
            A_LU,
            A_piv,
            A_star,
            strips,
            airspeed,
            density,
        )

    return force_fn


def make_vlm_coefficients_fn(
    lspt_plane: LSPT_Plane,
    airspeed: float,
    density: float,
):
    """Create a differentiable function: alpha_deg -> (CL, CD, Cm).

    Like make_vlm_force_fn but returns non-dimensional coefficients.

    Args:
        lspt_plane: LSPT_Plane with fully assembled geometry
        airspeed: Freestream velocity (m/s)
        density: Air density (kg/m^3)

    Returns:
        A callable: alpha_deg -> (CL, CD, Cm)
    """
    force_fn = make_vlm_force_fn(lspt_plane, airspeed, density)
    q_inf = 0.5 * density * airspeed**2
    S = lspt_plane.S
    MAC = lspt_plane.MAC

    def coeff_fn(alpha_deg: Float) -> tuple[Array, Array, Array]:
        lift, drag, My = force_fn(alpha_deg)
        CL = lift / (q_inf * S)
        CD = drag / (q_inf * S)
        Cm = My / (q_inf * S * MAC)
        return CL, CD, Cm

    return coeff_fn


# =============================================================================
# End-to-end differentiable pipeline: design params -> geometry -> VLM -> forces
# =============================================================================


def vlm_forces_from_geometry(
    panels: Array,
    panel_cps: Array,
    panel_normals: Array,
    alpha_deg: Array,
    airspeed: float,
    density: float,
    CG: Array,
    near_wake_panels: Array,
    near_wake_cps: Array,
    near_wake_normals: Array,
    panel_indices: Array,
    near_wake_indices: Array,
    wake_shedding_panel_indices: Array,
    strip_panel_offsets: Array,
    strip_M: int,
    strip_chords: Array,
    strip_widths: Array,
) -> tuple[Array, Array, Array]:
    """Compute VLM forces from raw panel geometry arrays.

    This is the core differentiable function that takes panel arrays directly
    (rather than an LSPT_Plane object), enabling differentiation through
    geometry generation.

    Args:
        panels: Surface panels (num_surf_panels, 4, 3)
        panel_cps: Surface control points (num_surf_panels, 3)
        panel_normals: Surface normals (num_surf_panels, 3)
        alpha_deg: Angle of attack in degrees (JAX scalar)
        airspeed: Freestream velocity (m/s)
        density: Air density (kg/m^3)
        CG: Center of gravity (3,)
        near_wake_panels: Near wake panels (num_wake, 4, 3)
        near_wake_cps: Near wake control points (num_wake, 3)
        near_wake_normals: Near wake normals (num_wake, 3)
        panel_indices: Indices of surface panels in combined array
        near_wake_indices: Indices of near wake panels in combined array
        wake_shedding_panel_indices: Indices of panels that shed wake
        strip_panel_offsets: Start index of each strip in surface panels (num_strips,)
        strip_M: Number of chordwise panels per strip
        strip_chords: Chord length per strip (num_strips,)
        strip_widths: Width per strip (num_strips,)

    Returns:
        Tuple of (lift, drag, My) as JAX scalars with x2 symmetry factor
    """
    from .biot_savart import voring

    # Assemble combined panel arrays (surface + wake)
    all_panels = jnp.concatenate([panels, near_wake_panels], axis=0)
    all_cps = jnp.concatenate([panel_cps, near_wake_cps], axis=0)
    all_normals = jnp.concatenate([panel_normals, near_wake_normals], axis=0)

    PANEL_NUM = all_panels.shape[0]

    # Build influence matrices using vectorized voring
    def compute_row_contribution(i):
        def compute_single(j):
            U, Ustar = voring(
                all_cps[i, 0], all_cps[i, 1], all_cps[i, 2],
                all_panels[j],
            )
            return jnp.dot(U, all_normals[i]), jnp.dot(Ustar, all_normals[i])

        a_row, b_row = vmap(compute_single)(jnp.arange(PANEL_NUM))
        return a_row, b_row

    A, A_star = vmap(compute_row_contribution)(jnp.arange(PANEL_NUM))

    # Apply wake kinematic boundary conditions
    A = A.at[near_wake_indices, :].set(0)
    A = A.at[near_wake_indices, near_wake_indices].set(1)
    A = A.at[near_wake_indices, wake_shedding_panel_indices].set(-1)

    A_star = A_star.at[near_wake_indices, :].set(0)
    A_star = A_star.at[near_wake_indices, near_wake_indices].set(1)
    A_star = A_star.at[near_wake_indices, wake_shedding_panel_indices].set(-1)

    # LU factorize and solve
    A_LU, A_piv = jax.scipy.linalg.lu_factor(A)

    # Compute RHS from velocity vector
    Q = compute_velocity_vector(alpha_deg, airspeed)
    RHS = -vmap(lambda n: jnp.dot(Q, n))(all_normals)
    RHS = RHS.at[near_wake_indices].set(0)

    # Solve for circulations
    gammas = jax.scipy.linalg.lu_solve((A_LU, A_piv), RHS)
    w_induced = jnp.matmul(A_star, gammas)

    # Compute forces per strip
    num_strips = strip_panel_offsets.shape[0]
    total_lift = jnp.array(0.0)
    total_drag = jnp.array(0.0)
    total_Mx = jnp.array(0.0)
    total_My = jnp.array(0.0)
    total_Mz = jnp.array(0.0)

    for s in range(num_strips):
        strip_idxs = strip_panel_offsets[s] + jnp.arange(strip_M)
        strip_panels_s = panels[strip_idxs]
        strip_gammas = gammas[strip_idxs]
        strip_w = w_induced[strip_idxs]

        # Delta-gamma
        delta_gammas = strip_gammas.copy()
        delta_gammas = delta_gammas.at[1:].set(strip_gammas[1:] - strip_gammas[:-1])

        # Panel dimensions
        _, mean_pw, _ = vmap(panel_dimensions)(strip_panels_s)

        # Forces
        panel_L = density * airspeed * delta_gammas * mean_pw
        panel_D = -density * strip_w * delta_gammas * mean_pw

        total_lift = total_lift + jnp.sum(panel_L)
        total_drag = total_drag + jnp.sum(panel_D)

        # Moments
        pcps = vmap(panel_cp)(strip_panels_s)
        pnormals = vmap(panel_cp_normal)(strip_panels_s)
        lever_arms = pcps - CG
        M_lift = jnp.sum(panel_L[:, None] * jnp.cross(lever_arms, pnormals), axis=0)
        M_drag = jnp.sum(panel_D[:, None] * jnp.cross(lever_arms, pnormals), axis=0)
        M = M_lift + M_drag
        total_Mx = total_Mx + M[0]
        total_My = total_My + M[1]
        total_Mz = total_Mz + M[2]

    # Symmetry factor
    return total_lift * 2, total_drag * 2, total_My * 2


def make_design_sensitivity_fn(
    surface: WingSurface,
    airspeed: float,
    density: float,
    alpha_deg: float = 5.0,
):
    """Create a function for computing design sensitivities.

    Returns a function that takes design parameters (chord_dist, span_dist,
    twist_angles) and returns aerodynamic forces. This function is
    differentiable w.r.t. all design parameters via jax.grad or jax.jacobian.

    Args:
        surface: WingSurface to extract static geometry config from
        airspeed: Freestream velocity (m/s)
        density: Air density (kg/m^3)
        alpha_deg: Angle of attack in degrees (default 5.0)

    Returns:
        A callable: (chord_dist, span_dist, twist_angles) -> (lift, drag, My)

    Example:
        >>> fn = make_design_sensitivity_fn(wing, airspeed=20.0, density=1.225)
        >>> lift, drag, My = fn(chord_dist, span_dist, twist_angles)
        >>> # Sensitivity of drag to chord distribution
        >>> ddrag_dchord = jax.grad(lambda c: fn(c, span_dist, twist_angles)[1])(chord_dist)
    """
    params = extract_geometry_params(surface)
    N = params["N"]
    M = params["M"]

    # Static params (not differentiated)
    camber_z = params["camber_z"]
    chord_eta = params["chord_eta"]
    R_MAT = params["R_MAT"]
    origin = params["origin"]
    norm_factors = params["norm_factors"]
    x_offsets_default = params["x_offsets"]
    z_offsets_default = params["z_offsets"]
    CG = jnp.array(origin)  # Approximate CG at origin

    # Near-wake geometry computation (static, pre-computed from baseline)
    from ICARUS.aero.lspt_surface import LSPTSurface
    from ICARUS.aero.lspt_surface import compute_near_wake_panel

    # We need the wake geometry. For now, compute it from the baseline surface
    # and treat it as static (wake doesn't change with design params in linear VLM)
    lspt_surf = LSPTSurface(surface=surface, wing_id=0)
    lspt_surf.add_near_wake_panels()

    near_wake_panels = lspt_surf.near_wake_panels
    near_wake_cps = lspt_surf.near_wake_panel_cps
    near_wake_normals = lspt_surf.near_wake_panel_normals
    wake_shedding_indices = lspt_surf.wake_shedding_panel_indices

    num_surf_panels = (N - 1) * (M - 1)
    num_wake_panels = near_wake_panels.shape[0]

    panel_indices = jnp.arange(num_surf_panels)
    near_wake_idx = num_surf_panels + jnp.arange(num_wake_panels)

    # Strip info
    num_strips = N - 1
    strip_M_val = M - 1
    strip_panel_offsets = jnp.arange(num_strips) * strip_M_val

    # Pre-compute strip chords and widths from surface
    strip_chords = jnp.asarray(
        [(surface._chord_dist[i] + surface._chord_dist[i + 1]) / 2 for i in range(num_strips)],
    )
    strip_widths = jnp.asarray(
        [surface._span_dist[i + 1] - surface._span_dist[i] for i in range(num_strips)],
    )

    def sensitivity_fn(
        chord_dist: Array,
        span_dist: Array,
        twist_angles: Array,
        alpha: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        """Compute forces from design parameters.

        Args:
            chord_dist: Chord distribution (N,)
            span_dist: Span positions (N,)
            twist_angles: Twist angles in radians (N,)
            alpha: Optional angle of attack override (degrees)

        Returns:
            Tuple of (lift, drag, My) as JAX scalars
        """
        alpha_val = jnp.asarray(alpha_deg if alpha is None else alpha, dtype=jnp.float64)

        # Generate panels from design params
        geom = generate_panels(
            chord_dist, span_dist, twist_angles,
            x_offsets_default, z_offsets_default, camber_z,
            chord_eta, R_MAT, origin, norm_factors,
            N=N, M=M,
        )

        return vlm_forces_from_geometry(
            panels=geom.panels,
            panel_cps=geom.panel_cps,
            panel_normals=geom.panel_normals,
            alpha_deg=alpha_val,
            airspeed=airspeed,
            density=density,
            CG=CG,
            near_wake_panels=near_wake_panels,
            near_wake_cps=near_wake_cps,
            near_wake_normals=near_wake_normals,
            panel_indices=panel_indices,
            near_wake_indices=near_wake_idx,
            wake_shedding_panel_indices=wake_shedding_indices,
            strip_panel_offsets=strip_panel_offsets,
            strip_M=strip_M_val,
            strip_chords=strip_chords,
            strip_widths=strip_widths,
        )

    return sensitivity_fn
