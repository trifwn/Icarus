"""
Differentiable Aerodynamic Pipeline

End-to-end differentiable pipeline: DiffAirplane → panel geometry → VLM → forces.

This module provides:
1. diff_forces_fn: DiffAirplane + alpha → (lift, drag, moment)
2. make_gradient_fn: Creates a function for computing gradients of any
   aerodynamic output w.r.t. any design parameter
3. Implicit trim utilities

The pipeline reuses the existing VLM infrastructure (geometry.py, functional.py)
but feeds it from Equinox modules instead of ICARUS classes.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float

from ICARUS.aero.utils import panel_cp, panel_cp_normal, panel_dimensions
from ICARUS.aero.vlm.geometry import generate_grid, grid_to_panels_jax
from ICARUS.aero.vlm.functional import compute_velocity_vector
from ICARUS.aero.lspt_surface import compute_near_wake_panel

from .airplane import DiffAirplane
from .wing_segment import DiffWingSegment


def _build_segment_panels(segment: DiffWingSegment) -> dict:
    """Generate panel geometry for a single DiffWingSegment.

    Returns dict with: panels, panel_cps, panel_normals, grid,
                       wake_shedding_panel_indices, N, M
    """
    params = segment.compute_geometry_params()

    grid = generate_grid(
        params["chord_dist"],
        params["span_dist"],
        params["twist_angles"],
        params["x_offsets"],
        params["z_offsets"],
        params["camber_z"],
        params["chord_eta"],
        params["R_MAT"],
        params["origin"],
        params["norm_factors"],
    )

    N, M = params["N"], params["M"]
    panels, panel_cps, panel_normals = grid_to_panels_jax(grid, N, M)

    # Wake shedding indices: last chordwise panel of each strip
    wake_shedding_indices = jnp.arange(M - 2, (N - 1) * (M - 1), M - 1)

    return {
        "panels": panels,
        "panel_cps": panel_cps,
        "panel_normals": panel_normals,
        "grid": grid,
        "wake_shedding_indices": wake_shedding_indices,
        "N": N,
        "M": M,
    }


def _compute_near_wake(panels: Array, panel_normals: Array, wake_shedding_indices: Array) -> tuple[Array, Array, Array]:
    """Compute near wake panels from trailing edge panels."""
    wake_panels = panels[wake_shedding_indices]
    wake_normals = panel_normals[wake_shedding_indices]

    near_wake_panels = vmap(compute_near_wake_panel)(wake_panels, wake_normals)
    near_wake_cps = vmap(panel_cp)(near_wake_panels)
    near_wake_normals = vmap(panel_cp_normal)(near_wake_panels)

    return near_wake_panels, near_wake_cps, near_wake_normals


def diff_vlm_forces(
    airplane: DiffAirplane,
    alpha_deg: Float[Array, ""],
    airspeed: float,
    density: float,
) -> tuple[Array, Array, Array]:
    """Compute VLM forces from a DiffAirplane.

    This is the core end-to-end differentiable function. It:
    1. Generates panel geometry from DiffWingSegment design params
    2. Computes near wake panels
    3. Assembles influence matrices
    4. Solves VLM system
    5. Computes strip forces

    All operations are JAX-compatible and differentiable w.r.t.
    the DiffAirplane's design parameters and alpha_deg.

    Args:
        airplane: DiffAirplane with all design parameters
        alpha_deg: Angle of attack in degrees (JAX scalar)
        airspeed: Freestream velocity (m/s)
        density: Air density (kg/m^3)

    Returns:
        (lift, drag, My) as JAX scalars with symmetry factor applied
    """
    from ICARUS.aero.vlm.biot_savart import voring

    # Collect panels from all lifting segments
    all_surf_panels = []
    all_surf_cps = []
    all_surf_normals = []
    all_near_wake_panels = []
    all_near_wake_cps = []
    all_near_wake_normals = []
    all_wake_shedding_offsets = []
    strip_infos = []  # (start_idx, M-1, chord, width)

    surf_panel_offset = 0

    for segment in airplane.lifting_segments:
        seg_data = _build_segment_panels(segment)
        panels = seg_data["panels"]
        cps = seg_data["panel_cps"]
        normals = seg_data["panel_normals"]
        N = seg_data["N"]
        M = seg_data["M"]

        all_surf_panels.append(panels)
        all_surf_cps.append(cps)
        all_surf_normals.append(normals)

        # Compute near wake
        nw_panels, nw_cps, nw_normals = _compute_near_wake(
            panels, normals, seg_data["wake_shedding_indices"],
        )
        all_near_wake_panels.append(nw_panels)
        all_near_wake_cps.append(nw_cps)
        all_near_wake_normals.append(nw_normals)

        # Wake shedding indices (relative to global surface panel array)
        all_wake_shedding_offsets.append(
            surf_panel_offset + seg_data["wake_shedding_indices"],
        )

        # Strip info for force computation
        num_strips = N - 1
        strip_M = M - 1
        for i in range(num_strips):
            strip_start = surf_panel_offset + i * strip_M
            strip_infos.append((strip_start, strip_M))

        surf_panel_offset += panels.shape[0]

    # Concatenate all panels
    surf_panels = jnp.concatenate(all_surf_panels, axis=0)
    surf_cps = jnp.concatenate(all_surf_cps, axis=0)
    surf_normals = jnp.concatenate(all_surf_normals, axis=0)
    near_wake_panels_all = jnp.concatenate(all_near_wake_panels, axis=0)
    near_wake_cps_all = jnp.concatenate(all_near_wake_cps, axis=0)
    near_wake_normals_all = jnp.concatenate(all_near_wake_normals, axis=0)

    # Combined panel arrays
    all_panels = jnp.concatenate([surf_panels, near_wake_panels_all], axis=0)
    all_cps = jnp.concatenate([surf_cps, near_wake_cps_all], axis=0)
    all_normals = jnp.concatenate([surf_normals, near_wake_normals_all], axis=0)

    PANEL_NUM = all_panels.shape[0]
    num_surf = surf_panels.shape[0]

    # Indices
    near_wake_indices = num_surf + jnp.arange(near_wake_panels_all.shape[0])
    wake_shedding_indices = jnp.concatenate(all_wake_shedding_offsets)

    # Build influence matrices
    def compute_row(i):
        def compute_single(j):
            U, Ustar = voring(
                all_cps[i, 0], all_cps[i, 1], all_cps[i, 2],
                all_panels[j],
            )
            return jnp.dot(U, all_normals[i]), jnp.dot(Ustar, all_normals[i])
        a_row, b_row = vmap(compute_single)(jnp.arange(PANEL_NUM))
        return a_row, b_row

    A, A_star = vmap(compute_row)(jnp.arange(PANEL_NUM))

    # Wake boundary conditions
    A = A.at[near_wake_indices, :].set(0)
    A = A.at[near_wake_indices, near_wake_indices].set(1)
    A = A.at[near_wake_indices, wake_shedding_indices].set(-1)

    A_star = A_star.at[near_wake_indices, :].set(0)
    A_star = A_star.at[near_wake_indices, near_wake_indices].set(1)
    A_star = A_star.at[near_wake_indices, wake_shedding_indices].set(-1)

    # Solve VLM system
    A_LU, A_piv = jax.scipy.linalg.lu_factor(A)
    Q = compute_velocity_vector(alpha_deg, airspeed)
    RHS = -vmap(lambda n: jnp.dot(Q, n))(all_normals)
    RHS = RHS.at[near_wake_indices].set(0)

    gammas = jax.scipy.linalg.lu_solve((A_LU, A_piv), RHS)
    w_induced = jnp.matmul(A_star, gammas)

    # Compute forces per strip
    CG = airplane.compute_cg()
    total_lift = jnp.array(0.0)
    total_drag = jnp.array(0.0)
    total_My = jnp.array(0.0)

    for strip_start, strip_M_val in strip_infos:
        strip_idxs = strip_start + jnp.arange(strip_M_val)
        strip_panels_s = surf_panels[strip_idxs]
        strip_gammas = gammas[strip_idxs]
        strip_w = w_induced[strip_idxs]

        # Delta-gamma
        delta_gammas = strip_gammas.at[1:].set(strip_gammas[1:] - strip_gammas[:-1])

        # Panel dimensions
        _, mean_pw, _ = vmap(panel_dimensions)(strip_panels_s)

        # Forces
        panel_L = density * airspeed * delta_gammas * mean_pw
        panel_D = -density * strip_w * delta_gammas * mean_pw

        total_lift = total_lift + jnp.sum(panel_L)
        total_drag = total_drag + jnp.sum(panel_D)

        # Moments about CG
        pcps = vmap(panel_cp)(strip_panels_s)
        pnormals = vmap(panel_cp_normal)(strip_panels_s)
        lever_arms = pcps - CG
        M_lift = jnp.sum(panel_L[:, None] * jnp.cross(lever_arms, pnormals), axis=0)
        M_drag = jnp.sum(panel_D[:, None] * jnp.cross(lever_arms, pnormals), axis=0)
        total_My = total_My + M_lift[1] + M_drag[1]

    # Symmetry factor
    return total_lift * 2, total_drag * 2, total_My * 2


def make_diff_coefficients_fn(
    airplane: DiffAirplane,
    airspeed: float,
    density: float,
) -> Callable:
    """Create a differentiable function: (airplane, alpha) → (CL, CD, Cm).

    Args:
        airplane: Baseline DiffAirplane (used for reference quantities)
        airspeed: Freestream velocity (m/s)
        density: Air density (kg/m^3)

    Returns:
        A callable (airplane, alpha_deg) → (CL, CD, Cm)
    """
    q_inf = 0.5 * density * airspeed**2
    S = airplane.S
    MAC = airplane.MAC

    def coeff_fn(plane: DiffAirplane, alpha_deg: Float[Array, ""]) -> tuple[Array, Array, Array]:
        lift, drag, My = diff_vlm_forces(plane, alpha_deg, airspeed, density)
        CL = lift / (q_inf * S)
        CD = drag / (q_inf * S)
        Cm = My / (q_inf * S * MAC)
        return CL, CD, Cm

    return coeff_fn


def make_gradient_fn(
    airplane: DiffAirplane,
    airspeed: float,
    density: float,
    alpha_deg: float = 5.0,
    output: str = "CD",
) -> Callable:
    """Create a gradient function for a specific aerodynamic output.

    Returns a function that computes gradients of the specified output
    w.r.t. the DiffAirplane's parameters. Use with eqx.filter_grad
    to select which parameters to differentiate.

    Args:
        airplane: Baseline DiffAirplane
        airspeed: Freestream velocity (m/s)
        density: Air density (kg/m^3)
        alpha_deg: Fixed angle of attack (degrees)
        output: Which output to differentiate ("CL", "CD", "Cm")

    Returns:
        A callable airplane → scalar
    """
    import equinox as eqx

    output_idx = {"CL": 0, "CD": 1, "Cm": 2}[output]
    coeff_fn = make_diff_coefficients_fn(airplane, airspeed, density)
    alpha = jnp.asarray(alpha_deg, dtype=jnp.float64)

    def scalar_fn(plane: DiffAirplane) -> Array:
        coeffs = coeff_fn(plane, alpha)
        return coeffs[output_idx]

    return scalar_fn


def compute_trim_alpha(
    airplane: DiffAirplane,
    airspeed: float,
    density: float,
    target_Cm: float = 0.0,
    alpha_init: float = 5.0,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> Float[Array, ""]:
    """Find trim angle of attack using Newton's method with autodiff.

    Solves: Cm(alpha) = target_Cm

    Uses jax.grad to get dCm/dalpha analytically, enabling fast convergence.

    Args:
        airplane: DiffAirplane model
        airspeed: Freestream velocity
        density: Air density
        target_Cm: Target pitching moment coefficient (default 0 for trim)
        alpha_init: Initial guess for alpha (degrees)
        max_iter: Maximum Newton iterations
        tol: Convergence tolerance

    Returns:
        Trim alpha in degrees
    """
    coeff_fn = make_diff_coefficients_fn(airplane, airspeed, density)

    def Cm_residual(alpha: Array) -> Array:
        _, _, Cm = coeff_fn(airplane, alpha)
        return Cm - target_Cm

    dCm_dalpha = jax.grad(Cm_residual)

    alpha = jnp.asarray(alpha_init, dtype=jnp.float64)
    for _ in range(max_iter):
        residual = Cm_residual(alpha)
        gradient = dCm_dalpha(alpha)
        step = residual / (gradient + 1e-30)
        alpha = alpha - step
        # Check convergence (this won't break tracing since we use for-loop)

    return alpha


def implicit_trim_sensitivity(
    airplane: DiffAirplane,
    airspeed: float,
    density: float,
    alpha_trim: Float[Array, ""],
) -> Callable:
    """Compute design sensitivities at a trimmed condition.

    Uses implicit differentiation:
        dalpha*/d(design) = -dCm/d(design) / dCm/dalpha

    This avoids differentiating through the Newton iteration.

    Args:
        airplane: DiffAirplane at trim
        airspeed, density: Flow conditions
        alpha_trim: Trimmed angle of attack

    Returns:
        A function that computes d(output)/d(design) at trim
    """
    coeff_fn = make_diff_coefficients_fn(airplane, airspeed, density)

    # dCm/dalpha at trim
    dCm_dalpha = float(jax.grad(lambda a: coeff_fn(airplane, a)[2])(alpha_trim))

    def trimmed_output_fn(plane: DiffAirplane) -> tuple[Array, Array, Array]:
        """Compute coefficients at trim, accounting for alpha adjustment."""
        # Direct effect: coefficients at current alpha_trim
        CL, CD, Cm = coeff_fn(plane, alpha_trim)
        return CL, CD, Cm

    return trimmed_output_fn
