"""
Differentiable Aerodynamic Pipeline

End-to-end differentiable pipeline: DiffAirplane → panel geometry → VLM → forces.

This module provides:
1. diff_vlm_forces: DiffAirplane + alpha → (lift, drag, moment)
2. make_diff_coefficients_fn: Creates coefficient function
3. make_gradient_fn: Creates gradient function for design optimization
4. compute_trim_alpha: Newton-based trim solver with autodiff
5. implicit_trim_sensitivity: Implicit differentiation at trim

The pipeline reuses the core VLM solver from aero.vlm.functional
(vlm_forces_from_geometry) to avoid code duplication. The Diff*
modules handle geometry parameterization; the shared solver handles
influence matrices, linear system solve, and strip force accumulation.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float

from ICARUS.aero.utils import panel_cp, panel_cp_normal
from ICARUS.aero.vlm.geometry import generate_grid, grid_to_panels_jax
from ICARUS.aero.vlm.functional import vlm_forces_from_geometry
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


def _compute_near_wake(
    panels: Array,
    panel_normals: Array,
    wake_shedding_indices: Array,
) -> tuple[Array, Array, Array]:
    """Compute near wake panels from trailing edge panels."""
    wake_panels = panels[wake_shedding_indices]
    wake_normals = panel_normals[wake_shedding_indices]

    near_wake_panels = vmap(compute_near_wake_panel)(wake_panels, wake_normals)
    near_wake_cps = vmap(panel_cp)(near_wake_panels)
    near_wake_normals = vmap(panel_cp_normal)(near_wake_panels)

    return near_wake_panels, near_wake_cps, near_wake_normals


def _assemble_geometry(airplane: DiffAirplane) -> dict:
    """Assemble panel geometry from all lifting segments of a DiffAirplane.

    Generates panels, computes wake, and builds the index arrays needed
    by vlm_forces_from_geometry().

    Returns:
        Dict with all arrays needed by vlm_forces_from_geometry()
    """
    all_surf_panels = []
    all_surf_cps = []
    all_surf_normals = []
    all_near_wake_panels = []
    all_near_wake_cps = []
    all_near_wake_normals = []
    all_wake_shedding_offsets = []
    all_strip_offsets = []
    strip_M = None

    surf_panel_offset = 0

    for segment in airplane.lifting_segments:
        seg_data = _build_segment_panels(segment)
        panels = seg_data["panels"]
        N = seg_data["N"]
        M = seg_data["M"]

        all_surf_panels.append(panels)
        all_surf_cps.append(seg_data["panel_cps"])
        all_surf_normals.append(seg_data["panel_normals"])

        # Near wake
        nw_panels, nw_cps, nw_normals = _compute_near_wake(
            panels, seg_data["panel_normals"], seg_data["wake_shedding_indices"],
        )
        all_near_wake_panels.append(nw_panels)
        all_near_wake_cps.append(nw_cps)
        all_near_wake_normals.append(nw_normals)

        # Wake shedding indices (global)
        all_wake_shedding_offsets.append(
            surf_panel_offset + seg_data["wake_shedding_indices"],
        )

        # Strip panel offsets
        num_strips = N - 1
        current_strip_M = M - 1
        if strip_M is None:
            strip_M = current_strip_M

        for i in range(num_strips):
            all_strip_offsets.append(surf_panel_offset + i * current_strip_M)

        surf_panel_offset += panels.shape[0]

    # Concatenate
    surf_panels = jnp.concatenate(all_surf_panels, axis=0)
    surf_cps = jnp.concatenate(all_surf_cps, axis=0)
    surf_normals = jnp.concatenate(all_surf_normals, axis=0)
    nw_panels = jnp.concatenate(all_near_wake_panels, axis=0)
    nw_cps = jnp.concatenate(all_near_wake_cps, axis=0)
    nw_normals = jnp.concatenate(all_near_wake_normals, axis=0)

    num_surf = surf_panels.shape[0]
    panel_indices = jnp.arange(num_surf)
    near_wake_indices = num_surf + jnp.arange(nw_panels.shape[0])
    wake_shedding_indices = jnp.concatenate(all_wake_shedding_offsets)
    strip_panel_offsets = jnp.array(all_strip_offsets)

    # Strip chords/widths — not used in force computation (panel_dimensions
    # computes widths from geometry) but required by the shared API signature.
    # Pass dummy values since vlm_forces_from_geometry doesn't use them for
    # force calculation (it uses vmap(panel_dimensions) on the actual panels).
    num_strips = strip_panel_offsets.shape[0]
    strip_chords = jnp.ones(num_strips)
    strip_widths = jnp.ones(num_strips)

    return {
        "panels": surf_panels,
        "panel_cps": surf_cps,
        "panel_normals": surf_normals,
        "near_wake_panels": nw_panels,
        "near_wake_cps": nw_cps,
        "near_wake_normals": nw_normals,
        "panel_indices": panel_indices,
        "near_wake_indices": near_wake_indices,
        "wake_shedding_panel_indices": wake_shedding_indices,
        "strip_panel_offsets": strip_panel_offsets,
        "strip_M": strip_M,
        "strip_chords": strip_chords,
        "strip_widths": strip_widths,
    }


def diff_vlm_forces(
    airplane: DiffAirplane,
    alpha_deg: Float[Array, ""],
    airspeed: float,
    density: float,
) -> tuple[Array, Array, Array]:
    """Compute VLM forces from a DiffAirplane.

    End-to-end differentiable: design params → geometry → VLM → forces.
    Uses the shared VLM solver (vlm_forces_from_geometry) from
    aero.vlm.functional to avoid code duplication.

    Args:
        airplane: DiffAirplane with all design parameters
        alpha_deg: Angle of attack in degrees (JAX scalar)
        airspeed: Freestream velocity (m/s)
        density: Air density (kg/m^3)

    Returns:
        (lift, drag, My) as JAX scalars with symmetry factor applied
    """
    geom = _assemble_geometry(airplane)
    CG = airplane.compute_cg()

    return vlm_forces_from_geometry(
        panels=geom["panels"],
        panel_cps=geom["panel_cps"],
        panel_normals=geom["panel_normals"],
        alpha_deg=alpha_deg,
        airspeed=airspeed,
        density=density,
        CG=CG,
        near_wake_panels=geom["near_wake_panels"],
        near_wake_cps=geom["near_wake_cps"],
        near_wake_normals=geom["near_wake_normals"],
        panel_indices=geom["panel_indices"],
        near_wake_indices=geom["near_wake_indices"],
        wake_shedding_panel_indices=geom["wake_shedding_panel_indices"],
        strip_panel_offsets=geom["strip_panel_offsets"],
        strip_M=geom["strip_M"],
        strip_chords=geom["strip_chords"],
        strip_widths=geom["strip_widths"],
    )


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

    def coeff_fn(
        plane: DiffAirplane,
        alpha_deg: Float[Array, ""],
    ) -> tuple[Array, Array, Array]:
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
    """Create a scalar function airplane → output for use with eqx.filter_grad.

    Args:
        airplane: Baseline DiffAirplane
        airspeed: Freestream velocity (m/s)
        density: Air density (kg/m^3)
        alpha_deg: Fixed angle of attack (degrees)
        output: Which output to differentiate ("CL", "CD", "Cm")

    Returns:
        A callable airplane → scalar
    """
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
) -> Float[Array, ""]:
    """Find trim angle of attack using Newton's method with autodiff.

    Solves: Cm(alpha) = target_Cm

    Args:
        airplane: DiffAirplane model
        airspeed: Freestream velocity
        density: Air density
        target_Cm: Target pitching moment coefficient (default 0)
        alpha_init: Initial guess for alpha (degrees)
        max_iter: Maximum Newton iterations

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
        alpha = alpha - residual / (gradient + 1e-30)

    return alpha


def implicit_trim_sensitivity(
    airplane: DiffAirplane,
    airspeed: float,
    density: float,
    alpha_trim: Float[Array, ""],
) -> Callable:
    """Create a function for computing coefficients at a trimmed condition.

    Uses implicit differentiation:
        dalpha*/d(design) = -dCm/d(design) / dCm/dalpha

    This avoids differentiating through the Newton iteration.

    Args:
        airplane: DiffAirplane at trim
        airspeed, density: Flow conditions
        alpha_trim: Trimmed angle of attack

    Returns:
        A function (DiffAirplane) → (CL, CD, Cm) at trim alpha
    """
    coeff_fn = make_diff_coefficients_fn(airplane, airspeed, density)

    def trimmed_output_fn(plane: DiffAirplane) -> tuple[Array, Array, Array]:
        return coeff_fn(plane, alpha_trim)

    return trimmed_output_fn
