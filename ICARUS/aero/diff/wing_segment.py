"""
Differentiable Wing Segment Module

Equinox-based replacement for ICARUS.vehicle.surface.WingSurface.
Contains all the design parameters needed for panel geometry generation
as differentiable fields.

The key insight: instead of storing panels (which are derived quantities),
we store the design parameters and compute panels on-the-fly through the
differentiable geometry generator in aero.vlm.geometry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .airfoil_camber import DiffAirfoilCamber, DiffNACA4Camber, morph_camber
from .control_surface import DiffControlSurface, apply_controls_to_camber
from .mass import DiffMass

if TYPE_CHECKING:
    from ICARUS.vehicle import WingSurface


class DiffWingSegment(eqx.Module):
    """Differentiable wing segment (lifting surface).

    This module stores the design parameters of a wing segment in a form
    suitable for JAX differentiation. Panel geometry is NOT stored; instead,
    it is computed on-the-fly by the differentiable pipeline.

    Dynamic fields (differentiable):
        chord_dist: Chord lengths at each spanwise station (N,)
        span_dist: Spanwise positions (N,)
        twist_angles: Twist angles in radians at each station (N,)
        x_offsets: Chordwise (sweep) offsets at each station (N,)
        z_offsets: Vertical (dihedral) offsets at each station (N,)
        structural_mass: Structural mass of the segment

    Static fields:
        name: Wing segment identifier
        N: Number of spanwise stations
        M: Number of chordwise stations
        R_MAT: 3x3 rotation matrix for wing orientation
        origin: Wing origin in 3D space (3,)
        chord_eta: Chordwise parametric positions [0, 1] of shape (M,)
        camber_z: Pre-evaluated camber values (M, N) - from airfoils
        norm_factors: Airfoil normalization factors per station (N,)
        root_camber: Root airfoil camber representation
        tip_camber: Tip airfoil camber representation
        controls: List of differentiable control surfaces
        is_lifting: Whether this surface generates lift
        is_symmetric_y: Y-symmetry flag
    """

    # Dynamic (differentiable) design parameters
    chord_dist: Float[Array, "N"]
    span_dist: Float[Array, "N"]
    twist_angles: Float[Array, "N"]
    x_offsets: Float[Array, "N"]
    z_offsets: Float[Array, "N"]
    structural_mass: Float[Array, ""]

    # Geometry configuration (non-differentiable but part of pytree for JAX compat)
    R_MAT: Float[Array, "3 3"]
    origin: Float[Array, "3"]
    chord_eta: Float[Array, "M"]
    camber_z: Float[Array, "M N"]
    norm_factors: Float[Array, "N"]

    # Static (non-array) fields
    name: str = eqx.field(static=True)
    N: int = eqx.field(static=True)
    M: int = eqx.field(static=True)

    # Airfoil and control surface modules (dynamic pytree nodes — not
    # typically differentiated, but keeping them dynamic avoids the
    # "JAX array set as static" warning since they contain JAX arrays)
    root_camber: DiffAirfoilCamber
    tip_camber: DiffAirfoilCamber
    controls: list[DiffControlSurface]

    # Flags
    is_lifting: bool = eqx.field(static=True)
    is_symmetric_y: bool = eqx.field(static=True)

    @property
    def num_panels(self) -> int:
        return (self.N - 1) * (self.M - 1)

    @property
    def span(self) -> float:
        """Total span (accounts for Y-symmetry)."""
        half_span = float(jnp.abs(self.span_dist[-1] - self.span_dist[0]))
        return half_span * 2 if self.is_symmetric_y else half_span

    @property
    def area(self) -> float:
        """Planform area (accounts for Y-symmetry)."""
        # Trapezoidal integration of chord over span
        s = float(jnp.trapezoid(self.chord_dist, self.span_dist))
        return s * 2 if self.is_symmetric_y else s

    @property
    def mean_aerodynamic_chord(self) -> float:
        """Mean aerodynamic chord."""
        c = self.chord_dist
        s = self.span_dist
        c_sq = c ** 2
        return float(jnp.trapezoid(c_sq, s) / jnp.trapezoid(c, s))

    def compute_geometry_params(self) -> dict:
        """Extract parameters for the VLM geometry generator.

        Returns a dict compatible with aero.vlm.geometry.generate_panels().
        Control surface deflections are applied to the camber here.
        """
        camber = self.camber_z

        # Apply control surface deflections to camber
        if self.controls:
            total_span = float(jnp.abs(self.span_dist[-1] - self.span_dist[0]))
            if total_span > 0:
                span_fractions = (self.span_dist - self.span_dist[0]) / total_span
            else:
                span_fractions = jnp.zeros_like(self.span_dist)
            camber = apply_controls_to_camber(
                camber, self.chord_eta, self.controls, span_fractions,
            )

        return {
            "chord_dist": self.chord_dist,
            "span_dist": self.span_dist,
            "twist_angles": self.twist_angles,
            "x_offsets": self.x_offsets,
            "z_offsets": self.z_offsets,
            "camber_z": camber,
            "chord_eta": self.chord_eta,
            "R_MAT": self.R_MAT,
            "origin": self.origin,
            "norm_factors": self.norm_factors,
            "N": self.N,
            "M": self.M,
        }


def from_wing_surface(surface: WingSurface) -> DiffWingSegment:
    """Convert an ICARUS WingSurface to a DiffWingSegment.

    Extracts all design parameters and pre-evaluates airfoil camber
    at the chordwise stations.

    Args:
        surface: A fully initialized WingSurface

    Returns:
        DiffWingSegment with all parameters as JAX arrays
    """
    import numpy as np
    from .airfoil_camber import from_airfoil
    from .control_surface import from_control_surface
    from ICARUS.vehicle import SymmetryAxes

    M = surface.M
    N = surface.N

    # Chordwise parametric positions
    chord_eta = np.array(
        [surface.chord_discretization_function(int(i)) for i in range(M)],
    )
    chord_eta[-1] -= 1e-7
    chord_eta[0] += 1e-7

    # Pre-compute camber z-values for each spanwise station
    camber_z = np.zeros((M, N))
    norm_factors = np.ones(N)
    for j in range(N):
        airf_j = surface.airfoils[j]
        norm_factors[j] = airf_j.norm_factor
        camber_z[:, j] = airf_j.camber_line(chord_eta)

    # Convert airfoils
    chord_eta_jax = jnp.asarray(chord_eta, dtype=jnp.float64)
    root_camber = from_airfoil(surface._root_airfoil, chord_eta_jax, M)
    tip_camber = from_airfoil(surface._tip_airfoil, chord_eta_jax, M)

    # Convert control surfaces
    controls = []
    for cs in surface.controls:
        if cs.name != "none":
            controls.append(from_control_surface(cs))

    # Check symmetry
    is_symmetric_y = SymmetryAxes.Y in surface.symmetries

    return DiffWingSegment(
        chord_dist=jnp.asarray(surface._chord_dist, dtype=jnp.float64),
        span_dist=jnp.asarray(surface._span_dist, dtype=jnp.float64),
        twist_angles=jnp.asarray(surface.twist_angles, dtype=jnp.float64),
        x_offsets=jnp.asarray(surface._xoffset_dist, dtype=jnp.float64),
        z_offsets=jnp.asarray(surface._zoffset_dist, dtype=jnp.float64),
        structural_mass=jnp.asarray(surface.structural_mass, dtype=jnp.float64),
        name=surface.name,
        N=N,
        M=M,
        R_MAT=jnp.asarray(surface.R_MAT, dtype=jnp.float64),
        origin=jnp.asarray(surface._origin, dtype=jnp.float64),
        chord_eta=chord_eta_jax,
        camber_z=jnp.asarray(camber_z, dtype=jnp.float64),
        norm_factors=jnp.asarray(norm_factors, dtype=jnp.float64),
        root_camber=root_camber,
        tip_camber=tip_camber,
        controls=controls,
        is_lifting=surface.is_lifting,
        is_symmetric_y=is_symmetric_y,
    )
