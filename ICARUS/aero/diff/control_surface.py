"""
Differentiable Control Surface Module

Equinox-based representation of aerodynamic control surfaces (ailerons,
flaps, elevators, rudders). The deflection angle is differentiable,
enabling gradient-based trim and control optimization.

The control surface modifies the camber line of affected panels by
rotating the aft portion of the airfoil about the hinge line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

if TYPE_CHECKING:
    from ICARUS.vehicle import ControlSurface


class DiffControlSurface(eqx.Module):
    """Differentiable control surface.

    Dynamic fields (differentiable):
        deflection: Control surface deflection angle in radians.
        gain: Control gain factor.

    Static fields:
        name: Control surface identifier
        control_var: Name of the control variable (e.g., "delta_e")
        span_start: Spanwise start position as fraction of total span [0, 1]
        span_end: Spanwise end position as fraction of total span [0, 1]
        chord_start: Hinge position as fraction of local chord [0, 1]
        chord_end: End position as fraction of local chord [0, 1]
        inverse_symmetric: Whether deflection is reversed on symmetric side
    """

    deflection: Float[Array, ""]
    gain: Float[Array, ""]

    name: str = eqx.field(static=True)
    control_var: str = eqx.field(static=True)
    span_start: float = eqx.field(static=True)
    span_end: float = eqx.field(static=True)
    chord_start: float = eqx.field(static=True)
    chord_end: float = eqx.field(static=True)
    inverse_symmetric: bool = eqx.field(static=True)

    def affects_station(self, eta_span: float) -> bool:
        """Check if this control surface affects a given spanwise station.

        Args:
            eta_span: Spanwise position as fraction of total span [0, 1]

        Returns:
            True if the control surface is active at this station
        """
        return self.span_start <= eta_span <= self.span_end

    def hinge_fraction(self, eta_span: float) -> float:
        """Get the hinge chord fraction at a given spanwise station.

        Linearly interpolates between chord_start and chord_end.

        Args:
            eta_span: Spanwise position as fraction of control span [0, 1]

        Returns:
            Hinge position as fraction of local chord
        """
        if self.span_end <= self.span_start:
            return self.chord_start
        local_eta = (eta_span - self.span_start) / (self.span_end - self.span_start)
        local_eta = max(0.0, min(1.0, local_eta))
        return (1 - local_eta) * self.chord_start + local_eta * self.chord_end

    def modify_camber(
        self,
        camber_z: Float[Array, "M"],
        chord_eta: Float[Array, "M"],
        eta_span: float,
    ) -> Float[Array, "M"]:
        """Apply control surface deflection to camber line.

        Rotates the camber line aft of the hinge point by the deflection
        angle. This is a thin-airfoil approximation where flap deflection
        modifies the effective camber.

        Args:
            camber_z: Original camber values (normalized by chord), shape (M,)
            chord_eta: Chordwise positions [0, 1], shape (M,)
            eta_span: Spanwise position as fraction of total span

        Returns:
            Modified camber values, shape (M,)
        """
        hinge = self.hinge_fraction(eta_span)
        effective_deflection = self.deflection * self.gain

        # For points aft of hinge, rotate about hinge point
        # Delta_z = -(x - x_hinge) * sin(deflection) (small angle: ~deflection)
        # Using exact trig for larger deflections
        delta_x = chord_eta - hinge
        aft_mask = (chord_eta >= hinge).astype(jnp.float64)

        # Rotation about hinge: z_new = z_hinge + (x-x_hinge)*sin(delta) + (z-z_hinge)*cos(delta)
        # For small deflections this simplifies to z_new ≈ z - (x-x_hinge)*delta
        # We use the exact form for correctness
        hinge_z = jnp.interp(hinge, chord_eta, camber_z)
        dz_from_hinge = camber_z - hinge_z

        rotated_dz = (
            delta_x * jnp.sin(effective_deflection)
            + dz_from_hinge * jnp.cos(effective_deflection)
        )

        modified_z = jnp.where(
            chord_eta >= hinge,
            hinge_z + rotated_dz,
            camber_z,
        )

        return modified_z


def from_control_surface(cs: ControlSurface) -> DiffControlSurface:
    """Convert an ICARUS ControlSurface to a DiffControlSurface.

    Args:
        cs: Source ControlSurface object

    Returns:
        DiffControlSurface with zero deflection
    """
    return DiffControlSurface(
        deflection=jnp.array(0.0),
        gain=jnp.asarray(cs.gain, dtype=jnp.float64),
        name=cs.name,
        control_var=cs.control_var,
        span_start=cs.span_percentage_start,
        span_end=cs.span_percentage_end,
        chord_start=cs.chord_percentage_start,
        chord_end=cs.chord_percentage_end,
        inverse_symmetric=cs.inverse_symmetric,
    )


def apply_controls_to_camber(
    camber_z: Float[Array, "M N"],
    chord_eta: Float[Array, "M"],
    controls: list[DiffControlSurface],
    span_fractions: Float[Array, "N"],
) -> Float[Array, "M N"]:
    """Apply all control surface deflections to a camber array.

    Iterates over control surfaces and spanwise stations, modifying
    the camber where each control is active.

    Args:
        camber_z: Original camber values (M, N) - M chordwise, N spanwise
        chord_eta: Chordwise positions [0, 1], shape (M,)
        controls: List of DiffControlSurface modules
        span_fractions: Spanwise station positions as fractions [0, 1], shape (N,)

    Returns:
        Modified camber array (M, N)
    """
    modified = camber_z
    for ctrl in controls:
        if ctrl.name == "none":
            continue
        for j in range(span_fractions.shape[0]):
            eta = float(span_fractions[j])
            if ctrl.affects_station(eta):
                col = ctrl.modify_camber(modified[:, j], chord_eta, eta)
                modified = modified.at[:, j].set(col)
    return modified
