"""
Differentiable Wing and Airplane Modules

Equinox-based replacements for ICARUS.vehicle.Wing and ICARUS.vehicle.Airplane.
These aggregate multiple DiffWingSegments and DiffMass objects into a complete
aircraft model suitable for end-to-end differentiation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .mass import DiffMass, from_mass
from .wing_segment import DiffWingSegment, from_wing_surface

if TYPE_CHECKING:
    from ICARUS.vehicle import Airplane, Wing


class DiffWing(eqx.Module):
    """Differentiable wing (aggregation of wing segments).

    A wing may consist of one or more DiffWingSegments (e.g., inboard +
    outboard sections with different taper or sweep).

    Dynamic fields propagate through segments.
    """

    segments: list[DiffWingSegment]
    name: str = eqx.field(static=True)

    @property
    def is_lifting(self) -> bool:
        return any(seg.is_lifting for seg in self.segments)

    @property
    def span(self) -> float:
        return sum(seg.span for seg in self.segments)

    @property
    def area(self) -> float:
        return sum(seg.area for seg in self.segments)

    @property
    def mean_aerodynamic_chord(self) -> float:
        """Area-weighted MAC."""
        total_area = self.area
        if total_area == 0:
            return 0.0
        weighted_mac = sum(seg.mean_aerodynamic_chord * seg.area for seg in self.segments)
        return weighted_mac / total_area

    @property
    def num_panels(self) -> int:
        return sum(seg.num_panels for seg in self.segments)


class DiffAirplane(eqx.Module):
    """Differentiable airplane model.

    Aggregates wings and point masses into a complete aircraft.
    The main wing is identified by name for reference area/span
    computation.

    Dynamic fields (differentiable):
        wings: List of DiffWing modules (contain all design params)
        point_masses: List of DiffMass modules
        cg_override: Optional CG override (3,). If set, used instead of
                     computing from masses. Matches ICARUS Airplane.CG behavior.

    Static fields:
        name: Aircraft identifier
        main_wing_name: Name of the primary wing
        has_cg_override: Whether cg_override should be used
    """

    wings: list[DiffWing]
    point_masses: list[DiffMass]
    cg_override: Float[Array, "3"]

    name: str = eqx.field(static=True)
    main_wing_name: str = eqx.field(static=True)
    has_cg_override: bool = eqx.field(static=True)

    @property
    def main_wing(self) -> DiffWing:
        for w in self.wings:
            if w.name == self.main_wing_name:
                return w
        raise ValueError(f"Main wing '{self.main_wing_name}' not found")

    @property
    def S(self) -> float:
        """Reference area (from main wing)."""
        return self.main_wing.area

    @property
    def span(self) -> float:
        """Reference span (from main wing)."""
        return self.main_wing.span

    @property
    def MAC(self) -> float:
        """Mean aerodynamic chord (from main wing)."""
        return self.main_wing.mean_aerodynamic_chord

    @property
    def all_segments(self) -> list[DiffWingSegment]:
        """Flat list of all wing segments."""
        segs = []
        for wing in self.wings:
            segs.extend(wing.segments)
        return segs

    @property
    def lifting_segments(self) -> list[DiffWingSegment]:
        """Only segments that generate lift."""
        return [seg for seg in self.all_segments if seg.is_lifting]

    def compute_cg(self) -> Float[Array, "3"]:
        """Compute center of gravity.

        If cg_override is set (from the original Airplane), uses that.
        Otherwise computes from wing masses and point masses.

        Returns:
            CG position as JAX array (3,)
        """
        if self.has_cg_override:
            return self.cg_override

        total_mass = jnp.array(0.0)
        weighted_pos = jnp.zeros(3)

        for wing in self.wings:
            for seg in wing.segments:
                total_mass = total_mass + seg.structural_mass
                mid_span_idx = seg.N // 2
                seg_pos = seg.origin + jnp.stack([
                    seg.x_offsets[mid_span_idx] + seg.chord_dist[mid_span_idx] * 0.25,
                    seg.span_dist[mid_span_idx],
                    seg.z_offsets[mid_span_idx],
                ])
                weighted_pos = weighted_pos + seg.structural_mass * seg_pos

        for pm in self.point_masses:
            total_mass = total_mass + pm.mass
            weighted_pos = weighted_pos + pm.mass * pm.position

        return jnp.where(total_mass > 0, weighted_pos / total_mass, jnp.zeros(3))

    @property
    def total_mass(self) -> float:
        m = sum(float(seg.structural_mass) for w in self.wings for seg in w.segments)
        m += sum(float(pm.mass) for pm in self.point_masses)
        return m


def from_airplane(airplane: Airplane) -> DiffAirplane:
    """Convert an ICARUS Airplane to a DiffAirplane.

    Captures the CG from the original Airplane for moment reference
    consistency. Recursively converts all wings, segments, and masses.

    Args:
        airplane: Source Airplane object

    Returns:
        DiffAirplane with all parameters as JAX arrays
    """
    diff_wings = []
    for wing in airplane.wings:
        diff_segments = []
        for segment in wing.get_separate_segments():
            diff_segments.append(from_wing_surface(segment))
        diff_wings.append(DiffWing(segments=diff_segments, name=wing.name))

    diff_masses = [from_mass(m) for m in airplane.point_masses]

    # Capture CG from original airplane for consistent moment reference
    cg = jnp.asarray(airplane.CG, dtype=jnp.float64)

    return DiffAirplane(
        wings=diff_wings,
        point_masses=diff_masses,
        cg_override=cg,
        name=airplane.name,
        main_wing_name=airplane.main_wing_name,
        has_cg_override=True,
    )
