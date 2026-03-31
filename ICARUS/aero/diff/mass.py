"""
Differentiable Mass Module

Equinox-based replacement for ICARUS.vehicle.base_classes.mass.Mass.
Position and mass value are differentiable; name and inertia are static.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

if TYPE_CHECKING:
    from ICARUS.vehicle.base_classes.mass import Mass


class DiffMass(eqx.Module):
    """Differentiable point mass.

    Dynamic fields (differentiable):
        position: 3D position vector [x, y, z]
        mass: mass value in kg

    Static fields:
        name: identifier string
        inertia: 6-element inertia vector [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    """

    position: Float[Array, "3"]
    mass: Float[Array, ""]

    name: str = eqx.field(static=True)
    inertia: Float[Array, "6"] = eqx.field(static=True)

    def inertia_about_point(self, point: Float[Array, "3"]) -> Float[Array, "6"]:
        """Compute inertia about an arbitrary point via parallel axis theorem.

        Returns:
            6-element array [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
        """
        r = self.position - point
        # Parallel axis theorem
        Ixx = self.inertia[0] + self.mass * (r[1] ** 2 + r[2] ** 2)
        Iyy = self.inertia[1] + self.mass * (r[0] ** 2 + r[2] ** 2)
        Izz = self.inertia[2] + self.mass * (r[0] ** 2 + r[1] ** 2)
        Ixy = self.inertia[3] + self.mass * r[0] * r[1]
        Ixz = self.inertia[4] + self.mass * r[0] * r[2]
        Iyz = self.inertia[5] + self.mass * r[1] * r[2]
        return jnp.array([Ixx, Iyy, Izz, Ixy, Ixz, Iyz])


def from_mass(mass: Mass) -> DiffMass:
    """Convert an existing ICARUS Mass to a DiffMass."""
    inertia_tensor = mass.inertia
    inertia_vec = jnp.array([
        inertia_tensor.I_xx,
        inertia_tensor.I_yy,
        inertia_tensor.I_zz,
        inertia_tensor.I_xy,
        inertia_tensor.I_xz,
        inertia_tensor.I_yz,
    ])
    return DiffMass(
        position=jnp.asarray(mass.position, dtype=jnp.float64),
        mass=jnp.asarray(mass.mass, dtype=jnp.float64),
        name=mass.name,
        inertia=inertia_vec,
    )


def to_mass(diff_mass: DiffMass) -> "Mass":
    """Convert a DiffMass back to an ICARUS Mass."""
    import numpy as np
    from ICARUS.vehicle.base_classes.mass import Mass
    from ICARUS.vehicle.base_classes import InertiaTensor

    inertia = InertiaTensor(
        I_xx=float(diff_mass.inertia[0]),
        I_yy=float(diff_mass.inertia[1]),
        I_zz=float(diff_mass.inertia[2]),
        I_xy=float(diff_mass.inertia[3]),
        I_xz=float(diff_mass.inertia[4]),
        I_yz=float(diff_mass.inertia[5]),
    )
    return Mass(
        name=diff_mass.name,
        position=np.array(diff_mass.position),
        mass=float(diff_mass.mass),
        inertia=inertia,
    )
