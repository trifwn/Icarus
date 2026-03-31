"""
Differentiable Airfoil Camber Module

Equinox-based representation of airfoil camber for the VLM pipeline.
The VLM only needs the camber line (not full thickness), so this module
stores pre-evaluated camber values at the chordwise stations used by the
geometry generator.

For NACA4 airfoils, the camber parameters (m, p) are differentiable,
enabling optimization of airfoil shape. For spline-based airfoils,
the camber is stored as a static array (evaluated once from the parent
Airfoil object).

Supports JAX-differentiable morphing between two airfoils via linear
interpolation of camber values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

if TYPE_CHECKING:
    from ICARUS.airfoils import Airfoil
    from ICARUS.airfoils import NACA4


class DiffAirfoilCamber(eqx.Module):
    """Differentiable airfoil camber representation.

    For VLM, only the camber line matters. This module stores either:
    - Pre-evaluated camber values (static, for spline airfoils)
    - NACA4 parameters m, p (dynamic, for parametric optimization)

    Dynamic fields (differentiable):
        camber_at_eta: Pre-evaluated camber z-values at chord_eta positions.
                       Shape (M,) where M is the number of chordwise stations.
                       These are normalized by chord (z/c).

    Static fields:
        name: Airfoil identifier
        norm_factor: Normalization factor for the airfoil chord
    """

    camber_at_eta: Float[Array, "M"]

    name: str = eqx.field(static=True)
    norm_factor: float = eqx.field(static=True)


class DiffNACA4Camber(eqx.Module):
    """Differentiable NACA4 camber with parametric shape control.

    Unlike DiffAirfoilCamber which stores pre-evaluated values, this
    module keeps the NACA4 parameters (m, p) as differentiable fields,
    enabling gradient-based airfoil optimization.

    Dynamic fields (differentiable):
        m: Maximum camber (e.g., 0.02 for 2%)
        p: Position of maximum camber (e.g., 0.4 for 40% chord)

    Static fields:
        name: Airfoil identifier
        xx: Maximum thickness (not needed for VLM camber, kept for reference)
        norm_factor: Normalization factor
    """

    m: Float[Array, ""]
    p: Float[Array, ""]

    name: str = eqx.field(static=True)
    xx: float = eqx.field(static=True)
    norm_factor: float = eqx.field(static=True)

    def camber_line(self, points: Float[Array, "M"]) -> Float[Array, "M"]:
        """Evaluate the NACA4 camber line at given chordwise positions.

        This is a JAX-differentiable version of NACA4.camber_line().

        Args:
            points: Chordwise positions in [0, 1], shape (M,)

        Returns:
            Camber z-values (normalized by chord), shape (M,)
        """
        p = self.p + 1e-19  # Avoid division by zero
        m = self.m
        xsi = points

        yc = jnp.select(
            [xsi <= 0, xsi < p],
            [
                0.0,
                (m / p**2) * (2 * p * xsi - xsi**2),
            ],
            default=(m / (1 - p) ** 2) * (1 - 2 * p + 2 * p * xsi - xsi**2),
        )
        return yc

    def evaluate_at(self, chord_eta: Float[Array, "M"]) -> Float[Array, "M"]:
        """Evaluate camber at specific chordwise stations.

        This is the entry point used by the geometry generator.
        """
        return self.camber_line(chord_eta)


def from_airfoil(
    airfoil: Airfoil,
    chord_eta: Float[Array, "M"] | None = None,
    M: int = 8,
) -> DiffAirfoilCamber:
    """Convert an ICARUS Airfoil to a DiffAirfoilCamber.

    Evaluates the camber line at the given chordwise stations and stores
    the result as a static array.

    Args:
        airfoil: Source Airfoil object
        chord_eta: Chordwise parametric positions (0..1). If None,
                   generates M linearly-spaced points.
        M: Number of chordwise stations (used if chord_eta is None)

    Returns:
        DiffAirfoilCamber with pre-evaluated camber values
    """
    import numpy as np

    if chord_eta is None:
        chord_eta_np = np.linspace(0, 1, M)
        chord_eta_np[0] += 1e-7
        chord_eta_np[-1] -= 1e-7
    else:
        chord_eta_np = np.asarray(chord_eta)

    camber_values = airfoil.camber_line(chord_eta_np)

    return DiffAirfoilCamber(
        camber_at_eta=jnp.asarray(camber_values, dtype=jnp.float64),
        name=airfoil.name,
        norm_factor=float(airfoil.norm_factor),
    )


def from_naca4(naca4: NACA4) -> DiffNACA4Camber:
    """Convert a NACA4 airfoil to a parametric DiffNACA4Camber.

    The m and p parameters become differentiable, enabling
    airfoil shape optimization.

    Args:
        naca4: Source NACA4 airfoil

    Returns:
        DiffNACA4Camber with differentiable m, p parameters
    """
    return DiffNACA4Camber(
        m=jnp.asarray(float(naca4.m), dtype=jnp.float64),
        p=jnp.asarray(float(naca4.p), dtype=jnp.float64),
        name=naca4.name,
        xx=float(naca4.xx),
        norm_factor=float(naca4.norm_factor),
    )


def morph_camber(
    camber1: DiffAirfoilCamber,
    camber2: DiffAirfoilCamber,
    eta: Float[Array, ""],
) -> DiffAirfoilCamber:
    """JAX-differentiable linear morph between two airfoil cambers.

    Args:
        camber1: Root airfoil camber
        camber2: Tip airfoil camber
        eta: Morph parameter in [0, 1]. 0 = camber1, 1 = camber2.

    Returns:
        Interpolated DiffAirfoilCamber
    """
    morphed = (1 - eta) * camber1.camber_at_eta + eta * camber2.camber_at_eta
    norm_factor = (1 - float(eta)) * camber1.norm_factor + float(eta) * camber2.norm_factor
    return DiffAirfoilCamber(
        camber_at_eta=morphed,
        name=f"morph({camber1.name},{camber2.name})",
        norm_factor=norm_factor,
    )
