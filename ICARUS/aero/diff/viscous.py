"""
Differentiable Viscous Drag Model

Provides JAX-differentiable profile drag estimation using pre-loaded
airfoil polars. The polar data is loaded once from the ICARUS database
(NumPy/Pandas) and converted to JAX arrays, enabling jax.grad through
the viscous drag computation.

Approach:
    1. Pre-load CL(alpha, Re) and CD(alpha, Re) from database as JAX arrays
    2. At solve time: compute effective alpha/Re per strip from VLM solution
    3. Interpolate polars using jnp.interp (differentiable)
    4. Compute strip viscous forces: D_visc = CD_2D * q * S_strip

This separates the non-differentiable I/O (database lookup) from the
differentiable computation (interpolation + force calculation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

if TYPE_CHECKING:
    from ICARUS.airfoils import Airfoil


class DiffPolarData(eqx.Module):
    """Pre-loaded airfoil polar data as JAX arrays.

    Stores CL(alpha), CD(alpha), Cm(alpha) at a single effective Reynolds
    number (or Reynolds-interpolated). The data is pre-loaded from the
    ICARUS database and stored as JAX arrays for differentiable interpolation.

    All fields are dynamic (part of pytree) for JAX compatibility.
    """

    aoa_data: Float[Array, "K"]    # Angle of attack values (degrees)
    cl_data: Float[Array, "K"]     # CL values
    cd_data: Float[Array, "K"]     # CD values
    cm_data: Float[Array, "K"]     # Cm values

    airfoil_name: str = eqx.field(static=True)

    def interpolate(
        self,
        aoa: Float[Array, ""],
    ) -> tuple[Array, Array, Array]:
        """Interpolate CL, CD, Cm at a given angle of attack.

        Uses jnp.interp which is JAX-differentiable.

        Args:
            aoa: Effective angle of attack in degrees

        Returns:
            (CL, CD, Cm) as JAX scalars
        """
        cl = jnp.interp(aoa, self.aoa_data, self.cl_data)
        cd = jnp.interp(aoa, self.aoa_data, self.cd_data)
        cm = jnp.interp(aoa, self.aoa_data, self.cm_data)
        return cl, cd, cm


def load_polar_data(
    airfoil: Airfoil,
    reynolds: float = 1e6,
    solver: str = "Xfoil",
) -> DiffPolarData | None:
    """Load polar data from the ICARUS database for a given airfoil.

    This is a non-differentiable setup step. Call once before the
    differentiable solve.

    Args:
        airfoil: ICARUS Airfoil object
        reynolds: Reynolds number for polar lookup
        solver: Solver name ("Xfoil", "Foil2Wake", etc.)

    Returns:
        DiffPolarData or None if polars not available
    """
    try:
        from ICARUS.database import Database
        import numpy as np

        db = Database.get_instance()
        airfoil_data = db.foils_db.data.get(airfoil.name)
        if airfoil_data is None:
            return None

        polar_map = airfoil_data.get_polars(solver)
        if polar_map is None:
            return None

        # Get the closest Reynolds number polar
        available_re = sorted(polar_map.polars.keys())
        if not available_re:
            return None

        # Find closest Reynolds
        closest_re = min(available_re, key=lambda r: abs(r - reynolds))
        polar = polar_map.polars[closest_re]

        aoa = np.array(polar.df["AoA"].values, dtype=float)
        cl = np.array(polar.df["CL"].values, dtype=float)
        cd = np.array(polar.df["CD"].values, dtype=float)
        cm = np.array(polar.df["Cm"].values, dtype=float) if "Cm" in polar.df.columns else np.zeros_like(cl)

        return DiffPolarData(
            aoa_data=jnp.asarray(aoa),
            cl_data=jnp.asarray(cl),
            cd_data=jnp.asarray(cd),
            cm_data=jnp.asarray(cm),
            airfoil_name=airfoil.name,
        )
    except Exception:
        return None


def make_flat_plate_polar(
    airfoil_name: str = "flat_plate",
    aoa_range: tuple[float, float] = (-15.0, 15.0),
    n_points: int = 61,
) -> DiffPolarData:
    """Create a simple flat-plate drag polar for testing.

    Uses the empirical approximation:
        CL = 2*pi*alpha (thin airfoil theory)
        CD = CD0 + CL^2 / (pi * e * AR_eff)  (simplified)

    For profile drag we use a simple quadratic:
        CD_profile = 0.008 + 0.0001 * alpha^2

    Args:
        airfoil_name: Name for the polar
        aoa_range: Range of angles of attack in degrees
        n_points: Number of data points

    Returns:
        DiffPolarData with synthetic polar
    """
    aoa = jnp.linspace(aoa_range[0], aoa_range[1], n_points)
    alpha_rad = jnp.deg2rad(aoa)

    cl = 2 * jnp.pi * alpha_rad
    cd = 0.008 + 0.0001 * aoa ** 2
    cm = -0.05 * jnp.ones_like(aoa)  # Roughly constant for symmetric airfoils

    return DiffPolarData(
        aoa_data=aoa,
        cl_data=cl,
        cd_data=cd,
        cm_data=cm,
        airfoil_name=airfoil_name,
    )


def compute_strip_viscous_forces(
    effective_aoa: Float[Array, ""],
    effective_velocity: Float[Array, ""],
    chord: Float[Array, ""],
    width: Float[Array, ""],
    density: float,
    polar: DiffPolarData,
) -> tuple[Array, Array]:
    """Compute viscous (profile) forces for a single strip.

    Args:
        effective_aoa: Effective angle of attack in degrees
        effective_velocity: Effective velocity magnitude
        chord: Strip chord length
        width: Strip spanwise width
        density: Air density
        polar: Pre-loaded polar data

    Returns:
        (viscous_lift, viscous_drag) as JAX scalars
    """
    cl_2d, cd_2d, _ = polar.interpolate(effective_aoa)
    q_eff = 0.5 * density * effective_velocity ** 2
    strip_area = chord * width

    visc_lift = cl_2d * q_eff * strip_area
    visc_drag = cd_2d * q_eff * strip_area

    return visc_lift, visc_drag
