from functools import partial

import jax
import jax.numpy as jnp
from interpax import Interpolator1D
from jaxtyping import Array
from jaxtyping import Float

from ICARUS.database import DB
from ICARUS.propulsion.engine import Engine
from ICARUS.vehicle.plane import Airplane


class MissionVehicle:
    def __init__(
        self,
        airplane: Airplane,
        engine: Engine,
        solver: str = "AVL",
    ) -> None:
        self.airplane: Airplane = airplane
        self.motor: Engine = engine
        self.polar_data = DB.vehicles_db.get_polars(airplane.name)
        self.solver_name: str = solver

        self.inertias: float = airplane.total_inertia[0]
        self.mass: float = airplane.M

        # Get the cl, cd, cm data
        cl = jnp.array(self.polar_data[f"{solver} CL"].values)
        cd = jnp.array(self.polar_data[f"{solver} CD"].values)
        cm = jnp.array(self.polar_data[f"{solver} Cm"].values)
        aoa = jnp.array(self.polar_data["AoA"].values)

        # Create interpolators
        self.cl_interpolator = Interpolator1D(aoa, cl, method="cubic", extrap=True)
        self.cd_interpolator = Interpolator1D(aoa, cd, method="cubic", extrap=True)
        self.cm_interpolator = Interpolator1D(aoa, cm, method="cubic", extrap=True)

    @partial(jax.jit, static_argnums=(0,))
    def interpolate_polars(
        self,
        aoa: Float[Array, "..."] | float,
    ) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
        _aoa = jnp.atleast_1d(aoa)
        cl = self.cl_interpolator(_aoa)
        cd = self.cd_interpolator(_aoa)
        cm = self.cm_interpolator(_aoa)
        return cl, cd, cm

    @partial(jax.jit, static_argnums=(0,))
    def get_aerodynamic_forces(
        self,
        velocity: Float[Array, "dim"] | float,
        aoa: Float[Array, "dim"] | float,
    ) -> tuple[Float[Array, "dim"], Float[Array, "dim"], Float[Array, "dim"]]:
        cl, cd, cm = self.interpolate_polars(aoa)

        density = 1.225
        lift = cl * 0.5 * density * velocity**2 * self.airplane.S
        drag = cd * 0.5 * density * velocity**2 * self.airplane.S
        torque = cm * 0.5 * density * velocity**2 * self.airplane.S * self.airplane.mean_aerodynamic_chord

        return lift, drag, torque
