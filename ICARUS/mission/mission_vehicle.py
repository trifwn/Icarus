from functools import partial

import jax
import jax.numpy as jnp
from interpax import Interpolator1D
from jaxtyping import Array
from jaxtyping import ArrayLike
from jaxtyping import Float

from ICARUS.database import Database
from ICARUS.propulsion.engine import Engine
from ICARUS.vehicle import Airplane


class MissionVehicle:
    def __init__(
        self,
        airplane: Airplane,
        engine: Engine,
        solver: str = "AVL",
    ) -> None:
        DB = Database.get_instance()

        self.airplane: Airplane = airplane
        self.motor: Engine = engine
        self.polar_data = DB.get_vehicle_polars(airplane.name)
        self.solver_name: str = solver

        self.inertias: float = airplane.inertia[0]
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
        velocity: Float[ArrayLike, "..."],
        aoa: Float[ArrayLike, "..."],
    ) -> tuple[
        Float[ArrayLike, "..."],
        Float[ArrayLike, "..."],
        Float[ArrayLike, "..."],
    ]:
        cl, cd, cm = self.interpolate_polars(aoa)

        density = 1.225
        lift = cl * 0.5 * density * velocity**2 * self.airplane.S
        drag = cd * 0.5 * density * velocity**2 * self.airplane.S
        torque = cm * 0.5 * density * velocity**2 * self.airplane.S * self.airplane.mean_aerodynamic_chord

        return lift, drag, torque
