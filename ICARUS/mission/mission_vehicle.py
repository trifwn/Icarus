import jax.numpy as jnp
import numpy as np
from pandas import DataFrame

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

        # self.elevator: Lifting_Surface = elevator
        # self.elevator_max_deflection = 30
        # self.Ixx: float = airplane.total_inertia[0]
        # self.l_m: float = -0.4

    @staticmethod
    def interpolate_polars(aoa: float, cldata: DataFrame) -> tuple[float, float, float]:
        cl = float(np.interp(aoa, cldata["AoA"], cldata[f"CL"]))
        cd = float(np.interp(aoa, cldata["AoA"], cldata[f"CD"]))
        cm = float(np.interp(aoa, cldata["AoA"], cldata[f"Cm"]))
        return cl, cd, cm

    def get_aerodynamic_forces(
        self,
        velocity: float,
        aoa: float 
    ) -> tuple[float, float, float]:

        aoa = float(np.rad2deg(aoa))
        cl, cd, cm = self.interpolate_polars(
            aoa,
            self.polar_data[
                [
                    f"{self.solver_name} CL",
                    f"{self.solver_name} CD",
                    f"{self.solver_name} Cm",
                    "AoA",
                ]
            ].rename(   
                columns= {
                    f"{self.solver_name} CL": "CL",
                    f"{self.solver_name} CD": "CD",
                    f"{self.solver_name} Cm": "Cm",
                } 
            ),
        )

        density = 1.225
        lift = cl * 0.5 * density * velocity**2 * self.airplane.S
        drag = cd * 0.5 * density * velocity**2 * self.airplane.S
        torque = cm * 0.5 * density * velocity**2 * self.airplane.S * self.airplane.mean_aerodynamic_chord

        return lift, drag, torque

    def get_aerodynamic_forces_jax(
        self,
        velocity: jnp.ndarray,
        aoa: jnp.ndarray
    )-> jnp.ndarray:
        pass