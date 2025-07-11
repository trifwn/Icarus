from __future__ import annotations

from typing import TYPE_CHECKING

from ICARUS.database import Database

from .. import GenuVP3Parameters
from .. import GenuVP7Parameters
from ..utils import GenuSurface
from .pertrubations import gnvp_disturbance_case

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane
    from ICARUS.vehicle import WingSurface


def sensitivities_serial(
    plane: Airplane,
    state: State,
    var: str,
    angle: float,
    solver_parameters: GenuVP3Parameters | GenuVP7Parameters,
) -> None:
    """For each pertrubation in the sensitivity attribute of the dynamic airplane
    object, run a simulation in GNVP3. Can be used mainly for a sensitivity
    analysis. This analysis is serial.

    Args:
        plane (Dynamic_Airplane): Dynamic Airplane Object
        var (str): Variable to be perturbed
        solver2D (str): 2D Solver to be used for foil data
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        angle_of_attack (float): Angle of attack in degrees
        solver_parameters (GenuVP3Parameters | GenuVP7Parameters): Solver Options

    """
    DB = Database.get_instance()
    bodies_dicts: list[GenuSurface] = []
    if solver_parameters.Split_Symmetric_Bodies:
        surfaces: list[WingSurface] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.wings

    for i, surface in enumerate(surfaces):
        genu_surf = GenuSurface(surface, i)
        bodies_dicts.append(genu_surf)

    for dst in state.sensitivities[var]:
        gnvp_disturbance_case(
            DB,
            plane,
            state,
            state.u_freestream,
            angle,
            state.environment,
            surfaces,
            bodies_dicts,
            dst,
            "Sensitivity",
            solver_parameters,
        )
