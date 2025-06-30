"""Class to define the parameters for the GenuVP solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from ICARUS.core.base_types import Struct
from ICARUS.core.types import FloatArray

from . import GenuSurface

if TYPE_CHECKING:
    from ICARUS.environment import Environment
    from ICARUS.vehicle import Airplane


class GenuParameters:
    def __init__(
        self,
        genu_bodies: list[GenuSurface],
        plane: Airplane,
        maxiter: int,
        timestep: float,
        u_freestream: float,
        angle_deg: float,
        environment: Environment,
        solver_parameters: dict[str, Any] | Struct,
    ) -> None:
        """Set the parameters for the gnvp3 solver.

        Args:
            bodies_dicts (list[dict[str, Any]]): List of dicts with the surface parameters
            plane (Airplane): Airplane Object
            maxiter (int): Max Iterations
            timestep (float): Timestep for the simulation
            u_freestream (float): U Freestream magnitude
            angle_deg (float): Angle of attack in degrees
            environment (Environment): Environment Object
            solver_parameters (dict[str, Any] | Struct): Solver Options

        Returns:
            dict[str, Any]: dict with all the parameters to define the simulation

        """
        n_bodies: int = len(genu_bodies)
        num_airfoils: int = len(plane.airfoils)
        angle: float = angle_deg * np.pi / 180
        dens: float = environment.air_density
        visc: float = environment.air_kinematic_viscosity

        airVelocity: list[float] = [
            u_freestream * np.cos(angle),
            0.0,
            u_freestream * np.sin(angle),
        ]

        self.name: str = plane.name
        self.nBods: int = n_bodies
        self.nBlades: int = num_airfoils
        self.CG: FloatArray = plane.CG
        self.maxiter: int = maxiter
        self.timestep: float = timestep
        self.u_freestream: list[float] = airVelocity
        self.rho: float = dens
        self.visc: float = visc
        self.Split_Symmetric_Bodies: bool = solver_parameters["Split_Symmetric_Bodies"]
        self.Use_Grid: bool = solver_parameters["Use_Grid"]
        # LOW LEVEL OPTION
        self.NMETH = solver_parameters["Integration_Scheme"]
        self.NEMTIP = solver_parameters["Tip_Emmision"]
        self.NTIMET = solver_parameters["Tip_Emmision_Begins"]
        self.NEMSLE = solver_parameters["Leading_Edge_Separation"]
        self.NTIMEL = solver_parameters["Leading_Edge_Separation_Begins"]
        self.RELAXS = solver_parameters["Relaxation_Factor"]
        self.EPSDS = solver_parameters["Pot_Convergence_Tolerence"]
        self.NLEVELT = solver_parameters["Movement_Levels"]
        self.NNEVP0 = solver_parameters["Vortex_Particle_Count"]
        self.RELAXU = solver_parameters["Vortex_Particle_Relaxation"]
        self.PARVEC = solver_parameters["Minimum_Width_Parameter"]
        self.NEMIS = solver_parameters["NEMIS"]
        self.EPSFB = solver_parameters["Bound_Vorticity_Cutoff"]
        self.EPSFW = solver_parameters["Wake_Vorticity_Cutoff"]
        self.EPSSR = solver_parameters["Cutoff_Length_Sources"]
        self.EPSDI = solver_parameters["Cutoff_Length_Sources2"]
        self.EPSVR = solver_parameters["Vortex_Cutoff_Length_f"]
        self.EPSO = solver_parameters["Vortex_Cutoff_Length_i"]
        self.EPSINT = solver_parameters["EPSINT"]
        self.COEF = solver_parameters["Particle_Dissipation_Factor"]
        self.RMETM = solver_parameters["Upper_Deformation_Rate"]
        self.IDEFW = solver_parameters["Wake_Deformation_Parameter"]
        self.REFLEN = solver_parameters["REFLEN"]
        self.IDIVVRP = solver_parameters["Particle_Subdivision_Parameter"]
        self.FLENSC = solver_parameters["Subdivision_Length_Scale"]
        self.NREWAK = solver_parameters["Wake_Particle_Merging_Parameter"]
        self.NMER = solver_parameters["Particle_Merging_Parameter"]
        self.XREWAK = solver_parameters["Merging_Starting_Distance"]
        self.RADMER = solver_parameters["Merging_Radius"]
        self.Elasticity_Solver = solver_parameters["Wake_Vorticity_Cutoff"]
        self.IYNELST = solver_parameters["Elasticity_Solver"]
