"""Class to define the parameters for the GenuVP solvers."""

from typing import Any

import numpy as np

from ICARUS.computation.solvers.GenuVP.utils.genu_surface import GenuSurface
from ICARUS.core.struct import Struct
from ICARUS.core.types import FloatArray
from ICARUS.environment.definition import Environment
from ICARUS.vehicle.airplane import Airplane


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
        solver_options: dict[str, Any] | Struct,
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
            solver_options (dict[str, Any] | Struct): Solver Options

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
        self.Split_Symmetric_Bodies: bool = solver_options["Split_Symmetric_Bodies"]
        self.Use_Grid: bool = solver_options["Use_Grid"]
        # LOW LEVEL OPTION
        self.NMETH = solver_options["Integration_Scheme"]
        self.NEMTIP = solver_options["Tip_Emmision"]
        self.NTIMET = solver_options["Tip_Emmision_Begins"]
        self.NEMSLE = solver_options["Leading_Edge_Separation"]
        self.NTIMEL = solver_options["Leading_Edge_Separation_Begins"]
        self.RELAXS = solver_options["Relaxation_Factor"]
        self.EPSDS = solver_options["Pot_Convergence_Tolerence"]
        self.NLEVELT = solver_options["Movement_Levels"]
        self.NNEVP0 = solver_options["Vortex_Particle_Count"]
        self.RELAXU = solver_options["Vortex_Particle_Relaxation"]
        self.PARVEC = solver_options["Minimum_Width_Parameter"]
        self.NEMIS = solver_options["NEMIS"]
        self.EPSFB = solver_options["Bound_Vorticity_Cutoff"]
        self.EPSFW = solver_options["Wake_Vorticity_Cutoff"]
        self.EPSSR = solver_options["Cutoff_Length_Sources"]
        self.EPSDI = solver_options["Cutoff_Length_Sources2"]
        self.EPSVR = solver_options["Vortex_Cutoff_Length_f"]
        self.EPSO = solver_options["Vortex_Cutoff_Length_i"]
        self.EPSINT = solver_options["EPSINT"]
        self.COEF = solver_options["Particle_Dissipation_Factor"]
        self.RMETM = solver_options["Upper_Deformation_Rate"]
        self.IDEFW = solver_options["Wake_Deformation_Parameter"]
        self.REFLEN = solver_options["REFLEN"]
        self.IDIVVRP = solver_options["Particle_Subdivision_Parameter"]
        self.FLENSC = solver_options["Subdivision_Length_Scale"]
        self.NREWAK = solver_options["Wake_Particle_Merging_Parameter"]
        self.NMER = solver_options["Particle_Merging_Parameter"]
        self.XREWAK = solver_options["Merging_Starting_Distance"]
        self.RADMER = solver_options["Merging_Radius"]
        self.Elasticity_Solver = solver_options["Wake_Vorticity_Cutoff"]
        self.IYNELST = solver_options["Elasticity_Solver"]
