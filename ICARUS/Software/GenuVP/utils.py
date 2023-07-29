"""
Defines the movement Class.
"""
from typing import Any

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray

from ICARUS.Core.struct import Struct
from ICARUS.Enviroment.definition import Environment
from ICARUS.Flight_Dynamics.disturbances import Disturbance
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.wing import Wing


class Movement:
    """
    Class to specify a generalized Movement as defined in the gnvp3 manual.
    """

    def __init__(
        self,
        name: str,
        Rotation: dict[str, Any],
        Translation: dict[str, Any],
    ) -> None:
        """
        Initialize the Movement Class

        Args:
            name (str): Name of the Movement
            Rotation (dict[str,Any]): Rotation Parameters. They must include:
                                        1) type: int
                                        2) axis: int
                                        3) t1: float
                                        4) t2: float
                                        5) a1: float
                                        6) a2: float
            Translation (dict[str,Any]): Translation Parameters. They must include:
                                        1) type: int
                                        2) axis: int
                                        3) t1: float
                                        4) t2: float
                                        5) a1: float
                                        6) a2: float
        """
        self.name: str = name
        self.rotation_type: int = Rotation["type"]

        self.rotation_axis: int = Rotation["axis"]

        self.rot_t1: float = Rotation["t1"]
        self.rot_t2: float = Rotation["t2"]

        self.rot_a1: float = Rotation["a1"]
        self.rot_a2: float = Rotation["a2"]

        self.translation_type: int = Translation["type"]

        self.translation_axis: int = Translation["axis"]

        self.translation_t1: float = Translation["t1"]
        self.translation_t2: float = Translation["t2"]

        self.translation_a1: float = Translation["a1"]
        self.translation_a2: float = Translation["a2"]


def define_movements(
    surfaces: list[Wing],
    CG: ndarray[Any, dtype[floating[Any]]],
    orientation: ndarray[Any, dtype[floating[Any]]] | list[float],
    disturbances: list[Disturbance] = [],
) -> list[list[Movement]]:
    """
    Define Movements for the surfaces.

    Args:
        surfaces (list[Wing]): List of Wing Objects
        CG (ndarray[Any, dtype[floating[Any]]]): Center of Gravity
        orientation (ndarray[Any, dtype[floating[Any]]] | list[float]): Orientation of the plane
        disturbances (list[Disturbance]): List of possible Disturbances. Defaults to empty list.

    Returns:
        list[list[Movement]]: A list of movements for each surface of the plane so that the center of gravity is at the origin.
    """
    movement: list[list[Movement]] = []
    all_axes = ("pitch", "roll", "yaw")
    all_ax_ids = (2, 1, 3)
    for _ in surfaces:
        sequence: list[Movement] = []
        for name, axis in zip(all_axes, all_ax_ids):
            Rotation: dict[str, Any] = {
                "type": 1,
                "axis": axis,
                "t1": -0.1,
                "t2": 0.0,
                "a1": orientation[axis - 1],
                "a2": orientation[axis - 1],
            }
            Translation: dict[str, Any] = {
                "type": 1,
                "axis": axis,
                "t1": -0.1,
                "t2": 0.0,
                "a1": -CG[axis - 1],
                "a2": -CG[axis - 1],
            }

            obj = Movement(name, Rotation, Translation)
            sequence.append(obj)

        for disturbance in disturbances:
            if disturbance.type is not None:
                sequence.append(distrubance2movement(disturbance))

        movement.append(sequence)
    return movement


def set_parameters(
    bodies_dicts: list[dict[str, Any]],
    plane: Airplane,
    maxiter: int,
    timestep: float,
    u_freestream: float,
    angle_deg: float,
    environment: Environment,
    solver_options: dict[str, Any] | Struct,
) -> dict[str, Any]:
    """
    Set the parameters for the gnvp3 solver.

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
    nBodies: int = len(bodies_dicts)
    nAirfoils: int = len(plane.airfoils)
    angle: float = angle_deg * np.pi / 180
    dens: float = environment.air_density
    visc: float = environment.air_dynamic_viscosity

    airVelocity: list[float] = [
        u_freestream * np.cos(angle),
        0.0,
        u_freestream * np.sin(angle),
    ]
    params: dict[str, Any] = {
        "nBods": nBodies,
        "nBlades": nAirfoils,
        "maxiter": maxiter,
        "timestep": timestep,
        "u_freestream": airVelocity,
        "rho": dens,
        "visc": visc,
        "Split_Symmetric_Bodies": solver_options["Split_Symmetric_Bodies"],
        "Use_Grid": solver_options["Use_Grid"],
        # LOW LEVEL OPTIONS
        "NMETH": solver_options["Integration_Scheme"],
        "NEMTIP": solver_options["Tip_Emmision"],
        "NTIMET": solver_options["Tip_Emmision_Begins"],
        "NEMSLE": solver_options["Leading_Edge_Separation"],
        "NTIMEL": solver_options["Leading_Edge_Separation_Begins"],
        "RELAXS": solver_options["Relaxation_Factor"],
        "EPSDS": solver_options["Pot_Convergence_Tolerence"],
        "NLEVELT": solver_options["Movement_Levels"],
        "NNEVP0": solver_options["Vortex_Particle_Count"],
        "RELAXU": solver_options["Vortex_Particle_Relaxation"],
        "PARVEC": solver_options["Minimum_Width_Parameter"],
        "NEMIS": solver_options["NEMIS"],
        "EPSFB": solver_options["Bound_Vorticity_Cutoff"],
        "EPSFW": solver_options["Wake_Vorticity_Cutoff"],
        "EPSSR": solver_options["Cutoff_Length_Sources"],
        "EPSDI": solver_options["Cutoff_Length_Sources2"],
        "EPSVR": solver_options["Vortex_Cutoff_Length_f"],
        "EPSO": solver_options["Vortex_Cutoff_Length_i"],
        "EPSINT": solver_options["EPSINT"],
        "COEF": solver_options["Particle_Dissipation_Factor"],
        "RMETM": solver_options["Upper_Deformation_Rate"],
        "IDEFW": solver_options["Wake_Deformation_Parameter"],
        "REFLEN": solver_options["REFLEN"],
        "IDIVVRP": solver_options["Particle_Subdivision_Parameter"],
        "FLENSC": solver_options["Subdivision_Length_Scale"],
        "NREWAK": solver_options["Wake_Particle_Merging_Parameter"],
        "NMER": solver_options["Particle_Merging_Parameter"],
        "XREWAK": solver_options["Merging_Starting_Distance"],
        "RADMER": solver_options["Merging_Radius"],
        "Elasticity_Solver": solver_options["Wake_Vorticity_Cutoff"],
    }
    return params


def make_surface_dict(surf: Wing, idx: int) -> dict[str, Any]:
    """
    Converts a Wing Object to a dict that can be used for making the input files of GNVP3

    Args:
        surf (Wing): Wing Object
        idx (int): IND of the surface to be assigned

    Returns:
        dict[str, Any]: Dict with the surface parameters
    """
    if surf.is_symmetric:
        N: int = 2 * surf.N - 1
        M: int = surf.M
    else:
        N = surf.N
        M = surf.M

    surface_dict: dict[str, Any] = {
        "NB": idx,
        "NACA": surf.airfoil.name,
        "name": surf.name,
        "bld": f"{surf.name}.bld",
        "cld": f"{surf.airfoil.name}.cld",
        "NNB": M,
        "NCWB": N,
        "x_0": surf.origin[0],
        "y_0": surf.origin[1],
        "z_0": surf.origin[2],
        "pitch": surf.orientation[0],
        "cone": surf.orientation[1],
        "wngang": surf.orientation[2],
        "x_end": surf.origin[0] + surf._offset_dist[-1],
        "y_end": surf.origin[1] + surf.span,
        "z_end": surf.origin[2] + surf._dihedral_dist[-1],
        "Root_chord": surf.chord[0],
        "Tip_chord": surf.chord[-1],
        "Offset": surf._offset_dist[-1],
        "Grid": surf.getGrid(),
    }
    return surface_dict


def distrubance2movement(disturbance: Disturbance) -> Movement:
    """
    Converts a disturbance to a movement

    Args:
        disturbance (Disturbance): Disturbance Object

    Raises:
        ValueError: If the disturbance type is not supported

    Returns:
        Movement: Movement Object
    """
    if disturbance.type == "Derivative":
        t1: float = -1
        t2: float = 0
        a1: float | None = 0
        a2: float | None = disturbance.amplitude
        distType = 8
    elif disturbance.type == "Value":
        t1 = -1
        t2 = 0.0
        a1 = disturbance.amplitude
        a2 = disturbance.amplitude
        distType = 1
    else:
        raise ValueError

    undisturbed: dict[str, Any] = {
        "type": 1,
        "axis": disturbance.axis,
        "t1": -1,
        "t2": 0,
        "a1": 0,
        "a2": 0,
    }

    disturbed: dict[str, Any] = {
        "type": distType,
        "axis": disturbance.axis,
        "t1": t1,
        "t2": t2,
        "a1": a1,
        "a2": a2,
    }

    if disturbance.isRotational:
        Rotation: dict[str, Any] = disturbed
        Translation: dict[str, Any] = undisturbed
    else:
        Rotation = undisturbed
        Translation = disturbed

    return Movement(disturbance.name, Rotation, Translation)
