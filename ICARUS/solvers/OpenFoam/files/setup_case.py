from __future__ import annotations

import os
import shutil
from enum import Enum
from subprocess import call
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from ICARUS import INSTALL_DIR
from ICARUS.database import Database
from ICARUS.database.utils import angle_to_directory
from ICARUS.solvers import setup_of_script

if TYPE_CHECKING:
    from ICARUS.airfoils.airfoil import Airfoil
    from ICARUS.core.types import FloatArray

OFBASE = os.path.join(INSTALL_DIR, "ICARUS", "Solvers", "OpenFoam", "files")


class MeshType(Enum):
    """Enum for Mesh Type"""

    # ! TODO: IMPLEMET Other Meshes
    structAirfoilMesher = 0
    copy_from = 1


def make_mesh(
    case_directory: str,
    airfoil_fname: str,
    mesh_type: MeshType,
) -> None:
    """
    Make the mesh for the simulation using the structAirfoilMesher.
    https://gitlab.com/before_may/structAirfoilMesher


    """
    if mesh_type == MeshType.structAirfoilMesher:
        # Check if struct.input exists
        filename = os.path.join(case_directory, "outPatch.out")
        if os.path.isfile(filename):
            print("Mesh is already Computed")
            return

        dst: str = os.path.join(case_directory, "struct.input")
        src: str = os.path.join("struct.input")
        shutil.copy(src, dst)

        call(["/bin/bash", "-c", f"{setup_of_script} -n {airfoil_fname}"], cwd=case_directory)

        src = os.path.join(OFBASE, "boundaryTemplate")
        dst = os.path.join(case_directory, "constant", "polyMesh", "boundary")
        shutil.copy(src, dst)
    elif mesh_type == MeshType.copy_from:
        pass


def init_case(
    case_directory: str,
    angle: float,
) -> None:
    """
    Make the zero folder for simulation

    Args:
        case_directory (str): Case Directory
        angle (float): Angle to run
    """
    src: str = os.path.join(OFBASE, "0")
    dst: str = os.path.join(case_directory, "0")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    filename: str = os.path.join(case_directory, "0", "U")
    with open(filename, encoding="UTF-8", newline="\n") as file:
        data: list[str] = file.readlines()
    data[26] = f"internalField uniform ( {np.cos(angle)} {np.sin(angle)} 0. );\n"
    with open(filename, "w", encoding="UTF-8") as file:
        file.writelines(data)


def constant_folder(
    case_directory: str,
    reynolds: float,
) -> None:
    """
    Make the constant folder for simulation

    Args:
        case_directory (str): Case Directory
        reynolds (float): Reynolds Number
    """
    src: str = os.path.join(OFBASE, "constant")
    dst: str = os.path.join(case_directory, "constant")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    filename: str = os.path.join(case_directory, "constant", "transportProperties")
    with open(filename, encoding="UTF-8", newline="\n") as file:
        data: list[str] = file.readlines()
    data[20] = (
        f"nu              [0 2 -1 0 0 0 0] {np.format_float_scientific(1 / reynolds, sign=False, precision=3)};\n"
    )
    with open(filename, "w", encoding="UTF-8") as file:
        file.writelines(data)


def system_folder(
    case_directory: str,
    angle: float,
    max_iterations: int,
) -> None:
    """
    Make the system folder for simulation

    Args:
        case_directory (str): Case Directory
        angle (float): Angle to run
        max_iterations (int): Max iterations for the simulation
    """
    src: str = os.path.join(OFBASE, "system")
    dst: str = os.path.join(case_directory, "system")
    shutil.copytree(src, dst, dirs_exist_ok=True)

    filename: str = os.path.join(case_directory, "system", "controlDict")
    with open(filename, encoding="UTF-8", newline="\n") as file:
        data: list[str] = file.readlines()
    data[36] = f"endTime {max_iterations}.;\n"
    data[94] = "\t\tCofR  (0.25 0. 0.);\n"
    data[95] = f"\t\tliftDir ({-np.sin(angle)} {np.cos(angle)} {0.0});\n"
    data[96] = f"\t\tdragDir ({np.cos(angle)} {np.sin(angle)} {0.0});\n"
    data[97] = "\t\tpitchAxis (0. 0. 1.);\n"
    data[98] = "\t\tmagUInf 1.;\n"
    data[110] = f"\t\tUInf ({np.cos(angle)} {np.sin(angle)} {0.0});\n"
    with open(filename, "w", encoding="UTF-8") as file:
        file.writelines(data)


def setup_open_foam(
    airfoil: Airfoil,
    reynolds: float,
    mach: float,
    angles: list[float] | FloatArray,
    solver_parameters: dict[str, Any],
) -> list[str]:
    """Function to setup OpenFoam cases for a given airfoil and Reynolds number

    Args:
        airfoil (Airfoil): Airfoil Object
        reynolds (float): Reynolds Number
        mach (float): Mach Number
        angles (list[float]): Angle to run simulation for (degrees)
        solver_parameters (dict[str, Any]): Solver Options
    """

    DB = Database.get_instance()
    _, AFDIR, REYNDIR, ANGLEDIRS = DB.generate_airfoil_directories(
        airfoil=airfoil,
        reynolds=reynolds,
        angles=angles,
    )
    airfoil_fname: str = airfoil.file_name

    if isinstance(angles, float):
        angles = [angles]
    mesh_dir: str = ""
    for i, angle in enumerate(angles):
        folder = angle_to_directory(angle)

        angle_directory: str = os.path.join(REYNDIR, folder)
        os.makedirs(angle_directory, exist_ok=True)

        angle_rad: float = angle * np.pi / 180
        # MAKE 0/ FOLDER
        init_case(angle_directory, angle_rad)

        # MAKE constant/ FOLDER
        constant_folder(angle_directory, reynolds)

        # MAKE system/ FOLDER
        max_iterations: int = solver_parameters["max_iterations"]
        system_folder(angle_directory, angle_rad, max_iterations)

        src: str = os.path.join(AFDIR, airfoil_fname)
        dst: str = os.path.join(angle_directory, airfoil_fname)
        shutil.copy(src, dst)

        mesh_type: MeshType = solver_parameters["mesh_type"]
        if i == 0:
            make_mesh(
                angle_directory,
                airfoil_fname,
                mesh_type,
            )
            mesh_dir = angle_directory
        else:
            src = os.path.join(mesh_dir, "constant", "polyMesh")
            dst = os.path.join(angle_directory, "constant", "polyMesh")
            shutil.copytree(src, dst, dirs_exist_ok=True)
            pass

        if solver_parameters["silent"] is False:
            pass

    return ANGLEDIRS
