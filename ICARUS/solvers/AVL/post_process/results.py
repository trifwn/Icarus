import os

import numpy as np
from pandas import DataFrame

from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.database import angle_to_directory
from ICARUS.database import disturbance_to_directory
from ICARUS.flight_dynamics import State
from ICARUS.solvers.AVL.post_process import AVLOutputParser
from ICARUS.solvers.AVL.post_process.output_parser import AVLEigenmode
from ICARUS.vehicle import Airplane


class AVLPostReadError(Exception):
    pass


def collect_avl_polar_forces(
    directory: str,
    plane: Airplane,
    state: State,
    angles: FloatArray | list[float],
) -> DataFrame:
    """POST-PROCESSING OF POLAR RUNS - RETURNS AN ARRAY WITH THE FOLLOWING ORDER OF VECTORS: AOA,CL,CD,CM

    Args:
        directory (str): Directory where the AVL results are stored
        plane (Airplane): Airplane object
        state (State): State object
        angles (FloatArray): Array of angles of attack

    Returns:
        FloatArray: Array with the following order of vectors: AOA,CL,CD,CM

    """

    # Empty dataframe to store the polar data
    all_forces = []
    for angle in angles:
        file_path = os.path.join(directory, f"{angle_to_directory(angle)}.txt")
        parser = AVLOutputParser(file_path)

        forces = parser.parse_forces()
        all_forces.append(forces)

    flat_forces = [force for sublist in all_forces for force in sublist]
    polar_df: DataFrame = AVLOutputParser.to_dataframe(flat_forces)

    Fz = polar_df["CL"] * plane.S * state.dynamic_pressure
    Fx = polar_df["CD"] * plane.S * state.dynamic_pressure
    My = (
        polar_df["Cm"] * plane.S * state.dynamic_pressure * plane.mean_aerodynamic_chord
    )

    polar_df = DataFrame(
        np.array([polar_df["Alpha"], Fz, Fx, My]).T,
        columns=["AoA", "AVL Fz", "AVL Fx", "AVL My"],
    )
    polar_df = polar_df.sort_values("AoA").reset_index(drop=True)
    return polar_df


def finite_difs_post(plane: Airplane, state: State) -> DataFrame:
    DB = Database.get_instance()
    DYNDIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="AVL",
        case="Dynamics",
    )

    aoa = state.trim["AoA"] * np.pi / 180
    results = []
    for dst in state.disturbances:
        casefile = os.path.join(DYNDIR, disturbance_to_directory(dst))
        if dst.var == "phi" or dst.var == "theta":
            Fx = 0.0
            Fy = 0.0
            Fz = 0.0
            M = 0.0
            N = 0.0
            L = 0.0
        else:
            with open(casefile, encoding="utf-8") as f:
                lines = f.readlines()

            x_axis = lines[19]
            y_axis = lines[20]
            z_axis = lines[21]

            try:
                CX = float(x_axis[11:19])
                CY = float(y_axis[11:19])
                CZ = float(z_axis[11:19])

                Cl = float(x_axis[33:41])
                Cm = float(y_axis[33:41])
                Cn = float(z_axis[33:41])

                # # Rotate the forces to the body frame
                # CX = CX * np.cos(aoa) - CZ * np.sin(aoa)
                # CY = CY
                # CZ = CX * np.sin(aoa) + CZ * np.cos(aoa)

                # Cl = Cl * np.cos(aoa) - Cn * np.sin(aoa)
                # Cm = Cm
                # Cn = Cl * np.sin(aoa) + Cn * np.cos(aoa)
            except ValueError:
                raise AVLPostReadError(f"Error reading file {casefile}")

            if dst.var == "u":
                dyn_pressure = float(
                    0.5
                    * state.environment.air_density
                    * float(
                        np.linalg.norm(
                            [
                                state.trim["U"] * np.cos(aoa) + dst.amplitude,
                                state.trim["U"] * np.sin(aoa),
                            ],
                        ),
                    )
                    ** 2.0,
                )
            elif dst.var == "w":
                dyn_pressure = float(
                    0.5
                    * state.environment.air_density
                    * float(
                        np.linalg.norm(
                            [
                                state.trim["U"] * np.cos(aoa),
                                state.trim["U"] * np.sin(aoa) + dst.amplitude,
                            ],
                        ),
                    )
                    ** 2.0,
                )
            else:
                dyn_pressure = state.trim_dynamic_pressure

            Fx = CX * dyn_pressure * plane.S
            Fy = CY * dyn_pressure * plane.S
            Fz = CZ * dyn_pressure * plane.S
            M = Cm * dyn_pressure * plane.S * plane.mean_aerodynamic_chord
            N = Cn * dyn_pressure * plane.S * plane.span
            L = Cl * dyn_pressure * plane.S * plane.span
        if dst.amplitude is None:
            ampl = 0.0
        else:
            ampl = float(dst.amplitude)
        results.append(np.array([ampl, dst.var, Fx, Fy, Fz, L, M, N]))
    df = DataFrame(results, columns=cols)
    df = df.sort_values("Type").reset_index(drop=True)
    df["Epsilon"] = df["Epsilon"].astype(float)
    perturbation_file: str = os.path.join(DYNDIR, "pertrubations.avl")
    df.to_csv(perturbation_file, index=False)
    return df


def implicit_dynamics_post(
    plane: Airplane,
    state: State,
) -> tuple[list[AVLEigenmode], list[AVLEigenmode]]:
    DB = Database.get_instance()
    DYNAMICS_DIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="AVL",
        case="Dynamics",
    )
    log = os.path.join(DYNAMICS_DIR, "eig_log.txt")
    parser = AVLOutputParser(log)

    eigenmodes = parser.parse_eigenmodes()
    if not eigenmodes:
        raise AVLPostReadError("No eigenmodes found in the log file.")

    longitudal_matrix: list[AVLEigenmode] = []
    lateral_matrix: list[AVLEigenmode] = []
    for mode in eigenmodes:
        if mode.mode_number <= 4:
            longitudal_matrix.append(mode)
        else:
            lateral_matrix.append(mode)

    return longitudal_matrix, lateral_matrix


cols: list[str] = [
    "Epsilon",
    "Type",
    "AVL Fx",
    "AVL Fy",
    "AVL Fz",
    "AVL Mx",
    "AVL My",
    "AVL Mz",
]
