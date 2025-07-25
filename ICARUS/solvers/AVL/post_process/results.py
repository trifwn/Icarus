import os

import numpy as np
from pandas import DataFrame

from ICARUS.database import Database
from ICARUS.database import directory_to_disturbance
from ICARUS.flight_dynamics import State
from ICARUS.solvers.AVL.post_process import AVLOutputParser
from ICARUS.solvers.AVL.post_process.output_parser import AVLEigenmode
from ICARUS.vehicle import Airplane

from ..files.cases import AVLRunSetup


class AVLPostReadError(Exception):
    pass


def collect_avl_polar_forces(
    directory: str,
    plane: Airplane,
    state: State,
    avl_run: AVLRunSetup,
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
    forces_file = avl_run.forces_file
    # strip_forces_file = "strip_forces.avl"
    file_path = os.path.join(directory, forces_file)
    parser = AVLOutputParser(file_path)
    all_forces = parser.parse_forces()

    polar_df: DataFrame = AVLOutputParser.to_dataframe(all_forces)

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


def finite_difs_post(
    plane: Airplane,
    state: State,
    run_setup: AVLRunSetup,
) -> DataFrame:
    DB = Database.get_instance()
    directory = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="AVL",
        case="Dynamics",
    )
    forces_file = run_setup.forces_file
    # strip_forces_file = "strip_forces.avl"
    file_path = os.path.join(directory, forces_file)
    parser = AVLOutputParser(file_path)
    all_forces = parser.parse_forces()

    polar_df: DataFrame = AVLOutputParser.to_dataframe(all_forces)

    aoa = state.trim["AoA"] * np.pi / 180
    results = []
    for index, row in polar_df.iterrows():
        dst = directory_to_disturbance(row["name"])

        CX = float(row["CX"])
        CY = float(row["CY"])
        CZ = float(row["CZ"])

        Cl = float(row["Cl"])
        Cm = float(row["Cm"])
        Cn = float(row["Cn"])

        if dst.var == "u":
            dyn_pressure = float(
                0.5
                * state.aero_state.density
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
                * state.aero_state.density
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

        ampl = float(dst.amplitude or 0.0)
        results.append(np.array([ampl, dst.var, Fx, Fy, Fz, L, M, N]))

    df = DataFrame(results, columns=cols)
    df = df.sort_values("Type").reset_index(drop=True)
    df["Epsilon"] = df["Epsilon"].astype(float)
    perturbation_file: str = os.path.join(directory, "pertrubations.avl")
    df.to_csv(perturbation_file, index=False)
    return df


def implicit_dynamics_post(
    plane: Airplane,
    state: State,
) -> list[AVLEigenmode]:
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

    return eigenmodes


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
