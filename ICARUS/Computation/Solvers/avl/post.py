import os

import numpy as np
import pandas
from pandas import DataFrame

from ICARUS.Computation.Solvers.AVL import AVL_HOME
from ICARUS.Core.types import FloatArray
from ICARUS.Database import DB
from ICARUS.Database import DB3D
from ICARUS.Database.utils import angle_to_case
from ICARUS.Database.utils import disturbance_to_case
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane


def process_avl_angle_run(plane: Airplane, angles: FloatArray) -> DataFrame:
    """POST-PROCESSING OF POLAR RUNS - RETURNS AN ARRAY WITH THE FOLLOWING ORDER OF VECTORS: AOA,CL,CD,CM

    Args:
        plane (Airplane): Airplane object
        angles (FloatArray): Array of angles of attack

    Returns:
        FloatArray: Array with the following order of vectors: AOA,CL,CD,CM
    """

    AoAs = []
    CLs = []
    CDs = []
    Cms = []
    RESULTS_DIR = os.path.join(DB3D, plane.directory, "AVL")
    for angle in angles:
        file = os.path.join(RESULTS_DIR, f"{angle_to_case(angle)}.txt")

        with open(file, encoding="utf-8") as f:
            con = f.readlines()

        CL = con[23]
        CD = con[24]
        Cm = con[20]

        AoAs.append(angle)
        if CL[11] == "-":
            CLs.append(float(CL[11:19]))
        else:
            CLs.append(float(CL[12:19]))
        if CD[11] == "-":
            CDs.append(float(CD[11:19]))
        else:
            CDs.append(float(CD[12:19]))
        if Cm[33] == "-":
            Cms.append(float(Cm[33:41]))
        else:
            Cms.append(float(Cm[34:41]))
    polar_df = DataFrame(
        np.array([AoAs, CLs, CDs, Cms]).T,
        columns=["AoA", "AVL CL", "AVL CD", "AVL Cm"],
    ).reset_index(drop=True)
    file_2_save = os.path.join(DB3D, plane.directory, "polars.avl")
    DB.vehicles_db.polars[plane.name] = polar_df
    polar_df.to_csv(file_2_save)
    return polar_df


def finite_difs_post(plane: Airplane, state: State) -> DataFrame:
    DYNDIR = os.path.join(DB3D, plane.name, "AVL", "Dynamics")
    results = []
    # pertrubation_df: DataFrame = DataFrame()

    for dst in state.disturbances:
        casefile = os.path.join(DYNDIR, disturbance_to_case(dst))
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

            CX = float(x_axis[11:19])
            Cl = float(x_axis[33:41])
            CY = float(y_axis[11:19])
            Cm = float(y_axis[33:41])
            CZ = float(z_axis[11:19])
            Cn = float(z_axis[33:41])

            if dst.var == "u":
                dyn_pressure = float(
                    0.5
                    * state.env.air_density
                    * np.linalg.norm(
                        [
                            state.trim["U"] * np.cos(state.trim["AoA"] * np.pi / 180) + dst.amplitude,
                            state.trim["U"] * np.sin(state.trim["AoA"] * np.pi / 180),
                        ],
                    ),
                )
            elif dst.var == "w":
                dyn_pressure = float(
                    0.5
                    * state.env.air_density
                    * np.linalg.norm(
                        [
                            state.trim["U"] * np.cos(state.trim["AoA"] * np.pi / 180),
                            state.trim["U"] * np.sin(state.trim["AoA"] * np.pi / 180) + dst.amplitude,
                        ],
                    ),
                )
            else:
                dyn_pressure = state.dynamic_pressure

            Fx = CX * dyn_pressure * plane.S
            Fy = CY * dyn_pressure * plane.S
            Fz = CZ * dyn_pressure * plane.S
            M = Cm * dyn_pressure * plane.S * plane.mean_aerodynamic_chord
            N = Cn * dyn_pressure * plane.S * plane.mean_aerodynamic_chord
            L = Cl * dyn_pressure * plane.S * plane.mean_aerodynamic_chord
        if dst.amplitude is None:
            ampl = 0.0
        else:
            ampl = float(dst.amplitude)
        results.append(np.array([ampl, dst.var, Fx, Fy, Fz, L, M, N]))

    pertrubation_df = DataFrame(results, columns=cols)
    pertrubation_df["Epsilon"] = pertrubation_df["Epsilon"].astype(float)
    return pertrubation_df


cols: list[str] = ["Epsilon", "Type", "Fx", "Fy", "Fz", "L", "M", "N"]
