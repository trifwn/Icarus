import os

import numpy as np
from pandas import DataFrame

from ICARUS.core.types import FloatArray
from ICARUS.database import DB
from ICARUS.database import DB3D
from ICARUS.database.utils import angle_to_case
from ICARUS.database.utils import disturbance_to_case
from ICARUS.flight_dynamics.state import State
from ICARUS.vehicle.plane import Airplane


class AVLPostReadError(Exception):
    pass


def csplit(input_file: str, pattern: str) -> list[str]:
    with open(input_file) as file:
        content = file.read()

    import re

    sections = re.split(pattern, content)
    sections = [section.strip() for section in sections if section.strip()]

    return sections

def moving_vars_assignment(plane,state):
    RESULTS_DIR = os.path.join(DB3D, plane.directory, "AVL")
    file = os.path.join(RESULTS_DIR, f"mov_log.txt")
    with open(file, encoding="utf-8") as f:
            con = f.readlines()
    for surf in plane.surfaces:
        surf.moving_surfs["avl_vars"] = []

        for n in surf.moving_surfs["names"]:
            for l in con:
                if l.find(f"->  {n}") != -1:
                    surf.moving_surfs["avl_vars"].append(l[2:4])


def collect_avl_polar_forces(plane: Airplane, state: State, angles: FloatArray | list[float]) -> DataFrame:
    """POST-PROCESSING OF POLAR RUNS - RETURNS AN ARRAY WITH THE FOLLOWING ORDER OF VECTORS: AOA,CL,CD,CM

    Args:
        plane (Airplane): Airplane object
        state (State): State object
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
        try:
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
        except ValueError:
            raise AVLPostReadError(f"Error reading file {file}")
    Fz = np.array(CLs) * plane.S * state.dynamic_pressure
    Fx = np.array(CDs) * plane.S * state.dynamic_pressure
    My = np.array(Cms) * plane.S * state.dynamic_pressure * plane.mean_aerodynamic_chord

    polar_df: DataFrame = DataFrame(
        np.array([AoAs, Fz, Fx, My]).T,
        columns=["AoA", "AVL Fz", "AVL Fx", "AVL My"],
    )
    polar_df = polar_df.sort_values("AoA").reset_index(drop=True)

    file_2_save = os.path.join(DB3D, plane.directory, "forces.avl")
    polar_df.to_csv(file_2_save, index=False, float_format="%.10f")
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

            try:
                CX = float(x_axis[11:19])
                Cl = float(x_axis[33:41])
                CY = float(y_axis[11:19])
                Cm = float(y_axis[33:41])
                CZ = float(z_axis[11:19])
                Cn = float(z_axis[33:41])
            except ValueError:
                raise AVLPostReadError(f"Error reading file {casefile}")

            if dst.var == "u":
                dyn_pressure = float(
                    0.5
                    * state.environment.air_density
                    * float(
                        np.linalg.norm(
                            [
                                state.trim["U"] * np.cos(state.trim["AoA"] * np.pi / 180) + dst.amplitude,
                                state.trim["U"] * np.sin(state.trim["AoA"] * np.pi / 180),
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
                                state.trim["U"] * np.cos(state.trim["AoA"] * np.pi / 180),
                                state.trim["U"] * np.sin(state.trim["AoA"] * np.pi / 180) + dst.amplitude,
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

    pertrubation_df = DataFrame(results, columns=cols)
    pertrubation_df["Epsilon"] = pertrubation_df["Epsilon"].astype(float)
    return pertrubation_df


def implicit_dynamics_post(plane: Airplane, state: State) -> tuple[list[complex], list[complex]]:
    PLANEDIR = os.path.join(DB3D, plane.directory)
    DYNAMICS_DIR = os.path.join(PLANEDIR, "AVL", "Dynamics")
    log = os.path.join(DYNAMICS_DIR, "eig_log.txt")
    sections = csplit(log, "1:")
    sec_2_use = sections[-1].splitlines()
    sec_2_use[0] = "  mode 1:  " + sec_2_use[0]

    def get_matrix(
        index: int,
        lines: list[str],
    ) -> tuple[FloatArray, FloatArray, complex]:
        """Extracts the EigenVector and EgienValue from AVL Output

        Args:
            index (int): Index in Reading File
            lines (list[str]): AVL output

        Returns:
            tuple[FloatArray, FloatArray, FloatArray]: Longitudal EigenVector, Lateral EigenVector, Mode
        """
        mode = complex(float(lines[index][10:22]), float(lines[index][24:36]))
        long_vecs = np.zeros((4), dtype=complex)
        lat_vecs = np.zeros((4), dtype=complex)
        for i in range(0, 4):
            long_vecs[i] = complex(float(lines[index + i + 1][8:18]), float(lines[index + i + 1][19:28]))

            lat_vecs[i] = complex(
                float(lines[index + i + 1][40:50]),
                float(lines[index + i + 1][51:60]),
            )

        return long_vecs, lat_vecs, mode

    longitudal_matrix = []
    lateral_matrix = []
    indexes = np.arange(0, 8, 1) * 6
    for i in indexes:
        long_vec, lat_vec, mode = get_matrix(i, sec_2_use)
        if float(np.mean(np.abs(long_vec))) < float(np.mean(np.abs(lat_vec))):
            lateral_matrix.append(mode)
        else:
            longitudal_matrix.append(mode)

    return longitudal_matrix, lateral_matrix


cols: list[str] = ["Epsilon", "Type", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
