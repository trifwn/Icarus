from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import deg2rad
from numpy import rad2deg

from ICARUS import AVL_exe
from ICARUS.core.types import FloatArray
from ICARUS.database.utils import angle_to_directory
from ICARUS.database.utils import disturbance_to_directory
from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane


@dataclass
class AVLRunCase:
    name: str
    alpha: float = 0.0  # Angle of attack [deg]
    beta: float = 0.0  # Sideslip angle [deg]
    pb_2V: float = 0.0  # Roll rate normalized by 2V
    qc_2V: float = 0.0  # Pitch rate normalized by 2V
    rb_2V: float = 0.0  # Yaw rate normalized by 2V
    control_inputs: dict[str, float] | None = (
        None  # Dict of control surface names and values
    )

    def to_avl_lines(self, case_number: int) -> list[str]:
        lines = [
            "-" * 45,
            f"Run case  {case_number}:  *{self.name}*",
            "",
            f"alpha        ->  alpha       =  {self.alpha:.9f}",
            f"beta         ->  beta        =  {self.beta:.9f}",
            f"pb/2V        ->  pb/2V       =  {self.pb_2V:.9f}",
            f"qc/2V        ->  qc/2V       =  {self.qc_2V:.9f}",
            f"rb/2V        ->  rb/2V       =  {self.rb_2V:.9f}",
        ]
        if self.control_inputs:
            lines += [
                f"{name:<12} ->  {name:<12} =  {value:.9f}"
                for name, value in self.control_inputs.items()
            ]
        return lines


class AVLRunSetup:
    def __init__(self, name: str, cases: list[AVLRunCase]) -> None:
        self.name = name

        self.forces_file = f"total_forces_{name.strip('.run')}.avl"
        self.strip_forces_file = f"strip_forces_{name.strip('.run')}.avl"

        self.cases = cases

    def write_run_file(self, directory: str) -> None:
        all_lines = []
        for i, rc in enumerate(self.cases, 1):
            all_lines += rc.to_avl_lines(i)

        output_path = Path(directory) / self.name
        with open(output_path, "w", encoding="utf-8") as f_io:
            for line in all_lines:
                f_io.write(f"{line}\n")

    @classmethod
    def aseq(
        cls,
        name: str,
        state: State,
        angles: list[float] | FloatArray,
    ) -> AVLRunSetup:
        """Create a run setup for polar analysis."""
        cases = [
            AVLRunCase(
                name=angle_to_directory(angle),
                alpha=angle,
                control_inputs=state.control_vector_dict,
            )
            for angle in angles
        ]
        return cls(name=name, cases=cases)

    @classmethod
    def stability_fd(
        cls,
        name: str,
        state: State,
        plane: Airplane,
    ) -> AVLRunSetup:
        """Create a run setup from a list of disturbances."""

        # Gather Variables
        aoa_trim = state.trim["AoA"]
        U = state.trim["U"]
        W = np.tan(deg2rad(aoa_trim)) * U

        cases = []

        for dst in state.disturbances:
            aoa = aoa_trim
            beta = 0.0
            pitch_rate = 0.0
            roll_rate = 0.0
            yaw_rate = 0.0

            if dst.amplitude is None:
                pass  # No disturbance, keep trim values
            else:
                match dst.var:
                    case "u":
                        aoa = rad2deg(np.arctan(W / (U + dst.amplitude)))
                    case "w":
                        aoa = rad2deg(np.arctan((W + dst.amplitude) / U))
                    case "q":
                        pitch_rate = (
                            dst.amplitude * plane.mean_aerodynamic_chord / (2 * U)
                        )
                    case "v":
                        beta = rad2deg(np.arctan(dst.amplitude / U))
                    case "p":
                        roll_rate = dst.amplitude * plane.span / (2 * U)
                    case "r":
                        yaw_rate = dst.amplitude * plane.span / (2 * U)
                    case "theta":
                        aoa = rad2deg(np.arctan(W / U)) + dst.amplitude
                    case "phi":
                        beta = rad2deg(np.arctan(W / U)) + dst.amplitude
                    case _:
                        print(f"Got unexpected var {dst.var}")
                        continue

            run_case = AVLRunCase(
                name=disturbance_to_directory(dst),
                alpha=aoa,
                beta=beta,
                pb_2V=roll_rate,
                qc_2V=pitch_rate,
                rb_2V=yaw_rate,
                control_inputs=state.control_vector_dict.copy(),  # assume static for all
            )
            cases.append(run_case)

        return cls(name=name, cases=cases)


def avl_run_cases(directory: str, plane: Airplane, avl_run: AVLRunSetup) -> None:
    # Create the run file
    avl_run.write_run_file(directory)

    # Ensure force output files exist
    open(os.path.join(directory, avl_run.forces_file), "w").close()
    open(os.path.join(directory, avl_run.strip_forces_file), "w").close()

    # Build script lines
    script_lines = [
        f"load {plane.name}.avl",
        f"mass {plane.name}.mass",
        f"case {avl_run.name}",
        "MSET 0",
        "oper",
        "MRF",
    ]

    for i, case in enumerate(avl_run.cases):
        script_lines.extend(
            [
                f"{i + 1}",
                "x",
                "FT",
                avl_run.forces_file,
                "a",
                "fs",
                avl_run.strip_forces_file,
                "a",
            ],
        )

    script_lines.extend(["", "quit"])  # "" for the empty/space line

    file_in = os.path.join(directory, f"{avl_run.name.strip('.run')}_script.in")
    file_out = os.path.join(directory, f"{avl_run.name.strip('.run')}_script.out")

    # Write input script to file
    with open(file_in, "w") as f:
        f.write("\n".join(script_lines))

    # Do the same with subprocess
    with open(file_in) as fin:
        with open(file_out, "w") as fout:
            res = subprocess.check_call(
                [AVL_exe],
                stdin=fin,
                stdout=fout,
                # stderr=fout,
                cwd=directory,
            )
    logging.debug(f"AVL return code: {res}")
