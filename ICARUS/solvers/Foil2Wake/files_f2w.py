from typing import Any

import numpy as np

from ICARUS import PLATFORM


def io_file(airfile: str, name: str) -> None:
    """Creates the io.files file for section f2w

    Args:
        airfile (str): Name of the file containing the airfoil geometry
        name (str): Positive or Negative Run

    """
    fname = f"io_{name}.files"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("***** input files  *****\n")
        f.write(f"design_{name}.inp\n")
        f.write(f"f2w_{name}.inp\n")
        f.write(f"{airfile}\n")
        f.write("--\n")
        f.write("--\n")
        f.write("***** OUTPUT FILES *****\n")
        f.write("AIRFOIL.OUT\n")
        f.write("TREWAKE.OUT\n")
        f.write("SEPWAKE.OUT\n")
        f.write("COEFPRE.OUT\n")
        f.write("AERLOAD.OUT\n")
        f.write("BDLAYER.OUT\n")
        f.write(f"SOL{name}.INI\n")
        f.write(f"SOL{name}.TMP\n")
        if PLATFORM == "Windows":
            f.write(f"TMP_{name}\\ \n")
        else:
            f.write(f"TMP_{name}/\n")
        f.write("\n")
        f.write("\n")


def design_file(
    number_of_angles: int,
    angles: list[float],
    name: str,
) -> None:
    """Generates the desing.inp file for section f2w. Depending on the name, it will generate
    the file for positive or negative angles

    Args:
        number_of_angles (int): Number of angles
        angles (list[float]): List of angles
        name (str): pos or neg. Meaning positive or negative run

    """
    fname: str = f"design_{name}.inp"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"{angles[0]}\n")
        f.write("0            ! ISOL\n")
        f.write(f"{number_of_angles}           ! No of ANGLES\n")

        for ang in angles:
            f.write(str(ang) + "\n")
        f.write("ANGLE DIRECTORIES (8 CHAR MAX!!!)\n")
        for ang in angles:
            if name == "pos":
                f.write(str(ang)[::-1].zfill(7)[::-1])
            else:
                f.write("m" + str(ang)[::-1].strip("-").zfill(6)[::-1])
            from ICARUS import PLATFORM

            if PLATFORM == "Windows":
                f.write("\\ \n")
            else:
                f.write("/\n")

        f.write("\n")
        f.write("\n")


def input_file(
    reynolds: float,
    mach: float,
    max_iter: float,
    timestep: float,
    ftrip_low: float,
    ftrip_upper: float,
    Ncrit: float,
    name: str,
    solver_parameters: dict[str, tuple[Any, str, Any]],
) -> None:
    """Creates the input file for section f2w program

    Args:
        Reynolds (float): Reynolds number for this simulation
        Mach (float): Mach Number for this simulation
        ftrip_low (dict[str, float]): Dictionary of lower transition points for positive and negative angles
        ftrip_upper (dict[str,float]): Dictionary of upper transition points for positive and negative angles
        name (str): _description_

    """
    fname: str = f"f2w_{name}.inp"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("0.        ! TEANGLE (deg)\n")
        f.write("1.        ! UINF\n")
        f.write(f"{max_iter}     ! NTIMEM\n")
        f.write(f"{timestep}     ! DT1\n")
        f.write(f"{timestep}     ! DT2\n")  # IS NOT IMPLEMENTED
        f.write(f"{solver_parameters['Cuttoff_1']}    ! EPS1\n")
        f.write(f"{solver_parameters['Cuttoff_2']}    ! EPS2\n")
        f.write(f"{solver_parameters['EPSCOE']}     ! EPSCOE\n")
        f.write(f"{solver_parameters['NWS']}        ! NWS\n")
        f.write(f"{solver_parameters['CCC1']}    ! CCC1\n")
        f.write(f"{solver_parameters['CCC2']}    ! CCC2\n")
        f.write(f"{solver_parameters['CCGON1']}      ! CCGON1\n")
        f.write(f"{solver_parameters['CCGON2']}      ! CCGON2\n")
        f.write(f"{solver_parameters['IMOVE']}        ! IMOVE\n")
        f.write(f" {solver_parameters['A0']}   ! A0\n")
        f.write(f" {solver_parameters['AMPL']}   ! AMPL\n")
        f.write(f" {solver_parameters['APHASE']}   ! APHASE\n")
        f.write(f" {solver_parameters['AKF']}   ! AKF\n")
        f.write(f"{solver_parameters['Chord_hinge']}     ! XC\n")
        f.write(f"{solver_parameters['ITEFLAP']}        ! ITEFLAP\n")
        f.write(f"{solver_parameters['XEXT']}     ! XEXT\n")
        f.write(f"{solver_parameters['YEXT']}      ! YEXT\n")
        f.write("\n")
        f.write(f"{solver_parameters['NTEWT']}        ! NTEWT\n")
        f.write(f"{solver_parameters['NTEST']}        ! NTEST\n")
        f.write("\n")
        f.write(f"{solver_parameters['IBOUNDL']}        ! IBOUNDL\n")
        f.write(f"{solver_parameters['boundary_layer_solve_time']}      ! NTIME_bl\n")
        f.write(f"{solver_parameters['IYNEXTERN']}        ! IYNEXTERN\n")
        f.write("\n")
        f.write(
            f"{np.format_float_scientific(reynolds, sign=False, precision=3, min_digits=3).zfill(8)}  ! Reynolds\n",
        )
        f.write("\n")
        f.write(f"{str(mach)[::-1].zfill(3)[::-1]}      ! Mach     Number\n")
        f.write("\n")
        f.write(f"{str(ftrip_low)[::-1].zfill(3)[::-1]}    1  ! TRANSLO\n")
        f.write(f"{str(ftrip_upper)[::-1].zfill(3)[::-1]}    2  ! TRANSLO\n")
        f.write(f"{int(Ncrit)}\t\t  ! AMPLUP_tr\n")
        f.write(f"{int(Ncrit)}\t\t  ! AMPLUP_tr\n")
        f.write("\n")
        f.write(
            f"{solver_parameters['ITSEPAR']}         ! ITSEPAR (1: 2 wake calculation)\n",
        )
        f.write(
            f"{solver_parameters['ISTEADY']}         ! ISTEADY (1: steady calculation)\n",
        )
        f.write("\n")
        f.write("\n")
        f.write("\n")
