import os
from typing import Any

import numpy as np

from ICARUS import platform_os
from ICARUS.Database import Foil_Section_exe


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
        f.write(f"AIRFOIL.OUT\n")
        f.write(f"TREWAKE.OUT\n")
        f.write(f"SEPWAKE.OUT\n")
        f.write(f"COEFPRE.OUT\n")
        f.write(f"AERLOAD.OUT\n")
        f.write(f"BDLAYER.OUT\n")
        f.write(f"SOL{name}.INI\n")
        f.write(f"SOL{name}.TMP\n")
        if platform_os == "Windows":
            f.write(f"TMP_{name}\\ \n")
        else:
            f.write(f"TMP_{name}/\n")
        f.write(f"\n")
        f.write(f"\n")


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
        f.write(f"0            ! ISOL\n")
        f.write(f"{number_of_angles}           ! No of ANGLES\n")

        for ang in angles:
            f.write(str(ang) + "\n")
        f.write("ANGLE DIRECTORIES (8 CHAR MAX!!!)\n")
        for ang in angles:
            if name == "pos":
                f.write(str(ang)[::-1].zfill(7)[::-1])
            else:
                f.write("m" + str(ang)[::-1].strip("-").zfill(6)[::-1])
            from ICARUS import platform_os

            if platform_os == "Windows":
                f.write("\\ \n")
            else:
                f.write("/\n")

        f.write(f"\n")
        f.write(f"\n")


def input_file(
    reynolds: float,
    mach: float,
    max_iter: float,
    timestep: float,
    ftrip_low: float,
    ftrip_upper: float,
    Ncrit: float,
    name: str,
    solver_options: dict[str, tuple[Any, str, Any]],
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
        f.write(f"{solver_options['Cuttoff_1']}    ! EPS1\n")
        f.write(f"{solver_options['Cuttoff_2']}    ! EPS2\n")
        f.write(f"{solver_options['EPSCOE']}     ! EPSCOE\n")
        f.write(f"{solver_options['NWS']}        ! NWS\n")
        f.write(f"{solver_options['CCC1']}    ! CCC1\n")
        f.write(f"{solver_options['CCC2']}    ! CCC2\n")
        f.write(f"{solver_options['CCGON1']}      ! CCGON1\n")
        f.write(f"{solver_options['CCGON2']}      ! CCGON2\n")
        f.write(f"{solver_options['IMOVE']}        ! IMOVE\n")
        f.write(f" {solver_options['A0']}   ! A0\n")
        f.write(f" {solver_options['AMPL']}   ! AMPL\n")
        f.write(f" {solver_options['APHASE']}   ! APHASE\n")
        f.write(f" {solver_options['AKF']}   ! AKF\n")
        f.write(f"{solver_options['Chord_hinge']}     ! XC\n")
        f.write(f"{solver_options['ITEFLAP']}        ! ITEFLAP\n")
        f.write(f"{solver_options['XEXT']}     ! XEXT\n")
        f.write(f"{solver_options['YEXT']}      ! YEXT\n")
        f.write(f"\n")
        f.write(f"{solver_options['NTEWT']}        ! NTEWT\n")
        f.write(f"{solver_options['NTEST']}        ! NTEST\n")
        f.write(f"\n")
        f.write(f"{solver_options['IBOUNDL']}        ! IBOUNDL\n")
        f.write(f"{solver_options['boundary_layer_solve_time']}      ! NTIME_bl\n")
        f.write(f"{solver_options['IYNEXTERN']}        ! IYNEXTERN\n")
        f.write(f"\n")
        f.write(f"{np.format_float_scientific(reynolds, sign=False, precision=3, min_digits=3).zfill(8)}  ! Reynolds\n")
        f.write(f"\n")
        f.write(f"{str(mach)[::-1].zfill(3)[::-1]}      ! Mach     Number\n")
        f.write(f"\n")
        f.write(f"{str(ftrip_low)[::-1].zfill(3)[::-1]}    1  ! TRANSLO\n")
        f.write(f"{str(ftrip_upper)[::-1].zfill(3)[::-1]}    2  ! TRANSLO\n")
        f.write(f"{int(Ncrit)}\t\t  ! AMPLUP_tr\n")
        f.write(f"{int(Ncrit)}\t\t  ! AMPLUP_tr\n")
        f.write(f"\n")
        f.write(f"{solver_options['ITSEPAR']}         ! ITSEPAR (1: 2 wake calculation)\n")
        f.write(f"{solver_options['ISTEADY']}         ! ISTEADY (1: steady calculation)\n")
        f.write(f"\n")
        f.write(f"\n")
        f.write(f"\n")


def setup_f2w(HOMEDIR: str, CASEDIR: str) -> None:
    """
    Sets up the f2w case copying and editing all necessary files

    Args:
        HOMEDIR (str): Home Directory
        CASEDIR (str): Case Directory
    """
    if "foil_section" not in next(os.walk(CASEDIR))[2]:
        src = Foil_Section_exe
        dst = os.path.join(CASEDIR, "foil_section")
        try:
            os.symlink(src, dst)
        except Exception as e:
            if e == FileExistsError:
                os.remove(dst)
                os.symlink(src, dst)
            elif e == OSError:
                import shutil

                # Permission Error so insted of symlink we do copy
                shutil.copyfile(src, dst)
