import os
import shutil

import numpy as np


def io_file(airfile: str, name: str) -> None:
    """Creates the io.files file for section f2w

    Args:
        airfile (str): Name of the file containing the airfoil geometry
        name (str): Positive or Negative Run
    """
    fname = f"io_{name}.files"
    with open(fname, encoding="utf-8") as file:
        data: list[str] = file.readlines()
    data[1] = f"design_{name}.inp\n"
    data[2] = f"f2w_{name}.inp\n"
    data[3] = f"{airfile}\n"

    data[13] = f"SOL{name}.INI\n"
    data[14] = f"SOL{name}.TMP\n"
    data[15] = f"zx_{name}"
    with open(fname, "w", encoding="utf-8") as file:
        file.writelines(data)


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
    with open(fname, encoding="utf-8") as file:
        data: list[str] = file.readlines()[:3]
    data[2] = f"{number_of_angles}           ! No of ANGLES\n"
    for ang in angles:
        data.append(str(ang) + "\n")
    data.append("ANGLE DIRECTORIES (8 CHAR MAX!!!)\n")
    for ang in angles:
        if name == "pos":
            data.append(str(ang)[::-1].zfill(7)[::-1] + "/\n")
        else:
            data.append("m" + str(ang)[::-1].strip("-").zfill(6)[::-1] + "/\n")
    with open(fname, "w", encoding="utf-8") as file:
        file.writelines(data)


def input_file(
    reynolds: float,
    mach: float,
    max_iter: float,
    timestep: float,
    ftrip_low: float,
    ftrip_upper: float,
    Ncrit: float,
    name: str,
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
    with open(fname, encoding="utf-8") as file:
        data: list[str] = file.readlines()

    data[2] = f"{max_iter}       ! NTIMEM\n"
    data[3] = f"{timestep}     ! DT1\n"
    data[4] = "5000      ! DT2\n"  # IS NOT IMPLEMENTED
    data[5] = "0.025     ! EPS1\n"
    data[6] = "0.025     ! EPS2\n"
    data[7] = "1.00      ! EPSCOE\n"
    data[27] = "400       ! NTIME_bl\n"
    data[30] = f"{np.format_float_scientific(reynolds, sign=False, precision=3, min_digits=3).zfill(8)}  ! Reynolds\n"
    data[32] = f"{str(mach)[::-1].zfill(3)[::-1]}      ! Mach     Number\n"
    data[34] = f"{str(ftrip_low)[::-1].zfill(3)[::-1]}    1  ! TRANSLO\n"
    data[35] = f"{str(ftrip_upper)[::-1].zfill(3)[::-1]}    2  ! TRANSLO\n"
    data[36] = f"{int(Ncrit)}\t\t  ! AMPLUP_tr\n"
    data[37] = f"{int(Ncrit)}\t\t  ! AMPLUP_tr\n"

    with open(fname, "w", encoding="utf-8") as file:
        file.writelines(data)


def setup_f2w(F2WBASE: str, HOMEDIR: str, CASEDIR: str) -> None:
    """
    Sets up the f2w case copying and editing all necessary files

    Args:
        F2WBASE (str): Base Case Directory for f2w
        HOMEDIR (str): Home Directory
        CASEDIR (str): Case Directory
    """
    filesNeeded: list[str] = [
        "design.inp",
        "design_neg.inp",
        "design_pos.inp",
        "f2w.inp",
        "f2w_neg.inp",
        "f2w_pos.inp",
        "io_neg.files",
        "io_pos.files",
        "write_out",
    ]
    for item in filesNeeded:
        src: str = os.path.join(F2WBASE, item)
        dst: str = os.path.join(CASEDIR, item)
        shutil.copy(src, dst)

    if "foil_section" not in next(os.walk(CASEDIR))[2]:
        src = os.path.join(HOMEDIR, "ICARUS", "foil_section")
        dst = os.path.join(CASEDIR, "foil_section")
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass
