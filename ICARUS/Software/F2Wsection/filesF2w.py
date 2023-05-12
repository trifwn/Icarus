import numpy as np


def io_file(airfile: str) -> None:
    """Creates the io.files file for section f2w

    Args:
        airfile (str): Name of the file containing the airfoil geometry
    """
    fname = "io.files"
    with open(fname, encoding="utf-8") as file:
        data: list[str] = file.readlines()
    data[1] = "design.inp\n"
    data[2] = "f2w.inp\n"
    data[3] = f"{airfile}\n"
    with open(fname, "w", encoding="utf-8") as file:
        file.writelines(data)


def desing_file(number_of_angles: int, angles: list[float], name: str) -> None:
    """Generates the desing.inp file for section f2w. Depending on the name, it will generate the file for positive or negative angles

    Args:
        number_of_angles (int): Number of angles
        angles (list[float]): List of angles
        name (str): pos or neg. Meaning positive or negative run
    """
    fname: str = f"design_{name}.inp"
    with open(fname, encoding="utf-8") as file:
        data = file.readlines()
    data[2] = f"{number_of_angles}           ! No of ANGLES\n"
    data: list[str] = data[:3]
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
    ftrip_low: dict[str, float],
    ftrip_upper: dict[str, float],
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

    data[2] = "201       ! NTIMEM\n"
    data[3] = "0.010     ! DT1\n"
    data[4] = "50000     ! DT2\n"  # IS NOT IMPLEMENTED
    data[5] = "0.025     ! EPS1\n"
    data[6] = "0.025     ! EPS2\n"
    data[7] = " 1.00     ! EPSCOE\n"
    data[27] = "200       ! NTIME_bl\n"
    data[
        30
    ] = f"{np.format_float_scientific(reynolds, sign=False, precision=2).zfill(8)}  ! Reynolds\n"
    data[32] = f"{str(mach)[::-1].zfill(3)[::-1]}      ! Mach     Number\n"
    data[34] = f"{str(ftrip_low[name])[::-1].zfill(3)[::-1]}    1  ! TRANSLO\n"
    data[35] = f"{str(ftrip_upper[name])[::-1].zfill(3)[::-1]}    2  ! TRANSLO\n"
    with open(fname, "w", encoding="utf-8") as file:
        file.writelines(data)
