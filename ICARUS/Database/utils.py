from ICARUS.Flight_Dynamics.disturbances import Disturbance


def angle_to_case(angle: float) -> str:
    """Convert angle to case folder name

    Args:
        angle (float): Angle of simulation

    Returns:
        str: folder name
    """

    if angle >= 0:
        folder: str = str(angle)[::-1].zfill(7)[::-1] + "_AoA"
    else:
        folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1] + "_AoA"
    return folder


def disturbance_to_case(dst: Disturbance) -> str:
    """Convert disturbance to case folder name

    Args:
        dst (Disturbance): Disturbance

    Returns:
        str: folder name
    """

    if dst.var == "Trim":
        folder: str = "Trim"
    elif dst.isPositive:
        folder = "p" + str(dst.amplitude)[::-1].zfill(6)[::-1] + f"_{dst.var}"
    else:
        folder = (
            "m" + str(dst.amplitude)[::-1].strip("-").zfill(6)[::-1] + f"_{dst.var}"
        )
    return folder
