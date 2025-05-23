from ICARUS.flight_dynamics import Disturbance


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


def case_to_angle(case: str) -> float:
    """Convert case folder name to angle

    Args:
        case (str): folder name

    Returns:
        float: Angle of simulation

    """
    if case[0] == "m":
        angle: float = -float(case[1:7])
    else:
        angle = float(case[:7])
    return angle


def disturbance_to_case(dst: Disturbance) -> str:
    """Convert disturbance to case folder name

    Args:
        dst (Disturbance): Disturbance

    Returns:
        str: folder name

    """
    if dst.var == "Trim" or dst.amplitude is None:
        return "Trim"

    amplitude: float = abs(dst.amplitude)
    # Round to 6 decimal in scientific notation
    amplitude_str = f"{amplitude:.3e}".zfill(9)
    if dst.is_positive:
        folder = "p" + amplitude_str + f"_{dst.var}"
    else:
        folder = "m" + amplitude_str + f"_{dst.var}"
    return folder


def case_to_disturbance(case: str) -> Disturbance:
    """Convert case folder name to disturbance

    Args:
        case (str): folder name

    Returns:
        Disturbance: Disturbance

    """
    if case == "Trim":
        dst: Disturbance = Disturbance("Trim", 0)
    elif case[0] == "m":
        print(case, case[8:], float(case[1:7]))
        dst = Disturbance(case[9:], -float(case[1:7]))
    else:
        dst = Disturbance(case[9:], float(case[1:7]))
    return dst
