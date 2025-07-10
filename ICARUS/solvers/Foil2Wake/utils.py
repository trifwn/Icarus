from ICARUS.core.types import FloatArray


def separate_angles(
    all_angles: list[float] | FloatArray,
) -> tuple[list[float], list[float]]:
    """Given A list of angles it separates them in positive and negative

    Args:
        all_angles (list[float]): Angles to separate
    Returns:
        tuple[list[float], list[float]]: Tuple of positive and negative angles

    """
    pangles: list[float] = []
    nangles: list[float] = []
    for ang in all_angles:
        ang = float(ang)
        if ang > 0:
            pangles.append(ang)
        elif ang == 0:
            pangles.append(ang)
        else:
            nangles.append(ang)
    nangles = nangles[::-1]
    return nangles, pangles
