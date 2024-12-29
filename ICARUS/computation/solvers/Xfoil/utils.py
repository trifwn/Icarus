def angles_sepatation(all_angles: list[float]) -> tuple[list[float], list[float]]:
    """Separate angles in positive and negative.

    Args:
        all_angles (_type_): _description_
    Returns:
        _type_: _description_

    """
    pangles: list[float] = []
    nangles: list[float] = []
    for ang in all_angles:
        if ang > 0:
            pangles.append(ang)
        elif ang == 0:
            pangles.append(ang)
            nangles.append(ang)
        else:
            nangles.append(ang)
    nangles = nangles[::-1]
    return nangles, pangles
