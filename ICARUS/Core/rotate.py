from typing import Any

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray


def y_rotation_stability_axes(angle: float) -> ndarray[Any, dtype[floating[Any]]]:
    """Returns the rotation matrix for a rotation around the y axis
    The convention for stability axes is with the x-axis pointing forward.
    So it is x and z are opposite of the general coordinate system.
    Args:
        angle (float): Angle of rotation in radians

    Returns:
        ndarray[Any, dtype[floating[Any]]]: Rotation matrix
    """
    return np.array(
        [
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)],
        ],
    )


def rotate_vector(
    vector: ndarray[Any, dtype[floating[Any]]],
    R: ndarray[Any, dtype[floating[Any]]],
) -> ndarray[Any, dtype[floating[Any]]]:
    """_Returns the rotated vectorsummary_

    Args:
        vector (ndarray[Any, dtype[floating[Any]]]): Vector to be rotated
        R (ndarray[Any, dtype[floating[Any]]]): Rotation matrix

    Returns:
        ndarray[Any, dtype[floating[Any]]]: Rotated vector
    """
    return np.array(np.dot(R, vector), dtype=float)
