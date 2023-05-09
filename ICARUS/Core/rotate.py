from typing import Any

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray


def y_rotation_stability_axes(angle: float) -> ndarray[Any, dtype[floating]]:
    """Returns the rotation matrix for a rotation around the y axis
    The convention for stability axes is with the x-axis pointing forward.
    So it is x and z are opposite of the general coordinate system.
    Args:
        angle (float): Angle of rotation in radians

    Returns:
        ndarray[Any, dtype[floating]]: Rotation matrix
    """
    return np.array(
        [
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)],
        ],
    )


def rotate_vector(
    vector: ndarray[Any, dtype[floating]],
    R: ndarray[Any, dtype[floating]],
) -> ndarray[Any, dtype[floating]]:
    """_Returns the rotated vectorsummary_

    Args:
        vector (ndarray[Any, dtype[floating]]): Vector to be rotated
        R (ndarray[Any, dtype[floating]]): Rotation matrix

    Returns:
        ndarray[Any, dtype[floating]]: Rotated vector
    """
    return np.dot(R, vector)
