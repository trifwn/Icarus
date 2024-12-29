import numpy as np

from ICARUS.core.types import FloatArray


def y_rotation_stability_axes(angle: float) -> FloatArray:
    """Returns the rotation matrix for a rotation around the y axis
    The convention for stability axes is with the x-axis pointing forward.
    So it is x and z are opposite of the general coordinate system.

    Args:
        angle (float): Angle of rotation in radians

    Returns:
        FloatArray: Rotation matrix

    """
    return np.array(
        [
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)],
        ],
    )


def rotate_vector(
    vector: FloatArray,
    R: FloatArray,
) -> FloatArray:
    """Returns the rotated vectorsummary

    Args:
        vector (FloatArray): Vector to be rotated
        R (FloatArray): Rotation matrix

    Returns:
        FloatArray: Rotated vector

    """
    return np.array(np.dot(R, vector), dtype=float)
