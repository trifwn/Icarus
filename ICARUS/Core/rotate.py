import numpy as np


def yRotationSB(angle):
    """Returns the rotation matrix for a rotation around the y axis
    The convention for stability axes is with the x-axis pointing forward.
    So it is x and z are opposite of the general coordinate system."""
    return np.array(
        [
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)],
        ],
    )


def rotate(vector, R):
    """Returns the rotated vector"""
    return np.dot(R, vector)
