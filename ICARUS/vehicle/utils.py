from __future__ import annotations

from enum import Enum
from typing import Callable

import numpy as np


class DiscretizationType(Enum):
    """Discretization types for ICARUS
     3.0        equal         |   |   |   |   |   |   |   |   |
     2.0        sine          || |  |   |    |    |     |     |
     1.0        cosine        ||  |    |      |      |    |  ||
     0.0        equal         |   |   |   |   |   |   |   |   |
    -1.0        cosine        ||  |    |      |      |    |  ||
    -2.0       -sine          |     |     |    |    |   |  | ||
    -3.0        equal         |   |   |   |   |   |   |   |   |
    """

    LINEAR = 3.0
    COSINE = 1.0
    SINE = 2.0
    INV_SINE = -2.0
    INV_COSINE = -1.0
    USER_DEFINED = -1000


class DistributionType(Enum):
    """Distribution types for ICARUS
    1.0        linear
    2.0        parabolic
    3.0        cubic
    4.0        exponential
    4.0        user-defined
    """

    LINEAR = 0.0
    PARABOLIC = 1.0
    CUBIC = 2.0
    EXPONENTIAL = 3.0
    ELLIPTIC = 4.0
    USER_DEFINED = 5.0


class SymmetryAxes(Enum):
    """Symmetry axes for ICARUS
    XZ plane = "Y"
    XY plane = "Z"
    YZ plane = "X"
    """

    NONE = "None"
    Y = "Y"
    Z = "Z"
    X = "X"


############################
# Descritization functions
############################
def equal_spacing_function(Ni: int, N: int, stretching: float) -> float:
    """Returns a linearly spaced array of length N."""
    return (Ni / (N - 1)) * stretching


def cosine_spacing_function(Ni: int, N: int, stretching: float, factor: float = 0.2) -> float:
    """Returns a cosine spaced array of length N."""
    cosine = 0.5 * (1 - np.cos(np.pi * Ni / (N - 1)))
    linear = Ni / (N - 1)
    y = (1 - factor) * cosine + factor * linear
    return y * stretching


def sine_spacing_function(Ni: int, N: int, stretching: float, factor: float = 0) -> float:
    """Returns a sine spaced array of length N."""
    if N < 0:
        N = -N
        y: float = 1.0 - np.sin(np.pi / 2 * (1 - (Ni / (N - 1))))
    else:
        y = np.sin(np.pi / (2 * (N - 1)) * Ni)
    y = (1 - factor) * y + factor * (Ni / (N - 1))
    y = y * stretching
    return y


def cspacer(Ni: int, N: int, cspace: float) -> float:
    """
    Discretize an airfoil chord into points using a blended spacing method.

    Parameters:
      nvc    : int, number of chordwise panels.
      N      : The point id.
      cspace : float, controls the blend between spacing methods.
               Its absolute value and integer part determine the blending weights.
      claf   : float, a factor influencing the control point location.

    Returns:
      xpt : numpy array of shape (nvc+1,), chordwise point locations (with 0 and 1 as endpoints)
      xvr : numpy array of shape (nvc,), first set of internal points
      xsr : numpy array of shape (nvc,), second set of internal points
    """
    pi = np.pi
    # --- Set blending weights (F0, F1, F2) ---
    if abs(cspace) > 2:
        f0 = abs(cspace) - 2.0
        f1 = 0.0
        f2 = 3.0 - abs(cspace)
    elif abs(cspace) > 1:
        f0 = 0.0
        f1 = 2.0 - abs(cspace)
        f2 = abs(cspace) - 1.0
    else:
        f0 = 1.0 - abs(cspace)
        f1 = abs(cspace)
        f2 = 0.0

    # --- Spacing parameters ---
    dth1 = pi / (4 * N + 2)  # for cosine spacing
    dth2 = 0.5 * pi / (4 * N + 1)  # for sine spacing
    dxc0 = 1.0 / (4 * N)  # for uniform spacing

    xpt = np.empty(N + 1, dtype=float)
    # --- Loop over panels ---
    # Fortran loops IVC = 1 to nvc. Here, we use ivc = i+1.
    for i in range(1, N):
        # --- Uniform spacing ---
        xc0 = (4 * i - 4) * dxc0
        xpt0 = xc0
        # --- Cosine spacing ---
        th1 = (4 * i - 3) * dth1
        xpt1 = 0.5 * (1.0 - np.cos(th1))
        # --- Sine spacing ---
        if cspace > 0.0:
            # Use sine spacing as 1 - cos(theta)
            th2 = (4 * i - 3) * dth2
            xpt2 = 1.0 - np.cos(th2)
        else:
            # Use negative sine spacing: sin(theta)
            th2 = (4 * i - 4) * dth2
            xpt2 = np.sin(th2)
        # --- Blend the three spacing approaches ---
        xpt[i] = f0 * xpt0 + f1 * xpt1 + f2 * xpt2
    xpt[1] = 0.0
    xpt[N] = 1.0
    return xpt[Ni + 1]


def equal_spacing_function_factory(
    N: int,
    stretching: float = 1,
) -> Callable[[int], float]:
    """Returns a function that returns a linearly spaced array of length N."""

    def equal_spacing_function(
        Ni: int,
    ) -> float:
        """Returns a linearly spaced array of length N."""
        return (Ni / (N - 1)) * stretching

    return equal_spacing_function


############################
# Distribution functions
############################


def linear_distribution_function_factory(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> Callable[[float], float]:
    """Returns a function that returns a linearly distributed array of length N."""

    def linear_distribution_function(
        x: float,
    ) -> float:
        if x0 == x1:
            return y0
        """Returns a linearly distributed array of length N."""
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    return linear_distribution_function
