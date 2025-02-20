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


def cosine_spacing_function(Ni: int, N: int, stretching: float) -> float:
    """Returns a cosine spaced array of length N."""
    return 0.5 * (1 - np.cos(np.pi * Ni / (N))) * stretching


def sine_spacing_function(Ni: int, N: int, stretching: float, factor: float = 0) -> float:
    """Returns a sine spaced array of length N."""
    if N < 0:
        N = -N
        y = 1 - np.sin(np.pi / 2 * (1 - (Ni / (N - 1))))
    else:
        y = np.sin(np.pi / (2 * (N - 1)) * Ni)
    y = (1 - factor) * y + factor * (Ni / (N - 1))
    y: float = y * stretching
    return y


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
