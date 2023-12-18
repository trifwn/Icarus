from enum import Enum
from functools import partial
from typing import Callable

import numpy as np

from ICARUS.Core.types import FloatArray


class DiscretizationType(Enum):
    """
    Discretization types for ICARUS
     3.0        equal         |   |   |   |   |   |   |   |   |
     2.0        sine          || |  |   |    |    |     |     |
     1.0        cosine        ||  |    |      |      |    |  ||
     0.0        equal         |   |   |   |   |   |   |   |   |
    -1.0        cosine        ||  |    |      |      |    |  ||
    -2.0       -sine          |     |     |    |    |   |  | ||
    -3.0        equal         |   |   |   |   |   |   |   |   |
    """

    EQUAL = 3.0
    COSINE = 1.0
    SINE = 2.0
    INV_SINE = -2.0
    INV_COSINE = -1.0
    UNKNOWN = -1000


class DistributionType(Enum):
    """
    Distribution types for ICARUS
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
    """
    Symmetry axes for ICARUS
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


def equal_spacing_function_factory(N: int, stretching: float = 1) -> Callable[[int], float]:
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


def linear_distribution_function_factory(x0: float, x1: float, y0: float, y1: float) -> Callable[[float], float]:
    """Returns a function that returns a linearly distributed array of length N."""

    def linear_distribution_function(
        x: float,
    ) -> float:
        if x0 == x1:
            return y0
        """Returns a linearly distributed array of length N."""
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    return linear_distribution_function
