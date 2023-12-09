from enum import Enum

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


def define_linear_span(
    sp: float,
    Ni: int,
) -> FloatArray:
    """Returns a linearly spaced span array."""
    return np.linspace(0, sp, Ni).round(12)


def define_linear_chord(
    Ni: int,
    ch1: float,
    ch2: float,
) -> FloatArray:
    """Returns a linearly spaced chord array."""
    return np.linspace(ch1, ch2, Ni).round(12)


def define_linear_twist(
    Ni: int,
    twist1: float,
    twist2: float,
) -> FloatArray:
    """Returns a linearly spaced twist array."""
    return np.linspace(twist1, twist2, Ni).round(12)
