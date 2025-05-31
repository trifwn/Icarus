from typing import Any
from typing import Callable
from typing import TypeVar
from typing import Union

import numpy as np
from numpy import complexfloating
from numpy import dtype
from numpy import floating
from numpy import ndarray

from .base_types import Struct

Numeric = Union[int, float, np.number[Any]]
DataDict = Union[dict[str, Any], Struct]

FloatArray = ndarray[Any, dtype[floating[Any]]]
ComplexArray = ndarray[Any, dtype[complexfloating[Any, Any]]]
FloatOrListArray = Union[FloatArray, list[float]]
AnyFloat = Union[float, int, np.number[Any], FloatArray, list[float]]

# Generic Type Variables
NumericVar = TypeVar("NumericVar", float, FloatArray)

# Type definitions
Vector3D = FloatArray  # Shape (3,)
Matrix3x3 = FloatArray  # Shape (3, 3)
DistributionFunc = Callable[[float, float, float], float]
