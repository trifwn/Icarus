from typing import Any
from typing import TypeVar
from typing import Union

import numpy as np
from numpy import complexfloating
from numpy import dtype
from numpy import floating
from numpy import ndarray

# npt.NDArray[np.float64]
# npt.ArrayLike


FloatArray = ndarray[Any, dtype[floating[Any]]]
ComplexArray = ndarray[Any, dtype[complexfloating[Any, Any]]]
FloatOrListArray = Union[FloatArray, list[float]]
AnyFloat = Union[float, int, np.number[Any], FloatArray, list[float]]

# Generic Type Variables
NumericVar = TypeVar("NumericVar", float, FloatArray)
