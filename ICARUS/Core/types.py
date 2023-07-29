from typing import Any
from typing import Union

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray

from ICARUS.Core.struct import Struct

Numeric = Union[int, float, np.number]
DataDict = Union[dict[str, Any], Struct]
FloatArray = ndarray[Any, dtype[floating[Any]]]
FloatOrListArray = Union[FloatArray, list[float]]
