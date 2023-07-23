from typing import Any
from typing import Union

import numpy as np
from nptyping import Float
from nptyping import NDArray

from ICARUS.Core.struct import Struct

Numeric = Union[int, float, np.number]
DataDict = Union[dict, Struct]
FloatArray = NDArray[Any, Float]
FloatOrListArray = Union[FloatArray, list[float]]
