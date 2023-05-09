from typing import Union

import numpy as np

from ICARUS.Core.struct import Struct

Numeric = Union[int, float, np.number]
NumericArray = Union[list[Numeric], np.ndarray]
DataDict = Union[dict, Struct]
