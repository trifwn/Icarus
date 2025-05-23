from .optimizable import Optimizable
from . import optimizers
from . import callbacks


import numpy as np
ii64 = np.iinfo(np.int64)
f64 = np.finfo(np.float64)
MAX_INT = ii64.max - 1
MAX_FLOAT = float(f64.max)



__all__ = [
    "optimizers",
    "callbacks",
    "Optimizable",
    "MAX_INT",
    "MAX_FLOAT",
]
