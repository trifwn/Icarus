"""

isort:skip_file
"""

import numpy as np

from . import callbacks
from . import optimizers

ii64 = np.iinfo(np.int64)
f64 = np.finfo(np.float64)
MAX_INT = ii64.max - 1
MAX_FLOAT = float(f64.max)


__all__ = [
    "optimizers",
    "callbacks",
    "MAX_INT",
    "MAX_FLOAT",
]
