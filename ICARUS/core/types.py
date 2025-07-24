from typing import Any
from typing import Callable
from typing import TypeVar
from typing import Union

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array
from jaxtyping import Float
from numpy import complexfloating
from numpy import dtype
from numpy import floating
from numpy import ndarray

# npt.NDArray[np.float64]
# npt.ArrayLike


FloatArray = ndarray[Any, dtype[floating[Any]]]
JaxArray = jnp.ndarray
ComplexArray = ndarray[Any, dtype[complexfloating[Any, Any]]]
FloatOrListArray = Union[FloatArray, list[float]]
AnyFloat = Union[float, int, np.number[Any], FloatArray, list[float]]

# Generic Type Variables
NumericVar = TypeVar("NumericVar", float, FloatArray)

# ArrayLike is a Union of all objects that can be implicitly converted to a
# standard JAX array (i.e. not including future non-standard array types like
# KeyArray and BInt). It's different than np.typing.ArrayLike in that it doesn't
# accept arbitrary sequences, nor does it accept string data.
ArrayType = Union[
    Array,  # JAX array type
    np.ndarray,  # NumPy array type
]

# Array definitions
Scalar = Float[ArrayLike, "1"]

Vector2D = Float[ArrayType, "2"]
Vector3D = Float[ArrayType, "3"]

Array1D = Float[ArrayType, "N"]
Array2D = Float[ArrayType, "N M"]
Array3D = Float[ArrayType, "N M P"]

Matrix3x3 = Float[ArrayType, "3 3"]  # Shape (3, 3)

# Panel definitions
Panel3D = Float[ArrayType, "4 3"]  # Shape (4, 3)
PanelArray3D = Float[ArrayType, "N 4 3"]  # Shape (N, 4, 3)
PanelGrid3D = Float[ArrayType, "N M 4 3"]  # Shape (N, M, 4, 3)

# Grid definitions
Grid3D = Float[ArrayType, "N M 3"]  # Shape (N, M, 3)
PointArray3D = Float[ArrayType, "N 3"]  # Shape (N, 3)

DistributionFunc = Callable[[float, float, float], float]
