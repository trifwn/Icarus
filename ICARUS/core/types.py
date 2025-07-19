from typing import Any
from typing import Callable
from typing import TypeVar
from typing import Union

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import ArrayLike
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

# Type definitions
Vector3D = FloatArray  # Shape (3,)
Matrix3x3 = Float[ArrayLike, "3 3"]  # Shape (3, 3)
DistributionFunc = Callable[[float, float, float], float]
