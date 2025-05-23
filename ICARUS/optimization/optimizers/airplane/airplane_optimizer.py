from copy import deepcopy
from typing import Any
from typing import Callable

from ICARUS.core.types import FloatArray
from ICARUS.optimization import MAX_FLOAT
from ICARUS.optimization import MAX_INT
from ICARUS.optimization.callbacks import OptimizationCallback
from ICARUS.vehicle import Airplane

from .. import General_SOO_Optimizer

class Airplane_Optimizer(General_SOO_Optimizer):
    def __init__(
        self,
        # Optimization Parameters
        plane: Airplane,
        design_variables: list[str],
        design_constants: dict[str, Any],
        bounds: dict[str, tuple[float, float]],
        # Objective Function
        f: Callable[..., float],
        jac: Callable[..., float] | None = None,
        linear_constraints: list[dict[str, FloatArray | str | float]] = [],
        non_linear_constraints: list[dict[str, Callable[..., float] | str | float]] = [],
        # Stop Parameters
        maxtime_sec: float = MAX_FLOAT,
        max_iter: int = MAX_INT,
        max_function_call: int = MAX_INT,
        optimization_tolerance: float = 1e-6,
        # Callback List
        callback_list: list[OptimizationCallback] = [],
    ) -> None:
        super().__init__(
            obj=plane,
            design_variables=design_variables,
            design_constants=design_constants,
            bounds=bounds,
            f=f,
            jac=jac,
            linear_constraints=linear_constraints,
            non_linear_constraints=non_linear_constraints,
            maxtime_sec=maxtime_sec,
            max_iter=max_iter,
            max_function_call=max_function_call,
            optimization_tolerance=optimization_tolerance,
            callback_list=callback_list,
        )
        self.current_obj: Airplane = deepcopy(plane)
        self.initial_obj: Airplane = deepcopy(plane)
        self.plane_name = plane.name

    def f(self, x: FloatArray) -> float:
        self.current_obj.name = f"{self.plane_name}_{self._nit}"
        return super().f(x)
