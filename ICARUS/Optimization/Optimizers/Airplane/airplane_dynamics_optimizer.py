from time import time
from typing import Any
from typing import Callable

from scipy.optimize import OptimizeResult

from ICARUS.Flight_Dynamics.state import State
from ICARUS.Optimization import MAX_FLOAT
from ICARUS.Optimization import MAX_INT
from ICARUS.Optimization.Callbacks.optimization_callback import OptimizationCallback
from ICARUS.Optimization.Optimizers.Airplane.airplane_optimizer import Airplane_Optimizer
from ICARUS.Vehicle.plane import Airplane


class Airplane_Dynamics_Optimizer(Airplane_Optimizer):
    def __init__(
        self,
        # Optimization Parameters
        plane: Airplane,
        state: State,
        design_variables: list[str],
        design_constants: dict[str, Any],
        bounds: dict[str, tuple[float, float]],
        # Objective Function
        f: Callable[..., float],
        jac: Callable[..., float] | None = None,
        # Stop Parameters
        maxtime_sec: float = MAX_FLOAT,
        max_iter: int = MAX_INT,
        max_function_call: int = MAX_INT,
        optimization_tolerance: float = 1e-6,
        # Callback List
        callback_list: list[OptimizationCallback] = [],
    ) -> None:
        super().__init__(
            plane,
            design_variables,
            design_constants,
            bounds,
            f,
            jac,
            maxtime_sec,
            max_iter,
            max_function_call,
            optimization_tolerance,
            callback_list,
        )
        self.state: State = state

    def run_all_callbacks(self, intermediate_result: OptimizeResult) -> None:
        # Run Callbacks
        for callback in self.callback_list:
            callback.update(
                plane=self.current_obj,
                state=self.state,
                iteration=self._nit,
                design_variables=self.design_variables,
                result=intermediate_result,
            )
