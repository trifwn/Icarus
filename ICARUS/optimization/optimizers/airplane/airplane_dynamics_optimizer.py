from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from scipy.optimize import OptimizeResult

from ICARUS.core.types import FloatArray
from ICARUS.flight_dynamics import State
from ICARUS.optimization import MAX_FLOAT
from ICARUS.optimization import MAX_INT
from ICARUS.optimization.callbacks import OptimizationCallback

from . import Airplane_Optimizer

if TYPE_CHECKING:
    from ICARUS.vehicle import Airplane


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
            plane=plane,
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
        self.state: State = state

    def run_all_callbacks(self, intermediate_result: OptimizeResult) -> None:
        # Run Callbacks
        for callback in self.callback_list:
            print(f"Running {callback} callback")
            callback.update(
                plane=self.current_obj,
                state=self.state,
                iteration=self._nit,
                design_variables=self.design_variables,
                result=intermediate_result,
                fitness=self.fitness[-1],
                penalty=self.penalties[-1],
                bounds=self.bounds,
            )
