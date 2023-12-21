from copy import deepcopy
from time import time
from typing import Any
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult

from ICARUS.Core.types import FloatArray
from ICARUS.Optimization import MAX_FLOAT
from ICARUS.Optimization import MAX_INT
from ICARUS.Optimization.Callbacks.optimization_callback import OptimizationCallback


class General_Optimizer:
    def __init__(
        self,
        # Optimization Parameters
        obj: object,
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
        verbosity: int = 1,
    ) -> None:
        # Basic Objects
        self.design_variables: dict[str, Any] = {design_variable: 0 for design_variable in design_variables}
        self.design_constants = design_constants
        self.initial_obj: object = deepcopy(obj)
        self.current_obj: object = deepcopy(obj)

        x0 = []
        for design_variable in self.design_variables.keys():
            x0.append(self.initial_obj.__getattribute__(design_variable))
        self.x0 = np.array(x0)

        # Constraints
        self.bounds = []
        for design_variable in self.design_variables.keys():
            self.bounds.append(bounds[design_variable])

        # Stop Parameters
        self.maxtime_sec: float = maxtime_sec
        self.max_function_call_count = max_function_call
        self.max_iter = max_iter
        self.tolerance = optimization_tolerance

        # Objective Function
        self.objective_fn = f
        # Jacobian
        self.jacobian = jac

        # Iteration Counters
        self._function_call_count: int = 0
        self._nit: int = 0

        # Callback List
        self.callback_list = callback_list
        self.verbosity = verbosity

    def f(self, x: FloatArray) -> float:
        if self._function_call_count > self.max_function_call_count:
            print(f"Function call count exceeded {self._function_call_count}")
            raise StopIteration

        # Update Function Call Count
        self._function_call_count += 1
        if self.verbosity > 1:
            print(f"Calculating OBJ {self._nit}")

        # Update Current Object
        for i, design_variable in enumerate(self.design_variables.keys()):
            self.current_obj.__setattr__(design_variable, x[i])
            self.design_variables[design_variable] = x[i]

        return self.objective_fn(self.current_obj, **self.design_constants)

    def jac(self, x: FloatArray) -> float:
        if self.jacobian is None:
            raise NotImplementedError("Jacobian not implemented")

        if self.verbosity > 1:
            print(f"\tCalculating J {self._nit}")
        # Update Current Object
        for i, design_variable in enumerate(self.design_variables.keys()):
            self.current_obj.__setattr__(design_variable, x[i])
        return self.jacobian(self.current_obj)

    def run_all_callbacks(self, intermediate_result: OptimizeResult) -> None:
        for callback in self.callback_list:
            callback.update(self.current_obj, intermediate_result, self._nit)

    def iteration_callback(self, intermediate_result: OptimizeResult) -> None:
        # callback to terminate if maxtime_sec is exceeded
        self._nit += 1
        elapsed_time = time() - self.start_time

        # Run Callbacks
        self.run_all_callbacks(intermediate_result)

        # Print Design Variables with the names of the design variables
        if self.verbosity > 0:
            print(f"Design Variables: ")
            for i, design_variable in enumerate(self.design_variables):
                print(f"\t{design_variable}: {intermediate_result.x[i]}")

            # Print Objective Function
            print(f"Fun: {intermediate_result.fun}")

        if elapsed_time > self.maxtime_sec:
            print(f"Optimization time exceeded {elapsed_time}")
            raise StopIteration
        else:
            print()
            print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(f"Iteration Number: {self._nit}")
            print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    def __call__(self, solver: str = "Nelder-Mead", options: dict[str, Any] = {}) -> OptimizeResult:
        for callback in self.callback_list:
            callback.setup()

        self.start_time = time()
        # set your initial guess to 'x0'
        # set your bounds to 'bounds'
        if solver == "Nelder-Mead":
            opt = minimize(
                self.f,
                x0=self.x0,
                method="Nelder-Mead",
                callback=self.iteration_callback,
                tol=self.tolerance,
                options={"maxiter": self.max_iter, "disp": True},
                bounds=self.bounds,
            )
        elif solver == "Newton-CG":
            opt = minimize(
                self.f,
                x0=self.x0,
                jac=self.jac if self.jacobian is not None else None,
                method="Newton-CG",
                callback=self.iteration_callback,
                tol=self.tolerance,
                options={"maxiter": 10, "disp": False},
                bounds=self.bounds,
            )
        else:
            raise NotImplementedError
        return opt
