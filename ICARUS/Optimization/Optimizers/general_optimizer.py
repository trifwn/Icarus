from time import time
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult

from ICARUS.Core.types import FloatArray

ii64 = np.iinfo(np.int64)
MAX_INT = ii64.max
f64 = np.finfo(np.float64)
MAX_FLOAT = float(f64.max)


class General_Optimizer:
    def __init__(
        self,
        objective_fn: Callable[..., float],
        x0: FloatArray,
        objective_jac: Callable[..., float] | None = None,
        maxtime_sec: float = MAX_FLOAT,
        max_iter: int = MAX_INT,
        max_function_call: int = MAX_INT,
        optimization_tolerance: float = 1e-6,
    ) -> None:
        # Basic Objects
        self.objective_fn = objective_fn
        self.x0 = x0

        # Stop Parameters
        self.maxtime_sec: float = maxtime_sec
        self.max_function_call_count = max_function_call
        self.max_iter = max_iter
        self.tolerance = optimization_tolerance

        # Jacobian
        if objective_jac is None:
            self.objective_jac = self.jac
        else:
            self.objective_jac = objective_jac

        self._function_call_count: int = 0
        self._nit: int = 0

    def f(self, x: FloatArray, tab: bool = False) -> float:
        if self._function_call_count > self.max_function_call_count:
            raise StopIteration
        self._function_call_count += 1
        if tab:
            print(f"\tCalculating OBJ {self._nit}")
        else:
            print(f"Calculating OBJ {self._nit}")
        return self.objective_fn(x)

    def jac(self, pay_x: FloatArray) -> float:
        print(f"Calculating JAC {self._nit}")
        inc = 1e-3

        O_f = self.f(pay_x + inc, tab=True)
        O_b = self.f(pay_x - inc, tab=True)
        j = (O_f - O_b) / (2 * inc)
        return j

    def callback(self, intermediate_result: OptimizeResult) -> None:
        # callback to terminate if maxtime_sec is exceeded
        self._nit += 1
        elapsed_time = time() - self.start_time

        if elapsed_time > self.maxtime_sec:
            print(elapsed_time)
            print(f"Fun: {intermediate_result.fun}")
            print(f"X : {intermediate_result.x}")
            raise StopIteration

        else:
            # you could print elapsed iterations and time
            print(f"Iteration Number: {self._nit}")

    def __call__(self) -> OptimizeResult:
        self.start_time = time()
        # set your initial guess to 'x0'
        # set your bounds to 'bounds'
        opt = minimize(
            self.f,
            x0=self.x0,
            jac=self.jac,
            method="Newton-CG",
            callback=self.callback,
            tol=self.tolerance,
            options={"maxiter": 10, "disp": False},
        )
        return opt
