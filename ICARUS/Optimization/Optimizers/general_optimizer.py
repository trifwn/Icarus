import inspect
import logging
from calendar import c
from copy import deepcopy
from re import A
from time import time
from typing import Any
from typing import Callable

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import OptimizeResult

from ICARUS.Core.types import FloatArray
from ICARUS.Optimization import MAX_FLOAT
from ICARUS.Optimization import MAX_INT
from ICARUS.Optimization.Callbacks.optimization_callback import OptimizationCallback


class General_SOO_Optimizer:
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
        linear_constraints: list[dict[str, FloatArray | str | float]] = [],
        non_linear_constraints: list[dict[str, Callable[..., float] | str | float]] = [],
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
        x0_norm = []
        for design_variable in self.design_variables.keys():
            x0.append(self.initial_obj.__getattribute__(design_variable))
            x0_norm.append(
                (x0[-1] - bounds[design_variable][0]) / (bounds[design_variable][1] - bounds[design_variable][0])
            )
        self.x0 = np.array(x0)
        self.x0_norm = np.array(x0_norm)

        # Bounds
        self.bounds = []
        self.bounds_norm = []
        for design_variable in self.design_variables.keys():
            self.bounds.append(bounds[design_variable])
            self.bounds_norm.append((0,1))


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
        self.succesful_iteration: bool = False

        # Callback List
        self.callback_list = callback_list
        self.verbosity = verbosity

        # Constraints
        self.linear_constraints: list[dict[str, FloatArray | str | float]] = linear_constraints
        self.non_linear_constraints: list[dict[str, Callable[..., float] | str | float]] = non_linear_constraints
        self.fitness: list[float] = []
        self.penalties: list[float] = []

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
            x_denorm = x[i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]
            self.current_obj.__setattr__(design_variable, x_denorm)
            self.design_variables[design_variable] = x_denorm

        # Calculate Fitness
        fitness = self.objective_fn(self.current_obj, **self.design_constants)
        print(f"Fitness is {fitness}")
        self.fitness.append(fitness)
        return fitness

    def jac(self, x: FloatArray) -> float:
        if self.jacobian is None:
            raise NotImplementedError("Jacobian not implemented")

        if self.verbosity > 1:
            print(f"\tCalculating J {self._nit}")
        # Update Current Object
        for i, design_variable in enumerate(self.design_variables.keys()):
            x_denorm = x[i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]
            self.current_obj.__setattr__(design_variable, x_denorm)
        return self.jacobian(self.current_obj)

    def run_all_callbacks(self, intermediate_result: OptimizeResult) -> None:
        if self.fitness[-1] > 1e9:
            return
        for callback in self.callback_list:
            callback.update(
                obj=self.current_obj,
                result=intermediate_result,
                iteration=self._nit,
                fitness=self.fitness[-1],
                bounds = self.bounds,
            )

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
                x_denorm = intermediate_result.x[i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]
                print(f"\t{design_variable}= {x_denorm},")

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
        # Setup Callbacks
        s_time = time()
        for callback in self.callback_list:
            callback.setup()
        e_time = time()
        print(f"Callback Setup Time: {e_time - s_time}")

        # If the solver is COBYLA, SLSQP or trust-constr we can add the constraints
        # else we need to add the constraints to the objective function as a penalty
        constraints = []
        if solver in ["COBYLA", "SLSQP", "trust-constr"]:
            for lin_constraint in self.linear_constraints:
                if not (isinstance(lin_constraint["lb"], float) and isinstance(lin_constraint["ub"], float)):
                    logging.warning(f"Linear Constraint {lin_constraint} has a non-float lb")
                    continue

                constraints.append(
                    LinearConstraint(
                        lin_constraint["A"],
                        lin_constraint["lb"],
                        lin_constraint["ub"],
                    ),
                )

            for non_lin_constraint in self.non_linear_constraints:
                if not (isinstance(non_lin_constraint["lb"], float) and isinstance(non_lin_constraint["ub"], float)):
                    logging.warning(f"Non-Linear Constraint {non_lin_constraint} has a non-float lb")
                    continue

                if "fun" not in non_lin_constraint.keys():
                    logging.warning(f"Non-Linear Constraint {non_lin_constraint} does not have a fun")
                    continue

                if not callable(non_lin_constraint["fun"]):
                    logging.warning(f"Non-Linear Constraint {non_lin_constraint} fun is not callable")
                    continue
                non_linear_fun: Callable[..., float] = non_lin_constraint["fun"]

                def fun_wrapper(x: FloatArray) -> float:
                    params = inspect.signature(non_linear_fun).parameters
                    if "x" in params:
                        # Add desing constants to the function call if they are in the function signature
                        return non_linear_fun(x, **self.design_constants)
                    else:
                        return non_linear_fun(**self.design_constants)

                constraints.append(
                    NonlinearConstraint(
                        fun_wrapper,
                        non_lin_constraint["lb"] if "lb" in non_lin_constraint else -np.inf,
                        non_lin_constraint["ub"] if "ub" in non_lin_constraint else np.inf,
                    ),
                )
                print(f"Added Non-Linear Constraint {non_lin_constraint}")
        elif solver in ["Nelder-Mead", "Newton-CG"]:
            print("Adding Constraints as Penalties")
            # Add Penalty for Linear Constraints
            linear_penalties: list[Callable[[FloatArray], float]] = []
            for lin_constraint in self.linear_constraints:
                if not (isinstance(lin_constraint["lb"], float) and isinstance(lin_constraint["ub"], float)):
                    logging.warning(f"Linear Constraint {lin_constraint} has a non-float lb")
                    continue

                A = lin_constraint["A"]
                lb = lin_constraint["lb"]
                ub = lin_constraint["ub"]
                linear_penalties.append(lambda x: max(0, np.dot(A, x) - ub) ** 2 + max(0, lb - np.dot(A, x)) ** 2)

            non_linear_penalties: list[Callable[[FloatArray], float]] = []
            for non_lin_constraint in self.non_linear_constraints:
                if not (isinstance(non_lin_constraint["lb"], float) and isinstance(non_lin_constraint["ub"], float)):
                    logging.warning(f"Non-Linear Constraint {non_lin_constraint} has a non-float lb")
                    continue

                if "fun" not in non_lin_constraint.keys():
                    logging.warning(f"Non-Linear Constraint {non_lin_constraint} does not have a fun")
                    continue

                if not callable(non_lin_constraint["fun"]):
                    logging.warning(f"Non-Linear Constraint {non_lin_constraint} fun is not callable")
                    continue
                non_linear_fun = non_lin_constraint["fun"]

                def fun_wrapper(x: FloatArray) -> float:
                    params = inspect.signature(non_linear_fun).parameters
                    if "x" in params:
                        # Add desing constants to the function call if they are in the function signature
                        return non_linear_fun(x, **self.design_constants)
                    else:
                        return non_linear_fun(**self.design_constants)

                lb = non_lin_constraint["lb"] if "lb" in non_lin_constraint else -np.inf
                ub = non_lin_constraint["ub"] if "ub" in non_lin_constraint else np.inf

                non_linear_penalties.append(
                    lambda x: max(0, fun_wrapper(x) - ub) ** 2 + max(0, lb - fun_wrapper(x)) ** 2,
                )

            def f_with_penalties(x: FloatArray) -> float:
                O = self.f(x)
                # Add Penalty for Linear Constraints
                penalty: float = 0.0
                for penalty_fun in linear_penalties:
                    penalty += penalty_fun(x)
                # Add Penalty for Non-Linear Constraints
                for penalty_fun in non_linear_penalties:
                    penalty += penalty_fun(x)

                self.penalties.append(penalty)
                return O + penalty

            print(f"Added {len(linear_penalties)} Linear Penalties")
            print(f"Added {len(non_linear_penalties)} Non Linear Penalties")

        else:
            raise NotImplementedError

        self.start_time = time()
        if solver == "Nelder-Mead":
            # Check if f_with_penalties is defined
            opt = minimize(
                f_with_penalties if "f_with_penalties" in locals() else self.f,
                x0=self.x0_norm,
                method="Nelder-Mead",
                callback=self.iteration_callback,
                tol=self.tolerance,
                options={"maxiter": self.max_iter, "disp": True},
                bounds=self.bounds_norm,
            )
        elif solver == "Newton-CG":
            opt = minimize(
                f_with_penalties if "f_with_penalties" in locals() else self.f,
                x0=self.x0_norm,
                jac=self.jac if self.jacobian is not None else None,
                method="Newton-CG",
                callback=self.iteration_callback,
                tol=self.tolerance,
                options={"maxiter": 10, "disp": False},
                bounds=self.bounds_norm,
            )
        elif solver == "COBYLA":
            opt = minimize(
                self.f,
                x0=self.x0_norm,
                method="COBYLA",
                callback=self.iteration_callback,
                tol=self.tolerance,
                options={"maxiter": self.max_iter, "disp": True},
                constraints=constraints,
                bounds = self.bounds_norm
            )
        elif solver == "SLSQP":
            opt = minimize(
                # f_with_penalties if "f_with_penalties" in locals() else self.f,
                self.f,
                x0=self.x0_norm,
                method="SLSQP",
                callback=self.iteration_callback,
                tol=self.tolerance,
                options={"maxiter": self.max_iter, "disp": True},
                constraints=constraints,
                bounds = self.bounds_norm
            )
        else:
            raise NotImplementedError
        return opt
