import inspect
import logging
from io import StringIO
from typing import Any
from typing import Callable

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import jsonpickle.ext.pandas as jsonpickle_pd
from pandas import DataFrame
from tabulate import tabulate

from .input import Input
from ICARUS.computation.solvers.solver_parameters import Parameter

jsonpickle_pd.register_handlers()
jsonpickle_numpy.register_handlers()


class Analysis:
    """
    Analysis Class. Used to define an analysis and store all the necessary information for it.
    The analysis can be run by calling the object. The results can be obtained by calling the get_results function.

    Args:
        solver_name (str): Name of the associated solver
        analysis_name (str): Name of the analysis
        run_function (Callable[..., Any]): Function to run the analysis
        options (Struct | dict[str, Any]): Analysis options
        solver_options (Struct | dict[str,Any], optional): Solver Options . Defaults to {}.
        unhook (Callable[...,Any] | None, optional): Function to run after the analysis Mainly for post processing. Defaults to None.

    """

    def __init__(
        self,
        solver_name: str,
        analysis_name: str,
        options: list[Input],
        execute_fun: Callable[..., Any],
        parallel_execute_fun: Callable[..., Any] | None = None,
        unhook: Callable[..., Any] | None = None,
    ) -> None:
        """
        Initializes an Analysis object

        Args:
            solver_name (str): Name of the associated solver
            analysis_name (str): Name of the analysis
            options (list[Options]): Analysis options
            run_function (Callable[..., Any]): Function to run the analysis
            unhook (Callable[...,Any] | None, optional): Function to run after the analysis Mainly for post processing. Defaults to None.
        """
        self.solver_name: str = solver_name
        self.name: str = analysis_name
        self.options: dict[str, Input] = {option.name: option for option in options}
        self.execute: Callable[..., Any] = execute_fun

        if callable(parallel_execute_fun):
            self.parallel_execute: Callable[..., Any] = parallel_execute_fun

        if callable(unhook):
            self.unhook: Callable[..., DataFrame | int] = unhook
        else:
            if unhook is not None:
                print("Unhook must be a function! Defaulting to None")
            self.unhook = lambda: 0

    def __str__(self) -> str:
        """
        String representation of the analysis

        Returns:
            str: Name and Options of the analysis
        """
        string = StringIO()
        string.write(f"Available Options of {self.solver_name} for {self.name}: \n\n")
        table: list[list[str]] = [["VarName", "Value", "Description"]]
        for opt in self.options.values():
            if opt.value is None:
                value: str = "None"
            elif hasattr(opt.value, "__str__"):
                if len(str(opt.value)) > 10:
                    value = "Complex Datatype"
                    if hasattr(opt.value, "name"):
                        value += f" ({opt.value.name})"
                else:
                    value = str(opt.value)
            elif hasattr(opt.value, "__len__"):
                if len(opt.value) > 3:
                    value = "Multiple Values"
                    if hasattr(opt.value, "name"):
                        value += f" ({opt.value.name})"
                else:
                    value = opt.value
            else:
                value = "N/A"
            table.append([opt.name, value, opt.description])  # TODO ADD __REPR__ INSTEAD OF __STR__
        string.write(tabulate(table[1:], headers=table[0], tablefmt="github"))
        string.write(
            "\n\nIf there are Multiple Values, or complex datatypes, or N/A you should inspect them sepretly by calling the option name\n",
        )

        return string.getvalue()

    def check_if_defined(self) -> bool:
        """
        Checks if all options have been set.

        Returns:
            bool: True if all options have been set, False otherwise
        """
        flag: bool = True
        for option in self.options.values():
            if option.value is None:
                print(f"Option {option} not set")
                flag = False
        return flag

    def __call__(self, solver_parameters: list[Parameter] | None, parallel: bool = False) -> Any:
        """
        Runs the analysis

        Returns:
            Any: Analysis Results as set by the unhook function
        """
        if self.check_if_defined():
            kwargs: dict[str, Any] = {option.name: option.value for option in self.options.values()}
            solver_options: dict[str, Any] = {}
            if solver_parameters is not None:
                solver_options = {option.name: option.value for option in solver_parameters}

            if not hasattr(self, "parallel_execute"):
                if parallel:
                    logging.info("Parallel Execution not supported for this analysis")
                logging.info("Running Serially")
                res: Any = self.execute(**kwargs, solver_options=solver_options)
            else:
                if parallel:
                    logging.info("Running Analysis in Parallel")
                    res = self.parallel_execute(**kwargs, solver_options=solver_options)
                else:
                    logging.info("Running Analysis in Serial")
                    res = self.execute(**kwargs, solver_options=solver_options)

            print("Analysis Completed")
            return res
        else:
            print(
                f"Options not set for {self.name} of {self.solver_name}. Here is what was passed:",
            )
            print(self)
            return -1

    def get_results(self) -> DataFrame | int:
        """
        Function to get the results. Calls the unhooks function.

        Returns:
            DataFrame | int: Results of the analysis or error code
        """
        print("Getting Results")
        args_needed = list(inspect.signature(self.unhook).parameters.keys())
        kwargs: dict[str, Any] = {
            option.name: option.value for _, option in self.options.items() if option.name in args_needed
        }
        return self.unhook(**kwargs)

    def encode_json(self) -> str:
        encoded: str = str(jsonpickle.encode(self))
        return encoded

    def __lshift__(self, other: dict[str, Any]) -> "Analysis":
        """
        overloading operator <<
        """
        if not isinstance(other, dict):
            raise TypeError("Can only << a dict")

        s: Analysis = self.__copy__()
        s.__dict__.update(other)
        return s

    def __copy__(self) -> "Analysis":
        return self.__class__(
            self.solver_name,
            self.name,
            [option for option in self.options.values()],
            self.execute,
            self.unhook,
        )

    def __getstate__(
        self,
    ) -> tuple[str, str, list[Input], Callable[..., Any], Callable[..., DataFrame | int]]:
        return (
            self.solver_name,
            self.name,
            [option for option in self.options.values()],
            self.execute,
            self.unhook,
        )

    def __setstate__(
        self,
        state: tuple[
            str,
            str,
            list[Input],
            Callable[..., Any],
            Callable[..., DataFrame | int],
        ],
    ) -> None:
        self.solver_name = state[0]
        self.name = state[1]
        self.options = {option.name: option for option in state[2]}
        self.execute = state[3]
        self.unhook = state[4]
