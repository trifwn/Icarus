import inspect
from io import StringIO
from typing import Any
from typing import Callable

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import jsonpickle.ext.pandas as jsonpickle_pd
from pandas import DataFrame
from tabulate import tabulate

from .options import Option
from ICARUS.Core.struct import Struct

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
        run_function: Callable[..., Any],
        options: Struct | dict[str, Any],
        solver_options: Struct | dict[str, Any] = {},
        unhook: Callable[..., Any] | None = None,
    ) -> None:
        """
        Initializes an Analysis object

        Args:
            solver_name (str): Name of the associated solver
            analysis_name (str): Name of the analysis
            run_function (Callable[..., Any]): Function to run the analysis
            options (Struct | dict[str, Any]): Analysis options
            solver_options (Struct | dict[str,Any], optional): Solver Options . Defaults to {}.
            unhook (Callable[...,Any] | None, optional): Function to run after the analysis Mainly for post processing. Defaults to None.
        """
        self.solver_name: str = solver_name
        self.name: str = analysis_name
        self.options: Struct = Struct()
        self.solver_options: Struct = Struct()
        self.execute: Callable[..., Any] = run_function

        for option in options.keys():
            desc, option_type = options[option]
            self.options[option] = Option(option, None, desc, option_type)

        if solver_options:
            for option in solver_options.keys():
                value, desc, option_type = solver_options[option]
                self.solver_options[option] = Option(option, value, desc, option_type)

        if callable(unhook):
            self.unhook: Callable[..., DataFrame | int] = unhook
        elif unhook is None:
            self.unhook = lambda: 0
        else:
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
        for _, opt in self.options.items():
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

    # __repr__: Callable[..., str] = __str__

    def get_solver_options(self, verbose: bool = False) -> Struct:
        """
        Get Solver Options for the current analysis. Solver Options refer to internal solver settings not
        the analysis itself.

        Args:
            verbose (bool, optional): Whether to also display the solver options and their description.
                                      Defaults to False.

        Returns:
            Struct: Object Reffering to the solver Options. Can be used to set the options
        """
        if verbose:
            string: str = f"Available Solver Parameters of {self.solver_name} for {self.name}: \n\n"
            table: list[list[str]] = [["VarName", "Value", "Description"]]
            for _, opt in self.solver_options.items():
                if opt.value is None:
                    table.append([opt.name, "None", opt.description])
                elif hasattr(opt.value, "__len__"):
                    if len(opt.value) > 2:
                        table.append([opt.name, "Multiple Values", opt.description])
                else:
                    table.append([opt.name, opt.value, opt.description])
            string += tabulate(table[1:], headers=table[0], tablefmt="github")
            string += "\n\nIf there are multiple values you should inspect them sepretly by calling the option name\n"
            print(string)
        return self.solver_options

    def get_options(self, verbose: bool = False) -> Struct:
        """
        Get the options for the current analysis. This referes to the neccessary values that have to be set
        to run the analysis

        Args:
            verbose (bool, optional): Whether to print the options. Defaults to False.

        Returns:
            Struct: Object Reffering to the Options. Can be used to set the options
        """
        if verbose:
            print(self)
        return self.options

    def set_option(self, option_name: str, option_value: Any) -> None:
        """
        Set an option for the current analysis.

        Args:
            option_name (str): Option Name
            option_value (Any): Option Value
        """
        try:
            self.options[option_name].value = option_value
        except KeyError:
            print(f"Option {option_name} not available")

    def set_all_options(self, options: Struct | dict[str, Any]) -> None:
        """
        Set all options for the current analysis.

        Args:
            options (Struct | dict[str,Any]): Object containing the options. Can be a dictionary or a Struct.
        """
        for option in options:
            self.set_option(option, options[option])

    def set_solver_param(self, param_name: str, param_value: Any) -> None:
        """
        Set a solver parameter for the current analysis.
        Solver Parameters refer to internal solver settings not the analysis itself.

        Args:
            param_name (str): Parameter Name
            param_value (Any): Parameter Value
        """
        try:
            self.solver_options[param_name].value = param_value
        except KeyError:
            print(f"Parameter {param_name} not available")

    def set_all_solver_params(self, params: Struct | dict[str, Any]) -> None:
        """
        Set all solver parameters for the current analysis.

        Args:
            params (Struct | dict[str, Any]): Object containing the parameters. Can be a dictionary or a Struct.
        """
        for param in params:
            self.set_solver_param(param, params[param])

    def check_options(self) -> bool:
        """
        Checks if all options have been set.

        Returns:
            bool: True if all options have been set, False otherwise
        """
        flag: bool = True
        for option in self.options:
            if self.options[option].value is None:
                print(f"Option {option} not set")
                flag = False
        return flag

    def check_has_run(self) -> bool:
        """
        Checks if the analysis has been run!! NOT IMPLEMENTED!!!
        """
        print("Checking Run")
        return True

    def __call__(self) -> Any:
        """
        Runs the analysis

        Returns:
            Any: Analysis Results as set by the unhook function
        """
        if self.check_options():
            kwargs: dict[str, Any] = {option: self.options[option].value for option in self.options.keys()}
            solver_options: dict[str, Any] = {
                option: self.solver_options[option].value for option in self.solver_options.keys()
            }
            res: Any = self.execute(**kwargs, solver_options=solver_options)
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
        kwargs: dict[str, Any] = {}
        for arg in args_needed:
            try:
                kwargs[arg] = self.options[arg].value
            except KeyError:
                print(f"Option {arg} not set")
                return -1
        return self.unhook(**kwargs)

    def copy(self) -> "Analysis":
        """
        Copy the analysis to a new object.

        Returns:
            Analysis: Copy of the Analysis
        """
        option_dict: dict[str, Any] = {k: v.description for k, v in self.options.items()}
        solver_options: dict[str, tuple[Any, Any]] = {
            k: (v.value, v.description) for k, v in self.solver_options.items()
        }
        return self.__class__(
            self.solver_name,
            self.name,
            self.execute,
            option_dict,
            solver_options,
        )

    def __copy__(self) -> "Analysis":
        option_dict: dict[str, tuple[str, Any]] = {k: (v.description, v.option_type) for k, v in self.options.items()}
        solver_options: dict[str, tuple[Any, str, Any]] = {
            k: (v.value, v.description, v.option_type) for k, v in self.solver_options.items()
        }
        return self.__class__(
            self.solver_name,
            self.name,
            self.execute,
            option_dict,
            solver_options,
        )

    def __getstate__(
        self,
    ) -> tuple[str, str, Callable[..., Any], Struct, Struct, Callable[..., DataFrame | int]]:
        return (
            self.solver_name,
            self.name,
            self.execute,
            self.options,
            self.solver_options,
            self.unhook,
        )

    def __setstate__(
        self,
        state: tuple[
            str,
            str,
            Callable[..., Any],
            Struct,
            Struct,
            Callable[..., DataFrame | int],
        ],
    ) -> None:
        (
            self.solver_name,
            self.name,
            self.execute,
            self.options,
            self.solver_options,
            self.unhook,
        ) = state

    def toJSON(self) -> str:
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
