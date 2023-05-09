import inspect
from io import StringIO
from tokenize import String
from typing import Any, Callable, Union

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import jsonpickle.ext.pandas as jsonpickle_pd
import numpy as np
from tabulate import tabulate

from .options import Option
from typing import Callable
from ICARUS.Core.struct import Struct

jsonpickle_pd.register_handlers()
jsonpickle_numpy.register_handlers()


class Analysis:
    def __init__(self, solverName, name, runFunc, options, solver_options, unhook=None) -> None:
        self.solverName: str = solverName
        self.name: str = name
        self.options = Struct()
        self.solver_options = Struct()
        self.execute: Callable = runFunc

        for option in options.keys():
            self.options[option] = Option(option, None, options[option])

        if solver_options is not None:
            for option in solver_options.keys():
                value, desc = solver_options[option]
                self.solver_options[option] = Option(option, value, desc)

        if callable(unhook):
            self.unhook = unhook
        elif unhook is None:
            self.unhook = lambda: 0
        else:
            print("Unhook must be a function! Defaulting to None")
            self.unhook = lambda: 0

    def __str__(self) -> str:
        string = StringIO()

        string.write(f"Available Options of {self.solverName} for {self.name}: \n\n")
        table: list[list[str]] = [["VarName", "Value", "Description"]]
        for _, opt in self.options.items():
            if opt.value is None:
                table.append([opt.name, "None", opt.description])
            elif hasattr(opt.value, "__len__"):
                if len(opt.value) > 2:
                    table.append([opt.name, "Multiple Values", opt.description])
            else:
                table.append([opt.name, opt.value, opt.description])
        string.write(tabulate(table[1:], headers=table[0], tablefmt="github"))
        string.write("\n\nIf there are multiple values you should inspect them sepretly by calling the option name\n")
        return string.read()

    __repr__ = __str__

    def getSolverOptions(self, verbose: bool = False) -> Struct:
        if verbose:
            string = f"Available Solver Parameters of {self.solverName} for {self.name}: \n\n"
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

    def getOptions(self, verbose=False) -> Struct:
        if verbose:
            print(self)
        return self.options

    def setOption(self, optionName, optionValue) -> None:
        try:
            self.options[optionName].value = optionValue
        except KeyError:
            print(f"Option {optionName} not available")

    def setOptions(self, options) -> None:
        for option in options:
            self.setOption(option, options[option])

    def checkOptions(self) -> bool:
        flag = True
        for option in self.options:
            if self.options[option].value is None:
                print(f"Option {option} not set")
                flag = False
        return flag

    def checkRun(self) -> bool:
        """
        Checks if the analysis has been run!! NOT IMPLEMENTED!!!
        """
        print("Checking Run")
        return True

    def __call__(self):
        if self.checkOptions():
            kwargs = {option: self.options[option].value for option in self.options.keys()}
            solver_options = {option: self.solver_options[option].value for option in self.solver_options.keys()}
            res = self.execute(**kwargs, solver_options=solver_options)
            print("Analysis Completed")
            return res
        else:
            print(
                f"Options not set for {self.name} of {self.solverName}. Here is what was passed:",
            )
            print(self)
            return -1

    def getResults(self) -> Union[np.ndarray, int]:
        print("Getting Results")
        args_needed = list(inspect.signature(self.unhook).parameters.keys())
        args = {}
        for arg in args_needed:
            try:
                args[arg] = self.options[arg].value
            except KeyError:
                print(f"Option {arg} not set")
                return -1
        return self.unhook(**args)

    def copy(self) -> "Analysis":
        optiondict = {k: v.description for k, v in self.options.items()}
        solver_options: dict[str, tuple[Any, Any]] = {
            k: (v.value, v.description) for k, v in self.solver_options.items()
        }
        return self.__class__(
            self.solverName,
            self.name,
            self.execute,
            optiondict,
            solver_options,
        )

    def __copy__(self) -> "Analysis":
        optiondict = {k: v.description for k, v in self.options.items()}
        solver_options = {k: (v.value, v.description) for k, v in self.solver_options.items()}
        return self.__class__(
            self.solverName,
            self.name,
            self.execute,
            optiondict,
            solver_options,
        )

    def __getstate__(self):
        return self.solverName, self.name, self.execute, self.options

    def __setstate__(self, state):
        self.solverName, self.name, self.execute, self.options = state

    def toJSON(self) -> str:
        encoded = jsonpickle.encode(self)
        return encoded

    def __lshift__(self, other) -> "Analysis":
        """overloading operator <<"""
        if not isinstance(other, dict):
            raise TypeError("Can only << a dict")

        s: Analysis = self.__copy__()
        s.__dict__.update(other)
        return s
