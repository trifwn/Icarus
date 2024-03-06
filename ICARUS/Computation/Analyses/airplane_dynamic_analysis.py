from typing import Any
from typing import Callable

from ICARUS.Computation.Analyses.analysis import Analysis
from ICARUS.Computation.Analyses.input import AirplaneInput
from ICARUS.Computation.Analyses.input import Input
from ICARUS.Computation.Analyses.input import StateInput
from ICARUS.Computation.Analyses.input import StrInput


airplane_option = AirplaneInput()
state_opion = StateInput()
solver_2D_option = StrInput("solver2D", "Name of 2D Solver from which to use computed polars")


class BaseDynamicAnalysis(Analysis):
    def __init__(
        self,
        solver_name: str,
        execute_fun: Callable[..., Any],
        parallel_execute_fun: Callable[..., Any] | None = None,
        unhook: Callable[..., Any] | None = None,
        extra_options: list[Input] = [],
    ) -> None:
        super().__init__(
            analysis_name="Dynamic Analysis",
            solver_name=solver_name,
            options=[airplane_option, state_opion, solver_2D_option, *extra_options],
            execute_fun=execute_fun,
            parallel_execute_fun=parallel_execute_fun,
            unhook=unhook,
        )