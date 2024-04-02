from typing import Any
from typing import Callable

from ICARUS.computation.analyses.analysis import Analysis
from ICARUS.computation.analyses.input import AirplaneInput
from ICARUS.computation.analyses.input import Input
from ICARUS.computation.analyses.input import ListFloatInput
from ICARUS.computation.analyses.input import StateInput
from ICARUS.computation.analyses.input import StrInput


airplane_option = AirplaneInput()
state_opion = StateInput()
solver_2D_option = StrInput("solver2D", "Name of 2D Solver from which to use computed polars")
angles = ListFloatInput("angles", "List of angles to run polars")


class BaseAirplanePolarAnalysis(Analysis):
    def __init__(
        self,
        solver_name: str,
        execute_fun: Callable[..., Any],
        parallel_execute_fun: Callable[..., Any] | None = None,
        unhook: Callable[..., Any] | None = None,
        extra_options: list[Input] = [],
    ) -> None:
        super().__init__(
            solver_name=solver_name,
            analysis_name="Aiplane Polar Analysis",
            options=[
                airplane_option,
                state_opion,
                solver_2D_option,
                angles,
                *extra_options,
            ],
            execute_fun=execute_fun,
            parallel_execute_fun=parallel_execute_fun,
            unhook=unhook,
        )
