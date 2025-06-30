from typing import Any
from typing import Callable

from . import AirplaneInput
from . import Analysis
from . import Input
from . import StateInput
from . import StrInput

airplane_option = AirplaneInput()
state_opion = StateInput()
solver_2D_option = StrInput(
    "solver2D",
    "Name of 2D Solver from which to use computed polars",
)


class BaseDynamicAnalysis(Analysis):
    def __init__(
        self,
        solver_name: str,
        execute_fun: Callable[..., Any],
        unhook: Callable[..., Any] | None = None,
        extra_options: list[Input] = [],
    ) -> None:
        super().__init__(
            analysis_name="Dynamic Analysis",
            solver_name=solver_name,
            inputs=[airplane_option, state_opion, solver_2D_option, *extra_options],
            execute_fun=execute_fun,
            unhook=unhook,
        )
