from typing import Any
from typing import Callable

from . import Analysis
from . import Input
from . import StrInput

casedir: StrInput = StrInput("CASEDIR", "Case Directory")


class BaseRerunAnalysis(Analysis):
    def __init__(
        self,
        solver_name: str,
        execute_fun: Callable[..., Any],
        unhook: Callable[..., Any] | None = None,
        extra_options: list[Input] = [],
    ) -> None:
        super().__init__(
            solver_name=solver_name,
            analysis_name="Rerun Analysis",
            inputs=[casedir, *extra_options],
            execute_fun=execute_fun,
            unhook=unhook,
        )
