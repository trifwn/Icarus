from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional

from ICARUS.computation.analyses.analysis_input import BaseAnalysisInput

from . import Analysis


@dataclass
class CaseBasedInput(BaseAnalysisInput):
    """Input based on a preconfigured case directory."""

    casedir: Optional[Path] = field(
        default=None,
        metadata={"description": "Path to the case directory containing setup and input files"},
    )


class CaseAnalysis(Analysis[CaseBasedInput]):
    def __init__(
        self,
        solver_name: str,
        execute_fun: Callable[..., Any],
        unhook: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            solver_name=solver_name,
            analysis_name="Rerun Analysis",
            execute_fun=execute_fun,
            post_execute_fun=unhook,
            input_type=CaseBasedInput(),
        )
