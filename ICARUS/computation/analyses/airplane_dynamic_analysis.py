from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Optional

from ICARUS.computation.analyses.analysis_input import BaseAnalysisInput
from ICARUS.flight_dynamics.state import State
from ICARUS.vehicle.airplane import Airplane

from . import Analysis


@dataclass
class AirplaneDynamicAnalysisInput(BaseAnalysisInput):
    """Input parameters for a dynamic analysis involving an airplane and its flight state."""

    airfoil: Optional[Airplane] = field(
        default=None,
        metadata={"description": "Airplane object to be analyzed dynamically"},
    )
    state: Optional[State] = field(
        default=None,
        metadata={"description": "Flight state describing velocity, altitude, etc."},
    )
    solver_2D: Optional[str] = field(
        default=None,
        metadata={"description": "Name of the 2D solver to be used for sectional analysis"},
    )


class BaseDynamicAnalysis(Analysis[AirplaneDynamicAnalysisInput]):
    def __init__(
        self,
        solver_name: str,
        execute_fun: Callable[..., Any],
        unhook: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            analysis_name="Dynamic Analysis",
            solver_name=solver_name,
            execute_fun=execute_fun,
            post_execute_fun=unhook,
            input_type=AirplaneDynamicAnalysisInput(),
        )
