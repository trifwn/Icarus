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
class AirplaneStaticAnalysisInput(BaseAnalysisInput):
    """Input for a multi-Reynolds airfoil polar analysis."""

    airfoil: Optional[Airplane] = field(
        default=None,
        metadata={"description": "Airplane object to be analyzed"},
    )
    state: Optional[State] = field(
        default=None,
        metadata={"description": "Flight state (e.g., speed, altitude, orientation)"},
    )
    solver_2D: Optional[str] = field(
        default=None,
        metadata={"description": "Name of the 2D solver used for aerodynamic sectional analysis"},
    )
    angles: Optional[list[float]] = field(
        default=None,
        metadata={"description": "List of angles of attack (in degrees) to evaluate"},
    )


class BaseAirplanePolarAnalysis(Analysis[AirplaneStaticAnalysisInput]):
    def __init__(
        self,
        solver_name: str,
        execute_fun: Callable[..., Any],
        unhook: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            solver_name=solver_name,
            analysis_name="Aiplane Polar Analysis",
            execute_fun=execute_fun,
            post_execute_fun=unhook,
            input_type=AirplaneStaticAnalysisInput(),
        )
