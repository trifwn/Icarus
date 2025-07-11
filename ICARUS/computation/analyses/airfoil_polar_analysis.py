from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Optional

from ICARUS.airfoils import Airfoil
from ICARUS.computation.analyses.analysis_input import iter_field
from ICARUS.core.types import FloatArray

from . import Analysis
from . import BaseAnalysisInput


@dataclass
class AirfoilPolarAnalysisInput(BaseAnalysisInput):
    """Input parameters for analyzing airfoil polar at specific angles."""

    airfoil: Optional[Airfoil | list[Airfoil]] = iter_field(
        order=2,
        default=None,
        metadata={"description": "Airfoil object to be analyzed"},
    )
    mach: Optional[float] = field(
        default=None,
        metadata={"description": "Mach number for the analysis"},
    )
    reynolds: Optional[float | list[float] | FloatArray] = iter_field(
        order=1,
        default=None,
        metadata={"description": "Reynolds number for the analysis"},
    )
    angles: Optional[list[float] | FloatArray | float] = iter_field(
        order=0,
        default=None,
        metadata={
            "description": "List of angles of attack (in degrees) to run polar analysis",
        },
    )


class BaseAirfoilPolarAnalysis(Analysis[AirfoilPolarAnalysisInput]):
    def __init__(
        self,
        solver_name: str,
        execute_fun: Callable[..., Any],
        post_execute_fun: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            analysis_name="Airfoil Polar Analysis",
            solver_name=solver_name,
            execute_fun=execute_fun,
            post_execute_fun=post_execute_fun,
            input_type=AirfoilPolarAnalysisInput(),
        )
