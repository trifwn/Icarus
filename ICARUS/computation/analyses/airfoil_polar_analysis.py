from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Optional

from ICARUS.airfoils import Airfoil

from . import Analysis
from . import BaseAnalysisInput


@dataclass
class AirfoilPolarAnalysisInput(BaseAnalysisInput):
    """Input parameters for analyzing airfoil polar at specific angles."""

    airfoil: Optional[Airfoil] = field(
        default=None,
        metadata={"description": "Airfoil object to be analyzed"},
    )
    mach: Optional[float] = field(
        default=None,
        metadata={"description": "Mach number for the analysis"},
    )
    reynolds: Optional[float] = field(
        default=None,
        metadata={"description": "Reynolds number for the analysis"},
    )
    angles: Optional[list[float]] = field(
        default=None,
        metadata={"description": "List of angles of attack (in degrees) to run polar analysis"},
    )


@dataclass
class AirfoilMultiReynsPolarAnalysisInput(BaseAnalysisInput):
    """Input for a multi-Reynolds airfoil polar analysis."""

    airfoil: Optional[Airfoil] = field(
        default=None,
        metadata={"description": "Airfoil object to be analyzed"},
    )
    mach: Optional[float] = field(
        default=None,
        metadata={"description": "Mach number for the analysis"},
    )
    reynolds: Optional[list[float]] = field(
        default=None,
        metadata={"description": "List of Reynolds numbers to run the analysis"},
    )
    angles: Optional[list[float]] = field(
        default=None,
        metadata={"description": "List of angles of attack (in degrees) for polar analysis"},
    )


class BaseAirfoilPolarAnalysis(Analysis[AirfoilPolarAnalysisInput]):
    def __init__(
        self,
        solver_name: str,
        execute_fun: Callable[..., Any],
        post_execute_fun: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            analysis_name="Airfoil Polar Analysis For a single Reynolds",
            solver_name=solver_name,
            execute_fun=execute_fun,
            post_execute_fun=post_execute_fun,
            input_type=AirfoilPolarAnalysisInput(),
        )


class BaseAirfoil_MultiReyn_PolarAnalysis(Analysis[AirfoilMultiReynsPolarAnalysisInput]):
    def __init__(
        self,
        solver_name: str,
        execute_fun: Callable[..., Any],
        post_execute_fun: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            solver_name=solver_name,
            analysis_name="Airfoil Polar Analysis for Multiple Reynold's Numbers",
            execute_fun=execute_fun,
            post_execute_fun=post_execute_fun,
            input_type=AirfoilMultiReynsPolarAnalysisInput(),
        )
