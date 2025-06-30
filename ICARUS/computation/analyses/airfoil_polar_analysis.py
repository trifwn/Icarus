from dataclasses import dataclass
from typing import Any
from typing import Callable

from ICARUS.airfoils import Airfoil

from . import AirfoilInput as AirfoilInputDef
from . import Analysis
from . import AnalysisInput
from . import FloatInput
from . import Input
from . import ListFloatInput

airfoil_option = AirfoilInputDef()
mach_option = FloatInput(name="mach", description="Mach number")

multi_reynolds_option = ListFloatInput(
    name="reynolds",
    description="List of Reynold's numbers to run",
)

reynolds_option = FloatInput(
    name="reynolds",
    description="Reynold's numbers to run",
)

angles = ListFloatInput("angles", "List of angles to run polars")


@dataclass
class AirfoilPolarAnalysisInput(AnalysisInput):
    """Input for a single Reynolds airfoil polar analysis."""

    airfoil: Airfoil
    mach: float
    reynolds: float
    angles: list[float]


@dataclass
class AirfoilMultiReynsPolarAnalysisInput(AnalysisInput):
    """Input for a multi-Reynolds airfoil polar analysis."""

    airfoil: Airfoil
    mach: float
    reynolds: list[float]
    angles: list[float]


class BaseAirfoilPolarAnalysis(Analysis[AirfoilPolarAnalysisInput]):
    def __init__(
        self,
        solver_name: str,
        execute_fun: Callable[..., Any],
        unhook: Callable[..., Any] | None = None,
        extra_options: list[Input] = [],
    ) -> None:
        super().__init__(
            analysis_name="Airfoil Polar Analysis For a single Reynolds",
            solver_name=solver_name,
            inputs=[
                airfoil_option,
                mach_option,
                reynolds_option,
                angles,
                *extra_options,
            ],
            execute_fun=execute_fun,
            unhook=unhook,
            input_type=AirfoilPolarAnalysisInput,
        )


class BaseAirfoil_MultiReyn_PolarAnalysis(Analysis[AirfoilMultiReynsPolarAnalysisInput]):
    def __init__(
        self,
        solver_name: str,
        execute_fun: Callable[..., Any],
        unhook: Callable[..., Any] | None = None,
        extra_options: list[Input] = [],
    ) -> None:
        super().__init__(
            solver_name=solver_name,
            analysis_name="Airfoil Polar Analysis for Multiple Reynold's Numbers",
            inputs=[
                airfoil_option,
                mach_option,
                multi_reynolds_option,
                angles,
                *extra_options,
            ],
            execute_fun=execute_fun,
            unhook=unhook,
            input_type=AirfoilMultiReynsPolarAnalysisInput,
        )
