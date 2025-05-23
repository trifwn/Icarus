from typing import Any
from typing import Callable

from . import AirfoilInput
from . import Analysis
from . import FloatInput
from . import Input
from . import ListFloatInput

airfoil_option = AirfoilInput()
mach_option = FloatInput(name="mach", description="Mach number")

multi_reynolds_option = ListFloatInput(
    name="reynolds",
    description="List of Reynold's numbers to run",
)

reynolds_option = FloatInput(
    name="reynolds",
    description="List of Reynold's numbers to run",
)

angles = ListFloatInput("angles", "List of angles to run polars")


class BaseAirfoilPolarAnalysis(Analysis):
    def __init__(
        self,
        solver_name: str,
        execute_function: Callable[..., Any],
        parallel_execute_function: Callable[..., Any] | None = None,
        unhook: Callable[..., Any] | None = None,
        extra_options: list[Input] = [],
    ) -> None:
        super().__init__(
            analysis_name="Airfoil Polar Analysis For a single Reynolds",
            solver_name=solver_name,
            options=[
                airfoil_option,
                mach_option,
                reynolds_option,
                angles,
                *extra_options,
            ],
            execute_fun=execute_function,
            parallel_execute_fun=parallel_execute_function,
            unhook=unhook,
        )


class BaseAirfoil_MultiReyn_PolarAnalysis(Analysis):
    def __init__(
        self,
        solver_name: str,
        execute_fun: Callable[..., Any],
        parallel_execute_fun: Callable[..., Any] | None = None,
        extra_options: list[Input] = [],
        unhook: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            solver_name=solver_name,
            analysis_name="Airfoil Polar Analysis for Multiple Reynold's Numbers",
            options=[
                airfoil_option,
                mach_option,
                multi_reynolds_option,
                angles,
                *extra_options,
            ],
            execute_fun=execute_fun,
            parallel_execute_fun=parallel_execute_fun,
            unhook=unhook,
        )
