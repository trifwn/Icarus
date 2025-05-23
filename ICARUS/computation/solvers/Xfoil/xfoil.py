from ICARUS.computation.analyses import AirfoilInput
from ICARUS.computation.analyses import Analysis
from ICARUS.computation.analyses import FloatInput
from ICARUS.computation.analyses import ListFloatInput
from ICARUS.computation.analyses.airfoil_polar_analysis import BaseAirfoilPolarAnalysis
from ICARUS.computation.solvers import BoolParameter
from ICARUS.computation.solvers import FloatParameter
from ICARUS.computation.solvers import IntParameter
from ICARUS.computation.solvers import Parameter
from ICARUS.computation.solvers import Solver
from ICARUS.computation.solvers.Xfoil.analyses.angles import multiple_reynolds_parallel
from ICARUS.computation.solvers.Xfoil.analyses.angles import (
    multiple_reynolds_parallel_seq,
)
from ICARUS.computation.solvers.Xfoil.analyses.angles import multiple_reynolds_serial
from ICARUS.computation.solvers.Xfoil.analyses.angles import (
    multiple_reynolds_serial_seq,
)

mach_option = FloatInput(name="mach", description="Mach number")

multi_reynolds_option = ListFloatInput(
    name="reynolds",
    description="List of Reynold's numbers to run",
)
min_angle = FloatInput(
    "min_aoa",
    "Minimum angle of attack",
)
max_angle = FloatInput(
    "max_aoa",
    "Maximum angle of attack",
)
aoa_step = FloatInput(
    "aoa_step",
    "Step between each angle of attack",
)


class Xfoil_Aseq_PolarAnalysis(Analysis):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            solver_name="Xfoil",
            analysis_name="Airfoil Polar Analysis For a multiple Reynolds using aseq",
            options=[
                AirfoilInput(),
                mach_option,
                multi_reynolds_option,
                min_angle,
                max_angle,
                aoa_step,
            ],
            execute_fun=multiple_reynolds_serial,
            parallel_execute_fun=multiple_reynolds_parallel,
            unhook=None,
        )


class Xfoil_PolarAnalysis(BaseAirfoilPolarAnalysis):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            solver_name="Xfoil",
            execute_function=multiple_reynolds_serial_seq,
            parallel_execute_function=multiple_reynolds_parallel_seq,
            unhook=None,
        )


solver_parameters: list[Parameter] = [
    IntParameter(
        "max_iter",
        100,
        "Maximum number of iterations",
    ),
    FloatParameter(
        "Ncrit",
        9,
        "Ncrit",
    ),
    Parameter(
        "xtr",
        (0.1, 0.1),
        "Transition points: Lower and upper",
        tuple[float],
    ),
    BoolParameter(
        "print",
        False,
        "Print xfoil output",
    ),
    IntParameter(
        "repanel_n",
        100,
        "Number of panels to repanel the airfoil",
    ),
]


class Xfoil(Solver):
    def __init__(self) -> None:
        super().__init__(
            name="XFoil",
            solver_type="2D-IBLM",
            fidelity=1,
            available_analyses=[
                Xfoil_PolarAnalysis(),
                Xfoil_Aseq_PolarAnalysis(),
            ],
            solver_parameters=solver_parameters,
        )


if __name__ == "__main__":
    xfoil = Xfoil()
