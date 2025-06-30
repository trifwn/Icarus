from dataclasses import dataclass
from dataclasses import field

from ICARUS.airfoils import Airfoil
from ICARUS.computation import Solver
from ICARUS.computation import SolverParameters
from ICARUS.computation.analyses import AirfoilInput
from ICARUS.computation.analyses import Analysis
from ICARUS.computation.analyses import AnalysisInput
from ICARUS.computation.analyses import FloatInput
from ICARUS.computation.analyses import ListFloatInput
from ICARUS.computation.analyses.airfoil_polar_analysis import (
    BaseAirfoil_MultiReyn_PolarAnalysis,
)
from ICARUS.solvers.Xfoil.analyses.angles import multiple_reynolds_serial
from ICARUS.solvers.Xfoil.analyses.angles import multiple_reynolds_serial_seq

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


@dataclass
class XfoilAseqPolarAnalysisInput(AnalysisInput):
    airfoil: Airfoil
    mach: float
    reynolds: list[float]
    min_aoa: float
    max_aoa: float
    aoa_step: float


class Xfoil_Aseq_PolarAnalysis(Analysis):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            solver_name="Xfoil",
            analysis_name="Airfoil Polar Analysis For a multiple Reynolds using aseq",
            inputs=[
                AirfoilInput(),
                mach_option,
                multi_reynolds_option,
                min_angle,
                max_angle,
                aoa_step,
            ],
            execute_fun=multiple_reynolds_serial,
            unhook=None,
            input_type=XfoilAseqPolarAnalysisInput,
        )


class Xfoil_PolarAnalysis(BaseAirfoil_MultiReyn_PolarAnalysis):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            solver_name="Xfoil",
            execute_fun=multiple_reynolds_serial_seq,
            unhook=None,
        )


@dataclass
class XfoilSolverParameters(SolverParameters):
    """Parameters for the Xfoil solver."""

    max_iter: int = field(
        default=100,
        metadata={"description": "Maximum number of iterations"},
    )
    Ncrit: float = field(
        default=9.0,
        metadata={"description": "Ncrit"},
    )
    xtr: tuple[float, float] = field(
        default=(0.1, 0.1),
        metadata={"description": "Transition points: Lower and upper"},
    )
    print: bool = field(
        default=False,
        metadata={"description": "Print xfoil output"},
    )
    repanel_n: int = field(
        default=0,
        metadata={"description": "Number of panels to repanel the airfoil. 0 for no repaneling"},
    )


class Xfoil(Solver[XfoilSolverParameters]):
    def __init__(self) -> None:
        super().__init__(
            name="XFoil",
            solver_type="2D-IBLM",
            fidelity=1,
            available_analyses=[
                Xfoil_PolarAnalysis(),
                Xfoil_Aseq_PolarAnalysis(),
            ],
            solver_parameters=XfoilSolverParameters(),
        )


if __name__ == "__main__":
    xfoil = Xfoil()
