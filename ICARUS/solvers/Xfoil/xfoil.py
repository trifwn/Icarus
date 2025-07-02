from dataclasses import dataclass
from dataclasses import field
from typing import Optional

from ICARUS.airfoils import Airfoil
from ICARUS.computation import Solver
from ICARUS.computation import SolverParameters
from ICARUS.computation.analyses import Analysis
from ICARUS.computation.analyses import BaseAnalysisInput
from ICARUS.computation.analyses.airfoil_polar_analysis import (
    BaseAirfoil_MultiReyn_PolarAnalysis,
)
from ICARUS.computation.analyses.analysis_input import iter_field
from ICARUS.solvers.Xfoil.analyses.angles import aseq_analysis, aseq_analysis_reset_bl
from ICARUS.solvers.Xfoil.post_process.polars import save_polar_results


@dataclass
class XfoilAseqInput(BaseAnalysisInput):
    """Input parameters for Xfoil airfoil polar analysis with angle of attack sweep."""

    airfoil: Optional[Airfoil] = field(
        default=None,
        metadata={"description": "Airfoil object to be analyzed"},
    )
    mach: Optional[float] = field(
        default=None,
        metadata={"description": "Mach number for the analysis"},
    )
    reynolds: Optional[list[float] | float] = iter_field(
        order=0,
        default=None,
        metadata={"description": "List of Reynolds numbers to analyze"},
    )
    min_aoa: Optional[float] = field(
        default=None,
        metadata={"description": "Minimum angle of attack in degrees"},
    )
    max_aoa: Optional[float] = field(
        default=None,
        metadata={"description": "Maximum angle of attack in degrees"},
    )
    aoa_step: Optional[float] = field(
        default=None,
        metadata={"description": "Angle of attack step in degrees"},
    )


class XfoilAseq(Analysis[XfoilAseqInput]):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            solver_name="Xfoil",
            analysis_name="Airfoil Polar Analysis For a multiple Reynolds using aseq",
            execute_fun=aseq_analysis,
            post_execute_fun=save_polar_results,
            input_type=XfoilAseqInput(),
        )


class XfoilAseqResetBL(BaseAirfoil_MultiReyn_PolarAnalysis):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            solver_name="Xfoil",
            execute_fun=aseq_analysis_reset_bl,
            post_execute_fun=save_polar_results,
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
                XfoilAseqResetBL(),
                XfoilAseq(),
            ],
            solver_parameters=XfoilSolverParameters(),
        )


if __name__ == "__main__":
    xfoil = Xfoil()
