from dataclasses import dataclass
from dataclasses import field
from typing import Literal
from typing import Optional

from ICARUS.computation import SolverParameters
from ICARUS.computation.analyses import BaseAirplaneAseq
from ICARUS.computation.base_solver import Solver
from ICARUS.computation.solver_parameters import IntOrNoneParameter
from ICARUS.computation.solver_parameters import Parameter
from ICARUS.computation.solver_parameters import StrParameter

from .analyses.polars import lspt_polars


class LSPT_PolarAnalysis(BaseAirplaneAseq):
    __call__ = staticmethod(lspt_polars)

    def __init__(self) -> None:
        super().__init__(
            solver_name="LSPT",
            execute_fun=lspt_polars,
            unhook=None,
        )


solver_parameters: list[Parameter] = [
    IntOrNoneParameter(
        "Ground_Effect",
        None,
        "Distance From Ground (m). None for no ground effect",
    ),
    StrParameter(
        "Wake_Geom_Type",
        "TE-Geometrical",
        "Type of wake geometry. The options are: -TE-Geometrical -Inflow-Uniform -Inflow-TE",
    ),
]


@dataclass
class LSPT_Parameters(SolverParameters):
    """Solver parameters with wake and ground effect options."""

    solver2D: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str = "Xfoil"

    ground_effect: Optional[int] = field(
        default=None,
        metadata={"description": "Distance From Ground (m). None for no ground effect"},
    )
    wake_type: str = field(
        default="TE-Geometrical",
        metadata={
            "description": (
                "Type of wake geometry. "
                "The options are: -TE-Geometrical -Inflow-Uniform -Inflow-TE"
            ),
        },
    )


class LSPT(Solver[LSPT_Parameters]):
    analyses = [LSPT_PolarAnalysis()]
    aseq = LSPT_PolarAnalysis()

    def __init__(self) -> None:
        super().__init__("LSPT", "3D VLM", 1, solver_parameters=LSPT_Parameters())
