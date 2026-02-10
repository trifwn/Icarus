from dataclasses import dataclass

from ICARUS.aero.vlm import lspt_polars
from ICARUS.computation.analyses import BaseAirplaneAseq
from ICARUS.computation.base_solver import Solver
from ICARUS.computation.solver_parameters import SolverParameters


class LSPT_PolarAnalysis(BaseAirplaneAseq):
    def __init__(self) -> None:
        super().__init__(
            solver_name="LSPT",
            execute_fun=lspt_polars,
            unhook=None,
        )


@dataclass
class LSPTParameters(SolverParameters):
    """Parameters for the LSPT solver."""

    Ground_Effect: int | None = None
    """Distance From Ground (m). None for no ground effect."""

    Wake_Geom_Type: str = "TE-Geometrical"
    """Type of wake geometry. The options are: -TE-Geometrical -Inflow-Uniform -Inflow-TE"""


class LSPT(Solver[LSPTParameters]):
    analyses = [LSPT_PolarAnalysis()]

    def __init__(self) -> None:
        super().__init__(
            "LSPT",
            "3D VLM",
            1,
            solver_parameters=LSPTParameters(),
        )
