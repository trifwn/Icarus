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


class LSPT(Solver):
    analyses = [LSPT_PolarAnalysis()]
    aseq = LSPT_PolarAnalysis()

    def __init__(self) -> None:
        super().__init__(
            "LSPT",
            "3D VLM",
            1,
            solver_parameters=solver_parameters,
        )
