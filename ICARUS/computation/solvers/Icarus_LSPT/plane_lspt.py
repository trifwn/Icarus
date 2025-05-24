from ICARUS.aero.vlm import run_lstp_angles
from ICARUS.computation.analyses import BaseAirplanePolarAnalysis
from ICARUS.computation.solvers import IntOrNoneParameter
from ICARUS.computation.solvers import Parameter
from ICARUS.computation.solvers import Solver
from ICARUS.computation.solvers import StrParameter


class LSPT_PolarAnalysis(BaseAirplanePolarAnalysis):
    def __init__(self) -> None:
        super().__init__(
            solver_name="LSPT",
            execute_fun=run_lstp_angles,
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
    def __init__(self) -> None:
        super().__init__(
            "LSPT",
            "3D VLM",
            1,
            [LSPT_PolarAnalysis()],
            solver_parameters=solver_parameters,
        )
