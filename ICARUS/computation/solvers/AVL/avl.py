from ICARUS.computation.analyses.airplane_dynamic_analysis import BaseDynamicAnalysis
from ICARUS.computation.analyses.airplane_polar_analysis import (
    BaseAirplanePolarAnalysis,
)
from ICARUS.computation.solvers.AVL import avl_dynamics_fd
from ICARUS.computation.solvers.AVL import avl_dynamics_implicit
from ICARUS.computation.solvers.AVL import avl_polars
from ICARUS.computation.solvers.AVL import process_avl_dynamics_fd
from ICARUS.computation.solvers.AVL import process_avl_dynamics_implicit
from ICARUS.computation.solvers.solver import Solver
from ICARUS.computation.solvers.solver_parameters import BoolParameter

use_avl_control_option = BoolParameter(
    name="use_avl_control",
    description="Use AVL control surface deflections",
    default_value=False,
)


class AVL_PolarAnalysis(BaseAirplanePolarAnalysis):
    def __init__(
        self,
    ) -> None:
        super().__init__("AVL", avl_polars, unhook=None)


# class AVL_TrimAnalysis(BasePolarAnalysis):
#     def __init__(
#         self,
#     ) -> None:
#         super().__init__("AVL",avl_polars,unhook = process_avl_polars)


class AVL_DynamicAnalysisFD(BaseDynamicAnalysis):
    def __init__(self) -> None:
        super().__init__("AVL", avl_dynamics_fd, unhook=process_avl_dynamics_fd)


class AVL_DynamicAnalysisImplicit(BaseDynamicAnalysis):
    def __init__(self) -> None:
        super().__init__(
            "AVL",
            avl_dynamics_implicit,
            unhook=process_avl_dynamics_implicit,
        )


class AVL(Solver):
    def __init__(self) -> None:
        super().__init__(
            "AVL",
            "3D VLM",
            1,
            [
                AVL_PolarAnalysis(),
                AVL_DynamicAnalysisFD(),
                AVL_DynamicAnalysisImplicit(),
            ],
            solver_parameters=[use_avl_control_option],
        )


# # EXAMPLE USAGE
if __name__ == "__main__":
    avl = AVL()
    available_analyses = avl.get_analyses_names()
    analysis = avl.analyses[available_analyses[0]]
    avl.select_analysis(analysis.name)
    options = avl.get_analysis_options()
    # etc...
