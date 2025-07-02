from ICARUS.computation.analyses.airplane_dynamic_analysis import BaseDynamicAnalysis
from ICARUS.computation.analyses.airplane_polar_analysis import (
    BaseAirplanePolarAnalysis,
)
from ICARUS.computation.base_solver import Solver
from ICARUS.computation.solver_parameters import BoolParameter
from ICARUS.solvers.AVL import avl_dynamics_fd
from ICARUS.solvers.AVL import avl_dynamics_implicit
from ICARUS.solvers.AVL import avl_polars
from ICARUS.solvers.AVL import process_avl_dynamics_fd
from ICARUS.solvers.AVL import process_avl_dynamics_implicit

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
