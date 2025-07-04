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
    __call__ = staticmethod(avl_polars)

    def __init__(
        self,
    ) -> None:
        super().__init__("AVL", avl_polars, unhook=None)


class AVL_DynamicAnalysisFD(BaseDynamicAnalysis):
    __call__ = staticmethod(avl_dynamics_fd)

    def __init__(self) -> None:
        super().__init__("AVL", avl_dynamics_fd, unhook=process_avl_dynamics_fd)


class AVL_DynamicAnalysisImplicit(BaseDynamicAnalysis):
    __call__ = staticmethod(avl_dynamics_implicit)

    def __init__(self) -> None:
        super().__init__(
            "AVL",
            avl_dynamics_implicit,
            unhook=process_avl_dynamics_implicit,
        )


class AVL(Solver):
    polars = AVL_PolarAnalysis()
    dynamics = AVL_DynamicAnalysisFD()
    dynamics_implicit = AVL_DynamicAnalysisImplicit()

    analyses = [
        AVL_PolarAnalysis(),
        AVL_DynamicAnalysisFD(),
        AVL_DynamicAnalysisImplicit(),
    ]

    def __init__(self) -> None:
        super().__init__(
            name="AVL",
            solver_type="3D VLM",
            fidelity=1,
            solver_parameters=[use_avl_control_option],
        )
