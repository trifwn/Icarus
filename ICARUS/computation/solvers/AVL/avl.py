from ICARUS.computation.analyses.airplane_dynamic_analysis import BaseDynamicAnalysis
from ICARUS.computation.analyses.airplane_polar_analysis import (
    BaseAirplanePolarAnalysis,
)
from ICARUS.computation.solvers.AVL.analyses.pertrubations import (
    avl_dynamic_analysis_fd,
)
from ICARUS.computation.solvers.AVL.analyses.pertrubations import (
    avl_dynamic_analysis_implicit,
)
from ICARUS.computation.solvers.AVL.analyses.pertrubations import process_avl_fd_res
from ICARUS.computation.solvers.AVL.analyses.pertrubations import process_avl_impl_res
from ICARUS.computation.solvers.AVL.analyses.polars import avl_angle_run
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
        super().__init__("AVL", avl_angle_run, unhook=None)


# class AVL_TrimAnalysis(BasePolarAnalysis):
#     def __init__(
#         self,
#     ) -> None:
#         super().__init__("AVL",avl_angle_run,unhook = process_avl_angles_run)


class AVL_DynamicAnalysisFD(BaseDynamicAnalysis):
    def __init__(self) -> None:
        super().__init__("AVL", avl_dynamic_analysis_fd, unhook=process_avl_fd_res)


class AVL_DynamicAnalysisImplicit(BaseDynamicAnalysis):
    def __init__(self) -> None:
        super().__init__(
            "AVL",
            avl_dynamic_analysis_implicit,
            unhook=process_avl_impl_res,
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
