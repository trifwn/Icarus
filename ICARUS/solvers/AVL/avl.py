from __future__ import annotations

from ICARUS.computation.analyses.airplane_dynamic_analysis import BaseStabilityAnalysis
from ICARUS.computation.analyses.airplane_polar_analysis import BaseAirplaneAseq
from ICARUS.computation.base_solver import Solver
from ICARUS.solvers.AVL import avl_polars
from ICARUS.solvers.AVL import avl_stability_fd
from ICARUS.solvers.AVL import avl_stability_implicit
from ICARUS.solvers.AVL import process_avl_dynamics_fd
from ICARUS.solvers.AVL import process_avl_dynamics_implicit

from .avl_parameters import AVLParameters


class AVL_PolarAnalysis(BaseAirplaneAseq):
    __call__ = staticmethod(avl_polars)

    def __init__(
        self,
    ) -> None:
        super().__init__("AVL", avl_polars, unhook=None)


class AVL_StabilityFD(BaseStabilityAnalysis):
    __call__ = staticmethod(avl_stability_fd)

    def __init__(self) -> None:
        super().__init__(
            "AVL",
            avl_stability_fd,
            post_execute_fun=process_avl_dynamics_fd,
        )


class AVL_StabilityImplicit(BaseStabilityAnalysis):
    __call__ = staticmethod(avl_stability_implicit)

    def __init__(self) -> None:
        super().__init__(
            "AVL",
            avl_stability_implicit,
            post_execute_fun=process_avl_dynamics_implicit,
        )


class AVL(Solver[AVLParameters]):
    aseq = AVL_PolarAnalysis()
    stability = AVL_StabilityFD()
    stability_implicit = AVL_StabilityImplicit()

    analyses = [
        AVL_PolarAnalysis(),
        AVL_StabilityFD(),
        AVL_StabilityImplicit(),
    ]

    def __init__(self) -> None:
        super().__init__(
            name="AVL",
            solver_type="3D VLM",
            fidelity=1,
            solver_parameters=AVLParameters(),
        )
