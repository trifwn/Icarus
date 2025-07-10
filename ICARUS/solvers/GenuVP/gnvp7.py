from typing import final

from ICARUS.computation.analyses import Analysis
from ICARUS.computation.base_solver import Solver

from .analyses import get_aseq_progress
from .analyses import get_stability_progress
from .analyses import gnvp_aseq
from .analyses import gnvp_stability
from .analyses import process_gnvp7_dynamics
from .analyses import process_gnvp_polars_7
from .gnvp7_parameters import GenuVP7Parameters
from .gnvp_inputs import GNVPAseqAnalysisInput
from .gnvp_inputs import GNVPStabilityAnalysisInput


@final
class GenuVP7_Aseq(Analysis[GNVPAseqAnalysisInput]):
    __call__ = staticmethod(gnvp_aseq)

    def __init__(self) -> None:
        super().__init__(
            solver_name="GenuVP3",
            execute_fun=gnvp_aseq,
            post_execute_fun=process_gnvp_polars_7,
            analysis_name="Aiplane Polar Analysis",
            input_type=GNVPAseqAnalysisInput(),
            monitor_progress_fun=get_aseq_progress,
        )


@final
class GenuVP7_Stability(Analysis[GNVPStabilityAnalysisInput]):
    __call__ = staticmethod(gnvp_stability)

    def __init__(self) -> None:
        super().__init__(
            solver_name="GenuVP3",
            analysis_name="Dynamic Analysis",
            execute_fun=gnvp_stability,
            post_execute_fun=process_gnvp7_dynamics,
            input_type=GNVPStabilityAnalysisInput(),
            monitor_progress_fun=get_stability_progress,
        )


class GenuVP7(Solver[GenuVP7Parameters]):
    analyses = [GenuVP7_Aseq(), GenuVP7_Stability()]
    aseq = GenuVP7_Aseq()
    stability = GenuVP7_Stability()

    def __init__(self) -> None:
        super().__init__(
            "GenuVP7",
            "3D VPM",
            2,
            solver_parameters=GenuVP7Parameters(),
        )
