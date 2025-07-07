from typing import final

from ICARUS.computation.analyses import Analysis
from ICARUS.computation.base_solver import Solver

from .analyses import get_aseq_progress
from .analyses import get_stability_progress
from .analyses import gnvp_aseq
from .analyses import gnvp_stability
from .analyses import process_gnvp3_dynamics
from .analyses import process_gnvp_polars_3
from .gnvp3_parameters import GenuVP3Parameters
from .gnvp_inputs import GNVPAseqAnalysisInput
from .gnvp_inputs import GNVPStabilityAnalysisInput


@final
class GenuVP3_Aseq(Analysis[GNVPAseqAnalysisInput]):
    __call__ = staticmethod(gnvp_aseq)

    def __init__(self) -> None:
        super().__init__(
            solver_name="GenuVP3",
            execute_fun=gnvp_aseq,
            post_execute_fun=process_gnvp_polars_3,
            analysis_name="Aiplane Polar Analysis",
            input_type=GNVPAseqAnalysisInput(),
            monitor_progress_fun=get_aseq_progress,
        )


@final
class GenuVP3_Stability(Analysis[GNVPStabilityAnalysisInput()]):
    __call__ = staticmethod(gnvp_stability)

    def __init__(self) -> None:
        super().__init__(
            solver_name="GenuVP3",
            analysis_name="Dynamic Analysis",
            execute_fun=gnvp_stability,
            post_execute_fun=process_gnvp3_dynamics,
            input_type=GNVPStabilityAnalysisInput(),
            monitor_progress_fun=get_stability_progress,
        )


class GenuVP3(Solver[GenuVP3Parameters]):
    analyses = [GenuVP3_Aseq(), GenuVP3_Stability()]
    aseq = GenuVP3_Aseq()
    stability = GenuVP3_Stability()

    def __init__(self) -> None:
        super().__init__(
            "GenuVP3",
            "3D VPM",
            2,
            solver_parameters=GenuVP3Parameters(),
        )


# EXAMPLE USAGE
if __name__ == "__main__":
    pass
