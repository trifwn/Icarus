from ICARUS.computation.analyses.airplane_dynamic_analysis import BaseDynamicAnalysis
from ICARUS.computation.analyses.airplane_polar_analysis import (
    BaseAirplanePolarAnalysis,
)
from ICARUS.computation.analyses.input import FloatInput
from ICARUS.computation.analyses.input import IntInput
from ICARUS.computation.analyses.rerun_analysis import BaseRerunAnalysis
from ICARUS.computation.solvers.GenuVP.analyses.angles import process_gnvp_angles_run_7
from ICARUS.computation.solvers.GenuVP.analyses.angles import run_gnvp7_angles
from ICARUS.computation.solvers.GenuVP.analyses.angles import run_gnvp7_angles_parallel
from ICARUS.computation.solvers.GenuVP.analyses.pertrubations import (
    proccess_pertrubation_res_7,
)
from ICARUS.computation.solvers.GenuVP.analyses.pertrubations import (
    run_gnvp7_pertrubation_parallel,
)
from ICARUS.computation.solvers.GenuVP.analyses.pertrubations import (
    run_gnvp7_pertrubation_serial,
)
from ICARUS.computation.solvers.GenuVP.files.gnvp7_interface import gnvp7_execute
from ICARUS.computation.solvers.solver import Solver
from ICARUS.computation.solvers.solver_parameters import BoolParameter
from ICARUS.computation.solvers.solver_parameters import FloatParameter
from ICARUS.computation.solvers.solver_parameters import IntParameter
from ICARUS.computation.solvers.solver_parameters import Parameter

timestep_option = FloatInput(
    "timestep",
    "Timestep = 0.05 * chord / u_inf",
)

maxiter_option = IntInput(
    "maxiter",
    "Maximum number of iterations",
)


class GenuVP7_RerunCase(BaseRerunAnalysis):
    def __init__(self) -> None:
        super().__init__(
            solver_name="GenuVP7",
            execute_fun=gnvp7_execute,
        )


class GenuVP7_PolarAnalysis(BaseAirplanePolarAnalysis):
    def __init__(self) -> None:
        super().__init__(
            solver_name="GenuVP7",
            execute_fun=run_gnvp7_angles,
            parallel_execute_fun=run_gnvp7_angles_parallel,
            unhook=process_gnvp_angles_run_7,
            extra_options=[timestep_option, maxiter_option],
        )


class GenuVP7_DynamicAnalysis(BaseDynamicAnalysis):
    def __init__(self) -> None:
        super().__init__(
            solver_name="GenuVP7",
            execute_fun=run_gnvp7_pertrubation_serial,
            parallel_execute_fun=run_gnvp7_pertrubation_parallel,
            unhook=proccess_pertrubation_res_7,
            extra_options=[timestep_option, maxiter_option],
        )


solver_parameters: list[Parameter] = [
    BoolParameter(
        "Split_Symmetric_Bodies",
        False,
        "Split Symmetric Bodies And Contstruct Them In GNVP3",
    ),
    BoolParameter(
        "Use_Grid",
        False,
        "Use the Grid generated in the plane object",
    ),
    IntParameter(
        "Integration_Scheme",
        2,
        "NMETHT=1 for Euler =2 for Adams Bashford time integrat. scheme",
    ),
    IntParameter(
        "Tip_Emmision",
        0,
        "NEMTIP=0,1. The latter means that tip-emission takes place",
    ),
    IntParameter(
        "Tip_Emmision_Begins",
        0,
        "NTIMET Time at which tip-emission begins",
    ),
    IntParameter(
        "Leading_Edge_Separation",
        0,
        "=0(no action), 1(leading-edge separ. takes place)",
    ),
    IntParameter(
        "Leading_Edge_Separation_Begins",
        0,
        "NTIMEL Time at which leading-edge separ. begins",
    ),
    FloatParameter(
        "Relaxation_Factor",
        1.0,
        "RELAXS relaxation factor for the singularity distributions",
    ),
    FloatParameter(
        "Pot_Convergence_Tolerence",
        1e-4,
        "EPSDS convergence tolerance of the potential calculations",
    ),
    IntParameter(
        "Movement_Levels",
        17,
        "NLEVELT number of movements levels",
    ),
    IntParameter(
        "Vortex_Particle_Count",
        1,
        "NNEVP0 Number of vortex particles created within a time step per near-wake element of a thin  wing",
    ),
    FloatParameter(
        "Vortex_Particle_Relaxation",
        1.0,
        "RELAXU relaxation factor for the emission velocity",
    ),
    FloatParameter(
        "Minimum_Width_Parameter",
        1.0,
        "PARVEC parameter for the minimum width of the near-wake element",
    ),
    IntParameter(
        "NEMIS",
        1,
        "1 or 2 UKNOWN",
    ),
    FloatParameter(
        "Bound_Vorticity_Cutoff",
        1e-3,
        "EPSFB  Cut-off length for the bound vorticity",
    ),
    FloatParameter(
        "Wake_Vorticity_Cutoff",
        1e-3,
        "EPSFW  Cut-off length for the near-wake vorticity",
    ),
    FloatParameter(
        "Cutoff_Length_Sources",
        0.003,
        "EPSSR  Cut-off length for source distributions",
    ),
    FloatParameter(
        "Cutoff_Length_Sources2",
        0.003,
        "EPSDI  Cut-off length for source distributions",
    ),
    FloatParameter(
        "Vortex_Cutoff_Length_f",
        1e-1,
        "EPSVR  Cut-off length for the free vortex particles (final)",
    ),
    FloatParameter(
        "Vortex_Cutoff_Length_i",
        1e-1,
        "EPSO   Cut-off length for the free vortex particles (init.) ",
    ),
    FloatParameter(
        "EPSINT",
        0.001,
        "EPSINT",
    ),
    FloatParameter(
        "Particle_Dissipation_Factor",
        0.0,
        "COEF    Factor for the disipation of particles",
    ),
    FloatParameter(
        "Upper_Deformation_Rate",
        0.001,
        "RMETM   Upper bound of the deformation rate",
    ),
    FloatParameter(
        "Wake_Deformation_Parameter",
        1,
        "IDEFW   Parameter for the deformation induced by the near wake ",
    ),
    FloatParameter(
        "REFLEN",
        1000.0,
        "REFLEN  Length used in VELEF for suppresing far-particle calc.",
    ),
    FloatParameter(
        "Particle_Subdivision_Parameter",
        0,
        "IDIVVRP Parameter for the subdivision of particles",
    ),
    FloatParameter(
        "Subdivision_Length_Scale",
        1000.0,
        "FLENSC  Length scale for the subdivision of particles",
    ),
    FloatParameter(
        "Wake_Particle_Merging_Parameter",
        0,
        "NREWAK  Parameter for merging of particles",
    ),
    FloatParameter(
        "Particle_Merging_Parameter",
        0,
        "NMER    Parameter for merging of particles",
    ),
    FloatParameter(
        "Merging_Starting_Distance",
        0,
        "XREWAK  X starting distance of merging",
    ),
    FloatParameter("Merging_Radius", 0, "RADMER  Radius for merging"),
    FloatParameter("Elasticity_Solver", 0, "IYNELST (1=BEAMDYN,2-ALCYONE,3=GAST)"),
]


class GenuVP7(Solver):
    def __init__(self) -> None:
        super().__init__(
            "GenuVP7",
            "3D VPM",
            2,
            [GenuVP7_PolarAnalysis(), GenuVP7_DynamicAnalysis(), GenuVP7_RerunCase()],
            solver_parameters=solver_parameters,
        )


# # EXAMPLE USAGE
if __name__ == "__main__":
    pass
