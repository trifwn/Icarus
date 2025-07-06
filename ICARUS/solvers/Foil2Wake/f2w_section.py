from dataclasses import dataclass
from dataclasses import field
from typing import final

from ICARUS.airfoils import Airfoil
from ICARUS.computation.analyses import Analysis, BaseAnalysisInput
from ICARUS.computation.analyses.analysis_input import iter_field
from ICARUS.computation.base_solver import Solver
from ICARUS.computation.solver_parameters import SolverParameters
from ICARUS.core.types import FloatArray
from ICARUS.solvers.Foil2Wake.post_process.polars import process_f2w_run
from ICARUS.solvers.Foil2Wake.analyses.angles import f2w_aseq


@dataclass
class Foil2WakeAseqInput(BaseAnalysisInput):
    """Input parameters for analyzing airfoil polar at specific angles."""

    airfoil: None | Airfoil | list[Airfoil] = iter_field(
        order=1,
        default=None,
        metadata={"description": "Airfoil object to be analyzed"},
    )
    mach: None | float = field(
        default=None,
        metadata={"description": "Mach number for the analysis"},
    )
    reynolds: None | float | list[float] | FloatArray = iter_field(
        order=0,
        default=None,
        metadata={"description": "Reynolds number for the analysis"},
    )
    angles: None | list[float] | FloatArray = field(
        default=None,
        metadata={
            "description": "List of angles of attack (in degrees) to run polar analysis"
        },
    )


@final
class Foil2WakeAseq(Analysis[Foil2WakeAseqInput]):
    __call__ = staticmethod(f2w_aseq)

    def __init__(self) -> None:
        super().__init__(
            analysis_name="Airfoil Polar Analysis",
            solver_name="Foil2Wake",
            execute_fun=f2w_aseq,
            post_execute_fun=process_f2w_run,
            input_type=Foil2WakeAseqInput(),
        )


@dataclass
class Foil2WakeSolverParameters(SolverParameters):
    """Complete solver parameters."""

    iterations: int = field(
        default=251, metadata={"description": "NTIMEM | Maximum number of iterations"}
    )
    timestep: float = field(
        default=0.01, metadata={"description": "DT1 | Simulation timestep"}
    )
    f_trip_low: float = field(
        default=0.1,
        metadata={"description": "TRANSLO | Transition points for lower surface"},
    )
    f_trip_upper: float = field(
        default=0.1,
        metadata={"description": "TRANSUP | Transition points for upper surface"},
    )
    Ncrit: float = field(
        default=9.0,
        metadata={"description": "N critical value for transition (e^N method)"},
    )
    boundary_layer_iteration_start: int = field(
        default=250, metadata={"description": "NTIME_bl | Start solving boundary layer"}
    )
    trailing_edge_angle: float = field(
        default=0.0, metadata={"description": "TEANGLE (deg) | Trailing edge angle"}
    )
    u_freestrem: float = field(
        default=1.0, metadata={"description": "UINF | Freestream velocity"}
    )
    Cuttoff_1: float = field(
        default=0.1, metadata={"description": "EPS1 | Cutoff parameter"}
    )
    Cuttoff_2: float = field(
        default=0.1, metadata={"description": "EPS2 | Cutoff parameter"}
    )
    EPSCOE: float = field(
        default=1.0, metadata={"description": "EPSCOE | Cutoff coefficient"}
    )
    NWS: int = field(default=3, metadata={"description": "NWS | ???"})
    CCC1: float = field(default=0.03, metadata={"description": "CCC1 | ???"})
    CCC2: float = field(default=0.03, metadata={"description": "CCC2 | ???"})
    CCGON1: float = field(default=30.0, metadata={"description": "CCGON1 | ???"})
    CCGON2: float = field(default=30.0, metadata={"description": "CCGON2 | ???"})
    IMOVE: int = field(default=1, metadata={"description": "IMOVE | ???"})
    A0: float = field(default=0.0, metadata={"description": "A0 | ???"})
    AMPL: float = field(default=0.0, metadata={"description": "AMPL | ???"})
    APHASE: float = field(default=0.0, metadata={"description": "APHASE | ???"})
    AKF: float = field(default=0.0, metadata={"description": "AKF | ???"})
    Chord_hinge: float = field(
        default=0.25,
        metadata={"description": "XC | Pitch point (0.25 = quarter chord)"},
    )
    ITEFLAP: int = field(
        default=1, metadata={"description": "ITEFLAP | 1: use flap, 0: don't use flap"}
    )
    XEXT: float = field(default=0.0, metadata={"description": "XEXT | ???"})
    YEXT: float = field(default=0.0, metadata={"description": "YEXT | ???"})
    NTEWT: int = field(default=9, metadata={"description": "NTEWT | ???"})
    NTEST: int = field(default=9, metadata={"description": "NTEST | ???"})
    IBOUNDL: int = field(
        default=1, metadata={"description": "IBOUNDL | 1: solve BL, 0: don't solve"}
    )
    IYNEXTERN: int = field(default=0, metadata={"description": "IYNEXTERN | ???"})
    ITSEPAR: int = field(default=0, metadata={"description": "ITSEPAR | ???"})
    ISTEADY: int = field(default=1, metadata={"description": "ISTEADY | ???"})


@final
class Foil2Wake(Solver[Foil2WakeSolverParameters]):
    analyses = [
        Foil2WakeAseq(),
    ]
    aseq = Foil2WakeAseq()

    def __init__(self) -> None:
        super().__init__(
            name="Foil2Wake",
            solver_type="2D-IBLM",
            fidelity=1,
            solver_parameters=Foil2WakeSolverParameters(),
        )
