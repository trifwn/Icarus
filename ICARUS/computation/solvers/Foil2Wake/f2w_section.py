from ICARUS.computation.analyses.airfoil_polar_analysis import (
    BaseAirfoil_MultiReyn_PolarAnalysis,
)
from ICARUS.computation.analyses.airfoil_polar_analysis import BaseAirfoilPolarAnalysis
from ICARUS.computation.solvers import FloatParameter
from ICARUS.computation.solvers import IntParameter
from ICARUS.computation.solvers import Parameter
from ICARUS.computation.solvers import Solver
from ICARUS.computation.solvers.Foil2Wake.analyses.angles import process_f2w_run
from ICARUS.computation.solvers.Foil2Wake.analyses.angles import (
    run_multiple_reynolds_parallel,
)
from ICARUS.computation.solvers.Foil2Wake.analyses.angles import (
    run_multiple_reynolds_sequentially,
)
from ICARUS.computation.solvers.Foil2Wake.analyses.angles import run_single_reynolds


class Foil2Wake_PolarAnalysis(BaseAirfoilPolarAnalysis):
    def __init__(self) -> None:
        super().__init__(
            solver_name="Foil2Wake",
            execute_function=run_single_reynolds,
            unhook=process_f2w_run,
        )


class Foil2Wake_MultiReyn_PolarAnanlysis(BaseAirfoil_MultiReyn_PolarAnalysis):
    def __init__(self) -> None:
        super().__init__(
            solver_name="Foil2Wake",
            execute_fun=run_multiple_reynolds_sequentially,
            parallel_execute_fun=run_multiple_reynolds_parallel,
            unhook=process_f2w_run,
        )


solver_parameters: list[Parameter] = [
    IntParameter(
        "max_iter",
        251,
        "NTIMEM | Maximum number of iterations",
    ),
    FloatParameter(
        "timestep",
        0.01,
        "DT1 | Simulation timestep",
    ),
    FloatParameter(
        "f_trip_low",
        0.1,
        "TRANSLO | Transition points for positive and negative angles for the lower surface",
    ),
    FloatParameter(
        "f_trip_upper",
        0.1,
        "TRANSUP | Transition points for positive and negative angles for the upper surface",
    ),
    FloatParameter(
        "Ncrit",
        9,
        "N critical value for transition according to e to N method",
    ),
    IntParameter(
        "boundary_layer_solve_time",
        250,
        "NTIME_bl | When to start solving the boundary layer",
    ),
    FloatParameter("trailing_edge_angle", 0.0, "TEANGLE (deg) | Trailing edge angle"),
    FloatParameter("u_freestrem", 1.0, "UINF | Freestream velocity"),
    FloatParameter("Cuttoff_1", 0.1, "EPS1 | ..."),
    FloatParameter("Cuttoff_2", 0.1, "EPS2 | ..."),
    FloatParameter("EPSCOE", 1.0, "EPSCOE | ..."),
    IntParameter("NWS", 3, "NWS | ..."),
    FloatParameter("CCC1", 0.03, "CCC1 | ..."),
    FloatParameter("CCC2", 0.03, "CCC2 | ..."),
    FloatParameter("CCGON1", 30.0, "CCGON1 | ..."),
    FloatParameter("CCGON2", 30.0, "CCGON2 | ..."),
    IntParameter("IMOVE", 1, "IMOVE | ..."),
    FloatParameter("A0", 0.0, "A0 | ..."),
    FloatParameter("AMPL", 0.0, "AMPL | ..."),
    FloatParameter("APHASE", 0.0, "APHASE | ..."),
    FloatParameter("AKF", 0.0, "AKF | ..."),
    FloatParameter(
        "Chord_hinge",
        0.25,
        "XC | Point from where to pitch the airfoil. 0.25 is the quarter chord",
    ),
    IntParameter(
        "ITEFLAP",
        1,
        "ITEFLAP | Whether to use flap or not. 1: use flap, 0: don't use flap",
    ),
    FloatParameter("XEXT", 0.0, "XEXT | ..."),
    FloatParameter("YEXT", 0.0, "YEXT | ..."),
    FloatParameter(
        "NTEWT",
        9,
        "NTEWT | ...",
    ),
    FloatParameter(
        "NTEST",
        9,
        "NTEST | ...",
    ),
    IntParameter(
        "IBOUNDL",
        1,
        "IBOUNDL | Whether to use solve the boundary layer or not. 1: solve, 0: don't solve",
    ),
    IntParameter("IYNEXTERN", 0, "IYNEXTERN | ..."),
    IntParameter("ITSEPAR", 0, "ITSEPAR | ..."),
    IntParameter("ISTEADY", 1, "ISTEADY | ..."),
]


class Foil2Wake(Solver):
    def __init__(self) -> None:
        super().__init__(
            name="Foil2Wake",
            solver_type="2D-IBLM",
            fidelity=1,
            available_analyses=[
                Foil2Wake_PolarAnalysis(),
                Foil2Wake_MultiReyn_PolarAnanlysis(),
            ],
            solver_parameters=solver_parameters,
        )


# Example Usage
if __name__ == "__main__":
    pass
