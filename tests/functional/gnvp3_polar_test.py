from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ICARUS.computation.solvers import Solver
from ICARUS.core.base_types import Struct
from ICARUS.core.types import FloatArray

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("run_parallel", [True, False])
def test_gnvp3_run(
    benchmark_airplane: Airplane,  # Assuming benchmark_plane is a fixture providing an Airplane instance
    benchmark_state: State,  # Assuming benchmark_state is a fixture providing a State instance
    run_parallel: bool,
):
    """Test GNVP3 solver execution in parallel and serial modes."""
    print(f"Testing GNVP3 Running ({'Parallel' if run_parallel else 'Serial'})...")

    # Get Solver
    from ICARUS.computation.solvers.GenuVP import GenuVP3

    gnvp3: Solver = GenuVP3()

    # Set Analysis
    polar_analysis: str = gnvp3.get_analyses_names()[0]
    gnvp3.select_analysis(polar_analysis)

    # Set Options
    options: Struct = gnvp3.get_analysis_options(verbose=True)
    solver_parameters: Struct = gnvp3.get_solver_parameters()

    AoAmin = -5
    AoAmax = 5
    NoAoA = (AoAmax - AoAmin) + 1
    angles_all: FloatArray = np.linspace(AoAmin, AoAmax, NoAoA)
    angles: list[float] = [ang for ang in angles_all]
    maxiter = 30
    timestep = 0.004

    options.plane = benchmark_airplane
    options.state = benchmark_state
    options.solver2D = "Xfoil"
    options.maxiter = maxiter
    options.timestep = timestep
    options.angles = angles

    solver_parameters.Split_Symmetric_Bodies = False
    solver_parameters.Use_Grid = True

    # Deformation
    solver_parameters.Bound_Vorticity_Cutoff = 0.003  # EPSFB
    solver_parameters.Wake_Vorticity_Cutoff = 0.003  # EPSFW
    solver_parameters.Vortex_Cutoff_Length_f = 1e-1  # EPSVR
    solver_parameters.Vortex_Cutoff_Length_i = 1e-1  # EPSO

    gnvp3.define_analysis(options, solver_parameters)
    _ = gnvp3.get_analysis_options(verbose=True)
    start_time: float = time.perf_counter()

    gnvp3.execute(parallel=run_parallel)

    end_time: float = time.perf_counter()
    execution_time = end_time - start_time
    mode = "Parallel" if run_parallel else "Serial"
    print(f"GNVP3 {mode} Run took: {execution_time:.3f} seconds")
    print("Testing GNVP3 Running... Done")

    results = gnvp3.get_results()

    # Assert that results were generated
    assert results is not None, "GNVP3 should return results"

    # Assert execution time is reasonable (less than 300 seconds)
    assert execution_time < 300.0, f"GNVP3 execution took too long: {execution_time:.3f}s"


if __name__ == "__main__":
    from .benchmark_plane_test import get_benchmark_plane
    from .benchmark_plane_test import get_benchmark_state

    airplane = get_benchmark_plane("bmark")
    state = get_benchmark_state(airplane)
    test_gnvp3_run(
        benchmark_airplane=airplane,
        benchmark_state=state,
        run_parallel=False,  # Change to True for parallel execution
    )
