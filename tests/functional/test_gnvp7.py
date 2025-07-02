from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ICARUS.computation import Solver
from ICARUS.computation.analyses.analysis import Analysis
from ICARUS.computation.core.types import ExecutionMode

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane

from ICARUS import PLATFORM


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("run_parallel", [True, False])
@pytest.mark.skipif(PLATFORM == "Windows", reason="GenuVP7 solver is not available in this environment")
def test_gnvp7_run(
    benchmark_airplane: Airplane,  # Assuming benchmark_plane is a fixture providing an Airplane instance
    benchmark_state: State,  # Assuming benchmark_state is a fixture providing a State instance
    run_parallel: bool,
):
    """Test GNVP7 solver execution in parallel and serial modes."""
    print(f"Testing GNVP7 Running ({'Parallel' if run_parallel else 'Serial'})...")

    # Get Solver
    from ICARUS.solvers.GenuVP import GenuVP7

    gnvp7: Solver = GenuVP7()

    # Set Analysis
    analysis: Analysis = gnvp7.get_analyses()[0]

    # Set Options
    inputs = analysis.get_analysis_input(verbose=True)

    AoAmin = -5
    AoAmax = 5
    NoAoA = (AoAmax - AoAmin) + 1
    angles = np.linspace(AoAmin, AoAmax, NoAoA)
    maxiter = 30
    timestep = 0.004

    inputs.plane = benchmark_airplane
    inputs.state = benchmark_state
    inputs.solver2D = "Xfoil"
    inputs.maxiter = maxiter
    inputs.timestep = timestep
    inputs.angles = angles

    solver_parameters = gnvp7.get_solver_parameters()
    solver_parameters.Split_Symmetric_Bodies = False
    solver_parameters.Use_Grid = True

    # DEFORMATION
    solver_parameters.Bound_Vorticity_Cutoff = 0.003  # EPSFB
    solver_parameters.Wake_Vorticity_Cutoff = 0.003  # EPSFW
    solver_parameters.Vortex_Cutoff_Length_f = 1e-1  # EPSVR
    solver_parameters.Vortex_Cutoff_Length_i = 1e-1  # EPSO

    execution_mode = ExecutionMode.MULTIPROCESSING if run_parallel else ExecutionMode.SEQUENTIAL
    start_time: float = time.perf_counter()
    results = gnvp7.execute(
        analysis=analysis,
        inputs=inputs,
        solver_parameters=solver_parameters,
        execution_mode=execution_mode,
    )
    end_time: float = time.perf_counter()
    execution_time = end_time - start_time

    print(f"GNVP7 {execution_mode} Run took: {execution_time:.3f} seconds")
    print("Testing GNVP7 Running... Done")

    # Assert that results were generated
    assert results is not None, "GNVP7 should return results"

    # Assert execution time is reasonable (less than 300 seconds)
    assert execution_time < 300.0, f"GNVP7 execution took too long: {execution_time:.3f}s"


if __name__ == "__main__":
    # Run the test directly if this script is executed
    pytest.main([__file__, "-v", "-s", "--tb=short"])
    # -v: verbose output
    # -s: disable output capturing
    # --tb=short: use short traceback format
