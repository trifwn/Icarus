from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ICARUS.core.types import FloatArray

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane


@pytest.mark.slow
@pytest.mark.integration
def test_lspt_run(
    benchmark_airplane: Airplane,  # Assuming benchmark_plane is a fixture providing an Airplane instance
    benchmark_state: State,  # Assuming benchmark_state is a fixture providing a State instance
) -> None:
    """Test LSPT solver execution."""
    print("Testing LSPT Running...")

    # Get Solver
    from ICARUS.solvers.Icarus_LSPT import LSPT

    lspt: LSPT = LSPT()

    # Set Analysis
    analysis = lspt.aseq

    # Set Options
    inputs = analysis.get_analysis_input(verbose=True)
    AoAmin = -3
    AoAmax = 3
    NoAoA = (AoAmax - AoAmin) + 1
    angles: FloatArray = np.linspace(AoAmin, AoAmax, NoAoA)

    inputs.plane = benchmark_airplane
    inputs.state = benchmark_state
    inputs.angles = angles

    solver_parameters = lspt.get_solver_parameters()
    solver_parameters.solver2D = "Xfoil"
    solver_parameters.ground_effect = True
    solver_parameters.wake_type = "TE-Geometrical"

    start_time: float = time.perf_counter()
    results = lspt.execute(
        analysis=analysis,
        inputs=inputs,
        solver_parameters=solver_parameters,
    )

    end_time: float = time.perf_counter()
    execution_time = end_time - start_time
    print(f"LSPT Run took: {execution_time:.3f} seconds")
    print("Testing LSPT Running... Done")

    # Assert that results were generated
    assert results is not None, "LSPT should return results"

    # Assert execution time is reasonable (less than 180 seconds)
    assert (
        execution_time < 180.0
    ), f"LSPT execution took too long: {execution_time:.3f}s"
