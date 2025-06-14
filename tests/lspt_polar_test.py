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
def test_lspt_run(
    benchmark_airplane: Airplane,  # Assuming benchmark_plane is a fixture providing an Airplane instance
    benchmark_state: State,  # Assuming benchmark_state is a fixture providing a State instance
):
    """Test LSPT solver execution."""
    print("Testing LSPT Running...")

    # Get Solver
    from ICARUS.computation.solvers.Icarus_LSPT import LSPT

    lspt: Solver = LSPT()

    # Set Analysis
    analysis: str = lspt.get_analyses_names()[0]

    lspt.select_analysis(analysis)

    # Set Options
    options: Struct = lspt.get_analysis_options(verbose=True)
    solver_parameters: Struct = lspt.get_solver_parameters()

    AoAmin = -3
    AoAmax = 3
    NoAoA = (AoAmax - AoAmin) + 1
    angles: FloatArray = np.linspace(AoAmin, AoAmax, NoAoA)

    options.plane = benchmark_airplane
    options.state = benchmark_state
    options.solver2D = "Xfoil"
    options.angles = angles

    solver_parameters.Ground_Effect = True
    solver_parameters.Wake_Geom_Type = "TE-Geometrical"

    lspt.define_analysis(options, solver_parameters)
    _ = lspt.get_analysis_options(verbose=True)

    start_time: float = time.perf_counter()
    lspt.execute()

    end_time: float = time.perf_counter()
    execution_time = end_time - start_time
    print(f"LSPT Run took: {execution_time:.3f} seconds")
    print("Testing LSPT Running... Done")

    results = lspt.get_results()

    # Assert that results were generated
    assert results is not None, "LSPT should return results"

    # Assert execution time is reasonable (less than 180 seconds)
    assert execution_time < 180.0, f"LSPT execution took too long: {execution_time:.3f}s"


# Backward compatibility function
def lspt_run() -> None:
    """Legacy function for backward compatibility."""
    from .benchmark_plane_test import get_benchmark_plane
    from .benchmark_plane_test import get_benchmark_state

    airplane = get_benchmark_plane("bmark")
    state = get_benchmark_state(airplane)

    test_lspt_run(
        benchmark_airplane=airplane,
        benchmark_state=state,
    )
