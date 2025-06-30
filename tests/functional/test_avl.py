from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ICARUS.computation import Solver
from ICARUS.core.base_types import Struct

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane


@pytest.mark.slow
@pytest.mark.integration
def test_avl_run(
    benchmark_airplane: Airplane,  # Assuming benchmark_plane is a fixture providing an Airplane instance
    benchmark_state: State,  # Assuming benchmark_state is a fixture providing a State instance
):
    """Test AVL solver execution."""
    print("Testing AVL Running ...")
    # Get Solver
    from ICARUS.solvers.AVL import AVL

    avl: Solver = AVL()
    analysis: str = avl.get_analyses_names()[0]

    avl.select_analysis(analysis)

    # Set Options
    options: Struct = avl.get_analysis_options(verbose=True)
    solver_parameters: Struct = avl.get_solver_parameters(verbose=True)

    AoAmin = -10
    AoAmax = 10
    NoAoA = (AoAmax - AoAmin) + 1
    angles = np.linspace(AoAmin, AoAmax, NoAoA)

    options.plane = benchmark_airplane
    options.state = benchmark_state
    options.solver2D = "Xfoil"
    options.angles = angles

    solver_parameters.use_avl_control = False

    avl.define_analysis(options=options, solver_parameters=solver_parameters)
    _ = avl.get_analysis_options(verbose=True)
    start_time: float = time.perf_counter()

    avl.execute()

    end_time: float = time.perf_counter()
    execution_time = end_time - start_time
    print(f"AVL Run took: {execution_time:.3f} seconds")
    print("Testing AVL Running... Done")

    results = avl.get_results()

    # Assert that results were generated
    assert results is not None, "AVL should return results"

    # Assert execution time is reasonable (less than 60 seconds)
    assert execution_time < 60.0, f"AVL execution took too long: {execution_time:.3f}s"
