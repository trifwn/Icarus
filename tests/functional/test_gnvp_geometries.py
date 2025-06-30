import sys

import numpy as np
import pytest

from ICARUS.core.types import FloatArray
from ICARUS.database import angle_to_directory
from ICARUS.flight_dynamics import State
from ICARUS.solvers.GenuVP.post_process import get_wake_data_3
from ICARUS.solvers.GenuVP.post_process import get_wake_data_7
from ICARUS.vehicle import Airplane
from ICARUS.visualization.gnvp import plot_gnvp3_wake
from ICARUS.visualization.gnvp import plot_gnvp7_wake

GNVP_VERSIONS = [3, 7]


@pytest.mark.integration
@pytest.mark.parametrize("gnvp_version", GNVP_VERSIONS)
@pytest.mark.parametrize("plot", [False, True])
def test_gnvp_geometry_all(benchmark_airplane: Airplane, benchmark_state: State, gnvp_version: int, plot: bool) -> None:
    """Test GNVP geometry comparison for all versions, optionally with plotting."""

    if plot:
        pytest.importorskip("matplotlib")

    if gnvp_version == 7 and sys.platform.startswith("win"):
        pytest.skip("GenuVP7 solver is not available on Windows")

    _gnvp_geometry(benchmark_airplane, benchmark_state, gnvp_version, plot)


def _gnvp_geometry(benchmark_airplane: Airplane, benchmark_state: State, gnvp_version: int, plot: bool = False) -> None:
    """Internal function to test GNVP geometry.

    Args:
        gnvp_version: GNVP version (3 or 7)
        plot: If True, plots the results

    Raises:
        ValueError: If gnvp_version is not 3 or 7
    """
    if gnvp_version not in [3, 7]:
        raise ValueError(f"GNVP Version error! Got Version {gnvp_version}. Expected 3 or 7.")

    # Get the correct wake data function
    if gnvp_version == 3:
        get_wake_data = get_wake_data_3
        plot_gnvp_wake = plot_gnvp3_wake
    else:  # gnvp_version == 7
        get_wake_data = get_wake_data_7
        plot_gnvp_wake = plot_gnvp7_wake

    case: str = angle_to_directory(0.0)

    XP, QP, VP, GP, near_wake, grid_gnvp = get_wake_data(benchmark_airplane, benchmark_state, case)
    mesh_grid_gnvp = np.meshgrid(grid_gnvp)

    lgrid_plane: list[FloatArray] = []
    for surface in benchmark_airplane.surfaces:
        surf_grid = surface.get_grid()
        if isinstance(surf_grid, list):
            lgrid_plane.extend(surf_grid)
        else:
            lgrid_plane.append(surf_grid)

    grid_plane: FloatArray = np.array(lgrid_plane)
    shape = grid_plane.shape
    grid_plane = grid_plane.reshape(shape[0] * shape[1] * shape[2], shape[3]) - benchmark_airplane.CG
    mesh_grid_plane = np.meshgrid(grid_plane)

    if plot:
        plot_gnvp_wake(benchmark_airplane, benchmark_state, case)

    np.testing.assert_almost_equal(mesh_grid_plane, mesh_grid_gnvp, decimal=3)
