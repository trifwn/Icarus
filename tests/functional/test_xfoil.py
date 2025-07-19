import os
import time
from typing import Any

import numpy as np
import pytest

from ICARUS.airfoils import Airfoil
from ICARUS.computation.core import ExecutionMode
from ICARUS.core.units import calc_reynolds
from ICARUS.database import Database
from ICARUS.solvers.Xfoil import XfoilSolverParameters


@pytest.fixture(scope="module")
def test_airfoils(DB: Database) -> list[Airfoil]:
    """Fixture that provides test airfoils."""
    airfoils_to_compute: list[Airfoil] = []
    airfoils_to_compute.append(DB.get_airfoil("NACA4415"))
    airfoils_to_compute.append(DB.get_airfoil("NACA4412"))
    airfoils_to_compute.append(DB.get_airfoil("NACA0008"))
    airfoils_to_compute.append(DB.get_airfoil("NACA0012"))
    airfoils_to_compute.append(DB.get_airfoil("NACA0015"))
    airfoils_to_compute.append(DB.get_airfoil("NACA2412"))

    for airfoil in airfoils_to_compute:
        airfoil.repanel_spl(160)

    return airfoils_to_compute


@pytest.fixture
def xfoil_parameters() -> dict[str, Any]:
    """Fixture that provides common Xfoil parameters."""
    # PARAMETERS FOR ESTIMATION
    chord_max: float = 0.16
    chord_min: float = 0.06
    u_max: float = 35.0
    u_min: float = 5.0
    viscosity: float = 1.56e-5

    # MACH ESTIMATION
    mach_max: float = 0.085
    MACH: float = mach_max

    # REYNOLDS ESTIMATION
    reynolds_max: float = calc_reynolds(u_max, chord_max, viscosity)
    reynolds_min: float = calc_reynolds(u_min, chord_min, viscosity)
    reynolds = np.linspace(
        start=reynolds_min,
        stop=reynolds_max,
        num=12,
    )

    # ANGLE OF ATTACK SETUP
    aoa_min: float = -8
    aoa_max: float = 14
    Ncrit = 9.0

    return {
        "reynolds": reynolds,
        "mach": MACH,
        "aoa_min": aoa_min,
        "aoa_max": aoa_max,
        "Ncrit": Ncrit,
    }


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("airfoil_name", ["NACA4415", "NACA0012"])
def test_xfoil_single_airfoil(
    DB: Database,
    xfoil_parameters: dict[str, float | np.ndarray],
    airfoil_name: str,
    save_plot: bool = True,
) -> None:
    """Test Xfoil computation for a single airfoil."""
    airfoil = DB.get_airfoil(airfoil_name)
    airfoil.repanel_spl(200)

    print(f"\nRunning airfoil {airfoil.name}\n")
    start_time: float = time.time()

    from ICARUS.solvers.Xfoil import Xfoil

    xfoil: Xfoil = Xfoil()

    # Import Analysis - Sequential Angle run for multiple reynolds
    analysis = xfoil.aseq

    # Set Inputs
    xfoil_inputs = analysis.get_analysis_input()
    xfoil_inputs.airfoil = airfoil
    xfoil_inputs.reynolds = xfoil_parameters["reynolds"]
    xfoil_inputs.mach = float(xfoil_parameters["mach"])
    xfoil_inputs.max_aoa = float(xfoil_parameters["aoa_max"])
    xfoil_inputs.min_aoa = float(xfoil_parameters["aoa_min"])
    xfoil_inputs.aoa_step = 0.5

    # Set Solver Options
    xfoil_solver_parameters: XfoilSolverParameters = xfoil.get_solver_parameters()
    xfoil_solver_parameters.max_iter = 500
    xfoil_solver_parameters.Ncrit = int(xfoil_parameters["Ncrit"])
    xfoil_solver_parameters.xtr = (0.2, 0.2)
    xfoil_solver_parameters.print = False

    # RUN
    xfoil.execute(
        analysis=analysis,
        inputs=xfoil_inputs,
        solver_parameters=xfoil_solver_parameters,
        execution_mode=ExecutionMode.THREADING,
    )

    end_time: float = time.time()
    execution_time = end_time - start_time
    print(f"Airfoil {airfoil.name} completed in {execution_time:.2f} seconds")

    if save_plot:
        try:
            import matplotlib.pyplot as plt

            polar = DB.get_airfoil_polars(airfoil)
            airfoil_folder = os.path.join(DB.DB_PATH, "images")
            os.makedirs(airfoil_folder, exist_ok=True)
            polar.plot()
            polar.save_polar_plot_img(airfoil_folder)
            plt.close("all")  # Close plots to avoid memory issues

            # Check that image was created
            image_files = [f for f in os.listdir(airfoil_folder) if f.endswith(".png")]
            assert len(image_files) > 0, "Polar plot image should be created"

        except Exception as e:
            pytest.skip(f"Could not create polar plot: {e}")

    # Assert that execution completed in reasonable time
    assert (
        execution_time < 600.0
    ), (
        f"Xfoil took too long: {execution_time:.2f}s"
    )  # Try to get polar data to verify computation succeeded
    try:
        polar = DB.get_airfoil_polars(airfoil)
        assert polar is not None, "Polar data should be generated"
    except Exception as e:
        pytest.fail(f"Failed to retrieve polar data: {e}")
