from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

import numpy as np

if TYPE_CHECKING:
    from ICARUS.airfoils import Airfoil
    from ICARUS.computation import Solver
    from ICARUS.core.types import FloatArray


def compute_airfoil_polars(
    airfoil: Airfoil,
    reynolds_numbers: list[float] | FloatArray,
    trips: tuple[float, float] = (1.0, 1.0),
    aoas: list[float] | FloatArray = np.linspace(-10, 16, 53),
    mach: float = 0.0,
    solver_name: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str = "Xfoil",
    repanel: int = 140,
    plot_polars: bool = False,
) -> None:
    if solver_name == "Xfoil":
        from ICARUS.solvers.Xfoil.xfoil import Xfoil

        solver: Solver = Xfoil()

        # Import Analysis
        analysis: str = solver.get_analyses_names()[1]  # Run
        solver.select_analysis(analysis)

        # Get Options
        options = solver.get_analysis_options(verbose=False)
        solver_parameters = solver.get_solver_parameters()

        # Set Options
        options.airfoil = airfoil
        options.mach = mach
        options.reynolds = reynolds_numbers
        options.min_aoa = min(aoas)
        options.max_aoa = max(aoas)
        options.aoa_step = aoas[1] - aoas[0]

        # Set Solver Options
        solver_parameters.max_iter = 200
        solver_parameters.Ncrit = 9
        solver_parameters.xtr = (trips[0], trips[1])
        solver_parameters.print = False
        solver_parameters.repanel_n = repanel

    elif solver_name == "Foil2Wake":
        from ICARUS.solvers.Foil2Wake.f2w_section import Foil2Wake

        solver = Foil2Wake()
        # Import Analysis
        analysis = solver.get_analyses_names()[1]  # Run
        solver.select_analysis(analysis)

        # Get Options
        options = solver.get_analysis_options(verbose=False)
        solver_parameters = solver.get_solver_parameters()

        # Set Options
        options.airfoil = airfoil
        options.reynolds = reynolds_numbers
        options.mach = mach
        options.angles = aoas

        solver_parameters.f_trip_upper = trips[0]
        solver_parameters.f_trip_low = trips[1]

    elif solver_name == "OpenFoam":
        from ICARUS.solvers.OpenFoam.open_foam import OpenFoam

        solver = OpenFoam()
        # Import Analysis
        analysis = solver.get_analyses_names()[1]  # Run
        solver.select_analysis(analysis)

        # Get Options
        options = solver.get_analysis_options(verbose=False)
        solver_parameters = solver.get_solver_parameters()

        # Set Options
        options.airfoil = airfoil
        options.angles = aoas
        options.reynolds = reynolds_numbers
        options.mach = mach

        # Set Solver Options
        from ICARUS.solvers.OpenFoam.files.setup_case import MeshType

        solver_parameters.mesh_type = MeshType.structAirfoilMesher
        solver_parameters.max_iterations = 2000
        solver_parameters.silent = False

    else:
        raise ValueError("Solver not recognized")

    # Run Solver
    solver.define_analysis(options, solver_parameters)
    solver.print_analysis_options()

    # RUN
    solver.execute()
    # Get polar
    if plot_polars:
        import matplotlib.pyplot as plt

        from ICARUS.database import Database

        DB = Database.get_instance()

        polar = DB.get_airfoil_polars(airfoil)
        fig = polar.plot()
        fig.show()
        plt.pause(10.0)
        plt.close()
