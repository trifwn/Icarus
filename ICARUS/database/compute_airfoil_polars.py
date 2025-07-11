from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

import numpy as np

if TYPE_CHECKING:
    from ICARUS.airfoils import Airfoil
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

        xfoil = Xfoil()

        # Import Analysis
        xf_analysis = xfoil.aseq  # Run

        # Get Options
        xf_inputs = xf_analysis.get_analysis_input(verbose=False)

        # Set Options
        xf_inputs.airfoil = airfoil
        xf_inputs.mach = mach
        xf_inputs.reynolds = reynolds_numbers
        xf_inputs.min_aoa = float(np.min(aoas))
        xf_inputs.max_aoa = float(np.max(aoas))
        xf_inputs.aoa_step = aoas[1] - aoas[0]

        # Set Solver Options
        xf_solver_parameters = xfoil.get_solver_parameters()
        xf_solver_parameters.max_iter = 200
        xf_solver_parameters.Ncrit = 9
        xf_solver_parameters.xtr = (trips[0], trips[1])
        xf_solver_parameters.print = False
        xf_solver_parameters.repanel_n = repanel

        # RUN
        xfoil.execute(
            analysis=xf_analysis,
            inputs=xf_inputs,
            solver_parameters=xf_solver_parameters,
        )
    elif solver_name == "Foil2Wake":
        from ICARUS.solvers.Foil2Wake import Foil2Wake

        foil2w = Foil2Wake()
        # Get Options
        f2w_analysis = foil2w.aseq
        f2w_inputs = f2w_analysis.get_analysis_input(verbose=False)

        # Set Options
        f2w_inputs.airfoil = airfoil
        f2w_inputs.reynolds = reynolds_numbers
        f2w_inputs.mach = mach
        f2w_inputs.angles = aoas

        f2w_solver_parameters = foil2w.get_solver_parameters()
        f2w_solver_parameters.f_trip_upper = trips[0]
        f2w_solver_parameters.f_trip_low = trips[1]

        # RUN
        foil2w.execute(
            analysis=f2w_analysis,
            inputs=f2w_inputs,
            solver_parameters=f2w_solver_parameters,
        )
    elif solver_name == "OpenFoam":
        from ICARUS.solvers.OpenFoam.open_foam import OpenFoam

        ofoam = OpenFoam()
        # Import Analysis
        of_analysis = ofoam.get_analyses()[1]  # Run

        # Set Inputs
        of_inputs = of_analysis.get_analysis_input(verbose=False)
        of_inputs.airfoil = airfoil
        of_inputs.angles = aoas
        of_inputs.reynolds = reynolds_numbers
        of_inputs.mach = mach

        # Get Solver Parameters
        solver_parameters = ofoam.get_solver_parameters()
        from ICARUS.solvers.OpenFoam.files.setup_case import MeshType

        solver_parameters.mesh_type = MeshType.structAirfoilMesher
        solver_parameters.max_iterations = 2000
        solver_parameters.silent = False

        # RUN
        ofoam.execute(
            analysis=of_analysis,
            inputs=of_inputs,
            solver_parameters=solver_parameters,
        )
    else:
        raise ValueError("Solver not recognized")

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
