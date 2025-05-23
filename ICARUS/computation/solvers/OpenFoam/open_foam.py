from ICARUS.computation.analyses.airfoil_polar_analysis import (
    BaseAirfoil_MultiReyn_PolarAnalysis,
)
from ICARUS.computation.solvers import BoolParameter
from ICARUS.computation.solvers import IntParameter
from ICARUS.computation.solvers import Parameter
from ICARUS.computation.solvers import Solver
from ICARUS.computation.solvers.OpenFoam.analyses.angles import angles_parallel
from ICARUS.computation.solvers.OpenFoam.analyses.angles import angles_serial
from ICARUS.computation.solvers.OpenFoam.files.setup_case import MeshType


class OpenFoam_MultiReyn_PolarAnanlysis(BaseAirfoil_MultiReyn_PolarAnalysis):
    def __init__(self) -> None:
        super().__init__(
            solver_name="Foil2Wake",
            execute_fun=angles_serial,
            parallel_execute_fun=angles_parallel,
            unhook=None,
        )


solver_parameters: list[Parameter] = [
    Parameter(
        "mesh_type",
        MeshType.structAirfoilMesher,
        "Type of mesh to use",
        MeshType,
    ),
    IntParameter(
        "max_iterations",
        400,
        "Maximum number of iterations",
    ),
    BoolParameter(
        "silent",
        False,
        "Whether to print progress or not",
    ),
]


class OpenFoam(Solver):
    def __init__(self) -> None:
        super().__init__(
            name="OpenFoam",
            solver_type="3D-RANS",
            fidelity=1,
            available_analyses=[
                OpenFoam_MultiReyn_PolarAnanlysis(),
            ],
            solver_parameters=solver_parameters,
        )


# Example Usage
if __name__ == "__main__":
    pass
    # open_foam = OpenFoam()
