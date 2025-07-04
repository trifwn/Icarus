from ICARUS.computation.analyses.airfoil_polar_analysis import (
    BaseAirfoil_MultiReyn_PolarAnalysis,
)
from ICARUS.computation.base_solver import Solver
from ICARUS.computation.solver_parameters import BoolParameter
from ICARUS.computation.solver_parameters import IntParameter
from ICARUS.computation.solver_parameters import Parameter
from ICARUS.solvers.OpenFoam.analyses.angles import angles_serial
from ICARUS.solvers.OpenFoam.files.setup_case import MeshType


class OpenFoam_MultiReyn_PolarAnanlysis(BaseAirfoil_MultiReyn_PolarAnalysis):
    def __init__(self) -> None:
        super().__init__(
            solver_name="Foil2Wake",
            execute_fun=angles_serial,
            post_execute_fun=None,
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
    analyses = [
        OpenFoam_MultiReyn_PolarAnanlysis(),
    ]

    def __init__(self) -> None:
        super().__init__(
            name="OpenFoam",
            solver_type="3D-RANS",
            fidelity=1,
            solver_parameters=solver_parameters,
        )


# Example Usage
if __name__ == "__main__":
    pass
    # open_foam = OpenFoam()
