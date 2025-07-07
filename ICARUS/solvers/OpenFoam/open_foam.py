from dataclasses import dataclass
from dataclasses import field

from ICARUS.computation import SolverParameters
from ICARUS.computation.analyses.airfoil_polar_analysis import BaseAirfoilPolarAnalysis
from ICARUS.computation.base_solver import Solver
from ICARUS.solvers.OpenFoam.analyses.angles import angles_serial
from ICARUS.solvers.OpenFoam.files.setup_case import MeshType


class OpenFoam_MultiReyn_PolarAnanlysis(BaseAirfoilPolarAnalysis):
    def __init__(self) -> None:
        super().__init__(
            solver_name="Foil2Wake",
            execute_fun=angles_serial,
            post_execute_fun=None,
        )


@dataclass
class OpenFoamParameters(SolverParameters):
    """Generic solver parameters."""

    mesh_type: MeshType = field(
        default=MeshType.structAirfoilMesher,
        metadata={"description": "Type of mesh to use"},
    )
    max_iterations: int = field(
        default=400,
        metadata={"description": "Maximum number of iterations"},
    )
    silent: bool = field(
        default=False,
        metadata={"description": "Whether to print progress or not"},
    )


class OpenFoam(Solver[OpenFoamParameters()]):
    analyses = [
        OpenFoam_MultiReyn_PolarAnanlysis(),
    ]

    def __init__(self) -> None:
        super().__init__(
            name="OpenFoam",
            solver_type="3D-RANS",
            fidelity=1,
            solver_parameters=OpenFoamParameters(),
        )
