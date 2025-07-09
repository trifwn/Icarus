from typing import Literal
from dataclasses import dataclass

from ICARUS.computation import SolverParameters

@dataclass
class AVLParameters(SolverParameters):
    """Parameters for the AVL solver."""

    use_avl_control: bool = False
    cd_parasitic: float = 0.0
    run_invscid: bool = False
    solver2D: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str = "Xfoil"
