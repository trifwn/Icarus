from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from ICARUS.computation import SolverParameters


@dataclass
class GenuVP7Parameters(SolverParameters):
    """Parameters for the GNVP7 solver."""

    solver2D: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str = "Xfoil"
    timestep: float = field(
        default=0.1,
        metadata={"description": "Time step for the simulation in seconds"},
    )
    iterations: int = field(
        default=100,
        metadata={"description": "Maximum number of iterations for the simulation"},
    )

    Split_Symmetric_Bodies: bool = field(
        default=False,
        metadata={"description": "Split Symmetric Bodies And Construct Them In GNVP3"},
    )
    Use_Grid: bool = field(
        default=False,
        metadata={"description": "Use the Grid generated in the plane object"},
    )
    Integration_Scheme: int = field(
        default=2,
        metadata={"description": "NMETHT=1 for Euler, =2 for Adams Bashforth"},
    )
    Tip_Emmision: int = field(
        default=0,
        metadata={
            "description": "NEMTIP=0,1. The latter means that tip-emission takes place",
        },
    )
    Tip_Emmision_Begins: int = field(
        default=0,
        metadata={"description": "NTIMET Time at which tip-emission begins"},
    )
    Leading_Edge_Separation: int = field(
        default=0,
        metadata={
            "description": "=0(no action), 1(leading-edge separation takes place)",
        },
    )
    Leading_Edge_Separation_Begins: int = field(
        default=0,
        metadata={"description": "NTIMEL Time at which leading-edge separation begins"},
    )
    Relaxation_Factor: float = field(
        default=1.0,
        metadata={
            "description": "RELAXS relaxation factor for singularity distributions",
        },
    )
    Pot_Convergence_Tolerence: float = field(
        default=1e-4,
        metadata={
            "description": "EPSDS convergence tolerance of the potential calculations",
        },
    )
    Movement_Levels: int = field(
        default=17,
        metadata={"description": "NLEVELT number of movement levels"},
    )
    Vortex_Particle_Count: int = field(
        default=1,
        metadata={
            "description": "NNEVP0 Vortex particles created per time step per element",
        },
    )
    Vortex_Particle_Relaxation: float = field(
        default=1.0,
        metadata={"description": "RELAXU relaxation factor for emission velocity"},
    )
    Minimum_Width_Parameter: float = field(
        default=1.0,
        metadata={"description": "PARVEC: minimum width of near-wake element"},
    )
    NEMIS: int = field(
        default=1,
        metadata={"description": "NEMIS=1 or 2. UNKNOWN"},
    )
    Bound_Vorticity_Cutoff: float = field(
        default=1e-3,
        metadata={"description": "EPSFB cut-off length for bound vorticity"},
    )
    Wake_Vorticity_Cutoff: float = field(
        default=1e-3,
        metadata={"description": "EPSFW cut-off length for near-wake vorticity"},
    )
    Cutoff_Length_Sources: float = field(
        default=0.003,
        metadata={"description": "EPSSR cut-off length for source distributions"},
    )
    Cutoff_Length_Sources2: float = field(
        default=0.003,
        metadata={"description": "EPSDI cut-off length for source distributions"},
    )
    Vortex_Cutoff_Length_f: float = field(
        default=0.1,
        metadata={"description": "EPSVR cut-off for free vortex particles (final)"},
    )
    Vortex_Cutoff_Length_i: float = field(
        default=0.1,
        metadata={"description": "EPSO cut-off for free vortex particles (initial)"},
    )
    EPSINT: float = field(
        default=0.001,
        metadata={"description": "EPSINT"},
    )
    Particle_Dissipation_Factor: float = field(
        default=0.0,
        metadata={"description": "COEF: Dissipation factor of particles"},
    )
    Upper_Deformation_Rate: float = field(
        default=0.001,
        metadata={"description": "RMETM: Upper bound of deformation rate"},
    )
    Wake_Deformation_Parameter: int = field(
        default=1,
        metadata={
            "description": "IDEFW:  Add contribution of the near wake elements optionally this can be performed eighter through vortex particle approximation (IDEFW=1) or through vortex lattice approximation (IDEFW=2))",
        },
    )
    REFLEN: float = field(
        default=1000.0,
        metadata={
            "description": "REFLEN: length to suppress far-particle calc in VELEF",
        },
    )
    Particle_Subdivision_Parameter: int = field(
        default=0,
        metadata={"description": "IDIVVRP: subdivision of particles"},
    )
    Subdivision_Length_Scale: int = field(
        default=1000,
        metadata={"description": "FLENSC: subdivision length scale"},
    )
    Wake_Particle_Merging_Parameter: int = field(
        default=0,
        metadata={"description": "NREWAK: wake particle merging parameter"},
    )
    Particle_Merging_Parameter: int = field(
        default=0,
        metadata={"description": "NMER: general merging parameter"},
    )
    Merging_Starting_Distance: float = field(
        default=0.0,
        metadata={"description": "XREWAK: X starting distance of merging"},
    )
    Merging_Radius: float = field(
        default=0.0,
        metadata={"description": "RADMER: radius for merging"},
    )
    Elasticity_Solver: float = field(
        default=0.0,
        metadata={"description": "IYNELST (1=BEAMDYN, 2=ALCYONE, 3=GAST)"},
    )

    genu_version: int = 7
