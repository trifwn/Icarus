from dataclasses import dataclass
from dataclasses import field

from ICARUS.computation.analyses import BaseAnalysisInput
from ICARUS.computation.analyses.analysis_input import iter_field
from ICARUS.core.types import FloatArray
from ICARUS.flight_dynamics import Disturbance
from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane


@dataclass
class GNVPAseqAnalysisInput(BaseAnalysisInput):
    """Input for a multi-Reynolds airfoil polar analysis."""

    plane: None | Airplane = field(
        default=None,
        metadata={"description": "Airplane object to be analyzed"},
    )
    state: None | State = field(
        default=None,
        metadata={"description": "Flight state (e.g., speed, altitude, orientation)"},
    )
    angles: None | float | list[float] | FloatArray = iter_field(
        order=0,
        default=None,
        metadata={"description": "List of angles of attack (in degrees) to evaluate"},
    )


@dataclass
class GNVPStabilityAnalysisInput(BaseAnalysisInput):
    """Input parameters for a dynamic analysis involving an airplane and its flight state."""

    plane: None | Airplane = field(
        default=None,
        metadata={"description": "Airplane object to be analyzed dynamically"},
    )
    state: None | State = field(
        default=None,
        metadata={"description": "Flight state describing velocity, altitude, etc."},
    )
    disturbances: None | Disturbance | list[Disturbance] = iter_field(
        order=0,
        default=None,
        metadata={
            "description": "List of disturbances to apply during the dynamic analysis",
        },
    )
