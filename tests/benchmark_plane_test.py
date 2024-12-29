import numpy as np

from ICARUS.core.types import FloatArray
from ICARUS.environment.definition import EARTH_ISA
from ICARUS.flight_dynamics.state import State
from ICARUS.vehicle.plane import Airplane
from ICARUS.vehicle.utils import SymmetryAxes
from ICARUS.vehicle.wing_segment import WingSegment


def get_bmark_plane(name: str) -> tuple[Airplane, State]:
    origin: FloatArray = np.array([0.0, 0.0, 0.0], dtype=float)
    wing_position: FloatArray = np.array(
        [0, 0.0, 0.0],
        dtype=float,
    )
    wing_orientation: FloatArray = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    Simplewing = WingSegment(
        name=name,
        root_airfoil="NACA0012",
        origin=origin + wing_position,
        orientation=wing_orientation,
        symmetries=SymmetryAxes.Y,
        span=2 * 2.5,
        sweep_offset=0.0,
        root_chord=0.8,
        tip_chord=0.8,
        N=10,
        M=5,
        mass=1,
    )
    airplane = Airplane(Simplewing.name, [Simplewing])
    # print(airplane.CG)
    # airplane.CG = np.array([0.337, 0, 0])

    u_inf = 20
    state = State(
        name="Unstick",
        airplane=airplane,
        u_freestream=u_inf,
        environment=EARTH_ISA,
    )

    return airplane, state
