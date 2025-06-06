import numpy as np

from ICARUS.airfoils import NACA4
from ICARUS.core.types import FloatArray
from ICARUS.environment import EARTH_ISA
from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane
from ICARUS.vehicle import SymmetryAxes
from ICARUS.vehicle import WingSegment


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
        root_airfoil=NACA4(M=0.04, P=0.4, XX=0.15),  # "NACA4415",
        origin=origin + wing_position,
        orientation=wing_orientation,
        symmetries=SymmetryAxes.Y,
        span=2 * 5,
        sweep_offset=0.0,
        root_chord=1.0,
        tip_chord=1.0,
        N=15,
        M=15,
        mass=1,
    )
    airplane = Airplane(Simplewing.name, main_wing=Simplewing)
    u_inf = 100
    state = State(
        name="Unstick",
        airplane=airplane,
        u_freestream=u_inf,
        environment=EARTH_ISA,
    )
    return airplane, state
