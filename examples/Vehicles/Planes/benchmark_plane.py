import numpy as np

from ICARUS.Computation.Solvers.XFLR5.polars import read_polars_2d
from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Database import DB
from ICARUS.Database import EXTERNAL_DB
from ICARUS.Environment.definition import EARTH_ISA
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.utils import SymmetryAxes
from ICARUS.Vehicle.wing_segment import Wing_Segment


def get_bmark_plane(name: str) -> tuple[Airplane, State]:
    read_polars_2d(EXTERNAL_DB)
    airfoils: Struct = DB.foils_db.airfoils

    origin: FloatArray = np.array([0.0, 0.0, 0.0], dtype=float)
    wing_position: FloatArray = np.array(
        [0, 0.0, 0.0],
        dtype=float,
    )
    wing_orientation: FloatArray = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    Simplewing = Wing_Segment(
        name=name,
        root_airfoil=DB.get_airfoil("NACA0015"),
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
    state = State(name="Unstick", airplane=airplane, u_freestream=u_inf, environment=EARTH_ISA)

    return airplane, state
