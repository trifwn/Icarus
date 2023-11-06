from ICARUS.Core.types import FloatArray
from ICARUS.Vehicle.plane import Airplane


def geom() -> (
    tuple[
        float,
        float,
        # float,
        FloatArray,
        FloatArray,
    ]
):
    print("Testing Geometry...")

    from examples.Vehicles.Planes.benchmark_plane import get_bmark_plane

    bmark: Airplane = get_bmark_plane("bmark")

    return (
        bmark.S,
        bmark.mean_aerodynamic_chord,
        # bmark.ar,
        bmark.CG,
        bmark.total_inertia,
    )
