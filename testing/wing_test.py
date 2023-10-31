from ICARUS.Core.types import FloatArray


def geom() -> (
    tuple[
        float,
        float,
        float,
        FloatArray,
        FloatArray,
    ]
):
    print("Testing Geometry...")

    from examples.Vehicles.Planes.simple_wing import Simplewing

    return (
        Simplewing.S,
        Simplewing.mean_aerodynamic_chord,
        Simplewing.area,
        Simplewing.CG,
        Simplewing.inertia,
    )
