from typing import Any

from numpy import dtype
from numpy import floating
from numpy import ndarray


def geom() -> (
    tuple[
        float,
        float,
        float,
        ndarray[Any, dtype[floating[Any]]],
        ndarray[Any, dtype[floating[Any]]],
    ]
):
    print("Testing Geometry...")

    from examples.Planes.simple_wing import Simplewing

    return (
        Simplewing.S,
        Simplewing.mean_aerodynamic_chord,
        Simplewing.area,
        Simplewing.CG,
        Simplewing.inertia,
    )
