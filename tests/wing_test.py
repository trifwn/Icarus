from typing import Any

from numpy import dtype
from numpy import floating
from numpy import ndarray


def geom() -> tuple[
    float,
    float,
    float,
    ndarray[Any, dtype[floating]],
    ndarray[Any, dtype[floating]],
]:
    print("Testing Geometry...")

    from Data.Planes.simple_wing import Simplewing

    return (
        Simplewing.S,
        Simplewing.mean_aerodynamic_chord,
        Simplewing.Area,
        Simplewing.CG,
        Simplewing.inertia,
    )
