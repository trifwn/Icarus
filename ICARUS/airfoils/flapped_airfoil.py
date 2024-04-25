from __future__ import annotations

from ICARUS.airfoils.airfoil import Airfoil
from ICARUS.core.types import FloatArray


class FlappedAirfoil(Airfoil):
    def __init__(
        self,
        upper: FloatArray,
        lower: FloatArray,
        parent: Airfoil,
        name: str | None = None,
    ):
        if name is None:
            name = parent.name + "_flapped"

        self.parent = parent
        super().__init__(
            upper=upper,
            lower=lower,
            name=name,
        )

    def flap(
        self,
        flap_hinge_chord_percentage: float,
        flap_angle: float,
        flap_hinge_thickness_percentage: float = 0.5,
        chord_extension: float = 1,
    ) -> FlappedAirfoil | Airfoil:
        return self.parent.flap(
            flap_hinge_chord_percentage,
            flap_angle,
            flap_hinge_thickness_percentage,
            chord_extension,
        )
