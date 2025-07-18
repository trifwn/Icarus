from __future__ import annotations

from jaxtyping import Float

from ICARUS.airfoils import Airfoil


class FlappedAirfoil(Airfoil):
    def __init__(
        self,
        upper: Float,
        lower: Float,
        parent: Airfoil,
        name: str | None = None,
    ) -> None:
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

    def __repr__(self) -> str:
        """Returns the string representation of the airfoil

        Returns:
            str: String representation of the airfoil

        """
        return f"(Flapped Airfoil): {self.name} with ({len(self._x_lower)} x {len(self._x_upper)}) points"
