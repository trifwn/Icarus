"""Defines the conceptual plane class"""

from typing import Any
from numpy import dtype, floating, ndarray

import numpy as np
from .wing_concept import ConceptualWing
from .fueselage_concept import ConceptualFueselage


class ConceptualPlane():
    """Defines a plane in the conceptual design context"""

    def __init__(
        self,
        wing: ConceptualWing,
        elevator: ConceptualWing,
        rudder: ConceptualWing,
        fueselage: ConceptualFueselage,
        center_gravity: float | None,
    ) -> None:
        """Initializes the plane"""

        self.main_wing: ConceptualWing = wing
        self.elevator: ConceptualWing = elevator
        self.rudder: ConceptualWing = rudder
        if center_gravity is None:
            self.cog: float = np.inf
        else:
            self.cog: float = center_gravity
        self.fueselage: ConceptualFueselage = fueselage

        # Get the coefficients
        self.cl_0: float = self.main_wing.cl_0
        self.cd_0: float = self.main_wing.cd_0
        self.cl: ndarray[Any, dtype[floating]] = self._lift_coefficient_with_fueselage()
        self.oswald_efficiency: float = 0.7

    def _lift_coefficient_with_fueselage(
        self
    ) -> ndarray[Any, dtype[floating]]:
        """Defines the lift coefficient of the plane with the fuselage"""
        cl: ndarray[Any, dtype[floating]] = self.main_wing.cl_3d*(
            1 - self.fueselage.k *
            (self.main_wing.cl_0 / self.main_wing.cl_3d) *
            (self.fueselage.area / self.main_wing.area)
        )
        return cl
