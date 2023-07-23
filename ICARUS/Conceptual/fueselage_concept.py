"""Defines the conceptual fueselage class"""


class ConceptualFueselage:
    """Defines a plane in the conceptual design context"""

    def __init__(self, fueselage_area: float, k: float = 1.0) -> None:
        self.area: float = fueselage_area
        self.k: float = k

    @property
    def weight(self) -> float:
        """Returns the weight of the fueselage"""
        return self._weight

    @weight.setter
    def add_weight(self, weight: float) -> None:
        """Adds a weight to the fueselage"""
        self._weight: float = weight
