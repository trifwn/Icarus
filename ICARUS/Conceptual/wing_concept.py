"""Defines the conceptual wing class"""

from typing import Any
from numpy import dtype, floating, intp, ndarray
import numpy as np
from ICARUS.Core.struct import Struct
from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Database.db import DB
from ICARUS.Airfoils.airfoilD import AirfoilD


class ConceptualWing():
    """Defines a wing in the conceptual design context"""

    def __init__(
        self,
        airfoil: str,
        position: list[float],
        chord: float,
        span: float,
        area: float
    ) -> None:
        """Initializes the wing

        Args:
            airfoil (str): Airfoil Name e.g. NACA4415
            position (list[float]): Position of the wing
            chord (float): Chord length (MAC)
            span (float): Wing span
            area (float): Wing Area
        """
        if airfoil.startswith('naca') or airfoil.startswith('NACA'):
            self.airfoil: AirfoilD = AirfoilD.NACA(airfoil[4:], 200)
        else:
            print("Airfoil not found")
        self.position: list[float] = position
        self.span: float = span
        self.chord: float = chord
        self.area: float = area
        self.aspect_ratio: float = self._aspect_ratio(
                                span=self.span,
                                chord=self.chord,
                                wing_area=self.area,
                            )
        self.cp_pos = self.position[0] + self.chord * 0.25

    def _aspect_ratio(
        self,
        span: float,
        chord: float,
        wing_area: float
    ) -> float:
        """Defines the wing Aspect Ration based on the projected area.

        Args:
            span (float): Wing span
            chord (float): Chord Length
            wing_area (float): Wing area

        Returns:
            AR (float): AR
        """
        aspect_ratio: float = (
            (span**2 / wing_area) *
            (span / chord)
        )
        return aspect_ratio

    def add_polars(
        self,
        database: DB,
        solver: str = "Xfoil",
        verbose: bool = True
    ) -> None:
        """Connect Conceptual Wing Object to Icarus Database
        to retrieve airfoil polars

        Args:
            database (DB): Databse Object
        """
        foilsdb: Database_2D = database.foilsDB
        airfoil_name: str = f"NACA{self.airfoil.name}"
        self.foil_polars: Struct = foilsdb.data[airfoil_name]
        foil_reynolds: list[str] = list(foilsdb.data[airfoil_name].keys())
        max_index: intp = np.argmax([float(i) for i in foil_reynolds])
        max_reynolds: str = foil_reynolds[max_index]
        self.aoa = self.foil_polars[str(max_reynolds)][solver]["AoA"]
        self.cl_2d = self.foil_polars[str(max_reynolds)][solver]["CL"]
        self.cd_2d = self.foil_polars[str(max_reynolds)][solver]["CD"]
        self.cl_3d = self.lift_coefficient(
                        cl_2d=self.cl_2d,
                        aspect_ratio=self.aspect_ratio
                        )
        self.zero_lift_angle = np.interp(0, self.cl_3d, self.aoa)
        self.cl_0 = float(np.interp(0, self.aoa, self.cl_3d))
        self.cd_0 = float(np.interp(0, self.aoa, self.cd_2d))
        if verbose:
            print(f"{self.zero_lift_angle=}\n{self.cl_0=}\n{self.cd_0=}")

    def lift_coefficient(
        self,
        cl_2d: ndarray[Any, dtype[floating]],
        aspect_ratio: float
    ) -> ndarray[Any, dtype[floating]]:
        """_summary_

        Args:
            cl_2d (list[float]): Cl polar of the airfoil
            aspect_ratio (float): Aspect Ration of the wing
        """

        cl_3d = cl_2d / (
            1 +
            (2 / aspect_ratio)
            )

        return cl_3d

    def add_weight(self, weight: float) -> None:
        """Adds a weight to the wing"

        Args:
            weight (float): Weight of the wing
        """

        self.weight: float = weight
