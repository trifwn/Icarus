from typing import Any

import airfoils as af
import numpy as np

from ICARUS.airfoils import Airfoil
from ICARUS.core.types import FloatArray


class NACA4(Airfoil):
    """
    NACA 4 digit airfoil class
    """

    @classmethod
    def from_name(cls, name: str) -> "NACA4":
        """
        Create a NACA 4 digit airfoil from a name

        Args:
            name (str): Name of the airfoil. Example: "NACA2412"

        Returns:
            NACA4: NACA 4 digit airfoil object
        """
        if not isinstance(name, str):
            raise TypeError("Name must be a string")

        # Keep only the digits
        name = "".join(filter(str.isdigit, name))
        if len(name) != 4:
            raise ValueError("Name must be 4 digits")
        if not name.isdigit():
            raise ValueError("Name must be 4 digits")
        M = int(name[0])
        P = int(name[1])
        XX = int(name[2:4])
        if M < 0 or M > 9:
            raise ValueError("M must be between 0 and 9")
        if P < 0 or P > 9:
            raise ValueError("P must be between 0 and 9")
        if XX < 0 or XX > 99:
            raise ValueError("XX must be between 0 and 99")
        return cls(M, P, XX)

    @classmethod
    def from_digits(cls, digits: str) -> "NACA4":
        """
        Create a NACA 4 digit airfoil from a list of digits

        Args:
            digits ( str): List of digits. Example: "2412"

        Returns:
            NACA4: NACA 4 digit airfoil object
        """
        M = int(digits[0])
        P = int(digits[1])
        XX = int(digits[2:4])
        return cls(M, P, XX)

    def __init__(self, M: int, P: int, XX: int, n_points: int = 200) -> None:
        """
        Initialize the NACA 4 digit airfoil

        Args:
            m (int): M is the maximum camber divided by 100. In the example M=2 so the camber is 0.02 or 2% of the chord
            p (int): P is the position of maximum camber divided by 10. In the example P=4 so the maximum camber is at 40% of the chord
            xx (int): XX is the maximum thickness divided by 100. In the example XX=12 so the maximum thickness is 0.12 or 12% of the chord
            n_points (int): Number of points to generate the airfoil. Default is 200.
        """

        name = f"naca{M}{P}{XX:02d}"

        m = M / 100
        p = P / 10
        xx = XX / 100

        upper, lower = af.gen_NACA4_airfoil(m, p, xx, n_points // 2)
        super().__init__(upper=upper, lower=lower, name=name)

        self.M: float = M
        self.P: float = P
        self.XX: float = XX

    def camber_line(
        self,
        points: float | FloatArray | list[float],
    ) -> FloatArray:
        """Function to generate the camber line for a NACA 4 digit airfoil.
        Returns the camber line for a given set of x coordinates.

        Args:
            points (FloatArray): X coordinates for which we need the camber line

        Returns:
            FloatArray: X,Y coordinates of the camber line

        """
        p: float = self.P / 10
        m: float = self.M / 100

        if isinstance(points, float):
            x: float = float(points)
            if x < p:
                result: float = m / p**2 * (2 * p * x - x**2)
            else:
                result = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2)
            return np.array(result)
        if isinstance(points, list):
            points = np.array(points, dtype=float)
        if isinstance(points, int):
            points = np.array(float(points))

        results: FloatArray = np.zeros_like(points)
        for i, xi in enumerate(points.tolist()):
            if xi < p:
                results[i] = m / p**2 * (2 * p * xi - xi**2)
            else:
                results[i] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * xi - xi**2)
        return results

    def __getstate__(self) -> dict[str, Any]:
        """Get the state of the object for pickling"""
        state = dict()
        state["m"] = self.M
        state["p"] = self.P
        state["xx"] = self.XX
        state["n_points"] = len(self._x_lower) * 2
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the object for unpickling"""
        NACA4.__init__(self, M=state["m"], P=state["p"], XX=state["xx"], n_points=state["n_points"])
