from typing import Any

import jax.numpy as jnp
import numpy as np

from ICARUS.airfoils import Airfoil
from ICARUS.core.types import FloatArray


class NACA4(Airfoil):
    """
    NACA 4 digit airfoil class
    """

    def __init__(self, M: float, P: float, XX: float, n_points: int = 200) -> None:
        """
        Initialize the NACA 4 digit airfoil. Example: NACA 2412

        Args:
            m (int): M is the maximum camber. In the example M=0.02 so the camber is 0.02 or 2% of the chord
            p (int): P is the position of maximum camber. In the example P=0.4 so the maximum camber is at 40% of the chord
            xx (int): XX is the maximum thickness. In the example XX=0.12 so the maximum thickness is 0.12 or 12% of the chord
            n_points (int): Number of points to generate the airfoil. Default is 200.
        """

        self.M: int = int(M * 100)
        self.P: int = int(P * 10)
        self.XX: int = int(XX * 100)

        self.m: float = M
        self.p: float = P
        self.xx: float = XX

        name = f"naca{self.M:01d}{self.P:01d}{self.XX:02d}"

        upper, lower = self.gen_NACA4_points(n_points // 2)
        super().__init__(upper=upper, lower=lower, name=name)

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
        M = int(name[0]) / 100.0
        P = int(name[1]) / 10.0
        XX = int(name[2:4]) / 100.0
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
        M = int(digits[0]) / 100.0
        P = int(digits[1]) / 10.0
        XX = int(digits[2:4]) / 100.0
        return cls(M, P, XX)

    def _camber_line(self, xsi):
        """
        Calculate the camber line and its derivative for a NACA 4 digit airfoil.
        Args:
            xsi (FloatArray): Non-dimensional x-coordinates (0 to 1)
        Returns:
            yc (FloatArray): Camber line y-coordinates
            dyc (FloatArray): Derivative of the camber line
        """
        p = self.p + 1e-19
        m = self.m

        # Camber line and its derivative using vectorized conditionals
        yc = jnp.where(
            xsi < p,
            (m / p**2) * (2 * p * xsi - xsi**2),
            (m / (1 - p) ** 2) * (1 - 2 * p + 2 * p * xsi - xsi**2),
        )
        dyc = jnp.where(
            xsi < p,
            (2 * m / p**2) * (p - xsi),
            (2 * m / (1 - p) ** 2) * (p - xsi),
        )
        return yc, dyc

    def thickness_distribution(self, xsi):
        xx = self.xx
        # Thickness distribution formula
        a0 = 0.2969
        a1 = -0.126
        a2 = -0.3516
        a3 = 0.2843
        a4 = -0.1036

        return (
            xx
            / 0.2
            * (a0 * jnp.sqrt(xsi) + a1 * xsi + a2 * xsi**2 + a3 * xsi**3 + a4 * xsi**4)
        )

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
        p: float = self.p
        m: float = self.m

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
        state["m"] = self.m
        state["p"] = self.p
        state["xx"] = self.xx
        state["n_points"] = len(self._x_lower) * 2
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the object for unpickling"""
        NACA4.__init__(
            self,
            M=state["m"],
            P=state["p"],
            XX=state["xx"],
            n_points=state["n_points"],
        )

    def gen_NACA4_points(self, n_points):
        """
        Generate upper and lower points for a NACA 4 airfoil using JAX.

        Args:
            m (float): maximum camber
            p (float): location of maximum camber
            xx (float): thickness
            n_points (int): number of points for discretization

        Returns:
            upper (2 x N array): x- and y-coordinates of the upper side
            lower (2 x N array): x- and y-coordinates of the lower side
        """

        beta = jnp.linspace(0, jnp.pi, n_points)
        xsi = 0.5 * (1 - jnp.cos(beta))  # cosine spacing

        yt = self.thickness_distribution(xsi)
        yc, dyc = self._camber_line(xsi)
        theta = jnp.arctan(dyc)

        x_upper = xsi - yt * jnp.sin(theta)
        y_upper = yc + yt * jnp.cos(theta)
        x_lower = xsi + yt * jnp.sin(theta)
        y_lower = yc - yt * jnp.cos(theta)

        upper = jnp.stack([x_upper, y_upper])
        lower = jnp.stack([x_lower, y_lower])

        return upper, lower
