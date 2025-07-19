from typing import Any
from typing import Self

import jax
from jax import numpy as jnp
from jaxtyping import Float
from jaxtyping import Int

from ICARUS.airfoils import Airfoil
from ICARUS.core.types import FloatArray
from ICARUS.core.types import JaxArray


@jax.tree_util.register_pytree_node_class
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
        self.M: Int = jnp.array(M * 100).astype(int)
        self.P: Int = jnp.array(P * 10).astype(int)
        self.XX: Int = jnp.array(XX * 100).astype(int)

        self.m: Float = jnp.asarray(M, dtype=float)
        self.p: Float = jnp.asarray(P, dtype=float)
        self.xx: Float = jnp.asarray(XX, dtype=float)

        upper, lower = self.gen_NACA4_points(n_points // 2)
        super().__init__(upper=upper, lower=lower)

    @property
    def name(self) -> str:
        """
        Name of the airfoil in the format NACAXXXX
        """
        if not hasattr(self, "_name") or self._name is None:
            self._name = f"NACA{self.M:01d}{self.P:01d}{self.XX:02d}"
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Sets the name of the airfoil"""
        if not isinstance(value, str):
            raise TypeError("Name must be a string")
        self._name = value.replace(" ", "")

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
        return cls.from_digits(name)

    @classmethod
    def from_digits(cls, digits: str) -> "NACA4":
        """
        Create a NACA 4 digit airfoil from a list of digits

        Args:
            digits ( str): List of digits. Example: "2412"

        Returns:
            NACA4: NACA 4 digit airfoil object
        """
        if len(digits) != 4:
            raise ValueError("Digits must be 4 characters long")
        if not digits.isdigit():
            raise ValueError("Digits must be numeric")
        M = int(digits[0]) / 100.0
        P = int(digits[1]) / 10.0
        XX = int(digits[2:4]) / 100.0

        if M < 0 or M > 9:
            raise ValueError("M must be between 0 and 9")
        if P < 0 or P > 9:
            raise ValueError("P must be between 0 and 9")
        if XX < 0 or XX > 99:
            raise ValueError("XX must be between 0 and 99")
        return cls(M, P, XX)

    def camber_line(self, points: Float) -> Float:
        p = self.p + 1e-19  # Avoid division by zero
        m = self.m
        c = 1.0

        xsi = jnp.asarray(points, dtype=float) / c  # Normalize points to [0, 1]

        # Camber line and its derivative using vectorized conditionals
        yc = jnp.select(
            [xsi <= 0, xsi < p],
            [
                0.0,  # xsi <= 0
                (m / p**2) * (2 * p * xsi - xsi**2),  # xsi < p
            ],
            default=(m / (1 - p) ** 2) * (1 - 2 * p + 2 * p * xsi - xsi**2),  # xsi >= p
        )
        return yc

    def camber_line_derivative(
        self,
        points: float | list[float] | FloatArray | JaxArray,
    ) -> JaxArray:
        p = self.p + 1e-19  # Avoid division by zero
        m = self.m
        c = 1.0  # Chord length is normalized to 1

        xsi = jnp.asarray(points, dtype=float) / c  # Normalize points to [0, 1]
        dyc = jnp.select(
            [xsi <= 0, xsi < p],
            [0.0, (2 * m / p**2) * (p - xsi)],
            default=(2 * m / (1 - p) ** 2) * (p - xsi),
        )
        return dyc

    def y_upper(self, ksi: Float) -> Float:
        # x-coordinate is between [0, 1]
        # x must be set between [min(x_upper), max(x_upper)]
        theta = jnp.arctan(self.camber_line_derivative(ksi))
        camber = self.camber_line(ksi)
        yt = self.thickness_distribution(ksi)
        return camber + yt * jnp.cos(theta)

    def y_lower(self, ksi: Float) -> Float:
        # x-coordinate is between [0, 1]
        # x must be set between [min(x_lower), max(x_lower)]
        theta = jnp.arctan(self.camber_line_derivative(ksi))
        yt = self.thickness_distribution(ksi)
        return self.camber_line(ksi) - yt * jnp.cos(theta)

    def thickness_distribution(self, xsi: Float) -> Float:
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

    def gen_NACA4_points(self, n_points: int) -> tuple[Float, Float]:
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
        yc = self.camber_line(xsi)
        dyc = self.camber_line_derivative(xsi)
        theta = jnp.arctan(dyc)

        x_upper = xsi - yt * jnp.sin(theta)
        y_upper = yc + yt * jnp.cos(theta)
        x_lower = xsi + yt * jnp.sin(theta)
        y_lower = yc - yt * jnp.cos(theta)

        upper = jnp.stack([x_upper, y_upper])
        lower = jnp.stack([x_lower, y_lower])
        return upper, lower

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

    def tree_flatten(self):
        M = jnp.asarray(self.M, dtype=jnp.int64).astype("float")
        P = jnp.asarray(self.P, dtype=jnp.int64).astype("float")
        XX = jnp.asarray(self.XX, dtype=jnp.int64).astype("float")
        num_points = jnp.asarray(self.n_points, dtype=jnp.int64)
        return ((M, P, XX), (num_points,))

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        return cls(
            M=children[0],
            P=children[1],
            XX=children[2],
            n_points=aux_data[0],
        )
