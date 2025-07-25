from typing import Any
from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jaxtyping import Int

from ICARUS.airfoils import Airfoil
from ICARUS.core.types import FloatArray
from ICARUS.core.types import JaxArray
from ICARUS.airfoils.core.operations import JaxAirfoilOps


@jax.tree_util.register_pytree_node_class
class NACA5(Airfoil):
    """
    NACA 5 digit airfoil class with JAX compatibility
    """

    def __init__(self, L: int, P: int, Q: int, XX: int, n_points: int = 200) -> None:
        """
        Initialize the NACA 5 digit airfoil

        Args:
            L (int): This digit controls the camber. It indicates the designed coefficient of lift (Cl) multiplied by 3/20. In the example L=2 so Cl=0.3
            P (int): The position of maximum camber divided by 20. In the example P=3 so maximum camber is at 0.15 or 15% chord
            Q (int): 0 = normal camber line, 1 = reflex camber line
            XX (int): Maximum thickness divided by 100. In the example XX=12 so the maximum thickness is 0.12 or 12% of the chord
            n_points (int): Number of points to generate
        """
        assert 0 <= L <= 9, "L must be between 0 and 9"
        assert 0 <= P <= 9, "P must be between 0 and 9"
        assert Q in [0, 1], "Q must be 0 or 1"
        assert 0 <= XX <= 99, "XX must be between 0 and 99"

        # Store integer versions for naming (avoid during vmap)
        try:
            self.L: Int = jnp.asarray(L, dtype=int)
            self.P: Int = jnp.asarray(P, dtype=int)
            self.Q: Int = jnp.asarray(Q, dtype=int)
            self.XX: Int = jnp.asarray(XX, dtype=int)
        except (TypeError, ValueError):
            # During vmap, just use placeholder values
            self.L: Int = jnp.asarray(2, dtype=int)
            self.P: Int = jnp.asarray(3, dtype=int)
            self.Q: Int = jnp.asarray(0, dtype=int)
            self.XX: Int = jnp.asarray(12, dtype=int)

        # Store float versions for calculations
        self.l: Float = jnp.asarray(L, dtype=float)
        self.p: Float = jnp.asarray(P, dtype=float)
        self.q: Float = jnp.asarray(Q, dtype=float)
        self.xx: Float = jnp.asarray(XX, dtype=float)

        # Generate coordinates using JAX operations
        upper, lower = self.gen_NACA5_points(n_points // 2)
        naca = f"{L}{P}{Q}{XX:02d}"
        super().__init__(upper=upper, lower=lower, name=f"naca{naca}")

    @property
    def name(self) -> str:
        """
        Name of the airfoil in the format NACAXXXXX
        """
        if not hasattr(self, "_name") or self._name is None:
            self._name = f"naca{self.L:01d}{self.P:01d}{self.Q:01d}{self.XX:02d}"
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Sets the name of the airfoil"""
        if not isinstance(value, str):
            raise TypeError("Name must be a string")
        self._name = value.replace(" ", "")

    @classmethod
    def from_name(cls, name: str) -> "NACA5":
        """
        Create a NACA 5 digit airfoil from a name

        Args:
            name (str): Name of the airfoil. Example: "NACA23012"

        Returns:
            NACA5: NACA 5 digit airfoil object
        """
        if not isinstance(name, str):
            raise TypeError("Name must be a string")

        # Keep only the digits
        name = "".join(filter(str.isdigit, name))
        return cls.from_digits(name)

    @classmethod
    def from_digits(cls, digits: str) -> "NACA5":
        """
        Create a NACA 5 digit airfoil from a list of digits

        Args:
            digits (str): List of digits. Example: "23012"

        Returns:
            NACA5: NACA 5 digit airfoil object
        """
        if len(digits) != 5:
            raise ValueError("Digits must be 5 characters long")
        if not digits.isdigit():
            raise ValueError("Digits must be numeric")

        L = int(digits[0])
        P = int(digits[1])
        Q = int(digits[2])
        XX = int(digits[3:5])

        if not (0 <= L <= 9):
            raise ValueError("L must be between 0 and 9")
        if not (0 <= P <= 9):
            raise ValueError("P must be between 0 and 9")
        if Q not in [0, 1]:
            raise ValueError("Q must be 0 or 1")
        if not (1 <= XX <= 99):
            raise ValueError("XX must be between 01 and 99")

        return cls(L, P, Q, XX)

    def gen_NACA5_points(self, n_points: int) -> tuple[Float, Float]:
        """
        Generate upper and lower points for a NACA 5 airfoil using JAX.

        Args:
            n_points (int): number of points for discretization

        Returns:
            upper (2 x N array): x- and y-coordinates of the upper side
            lower (2 x N array): x- and y-coordinates of the lower side
        """
        # Generate cosine-spaced x coordinates
        beta = jnp.linspace(0, jnp.pi, n_points)
        x = 0.5 * (1 - jnp.cos(beta))  # cosine spacing

        # Compute thickness distribution (same as NACA 4-digit)
        thickness = self.xx / 100.0
        yt = self.thickness_distribution(x, thickness)

        # Compute camber line
        design_cl = self.l * (3.0 / 2.0) / 10.0  # L * 3/20
        max_camber_pos = 0.5 * self.p / 10.0  # P/20

        # Compute both standard and reflex camber lines
        yc_standard, dyc_dx_standard = self.naca5_camber_line_standard(
            x, design_cl, max_camber_pos
        )
        yc_reflex, dyc_dx_reflex = self.naca5_camber_line_reflex(
            x, design_cl, max_camber_pos
        )

        # Use jnp.where for JAX compatibility
        yc = jnp.where(self.q == 1, yc_reflex, yc_standard)
        dyc_dx = jnp.where(self.q == 1, dyc_dx_reflex, dyc_dx_standard)

        # Compute angle of camber line
        theta = jnp.arctan(dyc_dx)

        # Compute upper and lower surface coordinates
        x_upper = x - yt * jnp.sin(theta)
        y_upper = yc + yt * jnp.cos(theta)
        x_lower = x + yt * jnp.sin(theta)
        y_lower = yc - yt * jnp.cos(theta)

        # Stack into coordinate arrays
        # Note: For proper selig format, upper surface should run from TE to LE
        # and lower surface should run from LE to TE
        upper_coords = jnp.stack(
            [x_upper[::-1], y_upper[::-1]]
        )  # Reverse upper surface
        lower_coords = jnp.stack([x_lower, y_lower])

        return upper_coords, lower_coords

    def thickness_distribution(self, x: Float, thickness: float) -> Float:
        """
        Compute NACA thickness distribution.

        Args:
            x: X coordinates (normalized to [0, 1])
            thickness: Maximum thickness as fraction of chord

        Returns:
            Thickness values at x coordinates
        """
        # NACA thickness distribution coefficients
        a0 = 0.2969
        a1 = -0.1260
        a2 = -0.3516
        a3 = 0.2843
        a4 = -0.1036  # For zero thickness trailing edge

        return (thickness / 0.2) * (
            a0 * jnp.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4
        )

    def naca5_camber_line_standard(
        self, x: Float, design_cl: float, max_camber_pos: float
    ) -> tuple[Float, Float]:
        """
        Compute NACA 5-digit standard camber line and its derivative.

        Args:
            x: X coordinates (normalized to [0, 1])
            design_cl: Design coefficient of lift (L * 3/20)
            max_camber_pos: Position of maximum camber (P/20)

        Returns:
            Tuple of (camber_line_y, camber_line_derivative)
        """
        # Standard NACA 5-digit parameters
        # These are lookup table values - simplified for JAX compatibility
        P_values = jnp.array([0.05, 0.10, 0.15, 0.20, 0.25])
        M_values = jnp.array([0.0580, 0.1260, 0.2025, 0.2900, 0.3910])
        K_values = jnp.array([361.4, 51.64, 15.957, 6.643, 3.230])

        # Interpolate to find m and k1 for the given position
        m = jnp.interp(max_camber_pos, P_values, M_values)
        k1 = jnp.interp(m, M_values, K_values)

        # Compute r (position parameter) - simplified calculation
        r = max_camber_pos + m * jnp.sqrt(m / 3)

        # Camber line calculation
        yc = jnp.where(
            x <= r,
            (design_cl / 0.3) * (k1 / 6) * (x**3 - 3 * r * x**2 + r**2 * (3 - r) * x),
            (design_cl / 0.3) * (k1 / 6) * r**3 * (1 - x),
        )

        # Camber line derivative
        dyc_dx = jnp.where(
            x <= r,
            (design_cl / 0.3) * (k1 / 6) * (3 * x**2 - 6 * r * x + r**2 * (3 - r)),
            -(design_cl / 0.3) * (k1 / 6) * r**3,
        )

        return yc, dyc_dx

    def naca5_camber_line_reflex(
        self, x: Float, design_cl: float, max_camber_pos: float
    ) -> tuple[Float, Float]:
        """
        Compute NACA 5-digit reflex camber line and its derivative.

        Args:
            x: X coordinates (normalized to [0, 1])
            design_cl: Design coefficient of lift (L * 3/20)
            max_camber_pos: Position of maximum camber (P/20)

        Returns:
            Tuple of (camber_line_y, camber_line_derivative)
        """
        # Reflex camber line parameters (simplified for JAX compatibility)
        r = 0.2170  # Fixed r value for reflex series
        k1 = 15.793  # Fixed k1 value for reflex series

        # Calculate k21 parameter
        k21 = (3 * (r - max_camber_pos) ** 2 - r**3) / (1 - r) ** 3

        # Camber line calculation for reflex
        yc = jnp.where(
            x < r,
            (design_cl / 0.3)
            * (k1 / 6)
            * ((x - r) ** 3 - k21 * (1 - r) ** 3 * x - r**3 * x + r**3),
            (design_cl / 0.3)
            * (k1 / 6)
            * (k21 * (x - r) ** 3 - k21 * (1 - r) ** 3 * x - r**3 * x + r**3),
        )

        # Camber line derivative for reflex
        dyc_dx = jnp.where(
            x < r,
            (design_cl / 0.3)
            * (k1 / 6)
            * (3 * (x - r) ** 2 - k21 * (1 - r) ** 3 - r**3),
            (design_cl / 0.3)
            * (k1 / 6)
            * (3 * k21 * (x - r) ** 2 - k21 * (1 - r) ** 3 - r**3),
        )

        return yc, dyc_dx

    def camber_line(self, points: Float) -> Float:
        """Compute camber line at given points."""
        design_cl = self.l * (3.0 / 2.0) / 10.0  # L * 3/20
        max_camber_pos = 0.5 * self.p / 10.0  # P/20

        x = jnp.asarray(points, dtype=float)

        # Compute both standard and reflex camber lines
        yc_standard, _ = self.naca5_camber_line_standard(x, design_cl, max_camber_pos)
        yc_reflex, _ = self.naca5_camber_line_reflex(x, design_cl, max_camber_pos)

        # Use jnp.where for JAX compatibility
        yc = jnp.where(self.q == 1, yc_reflex, yc_standard)

        return yc

    def camber_line_derivative(self, points: Float) -> Float:
        """Compute camber line derivative at given points."""
        design_cl = self.l * (3.0 / 2.0) / 10.0  # L * 3/20
        max_camber_pos = 0.5 * self.p / 10.0  # P/20

        x = jnp.asarray(points, dtype=float)

        # Compute both standard and reflex camber line derivatives
        _, dyc_dx_standard = self.naca5_camber_line_standard(
            x, design_cl, max_camber_pos
        )
        _, dyc_dx_reflex = self.naca5_camber_line_reflex(x, design_cl, max_camber_pos)

        # Use jnp.where for JAX compatibility
        dyc_dx = jnp.where(self.q == 1, dyc_dx_reflex, dyc_dx_standard)

        return dyc_dx

    def y_upper(self, ksi: Float) -> Float:
        """Upper surface y-coordinates."""
        theta = jnp.arctan(self.camber_line_derivative(ksi))
        camber = self.camber_line(ksi)
        thickness = self.xx / 100.0
        yt = self.thickness_distribution(ksi, thickness)
        return camber + yt * jnp.cos(theta)

    def y_lower(self, ksi: Float) -> Float:
        """Lower surface y-coordinates."""
        theta = jnp.arctan(self.camber_line_derivative(ksi))
        camber = self.camber_line(ksi)
        thickness = self.xx / 100.0
        yt = self.thickness_distribution(ksi, thickness)
        return camber - yt * jnp.cos(theta)

    def thickness(self, ksi: Float) -> Float:
        """Thickness distribution at given points."""
        thickness = self.xx / 100.0
        return self.thickness_distribution(ksi, thickness)

    def __getstate__(self) -> dict[str, Any]:
        """Get the state of the object for pickling"""
        state = dict()
        state["L"] = int(self.l)
        state["P"] = int(self.p)
        state["Q"] = int(self.q)
        state["XX"] = int(self.xx)
        state["n_points"] = len(self._x_lower) * 2
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the object for unpickling"""
        NACA5.__init__(
            self,
            L=state["L"],
            P=state["P"],
            Q=state["Q"],
            XX=state["XX"],
            n_points=state["n_points"],
        )

    def tree_flatten(self):
        """Enable JAX transformations by flattening the pytree structure."""
        # Return the original float values for JAX operations
        L = jnp.asarray(self.l, dtype=jnp.float64)
        P = jnp.asarray(self.p, dtype=jnp.float64)
        Q = jnp.asarray(self.q, dtype=jnp.float64)
        XX = jnp.asarray(self.xx, dtype=jnp.float64)
        # Use Python int for static arguments to avoid JAX array issues
        num_points = int(self.n_points)
        return ((L, P, Q, XX), (num_points,))

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        """Reconstruct the NACA5 object from flattened pytree structure."""

        # Convert JAX arrays back to Python ints for constructor
        def safe_int_convert(val):
            if hasattr(val, "item"):
                return int(val.item())
            elif hasattr(val, "__float__"):
                return int(float(val))
            else:
                return int(val)

        L_val = safe_int_convert(children[0])
        P_val = safe_int_convert(children[1])
        Q_val = safe_int_convert(children[2])
        XX_val = safe_int_convert(children[3])

        return cls(
            L=L_val,
            P=P_val,
            Q=Q_val,
            XX=XX_val,
            n_points=aux_data[0],
        )
