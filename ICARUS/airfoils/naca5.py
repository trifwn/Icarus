from typing import Any

import numpy as np

from ICARUS.airfoils import Airfoil
from ICARUS.airfoils._interpolate import interpolate
from ICARUS.core.types import FloatArray


class NACA5(Airfoil):
    """
    NACA 5 digit airfoil class
    """

    def __init__(self, L: int, P: int, Q: int, XX: int, n_points: int = 200) -> None:
        """
        Initialize the NACA 4 digit airfoil

        Args:
            l (int): This digit controls the camber. It indicates the designed coefficient of lift (Cl) multiplied by 3/20. In the examble L=2 so Cl=0.3
            p (int): The position of maximum camber divided by 20. In the examble P=3 so maximum camber is at 0.15 or 15% chord
            q (int): 0 = normal camber line, 1 = reflex camber line
            number (str): NACA 5 digit identifier
            xx (int): Maximum thickness divided by 100. In the example XX=12 so the maximum thickness is 0.12 or 12% of the chord
            n_points (int): Number of points to generate
        """
        assert 0 <= L <= 9, "L must be between 0 and 9"
        assert 0 <= P <= 9, "P must be between 0 and 9"
        assert Q in [0, 1], "Q must be 0 or 1"
        assert 0 <= XX <= 99, "XX must be between 0 and 99"

        naca = f"{L}{P}{Q}{XX}"
        upper, lower = gen_NACA5_airfoil(naca, n_points)
        super().__init__(upper=upper, lower=lower, name=f"naca{naca}")

        self.l: float = L
        self.p: float = P
        self.q: float = Q
        self.xx: float = XX

    def __getstate__(self) -> dict[str, Any]:
        """Get the state of the object for pickling"""
        state = dict()
        state["L"] = self.l
        state["P"] = self.p
        state["Q"] = self.q
        state["XX"] = self.xx
        state["n_points"] = len(self._x_lower) * 2
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the object for unpickling"""
        NACA5.__init__(self, L=state["L"], P=state["P"], Q=state["Q"], XX=state["XX"], n_points=state["n_points"])


def gen_NACA5_airfoil(
    number: str,
    n_points: int,
    finite_TE: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Generates a NACA 5 digit airfoil

    Args:
        number (str): NACA 5 digit identifier
        n_points (int): Number of points to generate
        finite_TE (bool, optional): Wheter to have a finite TE. Defaults to False.

    Returns:
        tuple[FloatArray, FloatArray]: Upper and lower surface coordinates

    """
    naca1 = int(number[0])
    naca23 = int(number[1:3])
    naca45 = int(number[3:])

    cld: float = naca1 * (3.0 / 2.0) / 10.0
    p: float = 0.5 * naca23 / 100.0
    t: float = naca45 / 100.0

    a0: float = +0.2969
    a1: float = -0.1260
    a2: float = -0.3516
    a3: float = +0.2843

    if finite_TE:
        a4: float = -0.1015  # For finite thickness trailing edge
    else:
        a4 = -0.1036  # For zero thickness trailing edge

    x = np.linspace(0.0, 1.0, n_points + 1)

    yt: list[float] = [
        5 * t * (a0 * np.sqrt(xx) + a1 * xx + a2 * pow(xx, 2) + a3 * pow(xx, 3) + a4 * pow(xx, 4)) for xx in x
    ]

    P: list[float] = [0.05, 0.1, 0.15, 0.2, 0.25]
    M: list[float] = [0.0580, 0.1260, 0.2025, 0.2900, 0.3910]
    K: list[float] = [361.4, 51.64, 15.957, 6.643, 3.230]

    m = interpolate(P, M, [p])[0]
    k1 = interpolate(M, K, [m])[0]

    xc1: list[float] = [xx for xx in x if xx <= p]
    xc2: list[float] = [xx for xx in x if xx > p]
    xc: list[float] = xc1 + xc2

    if p == 0:
        xu: list[float] | FloatArray = x
        yu: list[float] | FloatArray = yt

        xl: list[float] | FloatArray = x
        yl: list[float] | FloatArray = [-x for x in yt]

        zc = [0] * len(xc)
    else:
        yc1 = [k1 / 6.0 * (pow(xx, 3) - 3 * m * pow(xx, 2) + pow(m, 2) * (3 - m) * xx) for xx in xc1]
        yc2 = [k1 / 6.0 * pow(m, 3) * (1 - xx) for xx in xc2]
        zc = [cld / 0.3 * xx for xx in yc1 + yc2]

        dyc1_dx: list[float] = [
            cld / 0.3 * (1.0 / 6.0) * k1 * (3 * pow(xx, 2) - 6 * m * xx + pow(m, 2) * (3 - m)) for xx in xc1
        ]
        dyc2_dx: list[float] = [cld / 0.3 * -(1.0 / 6.0) * k1 * pow(m, 3)] * len(xc2)

        dyc_dx: list[float] = dyc1_dx + dyc2_dx
        theta: list[float] = [np.arctan(xx) for xx in dyc_dx]

        xu = [xx - yy * np.sin(zz) for xx, yy, zz in zip(x, yt, theta)]
        yu = [xx + yy * np.cos(zz) for xx, yy, zz in zip(zc, yt, theta)]

        xl = [xx + yy * np.sin(zz) for xx, yy, zz in zip(x, yt, theta)]
        yl = [xx - yy * np.cos(zz) for xx, yy, zz in zip(zc, yt, theta)]

    upper: FloatArray = np.array([xu, yu])
    lower: FloatArray = np.array([xl, yl])
    return upper, lower


# def camber_line(x, c, l, p, q, r_reflexed=None, k1_reflexed=None):
#     """
#     Compute the camber line (yc) and its derivative (dyc) for NACA 5-digit airfoils.

#     Parameters:
#         x: ndarray
#             Array of x-coordinates along the chord.
#         c: float
#             Chord length.
#         l: int
#             Camber parameter (first digit of NACA code).
#         p: int
#             Position of max camber parameter (second digit).
#         q: int
#             0 for standard camber, 1 for reflexed camber.
#         r_reflexed: float or None
#             Optional fixed r value for reflexed series.
#         k1_reflexed: float or None
#             Optional fixed k1 value for reflexed series.

#     Returns:
#         yc: ndarray
#             Camber line values at each x.
#         dyc: ndarray
#             Derivative of camber line at each x.
#     """
#     x = np.asarray(x)
#     n = x.size
#     yc  = np.zeros(n)
#     dyc = np.zeros(n)

#     x_norm = x / c
#     xcm = 0.1 * (p / 2)  # max camber position in chord fractions

#     if q == 0:  # standard series
#         # Fixed point iteration to solve for r
#         r_old = 0.1
#         tol = 1e-4
#         diff = 1
#         while diff > tol:
#             r = xcm + r_old * mt.sqrt(r_old / 3)
#             diff = abs(r - r_old)
#             r_old = r

#         # Compute k1
#         qm = ((3*r - 7*r**2 + 8*r**3 - 4*r**4) / mt.sqrt(r - r**2)
#               - 1.5*(1 - 2*r)*(mt.pi/2 - mt.asin(1 - 2*r)))
#         k1 = (6 * (0.3 * l / 2)) / qm

#         for i in range(n):
#             if x_norm[i] < r:
#                 yc[i]  = (c * k1 / 6) * (x_norm[i]**3 - 3*r*x_norm[i]**2 + r**2*(3 - r)*x_norm[i])
#                 dyc[i] = (k1 / 6) * (3*x_norm[i]**2 - 6*r*x_norm[i] + r**2*(3 - r))
#             else:
#                 yc[i]  = (c * k1 / 6) * r**3 * (1 - x_norm[i])
#                 dyc[i] = -(k1 / 6) * r**3

#     else:  # reflexed camber
#         r = r_reflexed if r_reflexed is not None else 0.2170
#         k1 = k1_reflexed if k1_reflexed is not None else 15.793
#         k21 = (3*(r - xcm)**2 - r**3) / (1 - r)**3

#         for i in range(n):
#             if x_norm[i] < r:
#                 yc[i] = (c*k1/6) * (((x_norm[i] - r)**3)
#                                     - k21*(1 - r)**3*x_norm[i]
#                                     - r**3*x_norm[i] + r**3)
#                 dyc[i] = (k1/6) * (3*(x_norm[i] - r)**2
#                                    - k21*(1 - r)**3 - r**3)
#             else:
#                 yc[i] = (c*k1/6) * (k21*(x_norm[i] - r)**3
#                                     - k21*(1 - r)**3*x_norm[i]
#                                     - r**3*x_norm[i] + r**3)
#                 dyc[i] = (k1/6) * (3*k21*(x_norm[i] - r)**2
#                                    - k21*(1 - r)**3 - r**3)

#     return yc, dyc
