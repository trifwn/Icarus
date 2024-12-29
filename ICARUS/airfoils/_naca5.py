import numpy as np

from ICARUS.airfoils._interpolate import interpolate
from ICARUS.core.types import FloatArray


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
