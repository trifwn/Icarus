import os
import re
import urllib.request
from typing import Any

import airfoils as af
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import dtype
from numpy import floating
from numpy import ndarray

from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray

# # Airfoil
# ##### 0 = Read from python module
# ##### 1 = Read from airfoiltools.com
# ##### 2 = load from file


class Airfoil(af.Airfoil):  # type: ignore
    """
    Class to represent an airfoil. Inherits from airfoil class from the airfoils module.
    Stores the airfoil data in the selig format.
    # ! TODO DEPRECATE THE DEPENDANCE ON THE airfoil MODULE. SPECIFICALLY WE HAVE
    # ! TO CHANGE THE MODULE SO THAT IT DOESNT NORMALIZE THE airfoil

    Args:
        af : Airfoil class from the airfoils module
    """

    def __init__(
        self,
        upper: FloatArray,
        lower: FloatArray,
        name: str,
        n_points: int,
    ) -> None:
        """
        Initialize the Airfoil class

        Args:
            upper (FloatArray): Upper surface coordinates
            lower (FloatArray): Lower surface coordinates
            naca (str): NACA 4 digit identifier (e.g. 0012)
            n_points (int): Number of points to be used to generate the airfoil. It interpolates between upper and lower
        """
        super().__init__(upper, lower)
        name = name.replace(" ", "")
        name = name.replace(".dat", "")
        name = name.replace("naca", "")
        if name.lower().startswith("naca"):
            self.file_name: str = name.upper()
        else:
            self.file_name = f"naca{name}"
        # self.file_name: str = f"{name}"
        name = name.replace("NACA", "")
        self.name: str = name

        self.n_points: int = n_points
        self.polars: dict[str, Any] | Struct = {}

        self.airfoil_to_selig()
        self.fix_le()

        # For Type Checking
        self._x_upper: FloatArray = self._x_upper
        self._y_upper: FloatArray = self._y_upper
        self._x_lower: FloatArray = self._x_lower
        self._y_lower: FloatArray = self._y_lower
        # self.getFromWeb()

    @classmethod
    def naca(cls, naca: str, n_points: int = 200) -> "Airfoil":
        """
        Initialize the Airfoil class from a NACA 4 digit identifier.

        Args:
            naca (str): NACA 4 digit identifier (e.g. 0012) can also take NACA0012
            n_points (int, optional): Number of points to generate. Defaults to 200.

        Raises:
            af.NACADefintionError: If the NACA identifier is not valid

        Returns:
            Airfoil: airfoil class object
        """
        re_4digits: re.Pattern[str] = re.compile(r"^\d{4}$")
        re_5digits: re.Pattern[str] = re.compile(r"^\d{5}$")

        if re_5digits.match(naca):
            l: float = float(naca[0]) / 10
            p: float = float(naca[1]) / 100
            q: float = float(naca[2]) / 1000
            xx: float = float(naca[3:5]) / 1000
            upper, lower = gen_NACA5_airfoil(naca, n_points)
            self: "Airfoil" = cls(upper, lower, naca, n_points)
            return self
        elif re_4digits.match(naca):
            p = float(naca[0]) / 10
            m: float = float(naca[1]) / 100
            xx = float(naca[2:4]) / 100
            upper, lower = af.gen_NACA4_airfoil(p, m, xx, n_points)
            self = cls(upper, lower, naca, n_points)
            self.set_naca4_digits(p, m, xx)
            return self
        else:
            raise af.NACADefintionError(
                "Identifier not recognised as valid NACA 4 definition",
            )

    @classmethod
    def load_from_file(cls, filename: str) -> "Airfoil":
        """
        Initialize the Airfoil class from a file.

        Args:
            filename (str): Name of the file to load the airfoil from

        Returns:
            Airfoil: Airfoil class object
        """
        x: list[float] = []
        y: list[float] = []

        # np.loadtxt(filename, skiprows=1, unpack=True)
        # pd.read_csv(
        #     filename,
        #     skiprows=1,
        #     sep=" ",
        #     on_bad_lines="skip",
        # )
        with open(filename) as file:
            for line in file:
                if line.startswith("NACA"):
                    continue
                if line == "\n":
                    continue
                try:
                    x.append(float(line.split()[0]))
                    y.append(float(line.split()[1]))
                except ValueError:
                    continue
        # find index of x = 0
        idx: int = x.index(0)

        upper: FloatArray = np.array([x[:idx], y[:idx]])
        lower: FloatArray = np.array([x[idx:], y[idx:]])
        self: "Airfoil" = cls(upper, lower, os.path.split(filename)[-1], len(x))
        return self

    def fix_le(self) -> None:
        if self._x_upper[0] < 0:
            # remove the first point
            self._x_upper = self._x_upper[1:]
            self._y_upper = self._y_upper[1:]
            self.fix_le()
        elif self._x_lower[0] > 0:
            # remove the first point
            self._x_lower = self._x_lower[1:]
            self._y_lower = self._y_lower[1:]
            self.fix_le()
        elif self._x_upper[0] != 0:
            # add a point at 0
            self._x_upper = np.hstack((0, self._x_upper))
            self._y_upper = np.hstack((0, self._y_upper))
            self.fix_le()
        elif self._x_lower[0] != 0:
            # add a point at 0
            self._x_lower = np.hstack((0, self._x_lower))
            self._y_lower = np.hstack((0, self._y_lower))
            self.fix_le()
        return None

    def flap_airfoil(
        self,
        flap_hinge: float,
        chord_extension: float,
        flap_angle: float,
        plotting: bool = False,
    ) -> "Airfoil":
        """
        Function to generate a flapped airfoil. The flap is defined by the flap hinge, the chord extension and the flap angle.

        Args:
            flap_hinge (float): Chordwise location of the flap hinge
            chord_extension (float): Chord extension of the flap
            flap_angle (float): Angle of the flap
            plotting (bool, optional): Whether to plot the new and old airfoil. Defaults to False.

        Returns:
            Airfoil: Flapped airfoil
        """
        # Find the index of the flap hinge
        idx_lower: int = int(np.argmin(np.abs(self._x_lower - flap_hinge)))
        idx_upper: int = int(np.argmin(np.abs(self._x_upper - flap_hinge)))
        # Rotate all the points to the right of the hinge
        # and then stretch them in the direction of the flap angle

        x_lower: FloatArray = self._x_lower[idx_lower:]
        x_upper: FloatArray = self._x_upper[idx_upper:]
        x_lower = x_lower - flap_hinge
        x_upper = x_upper - flap_hinge

        y_lower: FloatArray = self._y_lower[idx_lower:]
        temp = y_lower[0].copy()
        y_upper: FloatArray = self._y_upper[idx_upper:]
        y_lower = y_lower - temp
        y_upper = y_upper - temp

        # Rotate the points according to the hinge (located on the lower side)
        theta: float = -np.deg2rad(flap_angle)
        x_lower = x_lower * np.cos(theta) - y_lower * np.sin(theta)
        x_upper = x_upper * np.cos(theta) - y_upper * np.sin(theta)
        y_lower = x_lower * np.sin(theta) + y_lower * np.cos(theta)
        y_upper = x_upper * np.sin(theta) + y_upper * np.cos(theta)

        # Stretch the points so all points move the same amount
        x_lower = x_lower * (1 + chord_extension)
        x_upper = x_upper * (1 + chord_extension)

        # Translate the points back
        x_lower = x_lower + flap_hinge
        x_upper = x_upper + flap_hinge
        y_lower = y_lower + temp
        y_upper = y_upper + temp

        if plotting:
            self.plot()

        upper: FloatArray = np.array(
            [[*self._x_upper[:idx_upper], *x_upper], [*self._y_upper[:idx_upper], *y_upper]],
        )
        lower: FloatArray = np.array(
            [[*self._x_lower[:idx_lower], *x_lower], [*self._y_lower[:idx_lower], *y_lower]],
        )
        flapped = Airfoil(upper, lower, f"{self.name}_flapped", n_points=self.n_points)

        if plotting:
            flapped.plot()
            plt.show()
        return flapped

    def set_naca4_digits(self, p: float, m: float, xx: float) -> None:
        """
        Class to store the NACA 4 digits parameters for the airfoil in the object

        Args:
            p (float): Camber parameter
            m (float): Position of max camber
            xx (float): Thickness
        """
        self.p: float = p
        self.m: float = m
        self.xx: float = xx

    def set_naca5_digits(self, l: float, p: float, q: float, xx: float) -> None:
        """
        Class to store the NACA 4 digits parameters for the airfoil in the object

        Args:
            p (float): Camber parameter
            m (float): Position of max camber
            xx (float): Thickness
        """
        self.l: float = l
        self.p = p
        self.q: float = q
        self.xx = xx

    def camber_line_naca4(
        self,
        points: FloatArray,
    ) -> ndarray[Any, dtype[floating[Any]]]:
        """
        Function to generate the camber line for a NACA 4 digit airfoil.
        Returns the camber line for a given set of x coordinates.

        Args:
            points (FloatArray): X coordinates for which we need the camber line

        Returns:
            ndarray[Any, dtype[floating[Any]]]: X,Y coordinates of the camber line
        """
        p: float = self.p
        m: float = self.m

        res: ndarray[Any, dtype[floating[Any]]] = np.zeros_like(points)
        for i, x in enumerate(points):
            if x < p:
                res[i] = m / p**2 * (2 * p * x - x**2)
            else:
                res[i] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2)
        return res

    def airfoil_to_selig(self) -> None:
        """
        Returns the airfoil in the selig format
        """
        x_points: FloatArray = np.hstack((self._x_upper[::-1], self._x_lower[1:])).T
        y_points: FloatArray = np.hstack((self._y_upper[::-1], self._y_lower[1:])).T
        # y_points[0]=0
        # y_points[-1]=0
        self.selig: FloatArray = np.vstack((x_points, y_points))

    def load_from_web(self) -> None:
        """
        Fetches the airfoil data from the web. Specifically from the UIUC airfoil database.
        """
        link: str = "https://m-selig.ae.illinois.edu/ads/coord/naca" + self.name + ".dat"
        with urllib.request.urlopen(link) as url:
            site_data: str = url.read().decode("UTF-8")
        s: list[str] = site_data.split()
        s = s[2:]
        x: list[float] = []
        y: list[float] = []
        for i in range(int(len(s) / 2)):
            temp: float = float(s[2 * i])
            x.append(temp)
            temp = float(s[2 * i + 1])
            y.append(temp)
        # y[0] = 0
        # y[-1]= 0
        self.selig_web: FloatArray = np.vstack((x, y))

    def save_selig_te(self, directory: str | None = None, header: bool = False) -> None:
        """
        Saves the airfoil in the selig format.

        Args:
            directory (str, optional): Directory to save the airfoil. Defaults to None.
            header (bool, optional): Whether to include the header. Defaults to False.
        """

        if directory is not None:
            file_name = os.path.join(directory, self.file_name)
        else:
            file_name = self.file_name
        # if not file_name.endswith(".dat"):
        #     file_name += ".dat"
        # print(f"Saving airfoil to {file_name}")

        with open(file_name, "w") as file:
            if header:
                file.write(f"{self.name}\n\n")
            for x, y in self.selig.T:
                file.write(f" {x:.6f} {y:.6f}\n")

    def save_le(self, directory: str | None = None) -> None:
        """
        Saves the airfoil in the revese selig format.

        Args:
            directory (str, optional): Directory to save the airfoil. Defaults to None.
        """
        x = [*self._x_lower[::-1], *self._x_upper[::-1]]
        y = [*self._y_lower[::-1], *self._y_upper[::-1]]

        pts = np.vstack((x, y))
        if directory is not None:
            file_name = os.path.join(directory, self.file_name)
        else:
            file_name = self.file_name
        # if not file_name.endswith(".dat"):
        #     file_name += ".dat"

        # print(f"Saving airfoil to {file_name}")

        with open(file_name, "w") as file:
            file.write(f"{self.name}\n\n")
            for x, y in pts.T:
                file.write(f" {x:.6f} {y:.6f}\n")

    def plot(self) -> None:
        """
        Plots the airfoil in the selig format
        """
        pts = self.selig
        x, y = pts
        plt.plot(x[: self.n_points], y[: self.n_points], "r")
        plt.plot(x[self.n_points :], y[self.n_points :], "b")
        plt.axis("scaled")


def interpolate(
    xa: FloatArray | list[float],
    ya: FloatArray | list[float],
    queryPoints: FloatArray | list[float],
) -> FloatArray:
    """
    A cubic spline interpolation on a given set of points (x,y)
    Recalculates everything on every call which is far from efficient but does the job for now
    should eventually be replaced by an external helper class
    """

    # PreCompute() from Paint Mono which in turn adapted:
    # NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING
    # ISBN 0-521-43108-5, page 113, section 3.3.
    # http://paint-mono.googlecode.com/svn/trunk/src/PdnLib/SplineInterpolator.cs

    # number of points
    n: int = len(xa)
    u: FloatArray = np.zeros(n)
    y2: FloatArray = np.zeros(n)

    for i in range(1, n - 1):
        # This is the decomposition loop of the tridiagonal algorithm.
        # y2 and u are used for temporary storage of the decomposed factors.

        wx = xa[i + 1] - xa[i - 1]
        sig = (xa[i] - xa[i - 1]) / wx
        p = sig * y2[i - 1] + 2.0

        y2[i] = (sig - 1.0) / p

        ddydx = (ya[i + 1] - ya[i]) / (xa[i + 1] - xa[i]) - (ya[i] - ya[i - 1]) / (xa[i] - xa[i - 1])

        u[i] = (6.0 * ddydx / wx - sig * u[i - 1]) / p

    y2[n - 1] = 0

    # This is the backsubstitution loop of the tridiagonal algorithm
    # ((int i = n - 2; i >= 0; --i):
    for i in range(n - 2, -1, -1):
        y2[i] = y2[i] * y2[i + 1] + u[i]

    # interpolate() adapted from Paint Mono which in turn adapted:
    # NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING
    # ISBN 0-521-43108-5, page 113, section 3.3.
    # http://paint-mono.googlecode.com/svn/trunk/src/PdnLib/SplineInterpolator.cs

    results = np.zeros(n)

    # loop over all query points
    for i in range(len(queryPoints)):
        # bisection. This is optimal if sequential calls to this
        # routine are at random values of x. If sequential calls
        # are in order, and closely spaced, one would do better
        # to store previous values of klo and khi and test if

        klo = 0
        khi = n - 1

        while khi - klo > 1:
            k = (khi + klo) >> 1
            if xa[k] > queryPoints[i]:
                khi = k
            else:
                klo = k

        h = xa[khi] - xa[klo]
        a = (xa[khi] - queryPoints[i]) / h
        b = (queryPoints[i] - xa[klo]) / h

        # Cubic spline polynomial is now evaluated.
        results[i] = a * ya[klo] + b * ya[khi] + ((a * a * a - a) * y2[klo] + (b * b * b - b) * y2[khi]) * (h * h) / 6.0

    return results


def gen_NACA5_airfoil(number: str, n_points: int, finite_TE: bool = False) -> tuple[FloatArray, FloatArray]:
    """
    Generates a NACA 5 digit airfoil

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
