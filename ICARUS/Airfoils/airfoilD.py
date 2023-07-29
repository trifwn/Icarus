import os
import re
import urllib.request
from typing import Any

import airfoils as af
import matplotlib.pyplot as plt
import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray

from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray

# # Airfoil
# ##### 0 = Read from python module
# ##### 1 = Read from airfoiltools.com
# ##### 2 = load from file


class AirfoilD(af.Airfoil):  # type: ignore
    """
    Class to represent an airfoil. Inherits from Airfoil class from the airfoils module.
    Stores the airfoil data in the selig format.

    Args:
        af : Airfoil class from the airfoils module
    """

    def __init__(
        self,
        upper: FloatArray,
        lower: FloatArray,
        naca: str,
        n_points: int,
    ) -> None:
        """
        Initialize the AirfoilD class

        Args:
            upper (FloatArray): Upper surface coordinates
            lower (FloatArray): Lower surface coordinates
            naca (str): NACA 4 digit identifier (e.g. 0012)
            n_points (int): Number of points to be used to generate the airfoil. It interpolates between upper and lower
        """
        super().__init__(upper, lower)
        self.name: str = naca
        self.file_name: str = f"naca{naca}"
        self.n_points: int = n_points
        self.airfoil_to_selig()
        self.reynolds_ids: list[str] = []
        self.polars: dict[str, Any] | Struct = {}
        # self.getFromWeb()

    @classmethod
    def naca(cls, naca: str, n_points: int = 200) -> "AirfoilD":
        """
        Initialize the AirfoilD class from a NACA 4 digit identifier.

        Args:
            naca (str): NACA 4 digit identifier (e.g. 0012) can also take NACA0012
            n_points (int, optional): Number of points to generate. Defaults to 200.

        Raises:
            af.NACADefintionError: If the NACA identifier is not valid

        Returns:
            AirfoilD: AirfoilD class object
        """
        re_4digits: re.Pattern[str] = re.compile(r"^\d{4}$")

        if re_4digits.match(naca):
            p: float = float(naca[0]) / 10
            m: float = float(naca[1]) / 100
            xx: float = float(naca[2:4]) / 100
        else:
            raise af.NACADefintionError(
                "Identifier not recognised as valid NACA 4 definition",
            )

        upper, lower = af.gen_NACA4_airfoil(p, m, xx, n_points)
        self: "AirfoilD" = cls(upper, lower, naca, n_points)
        self.set_naca4_digits(p, m, xx)
        return self

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

    def save(self, directory: str | None = None) -> None:
        """
        Saves the airfoil in the selig format.

        Args:
            directory (str, optional): Directory to save the airfoil. Defaults to None.
        """

        pt0 = self.selig
        if directory is not None:
            file_name = os.path.join(directory, self.file_name)
        else:
            file_name = self.file_name
        np.savetxt(file_name, pt0.T)

    def plot(self) -> None:
        """
        Plots the airfoil in the selig format
        """
        pts = self.selig
        x, y = pts
        plt.plot(x[: self.n_points], y[: self.n_points], "r")
        plt.plot(x[self.n_points :], y[self.n_points :], "b")
        plt.axis("scaled")
