"""
Airfoil class to represent an airfoil. Inherits from airfoil class from the airfoils module.
The airfoil class is used to generate, store, and manipulate airfoils. To initialize the class
you need to pass the upper and lower surface coordinates. The class also contains alternative
constructors to generate airfoils from NACA 4 and 5 digit identifiers.

To initialize the Airfoil class, you need to pass the upper and lower surface coordinates.

>>> from ICARUS.Airfoils.airfoil import Airfoil
>>> naca0012 = Airfoil.naca("0012", n_points=200)
>>> naca0012.plot()

Alternatively, you can initialize the class from a file.

>>> naca0008 = Airfoil.load_from_file("naca0008.dat")
>>> naca0008.plot()

There is functionality to get the data from the web. This is done by fetching the data from the UIUC airfoil database.

>>> naca0008 = Airfoil.naca("0008", n_points=200)
>>> naca0008.load_from_web()
>>> naca0008.plot()

Finally, you can initialize the class from a morph between two airfoils.

>>> naca0008 = Airfoil.naca("0008", n_points=200)
>>> naca0012 = Airfoil.naca("0012", n_points=200)
>>> naca_merged = Airfoil.morph_new_from_two_foils(naca0008, naca0012, eta=0.5, n_points=200)

The class also contains methods to generate a flapped airfoil.

>>> naca0008_flapped = naca0008.flap_airfoil(flap_hinge=0.5, chord_extension=0.2, flap_angle=20, plotting=True)

The class contains methods to save the airfoil in the selig format (starting from the trailing edge) or in
the reverse selig format (starting from the leading edge).

>>> file_name = "naca0008"
>>> naca0008.save_selig_te(file_name)
>>> naca0008.save_le(file_name)

Finally, the class inherits from the airfoil class from the airfoils module. This means that you can use all the
methods from the original airfoil class which include but are not limited to:

>>> x = np.linspace(0, 1, 200)
>>> naca0008.camber_line(x)
>>> naca0008.camber_line_angle(x)
>>> naca0008.y_upper(x)
>>> naca0008.y_lower(x)
>>> naca0008.all_points()


"""
import os
import re
import urllib.request
from typing import Any
from typing import Union

import airfoils as af
import matplotlib.pyplot as plt
import numpy as np

from ICARUS.Airfoils._gen_NACA5_airfoil import gen_NACA5_airfoil
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
        # self.fix_le()

        # For Type Checking
        self._x_upper: FloatArray = self._x_upper
        self._y_upper: FloatArray = self._y_upper
        self._x_lower: FloatArray = self._x_lower
        self._y_lower: FloatArray = self._y_lower
        # self.getFromWeb()

    @classmethod
    def morph_new_from_two_foils(
        cls,
        airfoil1: "Airfoil",
        airfoil2: "Airfoil",
        eta: float,
        n_points: int,
    ) -> "Airfoil":
        """
        Returns a new airfoil morphed between two airfoils

        Notes:
            * This is an alternative constructor for the Airfoil class

        Args:
            airfoil1 (Airfoil): First airfoil
            airfoil2 (Airfoil): Second airfoil
            eta (float): Morphing parameter
            n_points (int): Number of points to generate

        Raises:
            ValueError: If eta is not in range [0,1]

        Returns:
            Airfoil: New airfoil morphed between the two airfoils
        """

        if not 0 <= eta <= 1:
            raise ValueError(f"'eta' must be in range [0,1], given eta is {float(eta):.3f}")

        x = np.linspace(0, 1, n_points)

        y_upper_af1 = airfoil1.y_upper(x)
        y_lower_af1 = airfoil1.y_lower(x)
        y_upper_af2 = airfoil2.y_upper(x)
        y_lower_af2 = airfoil2.y_lower(x)

        y_upper_new = y_upper_af1 * (1 - eta) + y_upper_af2 * eta
        y_lower_new = y_lower_af1 * (1 - eta) + y_lower_af2 * eta

        upper = np.array([x, y_upper_new])
        lower = np.array([x, y_lower_new])

        return cls(upper, lower, f"morphed_{airfoil1.name}_{airfoil2.name}_at_{eta}%", n_points)

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
        re_4digits: re.Pattern[str] = re.compile(r"\b(?:NACA\s*)?(\d{4})\b")
        re_5digits: re.Pattern[str] = re.compile(r"\b(?:NACA\s*)?(\d{5})\b")
        naca = naca.replace("naca", "")
        naca = naca.replace("NACA", "")
        naca = naca.replace(".", "")
        naca = naca.replace("-", "")
        naca = naca.replace("_", "")
        naca = naca.replace(" ", "")
        if re_5digits.match(naca):
            l: float = float(naca[0]) / 10
            p: float = float(naca[1]) / 100
            q: float = float(naca[2]) / 1000
            xx: float = float(naca[3:5]) / 1000
            upper, lower = gen_NACA5_airfoil(naca, n_points)
            self: "Airfoil" = cls(upper, lower, naca, n_points)
            return self
        elif re_4digits.match(naca):
            m: float = float(naca[0]) / 100
            p = float(naca[1]) / 10
            xx = float(naca[2:4]) / 100
            upper, lower = af.gen_NACA4_airfoil(m, p, xx, n_points // 2)
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

        # Stretch the points so all points move the same amount
        flap_chord_extension = chord_extension * np.cos(np.deg2rad(flap_angle))
        x_lower = x_lower * (flap_chord_extension)
        x_upper = x_upper * (flap_chord_extension)
        # x_lower = x_lower * (chord_extension)
        # x_upper = x_upper * (chord_extension)

        # Rotate the points according to the hinge (located on the lower side)
        theta: float = -np.deg2rad(flap_angle)
        x_lower = x_lower * np.cos(theta) - y_lower * np.sin(theta)
        x_upper = x_upper * np.cos(theta) - y_upper * np.sin(theta)
        y_lower = x_lower * np.sin(theta) + y_lower * np.cos(theta)
        y_upper = x_upper * np.sin(theta) + y_upper * np.cos(theta)

        # Translate the points back
        x_lower = x_lower + flap_hinge
        x_upper = x_upper + flap_hinge
        y_lower = y_lower + temp
        y_upper = y_upper + temp

        if plotting:
            self.plot()

        upper: FloatArray = np.array(
            [
                [*self._x_upper[:idx_upper], *x_upper],
                [*self._y_upper[:idx_upper], *y_upper],
            ],
        )
        lower: FloatArray = np.array(
            [
                [*self._x_lower[:idx_lower], *x_lower],
                [*self._y_lower[:idx_lower], *y_lower],
            ],
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
        points: float | FloatArray | list[float],
    ) -> FloatArray:
        """
        Function to generate the camber line for a NACA 4 digit airfoil.
        Returns the camber line for a given set of x coordinates.

        Args:
            points (FloatArray): X coordinates for which we need the camber line

        Returns:
            FloatArray: X,Y coordinates of the camber line
        """
        p: float = self.p
        m: float = self.m

        if isinstance(points, float):
            points_ = np.array([points])
        if isinstance(points, list):
            points_ = np.array(points)
        elif isinstance(points, np.ndarray):
            points_ = points
            points_ = points_.flatten()
        else:
            points_ = np.array(points)

        res: FloatArray = np.zeros_like(points_)
        for i, x in enumerate(points_):
            if x < p:
                res[i] = m / p**2 * (2 * p * x - x**2)
            else:
                res[i] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2)
        return res

    def camber_line(self, x: Union[float, FloatArray]) -> FloatArray:
        """"""
        if hasattr(self, "l"):
            # return self.camber_line_naca5(x)
            print("NACA 5 camber analytical solution not implemented yet")
            return super().camber_line(x)
        elif hasattr(self, "p"):
            return self.camber_line_naca4(x)
        else:
            return super().camber_line(x)

    def airfoil_to_selig(self) -> None:
        """
        Returns the airfoil in the selig format
        """
        x_points: FloatArray = np.hstack((self._x_lower[::-1], self._x_upper[1:])).T
        y_points: FloatArray = np.hstack((self._y_lower[::-1], self._y_upper[1:])).T
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

    def save_selig_te(self, directory: str | None = None, header: bool = False, inverse: bool = False) -> None:
        """
        Saves the airfoil in the selig format.

        Args:
            directory (str, optional): Directory to save the airfoil. Defaults to None.
            header (bool, optional): Whether to include the header. Defaults to False.
            inverse (bool, optional): Whether to save the airfoil in the reverse selig format. Defaults to False.
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
                file.write(f"{self.name} with {self.n_points}\n")
            if inverse:
                pts = self.selig.T[::-1]
            else:
                pts = self.selig.T

            for x, y in pts:
                file.write(f" {x:.6f} {y:.6f}\n")

    def save_le(self, directory: str | None = None) -> None:
        """
        Saves the airfoil in the revese selig format.

        Args:
            directory (str, optional): Directory to save the airfoil. Defaults to None.
        """
        x = [*self._x_lower[:], *self._x_upper[::-1]]
        y = [*self._y_lower[:], *self._y_upper[::-1]]

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

    def plot(self, camber: bool = False, scatter: bool = False) -> None:
        """
        Plots the airfoil in the selig format

        Args:
            camber (bool, optional): Whether to plot the camber line. Defaults to False.
            scatter (bool, optional): Whether to plot the airfoil as a scatter plot. Defaults to False.
        """
        pts = self.selig
        x, y = pts
        if scatter:
            plt.scatter(x[: self.n_points], y[: self.n_points], s=1)
            plt.scatter(x[self.n_points :], y[self.n_points :], s=1)
        else:
            plt.plot(x[: self.n_points], y[: self.n_points], "r")
            plt.plot(x[self.n_points :], y[self.n_points :], "b")

        if camber:
            x = np.linspace(0, 1, 100)
            y = self.camber_line(x)
            plt.plot(x, y, "k--")

        plt.axis("scaled")
