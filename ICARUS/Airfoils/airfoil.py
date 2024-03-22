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
import logging
import os
import re
import urllib.request
from typing import Any
from typing import Union

import airfoils as af
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from matplotlib.axes import Axes

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
        n_points: int = 200,
    ) -> None:
        """
        Initialize the Airfoil class

        Args:
            upper (FloatArray): Upper surface coordinates
            lower (FloatArray): Lower surface coordinates
            naca (str): NACA 4 digit identifier (e.g. 0012)
            n_points (int): Number of points to be used to generate the airfoil. It interpolates between upper and lower
        """

        lower, upper = self.close_airfoil(lower, upper)
        super().__init__(upper, lower)
        name = name.replace(" ", "")
        self.name: str = name
        self.file_name: str = name

        self.n_points: int = n_points
        # Repanel the airfoil
        # self.repanel(n_points=n_points, distribution="cosine")
        self.selig = self.to_selig()

        self.polars: dict[str, Any] | Struct = {}

        # For Type Checking
        self._x_upper: FloatArray = self._x_upper
        self._y_upper: FloatArray = self._y_upper

        self._x_lower: FloatArray = self._x_lower
        self._y_lower: FloatArray = self._y_lower
        self.n_upper = self._x_upper.shape[0]
        self.n_lower = self._x_lower.shape[0]

    def repanel_spl(self, n_points: int = 200, smoothing= 0.0) -> None:
        pts = self.to_selig()
        x = pts[0,:]
        y = pts[1,:]
        # Combine x and y coordinates into a single array of complex numbers
        complex_coords = x + 1j * y
        # Find unique complex coordinates
        unique_indices = np.sort(np.unique(complex_coords, return_index=True)[1])
        # Use the unique indices to get the unique x and y coordinates
        x = x[unique_indices]
        y = y[unique_indices]
        # x = np.hstack((x, x[0]))
        # y = np.hstack((y, y[0]))

        tck, u = splprep([x, y], s=smoothing)
    
        tnew = np.linspace(0,1,n_points)
        self.spline = splev(tnew, tck)
        lower, upper = self.split_sides(self.spline[0], self.spline[1])
        lower, upper = self.close_airfoil(lower, upper)

        self._x_upper = upper[0]
        self._y_upper = upper[1]
        self._x_lower = lower[0]
        self._y_lower = lower[1]
        #update for plot
        self.n_points = n_points
        self.n_upper = self._x_upper.shape[0]
        self.n_lower = self._x_lower.shape[0]
        self.selig = self.to_selig()

    def repanel_from_internal(self, n_points: int, distribution="cosine") -> None:
        """
        Repanels the airfoil to have n_points

        Args:
            n_points (int): Number of points to generate
        """
        if distribution == "cosine":
            beta = np.linspace(0, np.pi, int(n_points // 2))
            # apply cosine spacing to xsi
            xsi = 0.5 * (1 - np.cos(beta))
        elif distribution == "tanh":
            xsi = np.tanh(np.linspace(-3, 3, n_points))
            xsi = (xsi - np.min(xsi)) / (np.max(xsi) - np.min(xsi))
        else:
            xsi = np.linspace(0, 1, int(n_points // 2))

        _x_upper = xsi
        _x_lower = xsi
        _y_upper = self.y_upper(xsi)
        _y_lower = self.y_lower(xsi)

        lower = np.array([_x_lower, _y_lower], dtype=float)
        upper = np.array([_x_upper, _y_upper], dtype=float)

        #lower, upper = self.close_airfoil(lower, upper)

        self._x_lower = lower[0]
        self._y_lower = lower[1]
        self._x_upper = upper[0]
        self._y_upper = upper[1]

        self.n_points = n_points
        self.n_upper = self._x_upper.shape[0]
        self.n_lower = self._x_lower.shape[0]
        self.selig = self.to_selig()

    def close_airfoil(self, lower, upper):
        # Check if the airfoil is closed or not. Meaning that the upper and lower surface meet at the trailing edge and leading edge
        # If the airfoil is not closed, then it will be closed by adding a point at the trailing edge
        # Identify the upper surface trailing edge and leading edge
        f_upper = upper[0, 0]
        l_upper = upper[0, -1]
        if f_upper < l_upper:
            leading_upper = f_upper
            le_idx_upper = 0
            trailing_upper = l_upper
            te_idx_upper = -1
        else:
            leading_upper = l_upper
            le_idx_upper = -1
            trailing_upper = f_upper
            te_idx_upper = 0

        # Identify the lower surface trailing edge and leading edge
        f_lower = lower[0, 0]
        l_lower = lower[0, -1]
        if f_lower < l_lower:
            leading_lower = f_lower
            le_idx_lower = 0
            trailing_lower = l_lower
            te_idx_lower = -1
        else:
            leading_lower = l_lower
            le_idx_lower = -1
            trailing_lower = f_lower
            te_idx_lower = 0

        # Fix the trailing edge
        # Leading upper is the leftmost point. We need to add it to the surface with the rightmost point
        if leading_upper == leading_lower:
            pass
        elif leading_upper < leading_lower:
            if le_idx_lower == 0:
                lower = np.hstack((upper[:, le_idx_upper].reshape(2, 1), lower))
            elif le_idx_lower == -1:
                lower = np.hstack((lower, upper[:, le_idx_upper].reshape(2, 1)))
        elif leading_upper > leading_lower:
            if le_idx_upper == 0:
                upper = np.hstack((lower[:, le_idx_lower].reshape(2, 1), upper))
            elif le_idx_upper == -1:
                upper = np.hstack((upper, lower[:, le_idx_lower].reshape(2, 1)))

        # Fix the leading edge
        # Trailing upper is the rightmost point. We need to add it to the surface with the leftmost point
        if trailing_upper == trailing_lower:
            pass
        elif trailing_upper > trailing_lower:
            if te_idx_lower == -1:
                lower = np.hstack((lower, upper[:, te_idx_upper].reshape(2, 1)))
            elif te_idx_lower == 0:
                lower = np.hstack((upper[:, te_idx_upper].reshape(2, 1), lower))
        elif trailing_upper < trailing_lower:
            if te_idx_upper == -1:
                upper = np.hstack((upper, lower[:, te_idx_lower].reshape(2, 1)))
            elif te_idx_upper == 0:
                upper = np.hstack((lower[:, te_idx_lower].reshape(2, 1), upper))
        return lower, upper

    def thickness(self, x) -> FloatArray:
        """
        Returns the thickness of the airfoil at the given x coordinates

        Args:
            x (FloatArray): X coordinates

        Returns:
            FloatArray: _description_
        """
        thickness = self.y_upper(x) - self.y_lower(x)
        # Remove Nan
        thickness: FloatArray = thickness[~np.isnan(thickness)]
        return thickness

    def max_thickness(self) -> float:
        """
        Returns the maximum thickness of the airfoil

        Returns:
            float: Maximum thickness
        """
        thickness: FloatArray = self.thickness(np.linspace(0, 1, self.n_points))
        return float(np.max(thickness))

    def max_thickness_location(self) -> float:
        """
        Returns the location of the maximum thickness of the airfoil

        Returns:
            float: Location of the maximum thickness
        """
        thickness: FloatArray = self.thickness(np.linspace(0, 1, self.n_points))
        return float(np.argmax(thickness) / self.n_points)

    @staticmethod
    def split_sides(x,y):
        
        # Remove duplicate points from the array
        # A point is duplicated if it has the same x and y coordinates
        # This is done to avoid issues with the interpolation
        x_arr = np.array(x)
        y_arr = np.array(y)

        # Combine x and y coordinates into a single array of complex numbers
        complex_coords = x_arr + 1j * y_arr
        # Find unique complex coordinates
        unique_indices = np.sort(np.unique(complex_coords, return_index=True)[1])

        # Use the unique indices to get the unique x and y coordinates
        x_clean = x_arr[unique_indices]
        y_clean = y_arr[unique_indices]
        # Locate the trailing edge

        # Find Where x_arr = 0
        idxs = np.where(x_arr == 0)[0].flatten()
        if len(idxs) == 0:
            # Find where the x_arr is closest to 0
            # Check if the min values is duplicated in the array
            idx = np.argmin(np.abs(x_arr))
            # If it is duplicated, take the last one
            if len(np.where(x_arr == x_arr[idx])[0]) > 1:
                idx = np.where(x_arr == x_arr[idx])[0][-1]
        elif len(idxs) == 1:
            idx = idxs[0] + 1
        else:
            idx = idxs[-1]

        if idx == 1:
            idx = np.argmin(np.abs(x_arr[1:]))
            # If it is duplicated, take the last one
            if len(np.where(x_arr[1:] == x_arr[idx])[0]) > 1:
                idx = np.where(x_arr[1:] == x_arr[idx])[0][-1]
        # Calibrate idx to account for removed duplicates
        idx_int: int = int(unique_indices[unique_indices < idx].shape[0])
        lower: FloatArray = np.array([x_clean[idx_int:], y_clean[idx_int:]])
        upper: FloatArray = np.array([x_clean[:idx_int], y_clean[:idx_int]])

        return lower, upper

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

        ksi = np.linspace(0, np.pi, n_points//2)
        x = 0.5 * (1 - np.cos(ksi))

        y_upper_af1 = airfoil1.y_upper(x)
        y_lower_af1 = airfoil1.y_lower(x)
        y_upper_af2 = airfoil2.y_upper(x)
        y_lower_af2 = airfoil2.y_lower(x)

        y_upper_new = y_upper_af1 * (1 - eta) + y_upper_af2 * eta
        y_lower_new = y_lower_af1 * (1 - eta) + y_lower_af2 * eta

        upper = np.array([x, y_upper_new], dtype=float)
        lower = np.array([x, y_lower_new], dtype=float)

        # Remove nan values and duplicates
        nan_upper_idx = np.isnan(upper[1])
        upper = upper[:, ~nan_upper_idx]

        nan_lower_idx = np.isnan(lower[1])
        lower = lower[:, ~nan_lower_idx]
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
            self: "Airfoil" = cls(upper, lower, f"NACA{naca}", n_points)
            return self
        elif re_4digits.match(naca):
            m: float = float(naca[0]) / 100
            p = float(naca[1]) / 10
            xx = float(naca[2:4]) / 100
            upper, lower = af.gen_NACA4_airfoil(m, p, xx, n_points // 2)
            self = cls(upper, lower, f"NACA{naca}", n_points)
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
        logging.info(f"Loading airfoil from {filename}")
        with open(filename) as file:
            for line in file:
                line = line.strip()

                if line == "\n":
                    continue

                # Check if it contains two numbers
                if len(line.split()) != 2:
                    continue

                try:
                    x_i = float(line.split()[0])
                    y_i = float(line.split()[1])
                    if np.abs(x_i) > 2.0 or np.abs(y_i) > 2.0:
                        continue
                    x.append(x_i)
                    y.append(y_i)
                except (ValueError, IndexError):
                    continue
        lower,upper = cls.split_sides(x,y)
        try:
            self: "Airfoil" = cls(upper, lower, os.path.split(filename)[-1], len(x))
        except ValueError as e:
            print(f"Error loading airfoil from {filename}")
            raise (ValueError(e))
        return self
    
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
            x: float = float(points)
            if x < p:
                result: float = m / p**2 * (2 * p * x - x**2)
            else:
                result = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2)
            return np.array(result)
        else:
            if isinstance(points, list):
                points = np.array(points, dtype=float)
            if isinstance(points, int):
                points = np.array(float(points))

            results: FloatArray = np.zeros_like(points)
            for i, x in enumerate(points.tolist()):
                if x < p:
                    results[i] = m / p**2 * (2 * p * x - x**2)
                else:
                    results[i] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2)
            return results

    def camber_line(self, x: Union[float, list[float], FloatArray]) -> FloatArray:
        """"""
        if hasattr(self, "l"):
            # return self.camber_line_naca5(x)
            print("NACA 5 camber analytical solution not implemented yet")
            return np.array(super().camber_line(x), dtype=float)
        elif hasattr(self, "p"):
            return self.camber_line_naca4(x)
        else:
            return np.array(super().camber_line(x), dtype=float)

    def to_selig(self) -> FloatArray:
        """
        Returns the airfoil in the selig format.
        Meaning that the airfoil runs run from the trailing edge, round the leading edge,
        back to the trailing edge in either direction:
        """
        # Identify the upper and lower surface leading and trailing edges
        if self._x_upper[0] < self._x_upper[-1]:
            y_up = self._y_upper[::-1]
            x_up = self._x_upper[::-1]
        else:
            x_up = self._x_upper
            y_up = self._y_upper

        if self._x_lower[0] > self._x_lower[-1]:
            x_lo = self._x_lower[::-1]
            y_lo = self._y_lower[::-1]
        else:
            x_lo = self._x_lower
            y_lo = self._y_lower

        # Remove NaN values
        idx_nan = np.isnan(x_up) | np.isnan(y_up)
        x_up = x_up[~idx_nan]
        y_up = y_up[~idx_nan]

        idx_nan = np.isnan(x_lo) | np.isnan(y_lo)
        x_lo = x_lo[~idx_nan]
        y_lo = y_lo[~idx_nan]

        upper = np.array([x_up, y_up], dtype=float)
        lower = np.array([x_lo, y_lo], dtype=float)

        lower, upper = self.close_airfoil(lower, upper)

        x_up = upper[0]
        y_up = upper[1]

        x_lo = lower[0]
        y_lo = lower[1]

        x_points: FloatArray = np.hstack((x_up, x_lo)).T
        y_points: FloatArray = np.hstack((y_up, y_lo)).T

        return np.vstack((x_points, y_points))

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

    def plot(
        self,
        camber: bool = False,
        scatter: bool = False,
        max_thickness: bool = False,
        ax: Axes | None = None,
    ) -> None:
        """
        Plots the airfoil in the selig format

        Args:
            camber (bool, optional): Whether to plot the camber line. Defaults to False.
            scatter (bool, optional): Whether to plot the airfoil as a scatter plot. Defaults to False.
            max_thickness (bool, optional): Whether to plot the max thickness. Defaults to False.
        """
        pts = self.selig
        x, y = pts

        if ax is None:
            fig, _ax = plt.subplots()
        else:
            _ax: Axes = ax

        if scatter:
            _ax.scatter(x[: self.n_upper], y[: self.n_upper], s=1)
            _ax.scatter(x[self.n_upper :], y[self.n_upper :], s=1)
        else:
            _ax.plot(x[: self.n_upper], y[: self.n_upper], "r")
            _ax.plot(x[self.n_upper :], y[self.n_upper :], "b")

        if camber:
            x = np.linspace(0, 1, 100)
            y = self.camber_line(x)
            _ax.plot(x, y, "k--")

        if max_thickness:
            x = self.max_thickness_location()
            thick = self.max_thickness()
            y_up = self.y_upper(x)
            y_lo = self.y_lower(x)
            # Plot a line from the upper to the lower surface
            _ax.plot([x, x], [y_up, y_lo], "k--")
            # Add a text with the thickness
            _ax.text(x, y_lo, f"{thick:.3f}", ha="right", va="bottom")
        _ax.axis("scaled")
