"""
Airfoil class to represent an airfoil. Inherits from airfoil class from the airfoils module.
The airfoil class is used to generate, store, and manipulate airfoils. To initialize the class
you need to pass the upper and lower surface coordinates. The class also contains alternative
constructors to generate airfoils from NACA 4 and 5 digit identifiers.

To initialize the Airfoil class, you need to pass the upper and lower surface coordinates.

>>> from ICARUS.airfoils.airfoil import Airfoil
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

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import requests
from jaxtyping import Float
from matplotlib.axes import Axes

from ICARUS.core.types import FloatArray
from ICARUS.interpolation import JaxInterpolator1D

if TYPE_CHECKING:
    from .flapped_airfoil import FlappedAirfoil


class NACADefintionError(Exception):
    """Raised when the NACA identifier number is not valid"""

    pass


@jax.tree_util.register_pytree_node_class
class Airfoil:
    """Class to represent an airfoil. Inherits from airfoil class from the airfoils module.
    Stores the airfoil data in the selig format.
    """

    def __init__(
        self,
        upper: Float,
        lower: Float,
        name: str | None = None,
    ) -> None:
        """Initialize the Airfoil class

        Args:
            upper (FloatArray): Upper surface coordinates
            lower (FloatArray): Lower surface coordinates
            naca (str): NACA 4 digit identifier (e.g. 0012)
        """
        if name is not None:
            name = name.replace(" ", "")
            self._name: str | None  = name
        else:
            self._name = None

        lower, upper = self.remove_nan_points(lower, upper)  # remove nan points
        lower, upper = self.order_points(
            lower,
            upper,
        )  # order points according to LE/TE
        lower, upper = self.close_airfoil(lower, upper)  # close airfoil
        # Unpack coordinates
        self._x_upper = upper[0, :]
        self._y_upper = upper[1, :]
        self._x_lower = lower[0, :]
        self._y_lower = lower[1, :]

        self._y_upper_interp = JaxInterpolator1D(
            self._x_upper,
            self._y_upper,
            method="linear",
            extrap=True,
        )
        self._y_lower_interp = JaxInterpolator1D(
            self._x_lower,
            self._y_lower,
            method="linear",
            extrap=True,
        )

        self.min_x = np.min(self._x_upper)
        self.max_x = np.max(self._x_upper)
        self.norm_factor = 1

        # Repanel the airfoil
        # self.repanel(n_points=n_points, distribution="cosine")
        self.n_points: int = upper.shape[1]
        self.n_upper = self._x_upper.shape[0]
        self.n_lower = self._x_lower.shape[0]
        self.selig_original = self.to_selig()

    @property
    def name(self) -> str:
        """Returns the name of the airfoil"""
        if not hasattr(self, "_name") or self._name is None:
            return "Airfoil"
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Sets the name of the airfoil"""
        if not isinstance(value, str):
            raise TypeError("Name must be a string")
        self._name = value.replace(" ", "")

    @property
    def upper_surface(self) -> Float:
        """Returns the upper surface coordinates of the airfoil"""
        return jnp.array([self._x_upper, self._y_upper], dtype=float)

    @property
    def lower_surface(self) -> Float:
        """Returns the lower surface coordinates of the airfoil"""
        return jnp.array([self._x_lower, self._y_lower], dtype=float)

    def y_upper(self, ksi: Float) -> Float:
        # x-coordinate is between [0, 1]
        # x must be set between [min(x_upper), max(x_upper)]
        x = self.min_x + ksi * (self.max_x - self.min_x)
        return self._y_upper_interp(x)

    def y_lower(self, ksi: Float) -> Float:
        # x-coordinate is between [0, 1]
        # x must be set between [min(x_lower), max(x_lower)]

        x = self.min_x + ksi * (self.max_x - self.min_x)
        return self._y_lower_interp(x)

    def repanel_spl(self, n_points: int = 200) -> None:
        pts = self.selig_original
        x = pts[0, :]
        y = pts[1, :]
        # Combine x and y coordinates into a single array of complex numbers
        complex_coords = x + 1j * y
        # Find unique complex coordinates
        unique_indices = np.sort(np.unique(complex_coords, return_index=True)[1])
        # Use the unique indices to get the unique x and y coordinates
        x = x[unique_indices]
        y = y[unique_indices]

        from scipy.interpolate import CubicSpline

        # Airfoils 0 and 1 are defined by their cubic splines,
        #   x0(s0), y0(s0)       x1(s1), y1(s1)
        # with the discrete secant arc length parameters s computed from
        # the coordinates x(i),y(i):
        #   s(i) = s(i-1) + sqrt[ (x(i)-x(i-1))^2 + (y(i)-y(i-1))^2 ]
        s = np.zeros(x.shape)
        for i in range(1, x.shape[0]):
            s[i] = s[i - 1] + np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)
        unique_s_indices = np.sort(np.unique(s, return_index=True)[1])
        # Use the unique indices to get the unique arc length coordinates
        s = s[unique_s_indices]
        x = x[unique_s_indices]
        y = y[unique_s_indices]

        # Normalize the arc length
        s /= s[-1]
        spl = CubicSpline(s, y)

        ksi = np.linspace(0, np.pi, n_points // 2)
        # Apply cosine spacing to ksi
        tnew_1 = 0.5 * (1 - np.cos(ksi)) / 2
        tnew_2 = 0.5 + 0.5 * (1 - np.cos(ksi)) / 2
        tnew = np.hstack((tnew_1, tnew_2))
        y_new = np.array(spl(tnew), dtype=float)

        # Get the new x coordinates from the arc length
        x_new = np.interp(tnew, s, x)

        lower, upper = self.split_sides(x_new, y_new)
        lower, upper = self.remove_nan_points(lower, upper)  # remove nan points
        lower, upper = self.order_points(lower, upper)
        lower, upper = self.close_airfoil(lower, upper)

        self._x_upper = upper[0]
        self._y_upper = upper[1]
        self._x_lower = lower[0]
        self._y_lower = lower[1]
        # update for plot
        self.n_points = n_points
        self.n_upper = self._x_upper.shape[0]
        self.n_lower = self._x_lower.shape[0]

    def repanel_from_internal(
        self,
        n_points: int,
        distribution: str = "cosine",
    ) -> None:
        """Repanels the airfoil to have n_points

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

        _x_upper = self.min_x + (self.max_x - self.min_x) * xsi
        _x_lower = self.min_x + (self.max_x - self.min_x) * xsi
        _y_upper = self.y_upper(xsi)
        _y_lower = self.y_lower(xsi)

        lower = np.array([_x_lower, _y_lower], dtype=float)
        upper = np.array([_x_upper, _y_upper], dtype=float)

        # lower, upper = self.close_airfoil(lower, upper)

        self._x_lower = lower[0]
        self._y_lower = lower[1]
        self._x_upper = upper[0]
        self._y_upper = upper[1]

        self.n_points = n_points
        self.n_upper = self._x_upper.shape[0]
        self.n_lower = self._x_lower.shape[0]

    def to_selig(self) -> Float:
        """Returns the airfoil in the selig format.
        Meaning that the airfoil runs run from the trailing edge, round the leading edge,
        back to the trailing edge in either direction:
        """
        x_points = jnp.hstack((self._x_upper[::-1], self._x_lower))
        y_points = jnp.hstack((self._y_upper[::-1], self._y_lower))
        return jnp.vstack((x_points, y_points))

    @staticmethod
    def order_points(
        lower: Float,
        upper: Float,
    ) -> tuple[Float, Float]:
        """Orders the points of the airfoil so that the upper surface is on top and the lower surface is on the bottom.
        The points are ordered from leading edge to trailing edge.

        Args:
            lower (FloatArray): Lower surface coordinates
            upper (FloatArray): Upper surface coordinates

        Returns:
            tuple[FloatArray, FloatArray]: Ordered lower and upper surface coordinates

        """
        x_lower: Float = lower[0, :]
        y_lower: Float = lower[1, :]
        x_upper: Float = upper[0, :]
        y_upper: Float = upper[1, :]

        # Identify the upper and lower surface leading and trailing edges
        order_low = x_lower[0] > x_lower[-1]
        order_up = x_upper[0] > x_upper[-1]

        # Reorder the points so that the leading edge is at the beginning and the trailing edge is at the end
        x_upper = jnp.where(
            order_up,
            x_upper[::-1],
            x_upper,
        )

        y_upper = jnp.where(
            order_up,
            y_upper[::-1],
            y_upper,
        )

        x_lower = jnp.where(
            order_low,
            x_lower[::-1],
            x_lower,
        )

        y_lower = jnp.where(
            order_low,
            y_lower[::-1],
            y_lower,
        )

        lower = jnp.array([x_lower, y_lower], dtype=float)
        upper = jnp.array([x_upper, y_upper], dtype=float)
        return lower, upper

    @staticmethod
    def close_airfoil(
        lower: Float,
        upper: Float,
    ) -> tuple[Float, Float]:
        """
        Close airfoil by adding points at leading/trailing edges if needed.
        Simpler approach using pre-allocated arrays and conditional filling.
        """
        # Extract edge points
        upper_le = upper[:, 0:1]
        upper_te = upper[:, -1:]
        lower_le = lower[:, 0:1]
        lower_te = lower[:, -1:]

        upper_le_x = upper[0, 0]
        lower_le_x = lower[0, 0]
        upper_te_x = upper[0, -1]
        lower_te_x = lower[0, -1]

        # Determine what needs to be added
        add_lower_le_to_upper = lower_le_x < upper_le_x
        add_upper_le_to_lower = upper_le_x < lower_le_x
        add_lower_te_to_upper = upper_te_x < lower_te_x
        add_upper_te_to_lower = lower_te_x < upper_te_x

        # Create arrays with maximum possible size
        max_upper_cols = upper.shape[1] + 2  # original + potential LE + potential TE
        max_lower_cols = lower.shape[1] + 2

        # Initialize with NaN (or use zeros if you prefer)
        upper_extended = jnp.full((2, max_upper_cols), jnp.nan)
        lower_extended = jnp.full((2, max_lower_cols), jnp.nan)

        # Build upper surface
        upper_start_idx = jnp.where(add_lower_le_to_upper, 1, 0)
        upper_end_idx = upper_start_idx + upper.shape[1]

        # Place original upper surface
        upper_extended = upper_extended.at[:, upper_start_idx:upper_end_idx].set(upper)

        # Conditionally add leading edge
        upper_extended = jnp.where(
            add_lower_le_to_upper,
            upper_extended.at[:, 0:1].set(lower_le),
            upper_extended,
        )

        # Conditionally add trailing edge
        upper_extended = jnp.where(
            add_lower_te_to_upper,
            upper_extended.at[:, upper_end_idx : upper_end_idx + 1].set(lower_te),
            upper_extended,
        )

        # Build lower surface
        lower_start_idx = jnp.where(add_upper_le_to_lower, 1, 0)
        lower_end_idx = lower_start_idx + lower.shape[1]

        # Place original lower surface
        lower_extended = lower_extended.at[:, lower_start_idx:lower_end_idx].set(lower)

        # Conditionally add leading edge
        lower_extended = jnp.where(
            add_upper_le_to_lower,
            lower_extended.at[:, 0:1].set(upper_le),
            lower_extended,
        )

        # Conditionally add trailing edge
        lower_extended = jnp.where(
            add_upper_te_to_lower,
            lower_extended.at[:, lower_end_idx : lower_end_idx + 1].set(upper_te),
            lower_extended,
        )

        # Calculate actual sizes
        upper_actual_size = (
            upper.shape[1]
            + jnp.where(add_lower_le_to_upper, 1, 0)
            + jnp.where(add_lower_te_to_upper, 1, 0)
        )
        lower_actual_size = (
            lower.shape[1]
            + jnp.where(add_upper_le_to_lower, 1, 0)
            + jnp.where(add_upper_te_to_lower, 1, 0)
        )

        # Trim to actual sizes
        upper_final = upper_extended[:, :upper_actual_size]
        lower_final = lower_extended[:, :lower_actual_size]

        return lower_final, upper_final

    @staticmethod
    def remove_nan_points(lower: Float, upper: Float) -> tuple[Float, Float]:
        """Removes nan points from the airfoil"""
        upper = upper[:, ~jnp.isnan(upper[0, :]) | ~jnp.isnan(upper[1, :])]
        lower = lower[:, ~jnp.isnan(lower[0, :]) | ~jnp.isnan(lower[1, :])]
        return lower, upper

    def thickness(self, ksi: Float) -> Float:
        """Returns the thickness of the airfoil at the given x coordinates

        Args:
            Ksi (Float): Normalized x coordinates in the range [0, 1]

        Returns:
            Float: Thickness of the airfoil at the given ksi coordinates

        """
        thickness: Float = self.y_upper(ksi) - self.y_lower(ksi)
        # Remove Nan
        thickness = thickness[~jnp.isnan(thickness)]

        # Set 0 thickness for values after x_max
        thickness = thickness.at[ksi > self.max_x].set(0.0)
        return thickness

    @property
    def max_thickness(self) -> float:
        """Returns the maximum thickness of the airfoil

        Returns:
            float: Maximum thickness

        """
        thickness: FloatArray = self.thickness(np.linspace(0, 1, self.n_points))
        return np.max(thickness).astype(float)

    @property
    def max_thickness_location(self) -> float:
        """Returns the location of the maximum thickness of the airfoil

        Returns:
            float: Location of the maximum thickness

        """
        thickness: FloatArray = self.thickness(np.linspace(0, 1, self.n_points))
        return float(np.argmax(thickness) / self.n_points)

    @property
    def file_name(self) -> str:
        """Returns the file name of the airfoil"""
        if self.name is not None:
            return self.name
        return "Airfoil.dat"

    @staticmethod
    def split_sides(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
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
        airfoil1: Airfoil,
        airfoil2: Airfoil,
        eta: float,
        n_points: int,
    ) -> Airfoil:
        """Returns a new airfoil morphed between two airfoils

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
            raise ValueError(
                f"'eta' must be in range [0,1], given eta is {float(eta):.3f}",
            )
        # Round to 2 decimals
        eta = round(eta, 2)
        if eta == 0.0 or eta == 1.0:
            return airfoil1

        ksi = np.linspace(0, np.pi, n_points // 2)
        x = 0.5 * (1.0 - np.cos(ksi))

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
        # Create the name
        # Round eta to 2 decimals in string format
        eta_perc = int(eta * 100)
        eta_str = f"{eta_perc}"

        if airfoil1.name.startswith("morphed_"):
            # Check if the airfoils are coming from the same morphing
            airfoil1_parent_1 = airfoil1.name.split("_")[1]
            airfoil1_parent_2 = airfoil1.name.split("_")[2]
            airfoil1_eta = float(airfoil1.name.split("_")[4][:-1]) / 100
        else:
            airfoil1_parent_1 = airfoil1.name
            airfoil1_parent_2 = None
            airfoil1_eta = None

        if airfoil2.name.startswith("morphed_"):
            airfoil2_parent_1 = airfoil2.name.split("_")[1]
            airfoil2_parent_2 = airfoil2.name.split("_")[2]
            airfoil2_eta = float(airfoil2.name.split("_")[4][:-1]) / 100
        else:
            airfoil2_parent_1 = None
            airfoil2_parent_2 = airfoil2.name
            airfoil2_eta = None

        name = f"morphed_{airfoil1.name}_{airfoil2.name}_at_{eta_str}%"
        airfoil_parents = {
            airfoil1_parent_1,
            airfoil1_parent_2,
            airfoil2_parent_1,
            airfoil2_parent_2,
        }
        # Remove None values
        airfoil_parents = {x for x in airfoil_parents if x is not None}
        if len(airfoil_parents) == 1:
            return airfoil1
        # Remove None values
        airfoil_parents = {x for x in airfoil_parents if x is not None}
        if len(airfoil_parents) <= 2:
            # If the airfoils are coming from the same morphing
            if airfoil1_eta is not None:
                if airfoil2_eta is not None:
                    new_eta = airfoil1_eta * (1 - eta) + (airfoil2_eta) * (eta)
                else:  # airfoil2_eta is None:
                    new_eta = airfoil1_eta * (1 - eta)
            else:
                if airfoil2_eta is None:
                    new_eta = eta
                else:  # airfoil2_eta is not None:
                    new_eta = (airfoil2_eta) * (eta)

            # ROUND TO 2 DECIMALS
            new_eta = int(100 * new_eta)
            # Round to 2 decimals in string format
            new_eta_str = f"{new_eta}"
            name = f"morphed_{airfoil1_parent_1}_{airfoil2_parent_2}_at_{new_eta_str}%"
        return cls(upper, lower, name)

    @classmethod
    def naca(cls, naca: str, n_points: int = 200) -> Airfoil:
        """Initialize the Airfoil class from a NACA 4 digit identifier.

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
            from .naca5 import NACA5

            L = int(naca[0])
            P = int(naca[1])
            Q = int(naca[2])
            assert Q in [0, 1], "Q must be 0 or 1"

            XX = int(naca[3:5])
            naca5 = NACA5(L=L, P=P, Q=Q, XX=XX, n_points=n_points)
            return naca5
        if re_4digits.match(naca):
            from .naca4 import NACA4

            m = int(naca[0]) / 100
            p = int(naca[1]) / 10
            xx = int(naca[2:4]) / 100
            naca4 = NACA4(M=m, P=p, XX=xx, n_points=n_points)
            return naca4
        raise NACADefintionError(
            "Identifier not recognised as valid NACA 4 definition",
        )

    @classmethod
    def from_file(cls, filename: str) -> Airfoil:
        """Initialize the Airfoil class from a file.

        Args:
            filename (str): Name of the file to load the airfoil from

        Returns:
            Airfoil: Airfoil class object

        """
        x: list[float] = []
        y: list[float] = []

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
        x_arr = np.array(x)
        y_arr = np.array(y)
        lower, upper = cls.split_sides(x_arr, y_arr)
        try:
            self: Airfoil = cls(upper, lower, os.path.split(filename)[-1])
        except ValueError as e:
            print(f"Error loading airfoil from {filename}")
            raise (ValueError(e))
        return self

    def flap(
        self,
        flap_hinge_chord_percentage: float,
        flap_angle: float,
        flap_hinge_thickness_percentage: float = 0.5,
        chord_extension: float = 1,
    ) -> FlappedAirfoil | Airfoil:
        """Function to generate a flapped airfoil. The flap is defined by the flap hinge, the chord extension and the flap angle.

        Args:
            flap_hinge (float): Chordwise location of the flap hinge
            chord_extension (float): Chord extension of the flap
            flap_angle (float): Angle of the flap

        Returns:
            Airfoil: Flapped airfoil

        """
        flap_hinge_1 = (
            flap_hinge_chord_percentage * (self.max_x - self.min_x) + self.min_x
        )
        if flap_angle == 0 or flap_hinge_1 == 1.0:
            return self
        theta = np.deg2rad(flap_angle)

        x = np.linspace(self.min_x, flap_hinge_1, self.n_points)
        y_upper = self.y_upper(x)
        x_upper = x
        x_lower = x
        y_lower = self.y_lower(x)

        x_after_flap = np.linspace(flap_hinge_1, self.max_x, self.n_points)
        x_lower_after_flap = x_after_flap
        x_upper_after_flap = x_after_flap
        y_lower_after_flap = self.y_lower(x_after_flap)
        y_upper_after_flap = self.y_upper(x_after_flap)

        # Translate the points to the origin to rotate them
        x_lower_after_flap = x_lower_after_flap - flap_hinge_1
        x_upper_after_flap = x_upper_after_flap - flap_hinge_1

        hinge_upper = self.y_lower(flap_hinge_1)
        hinge_lower = self.y_upper(flap_hinge_1)
        eta = flap_hinge_thickness_percentage
        y_hinge = hinge_upper * (eta) + hinge_lower * (1 - eta)

        y_lower_after_flap = y_lower_after_flap - y_hinge
        y_upper_after_flap = y_upper_after_flap - y_hinge

        # Stretch the points so all points move the same amount
        x_lower_after_flap = x_lower_after_flap * (chord_extension)
        x_upper_after_flap = x_upper_after_flap * (chord_extension)

        # Rotate the points according to the hinge (located on the lower side)
        x = x_lower_after_flap
        y = y_lower_after_flap
        x_lower_after_flap = x * np.cos(theta) - y * np.sin(theta)
        y_lower_after_flap = x * np.sin(theta) + y * np.cos(theta)

        x = x_upper_after_flap
        y = y_upper_after_flap
        x_upper_after_flap = x * np.cos(theta) - y * np.sin(theta)
        y_upper_after_flap = x * np.sin(theta) + y * np.cos(theta)

        # Translate the points back
        x_lower_after_flap = x_lower_after_flap + flap_hinge_1
        x_upper_after_flap = x_upper_after_flap + flap_hinge_1
        y_lower_after_flap = y_lower_after_flap + y_hinge
        y_upper_after_flap = y_upper_after_flap + y_hinge

        # Remove the points where x < x_hinge
        problematic_indices = np.where(x_lower_after_flap < flap_hinge_1)
        x_lower_after_flap = np.delete(x_lower_after_flap, problematic_indices)
        y_lower_after_flap = np.delete(y_lower_after_flap, problematic_indices)

        problematic_indices = np.where(x_upper_after_flap < flap_hinge_1)
        x_upper_after_flap = np.delete(x_upper_after_flap, problematic_indices)
        y_upper_after_flap = np.delete(y_upper_after_flap, problematic_indices)

        # TODO: Add points in the upper surface to smooth the flap
        upper: FloatArray = np.array(
            [
                [*x_upper, *x_upper_after_flap],
                [*y_upper, *y_upper_after_flap],
            ],
        )
        lower: FloatArray = np.array(
            [
                [*x_lower, *x_lower_after_flap],
                [*y_lower, *y_lower_after_flap],
            ],
        )
        from .flapped_airfoil import FlappedAirfoil

        flapped = FlappedAirfoil(
            upper,
            lower,
            name=f"{self.name}_flapped_hinge_{flap_hinge_chord_percentage:.2f}_deflection_{flap_angle:.2f}",
            parent=self,
        )

        return flapped

    def flap_camber_line(
        self,
        flap_hinge: float,
        flap_angle: float,
        chord_extension: float = 1,
        # plot_flap: bool = False
    ) -> Airfoil:
        if flap_angle == 0 or flap_hinge == 1.0:
            return self
        flap_angle = np.deg2rad(flap_angle)

        n = self.n_points
        eta = (flap_hinge - self.min_x) / (self.max_x - self.min_x)
        n1 = int(n * eta)
        n2 = n - n1

        x_after = np.linspace(flap_hinge, self.max_x, n2)
        x_before = np.linspace(self.min_x, flap_hinge, n1)

        y_hinge = self.camber_line(flap_hinge)
        y_after = self.camber_line(x_after)

        y = y_after - y_hinge
        x = x_after - flap_hinge
        xnew = x * np.cos(flap_angle) - y * np.sin(flap_angle)
        ynew = x * np.sin(flap_angle) + y * np.cos(flap_angle)
        xnew += flap_hinge
        ynew += y_hinge

        # We need to take the xnew,ynew line and add thickness to both sides in the direction
        # of the normal to the camber line at each point. We can get the normal by taking the
        # derivative of the camber line. The normal will be the negative reciprocal of the
        # derivative. We can then add the thickness in the direction of the normal to get the
        # final points
        thickess = self.thickness(x_after)
        spacing = np.hstack((0, np.diff(y_after)))
        angle = np.arctan(np.gradient(x_after, spacing, edge_order=1))
        lower_y = ynew - np.sin(angle + flap_angle) * thickess / 2
        lower_x = xnew - np.cos(angle + flap_angle) * thickess / 2

        upper_y = ynew + np.sin(angle + flap_angle) * thickess / 2
        upper_x = xnew + np.cos(angle + flap_angle) * thickess / 2

        # Identiffy the problematic regions
        # Problematic are the regions of the first point of both the upper and lower surface
        # until the hinge point. We need to fill the gap that arises on one side and close
        # the gap that arises on the other side

        # Remove the points where x < x_hinge
        problematic_indices = np.where(upper_x < flap_hinge)
        upper_x = np.delete(upper_x, problematic_indices)
        upper_y = np.delete(upper_y, problematic_indices)

        problematic_indices = np.where(lower_x < flap_hinge)
        lower_x = np.delete(lower_x, problematic_indices)
        lower_y = np.delete(lower_y, problematic_indices)

        # Plot the original camber line
        y_upper_before = self.y_upper(x_before)
        y_lower_before = self.y_lower(x_before)

        x_upper = np.concatenate((x_before, upper_x))
        y_upper = np.concatenate((y_upper_before, upper_y))
        x_lower = np.concatenate((x_before, lower_x))
        y_lower = np.concatenate((y_lower_before, lower_y))

        # # Add the trailing edge point
        # x_te = self.max_x
        # y_te = self.camber_line(x_te)

        # # Rotate the trailing edge point
        # x = x_te - flap_hinge
        # y = y_te - y_hinge

        # x_te = x * np.cos(flap_angle) - y * np.sin(flap_angle)
        # y_te = x * np.sin(flap_angle) + y * np.cos(flap_angle)

        # x_te += flap_hinge
        # y_te += y_hinge

        # x_upper = np.concatenate((x_upper, [x_te]))
        # y_upper = np.concatenate((y_upper, [y_te]))

        upper = np.vstack([x_upper, y_upper])
        lower = np.vstack([x_lower, y_lower])

        # if plot_flap:
        #     y_camber_before = self.camber_line(x_before)
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.vlines(0.75, -1, 1)
        #     ax.scatter(upper_x, upper_y, color = 'red', s = 0.5)
        #     ax.scatter(lower_x, lower_y, color = 'black', s = 0.5)
        #     ax.plot(xnew, ynew, label="Flapped Camber Line", color = 'green')
        #     circle = plt.Circle((flap_hinge, y_hinge), thickess[0]/2, color='blue', fill=False)
        #     ax.add_artist(circle)
        #     # Make lines from upper_x, upper_y to xnew, ynew
        #     # for i in range(len(xnew)):
        #     #     ax.plot([xnew[i], upper_x[i]], [ynew[i], upper_y[i]], color = 'red', linewidth = 0.5)
        #     #     ax.plot([xnew[i], lower_x[i]], [ynew[i], lower_y[i]], color = 'black', linewidth = 0.5)
        #     ax.plot(x_before, y_camber_before, label="Original Camber Line", color = 'blue')
        #     ax.plot(x_before, y_upper_before, label="Original Upper Surface", color = 'red')
        #     ax.plot(x_before, y_lower_before, label="Original Lower Surface", color = 'black')
        #     ax.legend()
        #     ax.relim()
        #     ax.autoscale_view()
        #     ax.set_ylim(-0.1,0.1)
        #     ax.set_aspect('equal', 'box')
        #     plt.show()

        from .flapped_airfoil import FlappedAirfoil

        flapped = FlappedAirfoil(
            upper,
            lower,
            name=f"{self.name}_flapped_hinge_{flap_hinge:.2f}_deflection_{np.rad2deg(flap_angle):.2f}",
            parent=self,
        )
        return flapped

    def camber_line(self, points: float | list[float] | FloatArray) -> FloatArray:
        """Returns the camber line for a given set of x coordinates

        Args:
            points (float | list[float] | FloatArray): X coordinates

        Returns:
            FloatArray: Camber line

        """
        return np.array((self.y_upper(points) + self.y_lower(points)) / 2, dtype=float)

    @classmethod
    def load_from_web(cls, name: str) -> Airfoil:
        """Fetches the airfoil data from the web. Specifically from the UIUC airfoil database."""
        db_url = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
        base_url = "https://m-selig.ae.illinois.edu/ads/"
        response = requests.get(db_url)
        if response.status_code == 200:
            # Find all lines containing .dat filenames
            lines = response.text.split("\n")
            filenames = []
            for line in lines:
                match = re.search(r'href="(.*?)\.dat"', line)
                if match:
                    filenames.append(f"{match.group(1)}.dat")

            for filename in filenames:
                download_url = base_url + filename

                # Get the Airfoil name from the filename
                airfoil_name = filename.split(".")[0].split("/")[-1]
                if airfoil_name.upper() != name.upper():
                    continue

                # Download the file (handle potential errors)
                try:
                    response = requests.get(download_url)
                    if response.status_code == 200:
                        # Remove the .dat extension
                        filename = airfoil_name.lower()
                        # Save the downloaded data locally with the filename
                        dirname = airfoil_name.upper()

                        from ICARUS.database import Database

                        DB = Database.get_instance()
                        DB2D = DB.DB2D

                        os.makedirs(os.path.join(DB2D, dirname), exist_ok=True)
                        filename = os.path.join(DB2D, dirname, filename)
                        with open(filename, "wb") as f:
                            f.write(response.content)
                        print(
                            f"Downloaded: {filename} from {download_url}. Creating Airfoil obj...",
                        )
                        return cls.from_file(filename)
                    raise FileNotFoundError(
                        f"Error downloading {filename}: {response.status_code}",
                    )
                except requests.exceptions.RequestException as e:
                    raise FileNotFoundError(f"Error downloading {filename}: {e}")
        raise FileNotFoundError(f"Error fetching {db_url}: {response.status_code}")

    def save_selig(
        self,
        directory: str | None = None,
        header: bool = False,
        inverse: bool = False,
    ) -> None:
        """Saves the airfoil in the selig format.

        Args:
            directory (str, optional): Directory to save the airfoil. Defaults to None.
            header (bool, optional): Whether to include the header. Defaults to False.
            inverse (bool, optional): Whether to save the airfoil in the reverse selig format. Defaults to False.

        """
        if directory is not None:
            file_name = os.path.join(directory, self.file_name)
        else:
            file_name = self.file_name

        with open(file_name, "w") as file:
            if header:
                file.write(f"{self.name} with {self.n_points}\n")

            pts = self.to_selig()
            if inverse:
                pts = pts[:, ::-1]

            x = pts[0, :]
            y = pts[1, :]
            # Remove NaN values and duplicates
            x_arr = np.array(x)
            y_arr = np.array(y)

            # Combine x and y coordinates into a single array of complex numbers
            complex_coords = x_arr + 1j * y_arr
            # Round all the values to 6 decimals
            complex_coords = np.round(complex_coords, 5)

            # Find unique complex coordinates
            unique_indices = np.sort(np.unique(complex_coords, return_index=True)[1])

            # Use the unique indices to get the unique x and y coordinates
            x_clean = x_arr[unique_indices]
            y_clean = y_arr[unique_indices]

            # Remove NaN values
            idx_nan = np.isnan(x_clean) | np.isnan(y_clean)
            x_clean = x_clean[~idx_nan]
            y_clean = y_clean[~idx_nan]

            for x, y in zip(x_clean, y_clean):
                file.write(f"{x:.6f} {y:.6f}\n")

    def save_le(
        self,
        directory: str | None = None,
        header: bool = False,
    ) -> None:
        """Saves the airfoil in the revese selig format.

        Args:
            directory (str, optional): Directory to save the airfoil. Defaults to None.

        """
        x = [*self._x_lower[:], *self._x_upper[::-1]]
        y = [*self._y_lower[:], *self._y_upper[::-1]]

        pts = np.vstack((x, y))
        if directory is not None:
            # If directory does not exist, create it
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_name = os.path.join(directory, self.file_name)
        else:
            file_name = self.file_name

        with open(file_name, "w") as file:
            if header:
                file.write(f"{self.name} with {self.n_points}\n")
            for x, y in pts.T:
                file.write(f" {x:.6f} {y:.6f}\n")

    def plot(
        self,
        camber: bool = False,
        scatter: bool = False,
        max_thickness: bool = False,
        ax: Axes | None = None,
        overide_color: str | None = None,
    ) -> None:
        """Plots the airfoil in the selig format

        Args:
            camber (bool, optional): Whether to plot the camber line. Defaults to False.
            scatter (bool, optional): Whether to plot the airfoil as a scatter plot. Defaults to False.
            max_thickness (bool, optional): Whether to plot the max thickness. Defaults to False.

        """
        pts = self.to_selig()
        x, y = pts

        if ax is None:
            fig, _ax = plt.subplots()
        else:
            _ax = ax

        if scatter:
            _ax.scatter(x[: self.n_upper], y[: self.n_upper], s=1)
            _ax.scatter(x[self.n_upper :], y[self.n_upper :], s=1)
        else:
            if overide_color is not None:
                col_up = overide_color
                col_lo = overide_color
            else:
                col_up = "r"
                col_lo = "b"

            _ax.plot(x[: self.n_upper], y[: self.n_upper], col_up)
            _ax.plot(x[self.n_upper :], y[self.n_upper :], col_lo)

        if camber:
            x_min = np.min(x)
            x_max = np.max(x)
            x = np.linspace(x_min, x_max, 100)
            y = self.camber_line(x)
            _ax.plot(x, y, "k--")

        if max_thickness:
            x = self.max_thickness_location
            thick = self.max_thickness
            y_up = self.y_upper(x)
            y_lo = self.y_lower(x)
            # Plot a line from the upper to the lower surface
            _ax.plot([x, x], [y_up, y_lo], "k--")
            # Add a text with the thickness
            _ax.text(x, y_lo, f"{thick:.3f}", ha="right", va="bottom")
        _ax.axis("scaled")
        _ax.set_title(f"Airfoil {self.name}")

    def __repr__(self) -> str:
        """Returns the string representation of the airfoil

        Returns:
            str: String representation of the airfoil

        """
        return self.__str__()

    def __str__(self) -> str:
        """Returns the string representation of the airfoil

        Returns:
            str: String representation of the airfoil

        """
        if not hasattr(self, "name") or self.name is None:
            return f"Airfoil with ({len(self._x_lower)} x {len(self._x_upper)}) points"
        return f"Airfoil: {self.name} with ({len(self._x_lower)} x {len(self._x_upper)}) points"

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Sets the state of the airfoil

        Args:
            state (dict[str, Any]): State of the airfoil

        """
        upper = state["upper"]
        lower = state["lower"]
        # Convert to numpy arrays
        upper = np.array(upper, dtype=float)
        lower = np.array(lower, dtype=float)

        Airfoil.__init__(
            self,
            name=state["name"],
            upper=upper,
            lower=lower,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Returns the state of the airfoil

        Returns:
            dict[str, Any]: State of the airfoil

        """
        return {
            "name": self.name,
            "upper": np.vstack((self._x_upper, self._y_upper), dtype=float).tolist(),
            "lower": np.vstack((self._x_lower, self._y_lower), dtype=float).tolist(),
        }

    def tree_flatten(self):
        x_upper = jnp.array(self._x_upper, dtype=float)
        y_upper = jnp.array(self._y_upper, dtype=float)
        x_lower = jnp.array(self._x_lower, dtype=float)
        y_lower = jnp.array(self._y_lower, dtype=float)
        name = self.name

        return ((x_lower, y_lower, x_upper, y_upper), (name))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            lower=jnp.array(children[:2], dtype=float),
            upper=jnp.array(children[2:4], dtype=float),
            name=aux_data[0] if aux_data else None,
        )
