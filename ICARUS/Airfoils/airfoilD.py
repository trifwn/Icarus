import os
import re
import shutil
import urllib.request
from typing import Any
from typing import Callable

import airfoils as af
import matplotlib.pyplot as plt
import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray
from pandas import DataFrame

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
        link: str = (
            "https://m-selig.ae.illinois.edu/ads/coord/naca" + self.name + ".dat"
        )
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
        self.selig2: FloatArray = np.vstack((x, y))

    def access_db(self, HOMEDIR: str, DBDIR: str) -> None:
        """
        Connection to Database. Saves the airfoil in the database.
        TODO: TO BE DEPRECATED. It should be handled by the solver

        Args:
            HOMEDIR (str): Home directory
            DBDIR (str): Database directory
        """
        os.chdir(DBDIR)
        AFDIR: str = f"NACA{self.name}"
        os.makedirs(AFDIR, exist_ok=True)
        os.chdir(AFDIR)
        self.AFDIR: str = os.getcwd()
        self.HOMEDIR: str = HOMEDIR
        self.DBDIR: str = DBDIR
        os.chdir(HOMEDIR)
        exists = False
        for i in os.listdir(self.AFDIR):
            if i.startswith("naca"):
                self.airfile = os.path.join(self.AFDIR, i)
                exists = True
        if exists:
            self.save()

    def set_reynolds_case(self, reynolds_num: float) -> None:
        """
        Set current reynolds number and create a directory for it.
        TODO: TO BE DEPRECATED. It should be handled by the database and solver

        Args:
            Reynolds_number (float): _description_
        """
        reynolds_str: str = np.format_float_scientific(
            reynolds_num,
            sign=False,
            precision=3,
        )
        self.current_reynolds: str = reynolds_str
        self.reynolds_ids.append(self.current_reynolds)
        if self.current_reynolds not in self.polars.keys():
            self.polars[self.current_reynolds] = {}

        try:
            self.REYNDIR: str = os.path.join(
                self.AFDIR,
                f"Reynolds_{reynolds_str.replace('+', '')}",
            )
            os.makedirs(self.REYNDIR, exist_ok=True)
            shutil.copy(self.airfile, self.REYNDIR)
        except AttributeError:
            print("DATABASE is not initialized!")

    def save(self) -> None:
        """
        Saves the airfoil in the selig format.
        """
        self.airfile = os.path.join(self.AFDIR, f"naca{self.name}")
        pt0 = self.selig
        np.savetxt(self.airfile, pt0.T)

    def plot(self) -> None:
        """
        Plots the airfoil in the selig format
        """
        pts = self.selig
        x, y = pts
        plt.plot(x[: self.n_points], y[: self.n_points], "r")
        plt.plot(x[self.n_points :], y[self.n_points :], "b")
        plt.axis("scaled")

    def solver_run(
        self,
        solver: Callable[..., None],
        args: list[Any],
        kwargs: dict[str, Any] = {},
    ) -> None:
        """
        Runs the solver.
        TODO: TO BE DEPRECATED. It should be handled by the solver and analysis class

        Args:
            solver (_type_): Solver function
            args (_type_): Positional arguments for the solver
            kwargs (dict, optional): Keyword Arguments for the solver . Defaults to empty {}.
        """
        solver(*args, **kwargs)

    def solver_setup(
        self,
        setupsolver: Callable[..., None],
        args: list[Any],
        kwargs: dict[str, Any] = {},
    ) -> None:
        """
        Runs the solver setup (hook) function.
        TODO: TO BE DEPRECATED. It should be handled by the solver and analysis class

        Args:
            setupsolver (Callable[..., None]): Solver setup function
            args (list[Any]): Solver setup function positional arguments
            kwargs (dict[str, Any], optional): Solver Setup kwargs . Defaults to empty {}.
        """
        setupsolver(*args, **kwargs)

    def clean_results(
        self,
        clean_function: Callable[..., None],
        args: list[Any],
        kwargs: dict[str, Any] = {},
    ) -> None:
        """
        Cleans the results of the Analysis.
        TODO: TO BE DEPRECATED. It should be handled by the solver and analysis class

        Args:
            clean_function (_type_): Function to clean the results
            args (_type_): Positional arguments for the clean function
            kwargs (dict, optional): Keyword arguements for the clean function . Defaults to empty {}.
        """
        clean_function(*args, **kwargs)

    def make_polars(
        self,
        make_polars_function: Callable[..., DataFrame],
        solver_name: str,
        args: list[Any],
        kwargs: dict[str, Any] = {},
    ) -> DataFrame:
        """
        Function to make the polars for a given analysis.
        TODO: TO BE DEPRECATED. It should be handled by the solver and analysis class

        Args:
            make_polars_function (Callable[..., DataFrame]): Function to make the polars
            solver_name (str): Solver name
            args (list[Any]): Positional arguments for the make_polars function
            kwargs (dict[str, Any], optional): Keyword Arguements for the function . Defaults to {}.
        """
        polars_df: DataFrame = make_polars_function(*args, **kwargs)
        self.polars[self.current_reynolds][solver_name] = polars_df
        return polars_df
