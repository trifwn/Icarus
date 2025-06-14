from __future__ import annotations

import logging
import os
import re
import shutil
from time import sleep
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame

from ICARUS.airfoils import Airfoil
from ICARUS.airfoils import AirfoilData
from ICARUS.airfoils import AirfoilPolars
from ICARUS.airfoils import PolarNotAccurate
from ICARUS.airfoils import ReynoldsNotIncluded
from ICARUS.core.base_types import Struct
from ICARUS.database.utils import angle_to_directory

if TYPE_CHECKING:
    from ICARUS.core.types import FloatArray


class AirfoilNotFoundError(Exception):
    """Exception raised when an airfoil is not found in the database."""

    def __init__(self, airfoil_name: str) -> None:
        """Initialize the AirfoilNotFoundError class.

        Args:
            airfoil_name (str): Airfoil name

        """
        message = f"Airfoil {airfoil_name} not found in database!"
        super().__init__(message)


class PolarsNotFoundError(Exception):
    """Exception raised when polars are not found in the database."""

    def __init__(
        self,
        airfoil_name: str,
        solver: str | None = None,
        solvers_found: list[str] | None = None,
    ) -> None:
        """Initialize the PolarsNotFoundError class.

        Args:
            airfoil_name (str): Airfoil name
            solver (str): Solver name

        """
        if solver:
            message = f"Polars for airfoil {airfoil_name} not found in database for solver {solver}! The available solvers are {solvers_found}"
        else:
            message = f"Polars for airfoil {airfoil_name} not found in database!"
        super().__init__(message)


class Database_2D:
    """Database class to store 2d simulation objects (airfoils), analyses and results (polars)."""

    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Database_2D:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> Database_2D:
        if cls._instance is None:
            raise ValueError("Database_2D not initialized!")
        return cls._instance

    def __init__(
        self,
        location: str,
        EXTERNAL_DB: str | None = None,
    ) -> None:
        """Initialize the Database_2D class."""
        self.DB2D: str = location
        if not os.path.isdir(self.DB2D):
            os.makedirs(self.DB2D)

        # !TODO: Make data private
        self.polars: dict[str, AirfoilData] = {}
        self.airfoils = Struct()

        self.EXTERNAL_DB = EXTERNAL_DB

    def get_airfoil_names(self) -> list[str]:
        """Returns the available airfoils in the database.

        Returns:
            list[str]: List of airfoil names

        """
        return list(self.airfoils.keys())

    def get_airfoil(self, airfoil_name: str) -> Airfoil:
        """Returns the airfoil object from the database.

        Args:
            name (str): Airfoil name

        Returns:
            Airfoil: Airfoil object

        """
        airfoil_name = airfoil_name.upper()
        try:
            airf: Airfoil = self.airfoils[airfoil_name]
            return airf
        except KeyError:
            try:
                # Try to load the airfoil from the DB or EXTERNAL DB
                self.add_airfoil(airfoil_name)
                airf = self.airfoils[airfoil_name]
                return airf
            except FileNotFoundError:
                raise AirfoilNotFoundError(airfoil_name)

    def get_polars(self, airfoil: str | Airfoil, solver: str | None = None) -> AirfoilPolars:
        """Returns the polars object from the database.

        Args:
            airfoil_name (str): Airfoil name
            solver (str): Solver name

        Returns:
            Polars: Polars object

        """
        if isinstance(airfoil, str):
            airfoil = self.get_airfoil(airfoil)

        airfoil_name = airfoil.name.upper()
        if airfoil_name not in self.polars.keys():
            try:
                # Try to load the airfoil from the DB or EXTERNAL DB
                self.load_airfoil_data(airfoil_name)
                if airfoil_name not in self.polars.keys():
                    raise PolarsNotFoundError(airfoil_name)
            except (FileNotFoundError, StopIteration):
                raise AirfoilNotFoundError(airfoil_name)

        airfoil_data: AirfoilData = self.polars[airfoil_name]
        try:
            polar = airfoil_data.get_polars(solver=solver)
        except KeyError:
            raise PolarsNotFoundError(airfoil_name, solver, list(airfoil_data.polars.keys()))
        return polar

    def get_airfoil_data(self, airfoil: str | Airfoil) -> AirfoilData:
        """Returns the solvers available for a given airfoil.

        Args:
            airfoil_name (str): Airfoil name

        Returns:
            list[str]: List of solver names

        """
        if isinstance(airfoil, str):
            airfoil_name = airfoil.upper()
        elif isinstance(airfoil, Airfoil):
            airfoil_name = airfoil.name.upper()
        else:
            raise ValueError("airfoil must be a string or an Airfoil object")

        if airfoil_name not in self.airfoils.keys():
            self.add_airfoil(airfoil_name)

        airfoil_name = airfoil_name.upper()
        if airfoil_name.upper() not in self.polars.keys():
            try:
                # Try to load the airfoil from the DB or EXTERNAL DB
                self.load_airfoil_data(airfoil_name.upper())
                if airfoil_name.upper() not in self.polars.keys():
                    raise PolarsNotFoundError(airfoil_name)
            except (FileNotFoundError, StopIteration):
                raise AirfoilNotFoundError(airfoil_name)

        airfoil_data: AirfoilData = self.polars[airfoil_name.upper()]
        return airfoil_data

    def compute_polars(
        self,
        airfoil: Airfoil,
        reynolds: list[float] | FloatArray,
        angles: list[float] | FloatArray,
        solver_name: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str = "Xfoil",
        trips: tuple[float, float] = (0.3, 0.3),
    ) -> None:
        """Computes the polars for an airfoil at a given reynolds number and angles of attack.

        Args:
            airfoil (Airfoil): Airfoil object
            reynolds (float): Reynolds number
            angles (list[float]): List of angles of attack

        """
        if isinstance(reynolds, float):
            reynolds = [reynolds]
        airfoil.repanel_spl(400)

        from .compute_airfoil_polars import compute_airfoil_polars

        compute_airfoil_polars(
            airfoil=airfoil,
            reynolds_numbers=reynolds,
            aoas=angles,
            solver_name=solver_name,
            mach=0.0,
            plot_polars=False,
            repanel=110,
            trips=trips,
        )

    def get_or_compute_polars(
        self,
        airfoil: Airfoil,
        reynolds: float,
        aoa: list[float] | FloatArray,
        solver_name: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str = "Xfoil",
        REYNOLDS_BINS: list[float] | FloatArray | None = None,
    ) -> AirfoilPolars:
        if REYNOLDS_BINS is None:
            RE_MIN = 1e4
            RE_MAX = 1e8
            BINS_PER_DECADE = 5
            NUM_BINS = BINS_PER_DECADE * int(np.log10(RE_MAX / RE_MIN))
            REYNOLDS_BINS = np.logspace(np.log10(RE_MIN), np.log10(RE_MAX), NUM_BINS)
        else:
            NUM_BINS = len(REYNOLDS_BINS)
        assert REYNOLDS_BINS is not None

        try:
            polars = self.get_polars(airfoil.name, solver=solver_name)
            reyns_computed = polars.reynolds_nums
            # Check if the reynolds number is within the range of the computed polars
            # To be within the range of the computed polars the reynolds number must be
            # reyns_computed[i] - DR_REYNOLDS[matching] < reynolds_wanted < reyns_computed[i] + DR_REYNOLDS[matching]
            # If the reynolds number is not within the range of the computed polars, the polar is recomputed

            # Find the bin corresponding to the each computed reynolds number
            reyns_bin = int(np.digitize(x=reynolds, bins=REYNOLDS_BINS, right=True))
            reyns_bin = max(reyns_bin, 1)
            reyns_bin = min(reyns_bin, NUM_BINS - 1)

            reynolds_found = False
            DR_REYNOLDS = np.diff(REYNOLDS_BINS)
            if not (reyns_bin == 0 or reyns_bin == NUM_BINS):
                tolerance = DR_REYNOLDS[reyns_bin] / 2
                for computed_reyn in reyns_computed:
                    if abs(computed_reyn - reynolds) < tolerance:
                        reynolds_found = True

            if not reynolds_found:
                self.compute_polars(
                    airfoil=airfoil,
                    solver_name=solver_name,
                    reynolds=[REYNOLDS_BINS[reyns_bin - 1], REYNOLDS_BINS[reyns_bin]],
                    angles=aoa,
                    trips=(1.0, 1.0),
                )
            return self.get_polars(airfoil.name, solver=solver_name)
        except (
            PolarsNotFoundError,
            AirfoilNotFoundError,
            PolarNotAccurate,
            ReynoldsNotIncluded,
            FileNotFoundError,
        ):
            print(
                f"\tPolar for {airfoil.name} not found in database. Trying to recompute with stricter trip conditions...",
            )
            self.compute_polars(
                airfoil=airfoil,
                solver_name=solver_name,
                reynolds=[reynolds],
                angles=aoa,
                trips=(0.3, 0.3),
            )
            try:
                return self.get_polars(airfoil.name, solver=solver_name)
            except PolarsNotFoundError:
                print(
                    f"\tPolar for {airfoil.name} not found in database. Trying to recompute with even stricter trip conditions...",
                )
                self.compute_polars(
                    airfoil=airfoil,
                    solver_name=solver_name,
                    reynolds=[reynolds],
                    angles=aoa,
                    trips=(0.01, 0.1),
                )
                return self.get_polars(airfoil.name, solver=solver_name)

    def load_all_data(self) -> None:
        """Scans the filesystem and load all the data.
        Scans the filesystem and loads data if not already loaded.
        """
        # Accessing Database Directory
        if not os.path.isdir(self.DB2D):
            print(f"Creating self.DB2D directory at {self.DB2D}...")
            os.makedirs(self.DB2D, exist_ok=True)

        # Get Folders
        airfoil_folders: list[str] = next(os.walk(self.DB2D))[1]
        for airfoil_folder in airfoil_folders:
            logging.info(f"Scanning {airfoil_folder}...")
            # Load Airfoil Object
            try:
                self.add_airfoil(airfoil_folder)
            except FileNotFoundError:
                logging.exception(
                    f"Airfoil {airfoil_folder} not found in self.DB2D or EXTERNAL DB",
                )
                print(f"Airfoil {airfoil_folder} not found in self.DB2D or EXTERNAL DB")
                continue

            # Load Airfoil Data
            self.load_airfoil_data(airfoil_folder)

    def load_airfoil_data(self, airfoil: str | Airfoil) -> None:
        if isinstance(airfoil, Airfoil):
            airfoil_name = airfoil.name.upper()
        else:
            airfoil_name = airfoil.upper()

        # Load Computed Data
        data = self.read_airfoil_data_folder(airfoil_folder=airfoil_name)

        # Check if the data is empty
        if not data:
            logging.info(f"No data found for airfoil {airfoil_name}")
            return
        logging.info(f"Loading data for airfoil {airfoil_name}")

        # Create Polar Object
        inverted_data: dict[str, dict[str, DataFrame]] = {}
        for reynolds in data.keys():
            for solver in data[reynolds].keys():
                if solver not in inverted_data.keys():
                    inverted_data[solver] = {}
                inverted_data[solver][reynolds] = data[reynolds][solver]

        if airfoil_name not in self.polars.keys():
            self.polars[airfoil_name] = AirfoilData(
                name=airfoil_name,
                data=inverted_data,
            )
        else:
            self.polars[airfoil_name].add_data(
                data=inverted_data,
            )
        return

    def read_airfoil_data_folder(self, airfoil_folder: str) -> Struct:
        """Scans the reynolds subdirectories and loads the data.

        Args:
            airfoil_folder (str): Airfoil folder

        Returns:
            Struct: A struct containing the polars for all reynolds.

        """
        airfoil_data = Struct()
        # Read the reynolds subdirectories
        airfoil_folder_path = os.path.join(self.DB2D, airfoil_folder)
        folders: list[str] = next(os.walk(airfoil_folder_path))[1]  # folder = reynolds subdir

        for folder in folders:
            airfoil_data[folder[9:]] = self.scan_different_solver(
                airfoil_folder_path,
                folder,
            )
        return airfoil_data

    def scan_different_solver(self, airfoil_dir: str, airfoil_subdir: str) -> Struct:
        """Scans the different solver files and loads the data.

        Args:
            airfoil_dir (str): Airfoil directory
            airfoil_subdir (str): Airfoil subdirectories. (Reynolds)

        Raises:
            ValueError: If it encounters a solver not recognized.

        Returns:
            Struct: Struct containing the polars for all solvers.

        """
        current_reynolds_data: Struct = Struct()
        subdir = os.path.join(airfoil_dir, airfoil_subdir)
        files: list[str] = next(os.walk(subdir))[2]

        for file in files:
            if file.startswith("clcd"):
                solver: str = file[5:]
                if solver == "f2w":
                    name = "Foil2Wake"
                elif solver == "of":
                    name = "OpenFoam"
                elif solver == "xfoil":
                    name = "Xfoil"
                else:
                    raise ValueError("Solver not recognized!")
                filename = os.path.join(subdir, file)
                try:
                    current_reynolds_data[name] = pd.read_csv(filename, dtype=float)
                except ValueError:
                    current_reynolds_data[name] = pd.read_csv(
                        filename,
                        delimiter="\t",
                        dtype=float,
                    )
                # Check if the dataframe read is nan or empty
                try:
                    if current_reynolds_data[name]["CL"].isnull().values.all() or current_reynolds_data[name].empty:
                        del current_reynolds_data[name]
                except KeyError:
                    del current_reynolds_data[name]

        return current_reynolds_data

    def add_airfoil(self, airfoil_folder: str) -> None:
        """Adds an airfoil to the database.

        Args:
            airfoil_folder (str):

        Raises:
            FileNotFoundError: If the airfoil is not found in the self.DB2D or EXTERNAL DB and cant be generated from NACA Digits.

        """
        # Handle NACA airfoils first since we have analytical expressions for them
        if airfoil_folder.upper().startswith("NACA") and (
            len(airfoil_folder) == (4 + 4) or len(airfoil_folder) == (5 + 4)
        ):
            try:
                naca_foil = Airfoil.naca(airfoil_folder[4:], n_points=200)
                # Save the airfoil to the self.DB2D
                airfoil_dir = os.path.join(self.DB2D, airfoil_folder.upper())
                # Create the directory if it doesn't exist
                os.makedirs(airfoil_dir, exist_ok=True)
                naca_foil.save_selig(airfoil_dir)

                self.airfoils[airfoil_folder.upper()] = naca_foil
                logging.info(f"Loaded airfoil {airfoil_folder} from NACA Digits")
                return
            except Exception as e:
                print(
                    f"Error loading airfoil {airfoil_folder} from NACA Digits. Got error: {e}",
                )

        # Read the airfoil from the self.DB2D if it exists
        path_exists = os.path.exists(os.path.join(self.DB2D, airfoil_folder.upper()))
        if path_exists:
            try:
                filename = os.path.join(
                    self.DB2D,
                    airfoil_folder.upper(),
                    airfoil_folder.lower(),
                )
                self.airfoils[airfoil_folder] = Airfoil.from_file(filename)
                logging.info(f"Loaded airfoil {airfoil_folder} from self.DB2D")
            except Exception as e:
                print(
                    f"Error loading airfoil {airfoil_folder} from self.DB2D. Got error: {e}",
                )
        # Try to load from the WEB
        else:
            try:
                self.airfoils[airfoil_folder] = Airfoil.load_from_web(
                    airfoil_folder.lower(),
                )
                print(f"Loaded airfoil {airfoil_folder} from WEB")
                logging.info(
                    f"Loaded airfoil {airfoil_folder} from web and saved to DB",
                )
                return
            except FileNotFoundError:
                if self.EXTERNAL_DB is None:
                    raise FileNotFoundError(
                        f"Couldnt Find airfoil {airfoil_folder} in self.DB2D",
                    )
                # Search for the airfoil in the EXTERNAL DB
                folders: list[str] = os.walk(self.EXTERNAL_DB).__next__()[1]
                flag = False
                name: str = ""
                for folder in folders:
                    pattern = r"\([^)]*\)|[^0-9a-zA-Z]+"
                    cleaned_string: str = re.sub(pattern, " ", folder)
                    # Split the cleaned string into numeric and text parts
                    foil: str = "".join(filter(str.isdigit, cleaned_string))
                    text_part: str = "".join(filter(str.isalpha, cleaned_string))
                    if text_part.find("flap") != -1:
                        name = f"{foil + 'fl'}"
                    else:
                        name = foil

                    if name == airfoil_folder:
                        flag = True
                        name = folder

                if flag:
                    # list the files in the airfoil folder
                    flap_files: list[str] = os.listdir(
                        os.path.join(self.EXTERNAL_DB, name),
                    )
                    # check if the airfoil is in the flap folder
                    if name + ".dat" in flap_files:
                        # load the airfoil from the flap folder
                        filename = os.path.join(self.EXTERNAL_DB, name, name + ".dat")
                        self.airfoils[airfoil_folder] = Airfoil.from_file(filename)
                        logging.info(
                            f"Loaded airfoil {airfoil_folder} from EXTERNAL DB",
                        )
                else:
                    raise FileNotFoundError(
                        f"Couldnt Find airfoil {airfoil_folder} in self.DB2D or EXTERNAL DB",
                    )

    @staticmethod
    def generate_airfoil_directories(
        airfoil: Airfoil,
        reynolds: float,
        angles: list[float] | FloatArray,
    ) -> tuple[str, str, str, list[str]]:
        db = Database_2D.get_instance()
        AFDIR: str = os.path.join(
            db.DB2D,
            f"{airfoil.name.upper()}",
        )
        os.makedirs(AFDIR, exist_ok=True)
        exists = False
        for i in os.listdir():
            if i == airfoil.file_name:
                exists = True
        if not exists:
            airfoil.save_selig(AFDIR)
            sleep(0.1)

        reynolds_str: str = np.format_float_scientific(
            reynolds,
            sign=False,
            precision=3,
            min_digits=3,
        )

        REYNDIR: str = os.path.join(
            AFDIR,
            f"Reynolds_{reynolds_str.replace('+', '')}",
        )
        os.makedirs(REYNDIR, exist_ok=True)
        airfile = os.path.join(
            AFDIR,
            airfoil.file_name,
        )
        shutil.copy(airfile, REYNDIR)

        ANGLEDIRS: list[str] = []
        for angle in angles:
            folder = angle_to_directory(float(angle))
            ANGLEDIRS.append(os.path.join(REYNDIR, folder))

        from ICARUS import INSTALL_DIR

        return INSTALL_DIR, AFDIR, REYNDIR, ANGLEDIRS

    @staticmethod
    def fill_polar_table(df: DataFrame) -> DataFrame:
        """Fill Nan Values of Panda Dataframe Row by Row
        substituting first backward and then forward
        #! TODO: DEPRECATE THIS METHOD IN FAVOR OF POLAR CLASS

        Args:
            df (pandas.DataFrame): Dataframe with NaN values

        """
        CLs: list[str] = []
        CDs: list[str] = []
        CMs: list[str] = []
        for item in list(df.keys()):
            if item.startswith("CL"):
                CLs.append(item)
            if item.startswith("CD"):
                CDs.append(item)
            if item.startswith("Cm") or item.startswith("CM"):
                CMs.append(item)
        for colums in [CLs, CDs, CMs]:
            df[colums] = df[colums].interpolate(
                method="linear",
                limit_direction="backward",
                axis=1,
            )
            df[colums] = df[colums].interpolate(
                method="linear",
                limit_direction="forward",
                axis=1,
            )
        df.dropna(axis=0, subset=df.columns[1:], how="all", inplace=True)
        return df

    def interpolate_polars(
        self,
        reynolds: float,
        airfoil_name: str,
        aoa: float,
        solver: str,
    ) -> tuple[float, float, float]:
        """Interpolates the polars from the database.

        Args:
            reynolds (float): Reynolds number
            airfoil_name (str): airfoil Name
            aoa (float): Angle of Attack
            solver (str): Solver Name

        Returns:
            tuple[float, float, float]: CL, CD, Cm

        """
        if airfoil_name not in self.polars.keys():
            if f"NACA{airfoil_name}" in self.polars.keys():
                airfoil_name = f"NACA{airfoil_name}"
            else:
                raise ValueError(f"Airfoil {airfoil_name} not in database!")

        airfoil_data: AirfoilData = self.polars[airfoil_name]
        polars: AirfoilPolars = airfoil_data.get_polars(solver=solver)
        reynolds_stored: list[float] = polars.reynolds_nums
        max_reynolds_stored: float = max(reynolds_stored)
        min_reynolds_stored: float = min(reynolds_stored)

        if reynolds > max_reynolds_stored:
            raise ValueError(
                f"Reynolds {reynolds} not in database! Max Reynolds is {max_reynolds_stored}",
            )

        if reynolds < min_reynolds_stored:
            raise ValueError(
                f"Reynolds {reynolds} not in database! Min Reynolds is {min_reynolds_stored}",
            )

        reynolds_stored.sort()
        upper_reynolds: float | None = None
        lower_reynolds: float | None = None

        # Check if the reynolds is an exact match
        if reynolds in reynolds_stored:
            upper_reynolds = reynolds
            lower_reynolds = reynolds
        else:
            # Find the 2 reynolds numbers that our reynolds is between
            for i in range(1, len(reynolds_stored)):
                if reynolds_stored[i] > reynolds:
                    upper_reynolds = reynolds_stored[i]
                    lower_reynolds = reynolds_stored[i - 1]
                    break

        if not upper_reynolds or not lower_reynolds:
            raise ValueError(
                f"Reynolds {reynolds} not in database! Max Reynolds is {max_reynolds_stored}",
            )

        # Get the polars for the 2 reynolds numbers
        upper_polar: DataFrame = polars.get_reynolds_subtable(upper_reynolds)
        lower_polar: DataFrame = polars.get_reynolds_subtable(lower_reynolds)

        # Interpolate the CL, CD and Cm values for the given aoa for each reynolds
        try:
            CL_up: float = float(np.interp(aoa, upper_polar["AoA"], upper_polar["CL"]))
            CD_up: float = float(np.interp(aoa, upper_polar["AoA"], upper_polar["CD"]))
            Cm_up: float = float(np.interp(aoa, upper_polar["AoA"], upper_polar["Cm"]))

            CL_low: float = float(np.interp(aoa, lower_polar["AoA"], lower_polar["CL"]))
            CD_low: float = float(np.interp(aoa, lower_polar["AoA"], lower_polar["CD"]))
            Cm_low: float = float(np.interp(aoa, lower_polar["AoA"], lower_polar["Cm"]))

            # Interpolate between the 2 CL, CD and Cm values
            CL = float(
                np.interp(reynolds, [lower_reynolds, upper_reynolds], [CL_low, CL_up]),
            )
            CD = float(
                np.interp(reynolds, [lower_reynolds, upper_reynolds], [CD_low, CD_up]),
            )
            Cm = float(
                np.interp(reynolds, [lower_reynolds, upper_reynolds], [Cm_low, Cm_up]),
            )
        except KeyError as e:
            print(airfoil_name)
            print(upper_reynolds)
            print(lower_reynolds)
            raise KeyError(f"Key {e} not found in database!")

        return CL, CD, Cm

    def __str__(self) -> str:
        return "Foil Database"

    # def __enter__(self) -> None:
    #     """
    #     TODO: Implement this method.
    #     """
    #     pass

    # def __exit__(self) -> None:
    #     """
    #     TODO: Implement this method.
    #     """
    #     pass
