from __future__ import annotations

import logging
import os
import re
import shutil
from time import sleep
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import jsonpickle
import numpy as np

from ICARUS.airfoils import Airfoil
from ICARUS.airfoils import AirfoilData
from ICARUS.airfoils import AirfoilPolarMap
from ICARUS.airfoils import PolarNotAccurate
from ICARUS.airfoils import ReynoldsNotIncluded
from ICARUS.airfoils.metrics.polars import AirfoilPolar
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
        self.logger = logging.getLogger(self.__class__.__name__)

        self.DB2D: str = location
        if not os.path.isdir(self.DB2D):
            self.logger.info(f"Creating self.DB2D directory at {self.DB2D}...")
            os.makedirs(self.DB2D)

        # !TODO: Make data private
        self.polars: dict[str, AirfoilData] = {}
        self.airfoils: dict[str, Airfoil] = {}
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
            airfoil: Airfoil = self.airfoils[airfoil_name]
            return airfoil
        except KeyError:
            try:
                # Try to load the airfoil from the DB or EXTERNAL DB
                airfoil = self.load_airfoil(airfoil_name)
                return airfoil
            except FileNotFoundError:
                raise AirfoilNotFoundError(airfoil_name)

    def get_polars(
        self,
        airfoil: str | Airfoil,
        solver: str | None = None,
    ) -> AirfoilPolarMap:
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
                self.load_airfoil_data(airfoil)
                if airfoil_name not in self.polars.keys():
                    raise PolarsNotFoundError(airfoil_name)
            except (FileNotFoundError, StopIteration):
                raise AirfoilNotFoundError(airfoil_name)

        airfoil_data: AirfoilData = self.polars[airfoil_name]
        try:
            polar = airfoil_data.get_polars(solver=solver)
        except KeyError:
            raise PolarsNotFoundError(
                airfoil_name,
                solver,
                list(airfoil_data.polars.keys()),
            )
        return polar

    def get_airfoil_data(self, airfoil: str | Airfoil) -> AirfoilData:
        """Returns the solvers available for a given airfoil.

        Args:
            airfoil_name (str): Airfoil name

        Returns:
            list[str]: List of solver names

        """

        if isinstance(airfoil, Airfoil):
            self.add_airfoil(airfoil)
        elif isinstance(airfoil, str):
            # If the airfoil is a string, try to load it
            airfoil = self.load_airfoil(airfoil)
        else:
            raise ValueError("airfoil must be a string or an Airfoil object")

        airfoil_name = airfoil.name.upper()
        if airfoil_name not in self.polars:
            try:
                # Try to load the airfoil from the DB or EXTERNAL DB
                self.load_airfoil_data(airfoil)
                if airfoil_name not in self.polars:
                    raise PolarsNotFoundError(airfoil_name)
            except (FileNotFoundError, StopIteration):
                raise AirfoilNotFoundError(airfoil_name)

        airfoil_data: AirfoilData = self.polars[airfoil_name]
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
    ) -> AirfoilPolarMap:
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
            reyns_computed = polars.reynolds_numbers
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
            self.logger.info(
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
                self.logger.info(
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
        # Get Folders
        airfoil_folders: list[str] = next(os.walk(self.DB2D))[1]
        for airfoil_folder in airfoil_folders:
            self.logger.info(f"Scanning {airfoil_folder}...")

            try:
                # Load Airfoil Object
                airfoil = self.load_airfoil(airfoil_folder)
                # Load Airfoil Data
                self.load_airfoil_data(airfoil)
            except Exception as e:
                logging.error(
                    f"Airfoil {airfoil_folder} could not be loaded. Skipping...",
                    exc_info=e,
                )
                continue

    def load_airfoil_data(self, airfoil: Airfoil) -> None:
        airfoil_name = airfoil.name.upper()
        self.logger.info(f"Adding {airfoil_name} to the database")

        # Load Computed Data
        airfoil_data = self.read_airfoil_data_from_fiesystem(airfoil=airfoil)

        # Check if the data is empty
        if not airfoil_data:
            self.logger.info(f"No data found for airfoil {airfoil.name}")
            return
        else:
            self.logger.info(f"Loaded data for airfoil {airfoil.name}")

        self.polars[airfoil_name] = airfoil_data
        return

    def read_airfoil_data_from_fiesystem(self, airfoil: Airfoil) -> AirfoilData:
        """Scans the reynolds subdirectories and loads the data.

        Args:
            airfoil_folder (str): Airfoil folder

        Returns:
            Struct: A struct containing the polars for all reynolds.

        """
        airfoil_name = airfoil.name.upper()
        if airfoil_name not in self.polars.keys():
            airfoil_data = AirfoilData(airfoil_name=airfoil_name)
        else:
            airfoil_data = self.polars[airfoil_name]

        # Read the reynolds subdirectories
        airfoil_folder_path = os.path.join(self.DB2D, airfoil.name.upper())
        folders: list[str] = next(os.walk(airfoil_folder_path))[1]
        for folder in folders:
            # folder = reynolds subdir
            subdir = os.path.join(airfoil_folder_path, folder)
            self.scan_different_solver(subdir, airfoil_data)
        return airfoil_data

    def scan_different_solver(
        self,
        directory: str,
        airfoil_data: AirfoilData,
    ) -> None:
        """Scans the different solver files and loads the data.

        Args:
            directory (str): Directory to scan
            airfoil_data (AirfoilData): AirfoilData object to add the polars to

        Raises:
            ValueError: If it encounters a solver not recognized.

        """

        files: list[str] = next(os.walk(directory))[2]
        for file in files:
            if not file.startswith("polar"):
                continue

            if file.endswith("f2w"):
                name = "Foil2Wake"
            elif file.endswith("of"):
                name = "OpenFoam"
            elif file.endswith("xfoil"):
                name = "Xfoil"
            else:
                raise ValueError("Solver not recognized!")

            filename = os.path.join(directory, file)

            with open(filename, encoding="UTF-8") as f:
                json_obj: str = f.read()

            polar = jsonpickle.decode(json_obj)
            if not isinstance(polar, AirfoilPolar):
                # raise TypeError(f"Expected AirfoilPolar, got {type(polar)}")
                continue

            if polar.is_empty():
                continue
            # Add the polar to the polar data
            airfoil_data.add_polar(name, polar)

    def add_airfoil(self, airfoil: Airfoil) -> None:
        """Add an airfoil object to the database.

        Args:
            airfoil (Airfoil): Airfoil object to add

        """
        if not isinstance(airfoil, Airfoil):
            raise TypeError(f"Expected Airfoil, got {type(airfoil)}")
        airfoil_name = airfoil.name.upper()
        self.airfoils[airfoil_name] = airfoil
        self.logger.info(f"Added airfoil {airfoil_name} to the database.")

        # Check if the folder for the airfoil exists, if not create it
        airfoil_dir = os.path.join(self.DB2D, airfoil_name)
        os.makedirs(airfoil_dir, exist_ok=True)
        # Save the airfoil to the database
        if not os.path.exists(os.path.join(airfoil_dir, airfoil.file_name)):
            airfoil.save_selig(airfoil_dir)
            self.logger.info(f"Saved airfoil {airfoil_name} to the filesystem.")

    def load_airfoil(self, airfoil_name: str) -> Airfoil:
        """
        Add an airfoil to the database from various sources.

        The method searches for the airfoil in the following order:
        1.  NACA analytical generation
        2.  Local database
        3.  Web (airfoiltools.com)
        4.  External database (if configured)

        Args:
            airfoil_name (str): The name of the airfoil to add.

        Raises:
            FileNotFoundError: If the airfoil cannot be found in any of the sources.
        """
        airfoil_name = airfoil_name.upper()

        # Check if the airfoil is already in the database
        if airfoil_name in self.airfoils:
            self.logger.info(f"Airfoil {airfoil_name} already exists in the database.")
            return self.airfoils[airfoil_name]

        # Try loading from NACA digits first
        if self._try_load_from_naca(airfoil_name):
            return self.airfoils[airfoil_name]

        # Try loading from the local filesystem
        if self._try_load_from_filesystem(airfoil_name):
            return self.airfoils[airfoil_name]

        # Try loading from the web
        if self._try_load_from_web(airfoil_name):
            return self.airfoils[airfoil_name]

        # Try loading from the external database
        if self.EXTERNAL_DB and self._try_load_from_external_db(airfoil_name):
            return self.airfoils[airfoil_name]

        raise FileNotFoundError(
            f"Could not find or generate airfoil '{airfoil_name}' from any source.",
        )

    def _try_load_from_naca(self, airfoil_name: str) -> bool:
        if airfoil_name.startswith("NACA") and (len(airfoil_name) in [4 + 4, 5 + 4]):
            try:
                naca_foil = Airfoil.naca(airfoil_name[4:], n_points=200)
                self.add_airfoil(naca_foil)
                return True
            except Exception as e:
                self.logger.warning(
                    f"Could not generate NACA airfoil {airfoil_name}: {e}",
                )
        return False

    def _try_load_from_filesystem(self, airfoil_name: str) -> bool:
        airfoil_path = os.path.join(
            self.DB2D,
            airfoil_name.upper(),
            f"{airfoil_name.lower()}.airfoil",
        )

        if os.path.exists(airfoil_path):
            try:
                self.airfoils[airfoil_name] = Airfoil.from_file(airfoil_path)
                self.logger.info(f"Loaded airfoil {airfoil_name} from local DB.")
                return True
            except Exception as e:
                self.logger.error(
                    f"Error loading airfoil {airfoil_name} from local DB: {e}",
                )
        return False

    def _try_load_from_web(self, airfoil_name: str) -> bool:
        try:
            self.logger.info(f"Trying to fetch airfoil {airfoil_name} from web.")
            airfoil = Airfoil.load_from_web(airfoil_name.lower())
            self.add_airfoil(airfoil)
            return True
        except FileNotFoundError:
            return False

    def _try_load_from_external_db(self, airfoil_name: str) -> bool:
        if self.EXTERNAL_DB is None:
            return False

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

            if name == airfoil_name:
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
                self.airfoils[airfoil_name] = Airfoil.from_file(filename)
                self.logger.info(
                    f"Loaded airfoil {airfoil_name} from EXTERNAL DB",
                )
        else:
            raise FileNotFoundError(
                f"Couldnt Find airfoil {airfoil_name} in self.DB2D or EXTERNAL DB",
            )

        return False

    @staticmethod
    def generate_airfoil_directories(
        airfoil: Airfoil,
        reynolds: float | None = None,
        angles: float | list[float] | FloatArray = [],
    ) -> tuple[str, str, list[str]]:
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

        if reynolds is None:
            return AFDIR, str(None), []

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

        if isinstance(angles, float):
            angle_list: list[float] = [angles]
        elif isinstance(angles, list):
            angle_list = angles
        elif isinstance(angles, np.ndarray):
            angle_list = angles.tolist()
        else:
            raise TypeError(
                f"Expected float, list or np.ndarray for angles, got {type(angles)}",
            )

        for angle in angle_list:
            folder = angle_to_directory(float(angle))
            angle_dir = os.path.join(REYNDIR, folder)
            os.makedirs(angle_dir, exist_ok=True)
            ANGLEDIRS.append(angle_dir)

        return AFDIR, REYNDIR, ANGLEDIRS

    def interpolate_polars(
        self,
        reynolds: float,
        airfoil: Airfoil,
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
        airfoil_name = airfoil.name.upper()  # Ensure airfoil name is uppercase
        if airfoil_name not in self.polars.keys():
            if f"NACA{airfoil_name}" in self.polars.keys():
                airfoil_name = f"NACA{airfoil_name}"
            else:
                raise ValueError(f"Airfoil {airfoil_name} not in database!")

        airfoil_data: AirfoilData = self.polars[airfoil_name]
        polars: AirfoilPolarMap = airfoil_data.get_polars(solver=solver)
        reynolds_stored: list[float] = polars.reynolds_numbers
        max_reynolds_stored: float = max(reynolds_stored)
        min_reynolds_stored: float = min(reynolds_stored)

        if reynolds > max_reynolds_stored:
            raise ValueError(
                f"Airfoil: {airfoil.name} Reynolds {reynolds} not in database! Max Reynolds is {max_reynolds_stored}.",
            )

        if reynolds < min_reynolds_stored:
            raise ValueError(
                f"Airfoil: {airfoil.name} Reynolds {reynolds} not in database! Min Reynolds is {min_reynolds_stored}",
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
        upper_polar: AirfoilPolar = polars.get_polar(upper_reynolds)
        lower_polar: AirfoilPolar = polars.get_polar(lower_reynolds)

        # Interpolate the CL, CD and Cm values for the given aoa for each reynolds
        try:
            CL_up: float = float(
                np.interp(aoa, upper_polar.df["AoA"], upper_polar.df["CL"]),
            )
            CD_up: float = float(
                np.interp(aoa, upper_polar.df["AoA"], upper_polar.df["CD"]),
            )
            Cm_up: float = float(
                np.interp(aoa, upper_polar.df["AoA"], upper_polar.df["Cm"]),
            )

            CL_low: float = float(
                np.interp(aoa, lower_polar.df["AoA"], lower_polar.df["CL"]),
            )
            CD_low: float = float(
                np.interp(aoa, lower_polar.df["AoA"], lower_polar.df["CD"]),
            )
            Cm_low: float = float(
                np.interp(aoa, lower_polar.df["AoA"], lower_polar.df["Cm"]),
            )

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
            raise KeyError(f"Key {e} not found in database!")

        return CL, CD, Cm

    def __str__(self) -> str:
        return "Foil Database"
