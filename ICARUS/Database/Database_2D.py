import logging
import os
import re
import shutil
from time import sleep

import numpy as np
import pandas as pd
from pandas import DataFrame

from . import DB2D
from . import DB3D
from . import EXTERNAL_DB
from ICARUS import APPHOME
from ICARUS.Airfoils import airfoil
from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Airfoils.airfoil_polars import Polars
from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray


class Database_2D:
    """
    Database class to store 2d simulation objects (airfoils), analyses and results (polars).
    """

    def __init__(self) -> None:
        """
        Initialize the Database_2D class.
        """
        self.HOMEDIR: str = APPHOME
        self.DATADIR: str = DB2D
        if not os.path.isdir(self.DATADIR):
            os.makedirs(self.DATADIR)

        self.data: Struct = Struct()
        self.polars: Struct = Struct()
        self.airfoils: Struct = Struct()

    def get_airfoils(self) -> list[str]:
        """
        Returns the available airfoils in the database.

        Returns:
            list[str]: List of airfoil names
        """
        return list(self.airfoils.keys())

    def load_data(self) -> None:
        """
        Scans the filesystem and load all the data.
        """
        # Read all the data
        self.read_all_data()

    def read_all_data(self) -> None:
        """
        Scans the filesystem and loads data if not already loaded.
        """
        # Accessing Database Directory
        if not os.path.isdir(DB2D):
            print(f"Creating DB2D directory at {DB2D}...")
            os.makedirs(DB2D, exist_ok=True)

        data = Struct()
        # Get Folders
        airfoil_folders: list[str] = next(os.walk(DB2D))[1]
        for airfoil_folder in airfoil_folders:
            logging.info(f"Scanning {airfoil_folder}...")
            # Enter DB2D
            os.chdir(DB2D)
            airfoil_folder_path = os.path.join(DB2D, airfoil_folder)
            os.chdir(airfoil_folder_path)

            # Load Airfoil Object
            try:
                self.add_airfoil(airfoil_folder)
            except FileNotFoundError:
                logging.error(f"Airfoil {airfoil_folder} not found in DB2D or EXTERNAL DB")
                print(f"Airfoil {airfoil_folder} not found in DB2D or EXTERNAL DB")
                continue

            # Load Computed Data
            data[airfoil_folder] = self.load_airfoil_data(airfoil_folder)

            # Check if the data is empty
            if not data[airfoil_folder]:
                logging.info(f"No data found for airfoil {airfoil_folder}")
                continue
            # If not Create Data and Polars
            else:
                logging.info(f"Loaded data for airfoil {airfoil_folder}")
                self.data[airfoil_folder] = Struct()
                for j in data[airfoil_folder].keys():
                    for k in data[airfoil_folder][j].keys():
                        if k not in self.data[airfoil_folder].keys():
                            self.data[airfoil_folder][k] = Struct()
                        self.data[airfoil_folder][k][j] = data[airfoil_folder][j][k]

                # Create Polar Object
                self.polars[airfoil_folder] = Struct()
                for solver in self.data[airfoil_folder].keys():
                    self.polars[airfoil_folder][solver] = Polars(
                        name=airfoil_folder, data=self.data[airfoil_folder][solver],
                    )
        os.chdir(self.HOMEDIR)

    # def update_data()

    def load_airfoil_data(self, airfoil_folder: str) -> Struct:
        """
        Scans the reynolds subdirectories and loads the data.

        Args:
            airfoil_folder (str): Airfoil folder

        Returns:
            Struct: A struct containing the polars for all reynolds.
        """
        airfoil_data = Struct()
        # Read the reynolds subdirectories
        airfoil_folder_path = os.path.join(DB2D, airfoil_folder)
        folders: list[str] = next(os.walk(airfoil_folder_path))[1]  # folder = reynolds subdir

        for folder in folders:
            airfoil_data[folder[9:]] = self.scan_different_solver(airfoil_folder_path, folder)
        return airfoil_data

    def scan_different_solver(self, airfoil_dir: str, airfoil_subdir: str) -> Struct:
        """
        Scans the different solver files and loads the data.

        Args:
            airfoil_dir (str): Airfoil directory
            airfoil_subdir (str): Airfoil subdirectories. (Reynolds)

        Raises:
            ValueError: If it encounters a solver not recognized.

        Returns:
            Struct: Struct containing the polars for all solvers.
        """
        current_reynolds_data = Struct()
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
                except:
                    del current_reynolds_data[name]

        return current_reynolds_data

    def add_airfoil(self, airfoil_folder: str) -> None:
        """
        Adds an airfoil to the database.

        Args:
            airfoil_folder (str):

        Raises:
            FileNotFoundError: If the airfoil is not found in the DB2D or EXTERNAL DB and cant be generated from NACA Digits.
        """
        # Handle NACA airfoils first since we have analytical expressions for them
        if airfoil_folder.upper().startswith("NACA") and (
            len(airfoil_folder) == (4 + 4) or len(airfoil_folder) == (5 + 4)
        ):
            try:
                self.airfoils[airfoil_folder] = Airfoil.naca(airfoil_folder[4:], n_points=200)
                logging.info(f"Loaded airfoil {airfoil_folder} from NACA Digits")
            except Exception as e:
                print(f"Error loading airfoil {airfoil_folder} from NACA Digits. Got error: {e}")

        # Read the airfoil from the DB2D if it exists
        elif airfoil_folder.lower() in os.listdir(os.path.join(DB2D, airfoil_folder.upper())):
            try:
                filename = os.path.join(DB2D, airfoil_folder.upper(), airfoil_folder.lower())
                self.airfoils[airfoil_folder] = Airfoil.load_from_file(filename)
                logging.info(f"Loaded airfoil {airfoil_folder} from DB2D")
            except Exception as e:
                print(f"Error loading airfoil {airfoil_folder} from DB2D. Got error: {e}")

        # Search for the airfoil in the EXTERNAL DB
        else:
            folders: list[str] = os.walk(EXTERNAL_DB).__next__()[1]
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
                flap_files: list[str] = os.listdir(os.path.join(EXTERNAL_DB, name))
                # check if the airfoil is in the flap folder
                if name + ".dat" in flap_files:
                    # load the airfoil from the flap folder
                    filename = os.path.join(EXTERNAL_DB, name, name + ".dat")
                    self.airfoils[airfoil_folder] = Airfoil.load_from_file(filename)
                    logging.info(f"Loaded airfoil {airfoil_folder} from EXTERNAL DB")
            else:
                raise FileNotFoundError(f"Couldnt Find airfoil {airfoil_folder} in DB2D or EXTERNAL DB")

    @staticmethod
    def generate_airfoil_directories(
        airfoil: Airfoil,
        reynolds: float,
        angles: list[float] | FloatArray,
    ) -> tuple[str, str, str, list[str]]:
        AFDIR: str = os.path.join(
            DB2D,
            f"{airfoil.name.upper()}",
        )
        os.makedirs(AFDIR, exist_ok=True)
        exists = False
        for i in os.listdir():
            if i.startswith("naca"):
                exists = True
        if not exists:
            airfoil.save_selig_te(AFDIR)
            sleep(0.1)

        reynolds_str: str = np.format_float_scientific(reynolds, sign=False, precision=3, min_digits=3)

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
            folder = Database_2D.angle_to_dir(angle)
            ANGLEDIRS.append(os.path.join(REYNDIR, folder))

        return APPHOME, AFDIR, REYNDIR, ANGLEDIRS

    # STATIC METHODS
    @staticmethod
    def angle_to_dir(angle: float) -> str:
        """
        Converts an angle to a directory name.

        Args:
            angle (float): Angle

        Returns:
            str: Directory name
        """
        if angle >= 0:
            folder: str = str(angle)[::-1].zfill(7)[::-1]
        else:
            folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]
        return folder

    @staticmethod
    def dir_to_angle(folder: str) -> float:
        """
        Converts a directory name to an angle.

        Args:
            folder (str): Directory name

        Returns:
            float: Angle
        """
        if folder.startswith("m"):
            angle = -float(folder[1:])
        else:
            angle = float(folder)
        return angle

    @staticmethod
    def get_reynolds_from_dir(folder: str) -> float:
        """
        Gets the reynolds number from a directory name.

        Args:
            folder (str): Directory name

        Returns:
            float: Reynolds number
        """
        return float(folder[10:].replace("_", "e"))

    @staticmethod
    def get_dir_from_reynolds(reynolds: float) -> str:
        """
        Gets the directory name from a reynolds number.

        Args:
            reynolds (float): Reynolds number

        Returns:
            str: Directory name
        """
        return f"Reynolds_{reynolds}"

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
        """
        Interpolates the polars from the database.

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
        if solver not in self.polars[airfoil_name].keys():
            raise ValueError(f"Solver {solver} not in database!")

        polars: Polars = self.polars[airfoil_name][solver]
        reynolds_stored: list[float] = polars.reynolds_nums
        max_reynolds_stored: float = max(reynolds_stored)
        min_reynolds_stored: float = min(reynolds_stored)

        if reynolds > max_reynolds_stored:
            raise ValueError(f"Reynolds {reynolds} not in database! Max Reynolds is {max_reynolds_stored}")

        if reynolds < min_reynolds_stored:
            raise ValueError(f"Reynolds {reynolds} not in database! Min Reynolds is {min_reynolds_stored}")

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
            raise ValueError(f"Reynolds {reynolds} not in database! Max Reynolds is {max_reynolds_stored}")

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
            CL = float(np.interp(reynolds, [lower_reynolds, upper_reynolds], [CL_low, CL_up]))
            CD = float(np.interp(reynolds, [lower_reynolds, upper_reynolds], [CD_low, CD_up]))
            Cm = float(np.interp(reynolds, [lower_reynolds, upper_reynolds], [Cm_low, Cm_up]))
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
