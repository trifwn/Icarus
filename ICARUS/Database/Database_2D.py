import os
import re
import shutil
from time import sleep

import numpy as np
import pandas as pd
from pandas import DataFrame

from . import APPHOME
from . import DB2D
from . import EXTERNAL_DB
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

    def load_data(self) -> None:
        """
        Scans the filesystem and load all the data.
        """
        self.scan()
        self.airfoils: Struct = self.set_available_airfoils()
        self.polars = Struct()
        for airfoil, data in self.data.items():
            self.polars[airfoil] = Struct()
            for solver in data.keys():
                self.polars[airfoil][solver] = Polars(self.data[airfoil][solver])

    def scan(self) -> None:
        """
        Scans the filesystem and loads data if not already loaded.
        """
        # Accessing Database Directory
        try:
            os.chdir(DB2D)
        except FileNotFoundError:
            print(f"Database not found! Initializing Database at {DB2D}")
            os.makedirs(DB2D, exist_ok=True)
        # Get Folders
        folders: list[str] = next(os.walk("."))[1]
        data = Struct()
        for airfoil in folders:
            os.chdir(airfoil)
            data[airfoil] = self.scan_reynold_subdirs()
            os.chdir(DB2D)

        for airfoil in data.keys():
            if airfoil not in self.data.keys():
                self.data[airfoil] = Struct()

            for j in data[airfoil].keys():
                for k in data[airfoil][j].keys():
                    if k not in self.data[airfoil].keys():
                        self.data[airfoil][k] = Struct()
                    self.data[airfoil][k][j] = data[airfoil][j][k]
        os.chdir(self.HOMEDIR)

    def scan_reynold_subdirs(self) -> Struct:
        """
        Scans the reynolds subdirectories and loads the data.

        Returns:
            Struct: A struct containing the polars for all reynolds.
        """
        airfoil_data = Struct()
        folders: list[str] = next(os.walk("."))[1]  # folder = reynolds subdir
        for folder in folders:
            os.chdir(folder)
            airfoil_data[folder[9:]] = self.scan_different_solver()
            os.chdir("..")
        return airfoil_data

    def scan_different_solver(self) -> Struct:
        """
        Scans the different solver files and loads the data.

        Raises:
            ValueError: If it encounters a solver not recognized.

        Returns:
            Struct: Struct containing the polars for all solvers.
        """
        current_reynolds_data = Struct()
        files: list[str] = next(os.walk("."))[2]
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
                try:
                    current_reynolds_data[name] = pd.read_csv(file, dtype=float)
                except ValueError:
                    current_reynolds_data[name] = pd.read_csv(
                        file,
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

    def set_available_airfoils(self, verbose: bool = False) -> Struct:
        airfoils = Struct()
        for airf in list(self.data.keys()):
            try:
                airfoils[airf] = Airfoil.naca(airf[4:], n_points=200)
                if verbose:
                    print(f"Loaded airfoil {airf} from NACA Digits")
            except:
                # try to load the Airfoil from the DB2D
                try:
                    filename = os.path.join(DB2D, airf, airf.replace("NACA", "naca"))
                    airfoils[airf] = Airfoil.load_from_file(filename)
                    if verbose:
                        print(f"Loaded airfoil {airf} from DB2D")
                except:
                    #! TODO DEPRECATE THIS IT IS STUPID airfoilS SHOULD BE MORE ROBUST

                    # list the folders in the EXTERNAL DB
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

                        if len(airf) == 4 or len(airf) == 5:
                            name = "NACA" + name

                        if name == airf:
                            flag = True
                            name = folder

                    if flag:
                        # list the files in the airfoil folder
                        flap_files: list[str] = os.listdir(os.path.join(EXTERNAL_DB, name))
                        # check if the airfoil is in the flap folder
                        if name + ".dat" in flap_files:
                            # load the airfoil from the flap folder
                            filename = os.path.join(EXTERNAL_DB, name, name + ".dat")
                            airfoils[airf] = Airfoil.load_from_file(filename)
                            if verbose:
                                print(f"Loaded airfoil {airf} from EXTERNAL DB")
                    else:
                        raise FileNotFoundError(f"Couldnt Find airfoil {airf} in DB2D or EXTERNAL DB")
        return airfoils

    def get_airfoil_solvers(self, airfoil_name: str) -> list[str] | None:
        """
        Get the solvers for a given airfoil.

        Args:
            airfoil_name (str): airfoil Name

        Returns:
            list[str] | None: The solver names or None if the airfoil doesn't exist.
        """
        try:
            return list(self.data[airfoil_name].keys())
        except KeyError:
            print("Airfoil Doesn't exist! You should compute it first!")
            return None

    def get_airfoil_reynolds(self, airfoil_name: str) -> list[str] | None:
        """
        Returns the reynolds numbers computed for a given airfoil.

        Args:
            airfoil_name (str): airfoil Name

        Returns:
            list[str] | None: List of reynolds numbers computed or None if the airfoil doesn't exist.
        """
        try:
            reynolds: list[str] = []
            for solver in self.data[airfoil_name].keys():
                for reyn in self.data[airfoil_name][solver].keys():
                    reynolds.append(reyn)
            return reynolds
        except KeyError:
            print("Airfoil Doesn't exist! You should compute it first!")
            return None

    def generate_airfoil_directories(
        self,
        airfoil: Airfoil,
        reynolds: float,
        angles: list[float] | FloatArray,
    ) -> tuple[str, str, str, list[str]]:
        AFDIR: str = os.path.join(
            self.DATADIR,
            f"NACA{airfoil.name}",
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
            folder = self.angle_to_dir(angle)
            ANGLEDIRS.append(os.path.join(REYNDIR, folder))

        return self.HOMEDIR, AFDIR, REYNDIR, ANGLEDIRS

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
