import os
from typing import Any
from typing import Literal

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
import numpy as np
import pandas as pd
from numpy import floating
from pandas import DataFrame

from . import DB3D
from ICARUS import APPHOME
from ICARUS.Computation.Solvers.GenuVP.post_process.convergence import (
    get_loads_convergence_3,
)
from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane

jsonpickle_pd.register_handlers()


class Database_3D:
    """Class to represent the 3D Database. It contains all the information and results
    of the 3D analysis of the vehicles."""

    def __init__(self) -> None:
        self.HOMEDIR: str = APPHOME
        self.DATADIR: str = DB3D
        self.raw_data = Struct()
        self.data = Struct()
        self.planes = Struct()
        self.states = Struct()
        self.convergence_data = Struct()

    def load_data(self) -> None:
        self.scan_and_make_data()

    def scan_and_make_data(self) -> None:
        if not os.path.isdir(DB3D):
            # print(f"Creating {DB3D} directory...")
            os.makedirs(DB3D, exist_ok=True)

        planenames: list[str] = next(os.walk(DB3D))[1]
        for plane in planenames:  # For each plane planename == folder
            # Load Plane object
            file_plane: str = os.path.join(DB3D, plane, f"{plane}.json")
            plane_found: bool = self.load_plane_from_file(plane, file_plane)

            if plane_found:
                # Load Convergence Data
                if plane not in self.convergence_data.keys():
                    self.convergence_data[plane] = Struct()
                cases: list[str] = next(os.walk(os.path.join(DB3D, plane)))[1]
                for case in cases:
                    # Load States
                    if case.startswith("Dyn"):
                        self.states[plane] = self.load_plane_states(plane, case)
                        continue
                    if case.startswith("Sens"):
                        continue
                    if "gnvp3" in os.listdir(os.path.join(DB3D, plane, case)):
                        self.load_gnvp_case_convergence(plane, case, 3)
                    if "gnvp7" in os.listdir(os.path.join(DB3D, plane, case)):
                        self.load_gnvp_case_convergence(plane, case, 7)

            # Loading Forces from forces.* files
            file_gnvp_7: str = os.path.join(DB3D, plane, "forces.gnvp7")
            file_gnvp_3: str = os.path.join(DB3D, plane, "forces.gnvp3")
            file_lspt: str = os.path.join(DB3D, plane, "forces.lspt")

            self.load_gnvp_forces(plane, file_gnvp_7, genu_version=7)
            self.load_gnvp_forces(plane, file_gnvp_3, genu_version=3)
            self.load_lspt_forces(plane, file_lspt)

    def load_plane_states(self, plane: str, case: str) -> dict[str, Any]:
        """
        Function to load the states of the plane from the states.json file.

        Args:
            plane (str): Plane Name
            case (str): Case Directory
        """
        dynamics_directory: str = os.path.join(DB3D, plane, case)
        os.chdir(dynamics_directory)
        files: list[str] = next(os.walk(dynamics_directory))[2]
        states: dict[str, Any] = {}
        for file in files:
            if file.endswith(".json"):
                with open(file, encoding="UTF-8") as f:
                    json_obj: str = f.read()
                try:
                    obj: Any = jsonpickle.decode(json_obj)
                    if isinstance(obj, State):
                        state: State = obj
                    else:
                        raise TypeError(f"Expected State object, got {type(obj)}")
                    states[state.name] = state
                except Exception as error:
                    # print(f"Error decoding states object {plane}! Got error {error}")
                    pass
        os.chdir(self.HOMEDIR)
        return states

    def load_plane_from_file(self, name: str, file: str) -> bool:
        """Function to get Plane Object from file and decode it.

        Args:
            name (str): planename
            file (str): filename

        Returns:
            bool : whether the plane was found or not
        """
        try:
            with open(file, encoding="UTF-8") as f:
                json_obj: str = f.read()
                try:
                    self.planes[name] = jsonpickle.decode(json_obj)
                except Exception as error:
                    # print(f"Error decoding Plane object {name}! Got error {error}")
                    pass
            plane_found = True
        except FileNotFoundError:
            # print(f"No Plane object found in {name} folder at {file}!")
            plane_found = False
        return plane_found

    def load_gnvp_forces(self, planename: str, file: str, genu_version: int) -> None:
        """
        Load Forces from forces file and store them in the raw_data dict.
        If the file doesn't exist it tries to create it by loading the plane object
        and running the make_polars function. If that fails as weall it prints an error.

        TODO: Should get deprecated in favor of analysis logic in the future

        Args:
            planename (str): Planename
            file (str): Filename Containing FOrces (forces.gnvp3)
            genu_version (int): GNVP Version
        """
        try:
            self.raw_data[planename] = pd.read_csv(file)
            self.make_data_gnvp(planename, genu_version)
            return
        except FileNotFoundError:
            # print(f"No forces.gnvp3 file found in {planename} folder at {DB3D}!")
            pass
            # if planename in self.planes.keys():
            #     # print(
            #     #     "Since plane object exists with that name trying to create polars...",
            #     # )
            #     pln: Airplane = self.planes[planename]
            #     try:
            #         CASEDIR: str = os.path.join(DB3D, pln.CASEDIR)
            #         if genu_version == 3:
            #             from ICARUS.Solvers.GenuVP.files.gnvp3_interface import make_polars_3
            #             make_polars_3(CASEDIR, self.HOMEDIR)
            #         else:
            #             from ICARUS.Solvers.GenuVP.files.gnvp7_interface import make_polars_7

            #             make_polars_7(CASEDIR, self.HOMEDIR)

            #         self.raw_data[planename] = pd.read_csv(file)
            #         self.make_data_gnvp(planename, genu_version)
            #     except Exception as e:
            #         print(f"Failed to create Polars! Got Error:\n{e}")

    def load_lspt_forces(self, planename: str, file: str) -> None:
        """
        Load Forces from forces file and store them in the raw_data dict.

        Args:
            planename (str): _description_
            file (str): _description_
        """
        try:
            self.raw_data[f"{planename}_LSPT"] = pd.read_csv(file)
            self.make_data_lspt(planename)
        except FileNotFoundError:
            # print(f"No forces.lspt file found in {planename} folder at {DB3D}!")
            pass

    def load_gnvp_case_convergence(
        self,
        planename: str,
        case: str,
        genu_version: int,
    ) -> None:
        """
        Loads the convergence data from gnvp.out and LOADS_aer.dat and stores it in the
        convergence_data dict. If LOADS_aer.dat exists it tries to load it and then load
        the convergence data from gnvp.out. If successfull it adds the error data to the
        dataframe containing the loads and stores it in the convergence_data dict.

        Args:
            planename (str): Planename
            case (str): Case Name
            genu_version (int): GNVP Version
        """
        # Get Load Convergence Data from LOADS_aer.dat
        file: str = os.path.join(DB3D, planename, case, "LOADS_aer.dat")

        loads: DataFrame | None = get_loads_convergence_3(file)
        if loads is not None:
            # Get Error Convergence Data from gnvp.out
            file = os.path.join(DB3D, planename, case, f"gnvp{genu_version}.out")
            # self.Convergence[planename][case] = addErrorConvergence2df(file, loads) # IT OUTPUTS LOTS OF WARNINGS
            with open(file, encoding="UTF-8") as f:
                lines: list[str] = f.readlines()
            time: list[int] = []
            error: list[float] = []
            errorm: list[float] = []
            for line in lines:
                if not line.startswith(" STEP="):
                    continue

                a: list[str] = line[6:].split()
                time.append(int(a[0]))
                error.append(float(a[2]))
                errorm.append(float(a[6]))
            try:
                foo: int = len(loads["TTIME"])
                if foo > len(time):
                    loads = loads.tail(len(time))
                else:
                    error = error[-foo:]
                    errorm = errorm[-foo:]
                loads["ERROR"] = error
                loads["ERRORM"] = errorm
                self.convergence_data[planename][case] = loads
            except ValueError as e:
                print(f"Some Run Had Problems!\n{e}")

    def get_planenames(self) -> list[str]:
        """
        Returns the list of planenames in the database.

        Returns:
            list[str]: List of planenames
        """
        return list(self.planes.keys())

    def get_polar(self, plane: str, mode: str) -> DataFrame | None:
        """
        Gets the polar for a given plane and mode.

        Args:
            plane (str): Planename
            mode (str): Mode (Potential, 2D, ONERA)

        Returns:
            DataFrame | None: DataFrame containing the polars or None if it doesn't exist.
        """
        try:
            cols: list[str] = ["AoA", f"CL_{mode}", f"CD_{mode}", f"Cm_{mode}"]
            return self.data[plane][cols].rename(
                columns={f"CL_{mode}": "CL", f"CD_{mode}": "CD", f"Cm_{mode}": "Cm"},
            )
        except KeyError:
            print("Polar Doesn't exist! You should compute it first!")
        return None

    def make_data_gnvp(self, plane: str, gnvp_version: int) -> None:
        """
        Args:
            plane (str): Plane name

        Formats Polars from Forces, calculates the aerodynamic coefficients and stores them in the data dict.
        ! TODO: Should get deprecated in favor of analysis logic in the future. Handled by the unhook function.
        """
        pln: Airplane = self.planes[plane]
        if plane not in self.raw_data.keys():
            return None
        AoA: np.ndarray[Any, np.dtype[floating[Any]]] = self.raw_data[plane]["AoA"] * np.pi / 180

        for enc, name in zip(["", "2D", "DS2D"], ["Potential", "2D", "ONERA"]):
            Fx: FloatArray = self.raw_data[plane][f"TFORC{enc}(1)"]
            # Fy = self.raw_data[plane][f"TFORC{enc}(2)"]
            Fz: FloatArray = self.raw_data[plane][f"TFORC{enc}(3)"]

            # Mx = self.raw_data[plane][f"TAMOM{enc}(1)"]
            My: FloatArray = self.raw_data[plane][f"TAMOM{enc}(2)"]
            # Mz = self.raw_data[plane][f"TAMOM{enc}(3)"]

            Fx_new: FloatArray = Fx * np.cos(
                AoA,
            ) + Fz * np.sin(
                AoA,
            )
            # Fy_new = Fy
            Fz_new: FloatArray = -Fx * np.sin(
                AoA,
            ) + Fz * np.cos(
                AoA,
            )

            My_new: FloatArray = My

            Q: float = 0.5 * 1.225 * 20.0**2
            try:
                state: State = self.states[pln.name]["Unstick"]
                Q = state.dynamic_pressure
            except KeyError:
                try:
                    Q = pln.dynamic_pressure
                except AttributeError:
                    print(
                        f"Plane {plane} doesn't have loaded State! Using Default velocity of 20m/s",
                    )
                    pass
            finally:
                S: float = pln.S
                MAC: float = pln.mean_aerodynamic_chord
                df = pd.DataFrame()
                df["AoA"] = self.raw_data[plane]["AoA"].astype("float")
                df[f"GNVP{gnvp_version} {name} CL"] = Fz_new / (Q * S)
                df[f"GNVP{gnvp_version} {name} CD"] = Fx_new / (Q * S)
                df[f"GNVP{gnvp_version} {name} Cm"] = My_new / (Q * S * MAC)

                # Merge the new data with data from other solvers
                if plane not in self.data.keys():
                    self.data[plane] = df
                else:
                    # If the data contain old data from a previous run, then delete the old data
                    # and replace it with the new one
                    if f"GNVP{gnvp_version} {name} CL" in self.data[plane].columns:
                        self.data[plane].drop(
                            columns=[
                                f"GNVP{gnvp_version} {name} CL",
                                f"GNVP{gnvp_version} {name} CD",
                                f"GNVP{gnvp_version} {name} Cm",
                            ],
                            inplace=True,
                        )

                    # Merge the data with the new one on the AoA column
                    self.data[plane] = self.data[plane].merge(df, on="AoA", how="outer")

                    # Sort the dataframe by AoA
                    self.data[plane].sort_values(by="AoA", inplace=True)

    def make_data_lspt(self, plane: str) -> None:
        """
        Args:
            plane (str): Plane name

        Formats Polars from Forces, calculates the aerodynamic coefficients and stores them in the data dict.
        """

        pln: Airplane = self.planes[plane]
        if f"{plane}_LSPT" not in self.raw_data.keys():
            return None

        for enc, name in zip(["", "_2D"], ["Potential", "2D"]):
            AoA: np.ndarray[Any, np.dtype[floating[Any]]] = self.raw_data[f"{plane}_LSPT"]["AoA"] * np.pi / 180
            Fx: FloatArray = self.raw_data[f"{plane}_LSPT"][f"D{enc}"]
            # Fy = self.raw_data[plane][f"TFORC{enc}(2)"]
            Fz: FloatArray = self.raw_data[f"{plane}_LSPT"][f"L{enc}"]

            # Mx = self.raw_data[plane][f"TAMOM{enc}(1)"]
            My: FloatArray = self.raw_data[f"{plane}_LSPT"][f"My{enc}"]
            # Mz = self.raw_data[plane][f"TAMOM{enc}(3)"]

            Fx_new: FloatArray = Fx * np.cos(
                0 * AoA,
            ) + Fz * np.sin(
                0 * AoA,
            )
            # Fy_new = Fy
            Fz_new: FloatArray = -Fx * np.sin(
                0 * AoA,
            ) + Fz * np.cos(
                0 * AoA,
            )

            My_new: FloatArray = My

            Q: float = 0.5 * 1.225 * 20.0**2
            try:
                state: State = self.states[pln.name]["Unstick"]
                Q = state.dynamic_pressure
            except KeyError:
                try:
                    Q = pln.dynamic_pressure
                except AttributeError:
                    print(
                        f"Plane {plane} doesn't have loaded State! Using Default velocity of 20m/s",
                    )
                    pass
            finally:
                S: float = pln.S
                MAC: float = pln.mean_aerodynamic_chord
                if f"{plane}" not in self.data.keys():
                    self.data[f"{plane}"] = pd.DataFrame()
                    self.data[f"{plane}"][f"AoA"] = AoA * 180 / np.pi
                    self.data[f"{plane}"][f"LSPT {name} CL"] = Fz_new / (Q * S)
                    self.data[f"{plane}"][f"LSPT {name} CD"] = Fx_new / (Q * S)
                    self.data[f"{plane}"][f"LSPT {name} Cm"] = My_new / (Q * S * MAC)
                else:
                    # Create a new dataframe with the new data and merge it with the old one
                    # on the AoA column
                    df: DataFrame = pd.DataFrame()
                    df[f"AoA"] = AoA * 180 / np.pi
                    df[f"LSPT {name} CL"] = Fz_new / (Q * S)
                    df[f"LSPT {name} CD"] = Fx_new / (Q * S)
                    df[f"LSPT {name} Cm"] = My_new / (Q * S * MAC)
                    self.data[f"{plane}"] = self.data[f"{plane}"].merge(df, on="AoA", how="outer")

                    # Sort the dataframe by AoA
                    self.data[f"{plane}"].sort_values(by="AoA", inplace=True)

    def __str__(self) -> Literal["Vehicle Database"]:
        return "Vehicle Database"

    def __enter__(self) -> None:
        pass

    def __exit__(self) -> None:
        pass
