import os
from typing import Any
from typing import Literal

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
import numpy as np
import pandas as pd

from . import APPHOME
from . import DB3D
from ICARUS.Core.struct import Struct
from ICARUS.Software.GenuVP3.postProcess.convergence import getLoadsConvergence
from ICARUS.Vehicle.plane import Airplane

# from ICARUS.Software.GenuVP3.postProcess.convergence import addErrorConvergence2df

jsonpickle_pd.register_handlers()


class Database_3D:
    """Class to represent the 3D Database. It contains all the information and results
    of the 3D analysis of the vehicles."""

    def __init__(self):
        self.HOMEDIR = APPHOME
        self.DATADIR = DB3D
        self.raw_data = Struct()
        self.data = Struct()
        self.planes = Struct()
        self.dyn_planes = Struct()
        self.convergence_data = Struct()

    def loadData(self):
        self.scan()
        self.makeData()

    def scan(self):
        planenames = next(os.walk(DB3D))[1]
        for plane in planenames:  # For each plane planename == folder
            # if plane == 'bmark':
            #     continue
            plane_found = False
            # Load Plane object
            file = os.path.join(DB3D, plane, f"{plane}.json")
            plane_found = self.loadPlaneFromFile(plane, file)

            # Load DynPlane object
            file = os.path.join(DB3D, plane, f"dyn_{plane}.json")
            temp = self.loadDynPlaneFromFile(plane, file)

            plane_found = plane_found or temp

            # Loading Forces from forces.gnvp3 file
            file = os.path.join(DB3D, plane, "forces.gnvp3")
            self.loadGNVPForces(plane, file)

            if plane_found:
                self.convergence_data[plane] = Struct()
                cases = next(os.walk(os.path.join(DB3D, plane)))[1]
                for case in cases:
                    if case.startswith("Dyn") or case.startswith("Sens"):
                        continue
                    self.loadGNVPcaseConvergence(plane, case)

    def loadPlaneFromFile(self, name: str, file: str) -> bool:
        """Function to get Plane Object from file and decode it.

        Args:
            name (str): planename
            file (str): filename

        Returns:
            bool : whether the plane was found or not
        """
        try:
            with open(file, encoding="UTF-8") as f:
                json_obj = f.read()
                try:
                    self.planes[name] = jsonpickle.decode(json_obj)
                except Exception as error:
                    print(f"Error decoding Plane object {name}! Got error {error}")
            plane_found = True
        except FileNotFoundError:
            print(f"No Plane object found in {name} folder at {file}!")
            plane_found = False
        return plane_found

    def loadDynPlaneFromFile(self, name: str, file: str):
        """Function to retrieve Dyn Plane Object from file and decode it.

        Args:
            name (str): _description_
            file (str): _description_
        """
        try:
            with open(file, encoding="UTF-8") as f:
                json_obj = f.read()
                try:
                    self.dyn_planes[name] = jsonpickle.decode(json_obj)
                    if name not in self.planes.keys():
                        print("Plane object doesnt exist! Creating it...")
                        self.planes[name] = self.dyn_planes[name]
                except Exception as e:
                    print(f"Error decoding Dyn Plane object {name} ! Got error {e}")
        except FileNotFoundError:
            print(f"No Plane object found in {name} folder at {file}!")

    def loadGNVPForces(self, planename, file):
        # Should get deprecated in favor of analysis logic in the future
        try:
            self.raw_data[planename] = pd.read_csv(file)
            return
        except FileNotFoundError:
            print(f"No forces.gnvp3 file found in {planename} folder at {DB3D}!")
            if planename in self.planes.keys():
                print(
                    "Since plane object exists with that name trying to create polars...",
                )
                pln = self.planes[planename]
                try:
                    from ICARUS.Software.GenuVP3.filesInterface import makePolar

                    CASEDIR = os.path.join(DB3D, pln.CASEDIR)
                    makePolar(CASEDIR, self.HOMEDIR)
                    file = os.path.join(DB3D, planename, "forces.gnvp3")
                    self.raw_data[planename] = pd.read_csv(file)
                except Exception as e:
                    print(f"Failed to create Polars! Got Error:\n{e}")

    def loadGNVPcaseConvergence(self, planename, case):
        # Get Load Convergence Data from LOADS_aer.dat
        file = os.path.join(DB3D, planename, case, "LOADS_aer.dat")

        loads = getLoadsConvergence(file)
        if loads is not None:
            # Get Error Convergence Data from gnvp.out
            file = os.path.join(DB3D, planename, case, "gnvp.out")
            # self.Convergence[planename][case] = addErrorConvergence2df(file, loads) # IT OUTPUTS LOTS OF WARNINGS
            with open(file, encoding="UTF-8") as f:
                lines = f.readlines()
            time, error, errorm = [], [], []
            for line in lines:
                if not line.startswith(" STEP="):
                    continue

                a = line[6:].split()
                time.append(int(a[0]))
                error.append(float(a[2]))
                errorm.append(float(a[6]))
            try:
                foo = len(loads["TTIME"])
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

    def getPlanes(self) -> list[str]:
        return list(self.planes.keys())

    def getPolar(self, plane, mode):
        try:
            cols: list[str] = ["AoA", f"CL_{mode}", f"CD_{mode}", f"Cm_{mode}"]
            return self.data[plane][cols].rename(
                columns={f"CL_{mode}": "CL", f"CD_{mode}": "CD", f"Cm_{mode}": "Cm"},
            )
        except KeyError:
            print("Polar Doesn't exist! You should compute it first!")

    def makeData(self) -> None:
        for plane in list(self.planes.keys()):
            self.data[plane] = pd.DataFrame()
            pln: Airplane = self.planes[plane]
            self.data[plane]["AoA"] = self.raw_data[plane]["AoA"]
            AoA: np.ndarray[Any, np.dtype[np.floating]] = (
                self.raw_data[plane]["AoA"] * np.pi / 180
            )
            for enc, name in zip(["", "2D", "DS2D"], ["Potential", "2D", "ONERA"]):
                Fx: np.ndarray = self.raw_data[plane][f"TFORC{enc}(1)"]
                # Fy = self.raw_data[plane][f"TFORC{enc}(2)"]
                Fz: np.ndarray = self.raw_data[plane][f"TFORC{enc}(3)"]

                # Mx = self.raw_data[plane][f"TAMOM{enc}(1)"]
                My: np.ndarray = self.raw_data[plane][f"TAMOM{enc}(2)"]
                # Mz = self.raw_data[plane][f"TAMOM{enc}(3)"]

                Fx_new: np.ndarray = Fx * np.cos(AoA) + Fz * np.sin(AoA)
                # Fy_new = Fy
                Fz_new: np.ndarray = -Fx * np.sin(AoA) + Fz * np.cos(AoA)

                My_new: np.ndarray = My
                try:
                    Q: float = pln.dynamic_pressure
                    S: float = pln.S
                    mean_aerodynamic_chord: float = pln.mean_aerodynamic_chord
                    self.data[plane][f"CL_{name}"] = Fz_new / (Q * S)
                    self.data[plane][f"CD_{name}"] = Fx_new / (Q * S)
                    self.data[plane][f"Cm_{name}"] = My_new / (
                        Q * S * mean_aerodynamic_chord
                    )
                except AttributeError:
                    print("Plane doesn't have Q, S or mean_aerodynamic_chord!")

    def __str__(self) -> Literal["Vehicle Database"]:
        return "Vehicle Database"

    def __enter__(self, obj) -> None:
        pass

    def __exit__(self) -> None:
        pass
