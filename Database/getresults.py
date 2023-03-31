import numpy as np
from . import DB2D, DB3D
import os
import pandas as pd
from Airfoils import airfoil as af

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
jsonpickle_pd.register_handlers()


class Database_2D():
    def __init__(self, HOMEDIR):
        self.HOMEDIR = HOMEDIR
        self.Data = {}
        self.scan()
        self.airfoils = self.getAirfoils()

    def scan(self):
        os.chdir(DB2D)
        folders = next(os.walk('.'))[1]
        for folder in folders:
            os.chdir(folder)
            self.Data[folder] = self.scanReynolds()
            os.chdir(DB2D)
        os.chdir(self.HOMEDIR)

    def scanReynolds(self):
        airfoilDict = {}
        folders = next(os.walk('.'))[1]
        for folder in folders:
            os.chdir(folder)
            airfoilDict[folder[9:]] = self.scanSolvers()
            os.chdir('../')
        return airfoilDict

    def scanSolvers(self):
        reynDict = {}
        files = next(os.walk('.'))[2]
        for file in files:
            if file.startswith('clcd'):
                solver = file[5:]
                if solver == "f2w":
                    name = "Foil2Wake"
                elif solver == 'of':
                    name = "OpenFoam"
                elif solver == 'xfoil':
                    name = "Xfoil"
                reynDict[name] = pd.read_csv(file)
        return reynDict

    def getAirfoils(self):
        airfoils = {}
        for airf in list(self.Data.keys()):
            airfoils[airf] = af.AirfoilData.NACA(
                airf[4:], n_points=200)

        return airfoils

    def getReynolds(self, airfoil):
        try:
            return list(self.Data[str(airfoil)].keys())
        except KeyError:
            print("Airfoil Doesn't exist! You should compute it first!")


class Database_3D():
    def __init__(self, HOMEDIR):
        self.HOMEDIR = HOMEDIR
        self.rawData = {}
        self.Data = {}
        self.Planes = {}
        self.Convergence = {}
        self.scan()
        self.makeData()

    def scan(self):
        os.chdir(DB3D)
        folders = next(os.walk('.'))[1]
        for folder in folders:
            self.Convergence[folder] = {}
            os.chdir(folder)
            try:
                self.rawData[folder] = pd.read_csv(
                    f"{DB3D}/{folder}/clcd.genu")
                files = next(os.walk('.'))[2]
                for file in files:
                    if file.endswith(".json") and not file.startswith("dyn"):
                        with open(f"{file}", 'r') as f:
                            json_obj = f.read()
                            self.Planes[folder] = jsonpickle.decode(json_obj)
            except FileNotFoundError:
                print(f"Plane {folder} doesn't contain polars!")

            try:
                cases = next(os.walk('.'))[1]
                for case in cases:
                    os.chdir(case)
                    files = next(os.walk('.'))[2]
                    runExists = False
                    for file in files:
                        if file == "LOADS_aer.dat":
                            self.Convergence[folder][case] = pd.read_csv(
                                file, delim_whitespace=True, header=None,
                                names=cols)
                            runExists = True
                            break
                    if runExists:
                        for file in files:
                            if (file == "gnvp.out") and runExists:
                                with open(file, 'r') as f:
                                    a = f.readlines()
                                time = []
                                error = []
                                errorm = []
                                for line in a:
                                    if line.startswith(" STEP"):
                                        a = line.split()
                                        time.append(int(a[1]))
                                        error.append(float(a[3]))
                                        errorm.append(float(a[7]))
                                self.Convergence[folder][case]["ERROR"] = error
                                self.Convergence[folder][case]["ERRORM"] = errorm

                    os.chdir('../')
            except FileNotFoundError:
                print("Convergence data not found!")

            os.chdir(DB3D)
        os.chdir(self.HOMEDIR)

    def getPlanes(self):
        return list(self.Planes.keys())

    def importXFLRpolar(self, FILENAME):
        # import csv into pandas Dataframe and skip first 7 rows
        df = pd.read_csv(FILENAME, skiprows=7,
                         delim_whitespace=True, on_bad_lines="skip")
        # rename columns
        df.rename(columns={'alpha': 'AoA'}, inplace=True)

        # convert to float
        df = df.astype(float)
        self.Data["XFLR"] = df
        return df

    def getPolar(self, plane, mode):
        try:
            cols = ["AoA", f"CL_{mode}", f"CD_{mode}", f"Cm_{mode}"]
            return self.Data[plane][cols].rename(columns={f"CL_{mode}": "CL",
                                                          f"CD_{mode}": "CD",
                                                          f"Cm_{mode}": "Cm"})
        except KeyError:
            print("Polar Doesn't exist! You should compute it first!")

    def makeData(self):
        beta = 0
        for plane in list(self.Planes.keys()):
            self.Data[plane] = pd.DataFrame()
            pln = self.Planes[plane]

            self.Data[plane]["AoA"] = self.rawData[plane]["AoA"]
            AoA = self.rawData[plane]["AoA"] * np.pi/180
            for enc, name in zip(["", "2D", "DS2D"], ["Potential", "2D", "ONERA"]):
                Fx = self.rawData[plane][f"TFORC{enc}(1)"]
                Fy = self.rawData[plane][f"TFORC{enc}(2)"]
                Fz = self.rawData[plane][f"TFORC{enc}(3)"]

                Mx = self.rawData[plane][f"TAMOM{enc}(1)"]
                My = self.rawData[plane][f"TAMOM{enc}(2)"]
                Mz = self.rawData[plane][f"TAMOM{enc}(3)"]

                Fx_new = Fx * np.cos(-AoA) - Fz * np.sin(-AoA)
                Fy_new = Fy
                Fz_new = Fx * np.sin(-AoA) + Fz * np.cos(-AoA)

                My_new = My - \
                    Fx_new * pln.CG[2] + \
                    Fy_new * pln.CG[1] + \
                    Fz_new * pln.CG[0]

                self.Data[plane][f"CL_{name}"] = Fz_new / (pln.Q*pln.S)
                self.Data[plane][f"CD_{name}"] = Fx_new / (pln.Q*pln.S)
                self.Data[plane][f"Cm_{name}"] = My_new / (pln.Q*pln.S*pln.MAC)


cols = ["TTIME",
        "PSIB",
        "TFORC(1)",
        "TFORC(2)",
        "TFORC(3)",
        "TAMOM(1)",
        "TAMOM(2)",
        "TAMOM(3)",
        "TFORC2D(1)",
        "TFORC2D(2)",
        "TFORC2D(3)",
        "TAMOM2D(1)",
        "TAMOM2D(2)",
        "TAMOM2D(3)",
        "TFORCDS2D(1)",
        "TFORCDS2D(2)",
        "TFORCDS2D(3)",
        "TAMOMDS2D(1)",
        "TAMOMDS2D(2)",
        "TAMOMDS2D(3)"]
