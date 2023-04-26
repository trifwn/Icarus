import numpy as np
from . import DB3D,APPHOME
import os
import pandas as pd

from ICARUS.Core.struct import Struct

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
jsonpickle_pd.register_handlers()


class Database_3D():
    def __init__(self):
        self.HOMEDIR = APPHOME
        self.rawData = Struct()
        self.Data = Struct()
        self.Planes = Struct()
        self.dynPlanes = Struct()
        self.Convergence = Struct()
    
    def loadData(self):
        self.scan()
        self.makeData()

    def scan(self):
        os.chdir(DB3D)
        folders = next(os.walk('.'))[1]
        for folder in folders:
            self.Convergence[folder] = Struct()
            os.chdir(folder)
            try:
                clcdLOC = os.path.join(DB3D, folder, 'clcd.genu')
                self.rawData[folder] = pd.read_csv(clcdLOC)
                files = next(os.walk('.'))[2]
                for file in files:
                    if file.endswith(".json") and not file.startswith("dyn"):
                        with open(f"{file}", 'r') as f:
                            json_obj = f.read()
                            try:
                                self.Planes[folder] = jsonpickle.decode(json_obj)
                            except Exception as e:
                                print(f"Error loading Plane {file} , got error {e}")
                    if file.endswith(".json") and file.startswith("dyn"):
                        with open(f"{file}", 'r') as f:
                            json_obj = f.read()
                            try:
                                self.dynPlanes[folder] = jsonpickle.decode(json_obj)
                            except Exception as e:
                                print(f"Error loading Dynamic Plane {file}, got error {e}")
            except FileNotFoundError:
                print(f"Plane {folder} doesn't contain polars!")
                files = next(os.walk('.'))[2]
                for file in files:
                    if file.endswith(".json"):
                        with open(f"{file}", 'r') as f:
                            json_obj = f.read()
                            plane = jsonpickle.decode(json_obj)
                        print('Plane object exists. Trying to create Polars...')
                        try:
                            from ICARUS.Software.GenuVP3.filesInterface import makePolar
                            genuPolarArgs = [plane.CASEDIR, plane.HOMEDIR]
                            plane.makePolars(makePolar, genuPolarArgs)
                            self.Planes[folder] = plane
                            clcdLOC = os.path.join(DB3D, folder, 'clcd.genu')
                            self.rawData[folder] = pd.read_csv(clcdLOC)
                        except e:
                            print('Failed to create Polars! Got Error:')
                            print(e)

            try:
                cases = next(os.walk('.'))[1]
                for case in cases:
                    os.chdir(case)
                    files = next(os.walk('.'))[2]
                    runExists = False
                    for file in files:
                        if file == "LOADS_aer.dat":
                            self.Convergence[folder][case] = pd.read_csv(
                                file, delim_whitespace=True,
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
                                    if line.startswith(" STEP="):
                                        a = line[6:].split()
                                        time.append(int(a[0]))
                                        error.append(float(a[2]))
                                        errorm.append(float(a[6]))
                                try:
                                    foo = len(
                                        self.Convergence[folder][case]['TTIME'])
                                    if foo > len(time):
                                        self.Convergence[folder][case] = self.Convergence[folder][case].tail(len(time))
                                    else:
                                        error = error[-foo:]
                                        errorm = errorm[-foo:]
                                    self.Convergence[folder][case]["ERROR"] = error
                                    self.Convergence[folder][case]["ERRORM"] = errorm
                                except ValueError as e:
                                    print(
                                        f"Some Run Had Problems! {folder} {case}\n{e}")

                    os.chdir('..')
            except FileNotFoundError:
                print("Convergence data not found!")

            os.chdir(DB3D)
        os.chdir(self.HOMEDIR)

    def getPlanes(self):
        return list(self.Planes.keys())

    def getPolar(self, plane, mode):
        try:
            cols = ["AoA", f"CL_{mode}", f"CD_{mode}", f"Cm_{mode}"]
            return self.Data[plane][cols].rename(columns={f"CL_{mode}": "CL",
                                                          f"CD_{mode}": "CD",
                                                          f"Cm_{mode}": "Cm"})
        except KeyError:
            print("Polar Doesn't exist! You should compute it first!")

    def makeData(self):
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

                Fx_new = Fx * np.cos(AoA) + Fz * np.sin(AoA)
                Fy_new = Fy
                Fz_new = -Fx * np.sin(AoA) + Fz * np.cos(AoA)

                My_new = My
                try:
                    Q = pln.Q
                    S = pln.S
                    MAC = pln.MAC
                    self.Data[plane][f"CL_{name}"] = Fz_new / (Q*S)
                    self.Data[plane][f"CD_{name}"] = Fx_new / (Q*S)
                    self.Data[plane][f"Cm_{name}"] = My_new / (Q*S*MAC)
                except AttributeError:
                    print("Plane doesn't have Q, S or MAC!")


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


def ang2case(angle):
    if angle >= 0:
        folder = str(angle)[::-1].zfill(7)[::-1] + "_AoA"
    else:
        folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1] + "_AoA"
    return folder


def dst2case(dst):
    if dst.var == "Trim":
        folder = "Trim"
    elif dst.isPositive:
        folder = "p" + \
            str(dst.amplitude)[::-1].zfill(6)[::-1] + f"_{dst.var}"
    else:
        folder = "m" + \
            str(dst.amplitude)[::-1].strip("-").zfill(6)[::-1] + f"_{dst.var}"
    return folder
