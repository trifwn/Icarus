import os
from typing import Any

import pandas as pd

from . import APPHOME
from . import DB2D
from ICARUS.Airfoils.airfoilD import AirfoilD
from ICARUS.Core.struct import Struct


class Database_2D:
    def __init__(self):
        self.HOMEDIR = APPHOME
        self.DATADIR = DB2D
        self.data: Struct = Struct()

    def loadData(self):
        self.scan()
        self.airfoils = self.getAirfoils()

    def scan(self) -> None:
        try:
            os.chdir(DB2D)
        except FileNotFoundError:
            print(f"Database not found! Initializing Database at {DB2D}")
            os.makedirs(DB2D, exist_ok=True)
        folders: list[str] = next(os.walk("."))[1]
        data = Struct()
        for folder in folders:
            os.chdir(folder)
            data[folder] = self.scanReynolds()
            os.chdir(DB2D)

        for i in data.keys():
            if i not in self.data.keys():
                self.data[i] = Struct()
                continue

            for j in data[i].keys():
                for k in data[i][j].keys():
                    if k not in self.data[i].keys():
                        self.data[i][k] = Struct()
                    self.data[i][k][j] = data[i][j][k]
        os.chdir(self.HOMEDIR)

    def scanReynolds(self) -> Struct:
        airfoilDict = Struct()
        folders: list[str] = next(os.walk("."))[1]
        for folder in folders:
            os.chdir(folder)
            airfoilDict[folder[9:]] = self.scanSolvers()
            os.chdir("..")
        return airfoilDict

    def scanSolvers(self) -> Struct:
        reynDict = Struct()
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
                reynDict[name] = pd.read_csv(file)
        return reynDict

    def getAirfoils(self) -> Struct:
        airfoils = Struct()
        for airf in list(self.data.keys()):
            airfoils[airf] = AirfoilD.NACA(airf[4:], n_points=200)

        return airfoils

    def getSolver(self, airf) -> list[Any] | None:
        try:
            return list(self.data[str(airf)].keys())
        except KeyError:
            print("Airfoil Doesn't exist! You should compute it first!")
            return None

    def getReynolds(self, airf):
        try:
            reynolds = []
            for solver in self.data[str(airf)].keys():
                for reyn in self.data[str(airf)][solver].keys():
                    reynolds.append(reyn)
            return reynolds
        except KeyError:
            print("Airfoil Doesn't exist! You should compute it first!")

    def __str__(self):
        return "Foil Database"

    def __enter__(self, obj):
        pass

    def __exit__(self):
        pass
