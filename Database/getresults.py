import numpy as np
from . import DB2D, DB3D
import os
import pandas as pd
from Airfoils import airfoil as af


class Database_2D():
    def __init__(self, HOMEDIR):
        self.HOMEDIR = HOMEDIR
        self.Data = {}
        self.scan()
        self.airfoils = {}

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
                elif solver == 'xf':
                    name = "Xfoil"
                reynDict[name] = pd.read_csv(file)
        return reynDict

    def getAirfoils(self):
        for airfoil in list(self.Data.keys()):
            self.airfoils[airfoil] = af.AirfoilData.NACA(
                airfoil[4:], n_points=200)

        return self.airfoils

    def getReynolds(self, airfoil):
        try:
            return list(self.Data[str(airfoil)].keys())
        except KeyError:
            print("Airfoil Doesn't exist! You should compute it first!")


class Database_3D():
    def __init__(self, HOMEDIR):
        self.HOMEDIR = HOMEDIR
        self.Data = {}
        self.scan()

    def scan(self):
        os.chdir(DB3D)
        folders = next(os.walk('.'))[1]
        for folder in folders:
            os.chdir(folder)
            self.Data[folder] = pd.read_csv(f"{DB3D}/{folder}/clcd.genu")
            os.chdir(DB3D)
        os.chdir(self.HOMEDIR)

    def getPlanes(self):
        return list(self.Data.keys())

    def getPolar(self, Plane):
        try:
            return self.Data[str(Plane)]
        except KeyError:
            print("Plane Doesn't exist! You should compute it first!")

    def makeDimensionless(self, plane, Q, S, MAC):

        self.Data[plane]["CD_Pot"] = - \
            self.Data[plane]["TFORC(1)"]*np.sin(self.Data[plane]
                                                ["AoA"]*np.pi/180) / (Q*S)
        self.Data[plane]["CD_2D"] = -self.Data[plane]["TFORC2D(1)"]*np.sin(
            self.Data[plane]["AoA"]*np.pi/180) / (Q*S)
        self.Data[plane]["CD_ONERA"] = -self.Data[plane]["TFORCDS2D(1)"]*np.sin(
            self.Data[plane]["AoA"]*np.pi/180) * (Q*S)

        self.Data[plane]["CL_Pot"] = self.Data[plane]["TFORC(3)"]*np.cos(
            self.Data[plane]["AoA"]*np.pi/180) / (Q*S)
        self.Data[plane]["CL_2D"] = self.Data[plane]["TFORC2D(3)"]*np.cos(
            self.Data[plane]["AoA"]*np.pi/180) / (Q*S)
        self.Data[plane]["CL_ONERA"] = self.Data[plane]["TFORCDS2D(3)"]*np.cos(
            self.Data[plane]["AoA"]*np.pi/180) / (Q*S)

        self.Data[plane]["Cm_Pot"] = self.Data[plane]["TAMOM(2)"] / (Q*S*MAC)
        self.Data[plane]["Cm_2D"] = self.Data[plane]["TAMOM2D(2)"] / (Q*S*MAC)
        self.Data[plane]["Cm_ONERA"] = self.Data[plane]["TAMOMDS2D(2)"] / (
            Q*S*MAC)
