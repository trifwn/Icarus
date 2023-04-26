from ICARUS.Airfoils.airfoilD import AirfoilD
from ICARUS.Core.struct import Struct
from . import DB2D, APPHOME
import pandas as pd
import os


class Database_2D():
    def __init__(self):
        self.HOMEDIR = APPHOME
        self.Data = Struct()
        
    def loadData(self):
        self.scan()
        self.airfoils = self.getAirfoils()

    def scan(self):
        try:
            os.chdir(DB2D)
        except FileNotFoundError:
            print(f'Database not found! Initializing Database at {DB2D}')
            os.makedirs(DB2D,exist_ok=True)
        folders = next(os.walk('.'))[1]
        Data = Struct()
        for folder in folders:
            os.chdir(folder)
            Data[folder] = self.scanReynolds()
            os.chdir(DB2D)

        self.Data = Struct()
        for i in Data.keys():
            if i not in self.Data.keys():
                self.Data[i] = Struct()
            for j in Data[i].keys():
                for k in Data[i][j].keys():
                    if k not in self.Data[i].keys():
                        self.Data[i][k] = Struct()
                    self.Data[i][k][j] = Data[i][j][k]
        os.chdir(self.HOMEDIR)

    def scanReynolds(self):
        airfoilDict = Struct()
        folders = next(os.walk('.'))[1]
        for folder in folders:
            os.chdir(folder)
            airfoilDict[folder[9:]] = self.scanSolvers()
            os.chdir('..')
        return airfoilDict

    def scanSolvers(self):
        reynDict = Struct()
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
        airfoils = Struct()
        for airf in list(self.Data.keys()):
            airfoils[airf] = AirfoilD.NACA(
                airf[4:], n_points=200)

        return airfoils

    def getSolver(self, airf):
        try:
            return list(self.Data[str(airf)].keys())
        except KeyError:
            print("Airfoil Doesn't exist! You should compute it first!")

    def getReynolds(self, airf):
        try:
            reynolds = []
            for solver in self.Data[str(airf)].keys():
                for reyn in self.Data[str(airf)][solver].keys():
                    reynolds.append(reyn)
            return reynolds
        except KeyError:
            print("Airfoil Doesn't exist! You should compute it first!")
