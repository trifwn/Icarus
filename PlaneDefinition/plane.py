import numpy as np
import matplotlib.pyplot as plt
import os


class Airplane():
    def __init__(self, name, surfaces):

        self.CASENAME = name
        self.surfaces = surfaces

        toRemove = []
        for i, surface in enumerate(surfaces):
            if surface.isSymmetric == True:
                toRemove.append(i)
                l, r = surface.splitSymmetric()
                surfaces.append(l)
                surfaces.append(r)
        self.surfaces = [j for i, j in enumerate(
            self.surfaces) if i not in toRemove]

        self.airfoils = self.getAirfoils()
        self.Polars = {}
        self.angles = []
        self.bodies = []

    def getAirfoils(self):
        airfoils = []
        for surface in self.surfaces:
            if f"NACA{surface.airfoil.name}" not in airfoils:
                airfoils.append(f"NACA{surface.airfoil.name}")
        return airfoils

    def accessDB(self, HOMEDIR, DBDIR):
        os.chdir(DBDIR)
        CASEDIR = self.CASENAME
        os.system(f"mkdir -p {CASEDIR}")
        os.chdir(CASEDIR)
        self.CASEDIR = os.getcwd()
        self.HOMEDIR = HOMEDIR
        self.DBDIR = DBDIR
        os.chdir(HOMEDIR)

    def visAirplane(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for surface in self.surfaces:
            surface.plotWing(fig, ax)

        ax.set_title(self.CASENAME)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.axis('scaled')
        ax.view_init(30, 150)

    def angleCASE(self, angle):
        self.currAngle = angle
        self.angles.append(self.currAngle)
        if self.currAngle in self.Polars.keys():
            pass
        else:
            self.Polars[self.currAngle] = {}

        if angle >= 0:
            folder = str(angle)[::-1].zfill(7)[::-1] + "/"
        else:
            folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1] + "/"

        try:
            self.ANGLEDIR = f"{self.CASEDIR}/{folder}"
            os.system(f"mkdir -p {self.ANGLEDIR}")
        except AttributeError:
            print("DATABASE is not initialized!")

    def batchangles(self, angles):
        for angle in angles:
            self.angles.append(angle)
            if angle in self.Polars.keys():
                pass
            else:
                self.Polars[angle] = {}

    def runSolver(self, solver, args, kwargs={}):
        solver(*args, **kwargs)

    def setupSolver(self, setupsolver, args, kwargs={}):
        setupsolver(*args, **kwargs)

    def cleanRes(self, cleanFun, args, kwargs={}):
        cleanFun(*args, **kwargs)

    def makePolars(self, makePolFun, solverName, args, kwargs={}):
        polarsdf = makePolFun(*args, **kwargs)
        self.Polars = polarsdf

    def savePlane(self):
        print("SAVE not Implemented Yet")
