import numpy as np
import matplotlib.pyplot as plt
import os
import jsonpickle


class Airplane():
    def __init__(self, name, surfaces, disturbances=None, orientation=None):

        self.name = name
        self.surfaces = surfaces
        self.masses = []

        if disturbances is None:
            self.disturbances = []
        else:
            self.disturbances = disturbances

        if orientation is None:
            self.orientation = [0., 0., 0.]
        else:
            self.orientation = orientation

        toRemove = []
        for i, surface in enumerate(surfaces):
            if surface.name == "wing":
                # self.mainWing = surface
                self.S = surface.S
                self.MAC = surface.MAC
                self.AR = surface.AR

            if surface.isSymmetric == True:
                toRemove.append(i)
                l, r = surface.splitSymmetric()
                surfaces.append(l)
                surfaces.append(r)
        self.surfaces = [j for i, j in enumerate(
            self.surfaces) if i not in toRemove]

        self.airfoils = self.getAirfoils()
        self.bodies = []
        self.masses = []
        self.Inertia = []

        self.M = 0
        for surface in self.surfaces:
            mass = (surface.mass, surface.CG)
            mom = (surface.I)

            self.M += surface.mass
            self.Inertia.append(mom)
            self.masses.append(mass)

        self.CG = self.findCG()
        self.I = self.findInertia(self.CG)

    def findCG(self):
        x_cm = 0
        y_cm = 0
        z_cm = 0
        self.M = 0
        for m, r in self.masses:
            self.M += m
            x_cm += m * r[0]
            y_cm += m * r[1]
            z_cm += m * r[2]
        return np.array((x_cm, y_cm, z_cm)) / self.M

    def findInertia(self, point):
        I_xx = 0
        I_yy = 0
        I_zz = 0
        self.M = 0
        for I in self.Inertia:
            I_xx += I[0]
            I_yy += I[1]
            I_zz += I[2]

        for m, r_bod in self.masses:
            r = point - r_bod
            I_xx += m * (r[1]**2 + r[2]**2)
            I_yy += m * (r[0]**2 + r[2]**2)
            I_zz += m * (r[0]**2 + r[1]**2)
        return np.array((I_xx, I_yy, I_zz))

    def getAirfoils(self):
        airfoils = []
        for surface in self.surfaces:
            if f"NACA{surface.airfoil.name}" not in airfoils:
                airfoils.append(f"NACA{surface.airfoil.name}")
        return airfoils

    def accessDB(self, HOMEDIR, DBDIR):
        os.chdir(DBDIR)
        CASEDIR = self.name
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

        ax.set_title(self.name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.axis('scaled')
        ax.view_init(30, 150)

    def defineSim(self, U, dens):
        self.U = U
        self.dens = dens
        self.Q = 0.5 * dens * U**2

    def runSolver(self, solver, args, kwargs={}):
        solver(*args, **kwargs)

    def setupSolver(self, setupsolver, args, kwargs={}):
        setupsolver(*args, **kwargs)

    def cleanRes(self, cleanFun, args, kwargs={}):
        cleanFun(*args, **kwargs)

    def makePolars(self, makePolFun, args, kwargs={}):
        polarsdf = makePolFun(*args, **kwargs)
        self.Polars = polarsdf

    def toJSON(self):
        return jsonpickle.encode(self)

    def save(self):
        os.chdir(self.CASEDIR)
        with open(f'{self.name}.json', 'w') as f:
            f.write(self.toJSON())
        os.chdir(self.HOMEDIR)
