
import airfoils as af

import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import os
import re

# # Airfoil
# ##### 0 = Read from python module
# ##### 1 = Read from airfoiltools.com
# ##### 2 = load from file


class AirfoilD(af.Airfoil):
    def __init__(self, upper, lower, naca, n_points):
        super().__init__(upper, lower)
        self.name = naca
        self.fname = f"naca{naca}"
        self.n_points = n_points
        self.airfoil2Selig()
        self.Reynolds = []
        self.Polars = {}
        # self.getFromWeb()

    @classmethod
    def NACA(cls, naca, n_points=200):
        re_4digits = re.compile(r"^\d{4}$")

        if re_4digits.match(naca):
            p = float(naca[0])/10
            m = float(naca[1])/100
            xx = float(naca[2:4])/100
        else:
            raise af.NACADefintionError(
                "Identifier not recognised as valid NACA 4 definition")

        upper, lower = af.gen_NACA4_airfoil(p, m, xx, n_points)
        self = cls(upper, lower, naca, n_points)
        self.p = p 
        self.m = m
        self.xx = xx
                
        return self
    
    def camber_line_NACA4(self,points):
        p = self.p
        m = self.m
        xx = self.xx
        
        res = np.zeros_like(points)
        for i,x in enumerate(points):
            if x < p:
                res[i] = m/p**2 * (2*p*x - x**2)
            else:
                res[i] = m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2)
        return res

    def airfoil2Selig(self):
        x_points = np.hstack((self._x_upper[::-1], self._x_lower[1:])).T
        y_points = np.hstack((self._y_upper[::-1], self._y_lower[1:])).T
        # y_points[0]=0
        # y_points[-1]=0
        self.selig = np.vstack((x_points, y_points))

    def getFromWeb(self):
        link = "https://m-selig.ae.illinois.edu/ads/coord/naca" + self.name + ".dat"
        with urllib.request.urlopen(link) as url:
            s = url.read().decode("UTF-8")
        s = s.split()
        s = s[2:]
        x, y = list(), list()
        for i in range(int(len(s) / 2)):
            x.append(float(s[2 * i]))
            y.append(float(s[2 * i + 1]))
        # y[0] = 0
        # y[-1]= 0
        self.selig2 = np.vstack((x, y))

    def accessDB(self, HOMEDIR, DBDIR):
        os.chdir(DBDIR)
        AFDIR = f"NACA{self.name}"
        os.system(f"mkdir -p {AFDIR}")
        os.chdir(AFDIR)
        self.AFDIR = os.getcwd()
        self.HOMEDIR = HOMEDIR
        self.DBDIR = DBDIR
        os.chdir(HOMEDIR)
        exists = False
        for i in os.listdir(self.AFDIR):
            if i.startswith("naca"):
                self.airfile = f"{self.AFDIR}/{i}"
                exists = True
        if True:
            self.save()

    def reynCASE(self, Reyn):
        self.currReyn = np.format_float_scientific(
            Reyn, sign=False, precision=3)
        self.Reynolds.append(self.currReyn)
        if self.currReyn in self.Polars.keys():
            pass
        else:
            self.Polars[self.currReyn] = {}

        try:
            self.REYNDIR = f"{self.AFDIR}/Reynolds_{np.format_float_scientific(Reyn,sign=False,precision=3).replace('+', '')}"
            os.system(f"mkdir -p {self.REYNDIR}")
            os.system(f"cp {self.airfile} {self.REYNDIR}")
        except AttributeError:
            print("DATABASE is not initialized!")

    def save(self):
        self.airfile = f"{self.AFDIR}/naca{self.name}"
        pt0 = self.selig
        np.savetxt(self.airfile, pt0.T)

    def plotAirfoil(self):
        pts = self.selig
        x, y = pts
        plt.plot(x[: self.n_points], y[: self.n_points], "r")
        plt.plot(x[self.n_points:], y[self.n_points:], "b")
        plt.axis("scaled")

    def runSolver(self, solver, args, kwargs={}):
        solver(*args, **kwargs)

    def setupSolver(self, setupsolver, args, kwargs={}):
        setupsolver(*args, **kwargs)

    def cleanRes(self, cleanFun, args, kwargs={}):
        cleanFun(*args, **kwargs)

    def makePolars(self, makePolFun, solverName, args, kwargs={}):
        polarsdf = makePolFun(*args, **kwargs)
        self.Polars[self.currReyn][solverName] = polarsdf
