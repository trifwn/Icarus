import os
import re
import shutil
import urllib.request
from typing import Any

import airfoils as af
import matplotlib.pyplot as plt
import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray

# # Airfoil
# ##### 0 = Read from python module
# ##### 1 = Read from airfoiltools.com
# ##### 2 = load from file


class AirfoilD(af.Airfoil):
    def __init__(self, upper, lower, naca: str, n_points: int) -> None:
        super().__init__(upper, lower)
        self.name: str = naca
        self.fname: str = f"naca{naca}"
        self.n_points: int = n_points
        self.airfoil2Selig()
        self.Reynolds: list[float] = []
        self.Polars: dict = {}
        # self.getFromWeb()

    @classmethod
    def NACA(cls, naca: str, n_points: int = 200) -> "AirfoilD":
        re_4digits: re.Pattern[str] = re.compile(r"^\d{4}$")

        if re_4digits.match(naca):
            p: float = float(naca[0]) / 10
            m: float = float(naca[1]) / 100
            xx: float = float(naca[2:4]) / 100
        else:
            raise af.NACADefintionError(
                "Identifier not recognised as valid NACA 4 definition",
            )

        upper, lower = af.gen_NACA4_airfoil(p, m, xx, n_points)
        self = cls(upper, lower, naca, n_points)
        self.set_NACA4_digits(p, m, xx)

        return self

    def set_NACA4_digits(self, p: float, m: float, xx: float) -> None:
        self.p: float = p
        self.m: float = m
        self.xx: float = xx

    def camber_line_NACA4(self, points) -> ndarray[Any, dtype[floating[Any]]]:
        p: float = self.p
        m: float = self.m

        res: ndarray[Any, dtype[floating[Any]]] = np.zeros_like(points)
        for i, x in enumerate(points):
            if x < p:
                res[i] = m / p**2 * (2 * p * x - x**2)
            else:
                res[i] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2)
        return res

    def airfoil2Selig(self) -> None:
        x_points = np.hstack((self._x_upper[::-1], self._x_lower[1:])).T
        y_points = np.hstack((self._y_upper[::-1], self._y_lower[1:])).T
        # y_points[0]=0
        # y_points[-1]=0
        self.selig = np.vstack((x_points, y_points))

    def getFromWeb(self) -> None:
        link: str = (
            "https://m-selig.ae.illinois.edu/ads/coord/naca" + self.name + ".dat"
        )
        with urllib.request.urlopen(link) as url:
            site_data: str = url.read().decode("UTF-8")
        s: list[str] = site_data.split()
        s = s[2:]
        x, y = [], []
        for i in range(int(len(s) / 2)):
            temp: float = float(s[2 * i])
            x.append(temp)
            temp = float(s[2 * i + 1])
            y.append(temp)
        # y[0] = 0
        # y[-1]= 0
        self.selig2 = np.vstack((x, y))

    def accessDB(self, HOMEDIR: str, DBDIR: str) -> None:
        os.chdir(DBDIR)
        AFDIR: str = f"NACA{self.name}"
        os.makedirs(AFDIR, exist_ok=True)
        os.chdir(AFDIR)
        self.AFDIR: str = os.getcwd()
        self.HOMEDIR: str = HOMEDIR
        self.DBDIR: str = DBDIR
        os.chdir(HOMEDIR)
        exists = False
        for i in os.listdir(self.AFDIR):
            if i.startswith("naca"):
                self.airfile = os.path.join(self.AFDIR, i)
                exists = True
        if exists:
            self.save()

    def reynCASE(self, Reynolds_number: float) -> None:
        Reyn: str = np.format_float_scientific(Reynolds_number, sign=False, precision=3)
        self.currReyn: str = Reyn
        self.Reynolds.append(self.currReyn)
        if self.currReyn in self.Polars.keys():
            pass
        else:
            self.Polars[self.currReyn] = {}

        try:
            self.REYNDIR = os.path.join(self.AFDIR, f"Reynolds_{Reyn.replace('+', '')}")
            os.makedirs(self.REYNDIR, exist_ok=True)
            shutil.copy(self.airfile, self.REYNDIR)
        except AttributeError:
            print("DATABASE is not initialized!")

    def save(self) -> None:
        self.airfile: str = os.path.join(self.AFDIR, f"naca{self.name}")
        pt0 = self.selig
        np.savetxt(self.airfile, pt0.T)

    def plotAirfoil(self) -> None:
        pts = self.selig
        x, y = pts
        plt.plot(x[: self.n_points], y[: self.n_points], "r")
        plt.plot(x[self.n_points :], y[self.n_points :], "b")
        plt.axis("scaled")

    def runSolver(self, solver, args, kwargs={}) -> None:
        solver(*args, **kwargs)

    def setupSolver(self, setupsolver, args, kwargs={}) -> None:
        setupsolver(*args, **kwargs)

    def cleanRes(self, cleanFun, args, kwargs={}) -> None:
        cleanFun(*args, **kwargs)

    def makePolars(self, makePolFun, solverName, args, kwargs={}) -> None:
        polarsdf = makePolFun(*args, **kwargs)
        self.Polars[self.currReyn][solverName] = polarsdf
