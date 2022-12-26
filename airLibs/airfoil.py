
from airfoils import Airfoil
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import os
import sys
import getopt

# # Airfoil
# ##### 0 = Read from python module
# ##### 1 = Read from airfoiltools.com
# ##### 2 = load from file


class AirfoilS(Airfoil):
    def __init__(self, upper, lower):
        super().__init__(upper, lower)
        self.airfoil2Selig()
        self.getFromWeb()

    @classmethod
    def NACA(self, naca, n_points=200):
        self.name = naca
        if len(naca) == 4:
            return self.NACA4(naca, n_points)
        else:
            print("ERROR NOT 4 DIGITS")

    def airfoil2Selig(self):
        x_points = np.hstack((self._x_upper[::-1], self._x_lower)).T
        y_points = np.hstack((self._y_upper[::-1], self._y_lower)).T
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


def saveAirfoil(argv):
    options = argv
    if options == []:
        print("No options defined try -h")
        return 0
    elif options[0] == '-h':
        print('options: \n-s Save to file\n Usage: python airfoil.py -s file naca mode\
                \n\nnaca = 4 or 5 digit NACA\nmode: 0-> Load from lib, 1-> Load from File, 2-> Load from Web')
        return 0
    else:
        save = 's' in options[0]
        filen = str(options[1])
        Airfoiln = str(options[2])
        mode = int(options[3])
        n_points = int(options[4])
    f = AirfoilS.NACA(Airfoiln, n_points=n_points)

    # # Return and Save to file
    if mode == 0:
        # # Load from Lib
        pt0 = f.selig
        if save == True:
            np.savetxt(filen, pt0.T)
    elif mode == 1:
        # # Load from the file mode 1
        pt1 = np.loadtxt(filen)
        if save == True:
            np.savetxt(filen, pt1)
        return pt1
    elif mode == 2:
        # # Fetch from the web mode 2
        pt2 = f.selig2
        if save == True:
            np.savetxt(filen, pt2.T)
        return pt2.T
    return pt0.T


if __name__ == "__main__":
    saveAirfoil(sys.argv[1:])
