import sys

import numpy as np

from ICARUS.Airfoils.airfoilD import AirfoilD
from ICARUS.Core.types import FloatArray


def saveAirfoil(options: list[str]) -> FloatArray | int | None:
    if options == []:
        print("No options defined try -h")
        return 0
    elif options[0] == "-h":
        print(
            "options: \n-s Save to file\n Usage: python airfoil.py -s file naca mode\
                \n\nnaca = 4 or 5 digit NACA\nmode: 0-> Load from lib, 1-> Load from File, 2-> Load from Web",
        )
        return 0
    else:
        save: bool = "s" in options[0]
        filen = str(options[1])
        Airfoiln = str(options[2])
        mode = int(options[3])
        n_points = int(options[4])
    f: AirfoilD = AirfoilD.naca(Airfoiln, n_points=n_points)

    # # Return and Save to file
    if mode == 0:
        # # Load from Lib
        pt0: FloatArray = f.selig
        if save:
            np.savetxt(filen, pt0.T)
        return pt0.T
    elif mode == 1:
        # # Load from the file mode 1
        pt1: FloatArray = np.loadtxt(filen)
        if save:
            np.savetxt(filen, pt1)
        return pt1
    elif mode == 2:
        # # Fetch from the web mode 2
        pt2: FloatArray = f.selig_web
        if save:
            np.savetxt(filen, pt2.T)
        return pt2.T
    else:
        print("Invalid Mode")
        return None


if __name__ == "__main__":
    # python airfoil.py -s file naca mode
    # saveAirfoil('-s','naca3123','3123','0')
    saveAirfoil(sys.argv[1:])
