import sys

import numpy as np

from ICARUS.Airfoils.airfoilD import AirfoilD


def saveAirfoil(argv):
    options = argv
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
        save = "s" in options[0]
        filen = str(options[1])
        Airfoiln = str(options[2])
        mode = int(options[3])
        n_points = int(options[4])
    f = AirfoilD.NACA(Airfoiln, n_points=n_points)

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
    # saveAirfoil('-s','naca3123','3123','0')
