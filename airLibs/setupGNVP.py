import os
import numpy as np


def runGNVP(Reynolds, Mach, angle):

    # INPUT
    fname = 'input'
    with open(fname, 'r') as file:
        data = file.readlines()
    # data[1] =
    # data[2] =
    # data[3] =
    with open(fname, 'w') as file:
        file.writelines(data)

    # DFILE
    fname = 'dfile.yours'
    with open(fname, 'r') as file:
        data = file.readlines()
    #
    #
    #
    with open(fname, 'w') as file:
        file.writelines(data)

    # HERMES.GEO
    fname = 'hermes.geo'
    with open(fname, 'r') as file:
        data = file.readlines()
    #
    #
    #
    with open(fname, 'w') as file:
        file.writelines(data)

    # BLD FILES
    surfaces = ['Lwing', 'Rwing', 'Ltail', 'Rtail', 'rudder']
    for surface in surfaces:
        fname = f"{surface}.bld"
        with open(fname, 'r') as file:
            data = file.readlines()
        #
        #
        #
        with open(fname, 'w') as file:
            file.writelines(data)

    # CLD FILES
    airfoils = ['4415', '0008']
    for airfoil in airfoils:
        fname = f"{airfoil}.cld"
        with open(fname, 'r') as file:
            data = file.readlines()
            #
            #
            #
            with open(fname, 'w') as file:
                file.writelines(data)

    # RUN GNVP
    # os.system('./gnvp > gnvp.out')
    return 'Not implemented yet'


def removeResults():
    return 'Not implemented yet'
