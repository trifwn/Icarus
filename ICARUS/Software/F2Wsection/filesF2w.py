import os
import shutil
import numpy as np


def iofile(airfile):
    fname = 'io.files'
    with open(fname, 'r') as file:
        data = file.readlines()
    data[1] = 'design.inp\n'
    data[2] = 'f2w.inp\n'
    data[3] = airfile + '\n'
    with open(fname, 'w') as file:
        file.writelines(data)


def designFile(NoA, angles, name):
    fname = 'design_' + name+'.inp'
    with open(fname, 'r') as file:
        data = file.readlines()
    data[2] = str(NoA) + '           ! No of ANGLES\n'
    data = data[:3]
    for ang in angles:
        data.append(str(ang) + '\n')
    data.append('ANGLE DIRECTORIES (8 CHAR MAX!!!)\n')
    for ang in angles:
        if name == 'pos':
            data.append(str(ang)[::-1].zfill(7)[::-1] + '/\n')
        else:
            data.append(
                'm'+str(ang)[::-1].strip('-').zfill(6)[::-1] + '/\n')
    with open(fname, 'w') as file:
        file.writelines(data)


def f2winp(Reynolds, Mach, ftripL, ftripU, name):
    fname = 'f2w_' + name+'.inp'
    with open(fname, 'r') as file:
        data = file.readlines()

    data[2] = "201       ! NTIMEM\n"
    data[3] = "0.010     ! DT1\n"
    data[4] = "50000     ! DT2\n"  # IS NOT IMPLEMENTED
    data[5] = "0.025     ! EPS1\n"
    data[6] = "0.025     ! EPS2\n"
    data[7] = " 1.00     ! EPSCOE\n"

    data[27] = "200       ! NTIME_bl\n"

    data[30] = np.format_float_scientific(
        Reynolds, sign=False, precision=2).zfill(8) + '  ! Reynolds\n'
    data[32] = str(Mach)[::-1].zfill(3)[::-1] + '      ! Mach     Number\n'
    data[34] = str(ftripL[name])[::-1].zfill(3)[::-1] + \
        '    1  ! TRANSLO\n'
    data[35] = str(ftripU[name])[::-1].zfill(3)[::-1] + \
        '    2  ! TRANSLO\n'
    with open(fname, 'w') as file:
        file.writelines(data)
