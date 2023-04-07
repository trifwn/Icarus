import matplotlib.pyplot as plt
import numpy as np


def GNVPwake(plane, case, maxiter):

    fname = f"{plane.CASEDIR}/{case}/YOURS.WAK"
    with open(fname, "r") as file:
        data = file.readlines()
    a = []
    b = []
    c = []
    iteration = 0
    flag = True
    for i, line in enumerate(data):
        if line.startswith('  WAKE'):
            foo = line.split()
            iteration = int(foo[3])
            continue
        if iteration == maxiter:
            foo = line.split()
            if (len(foo) == 4) and flag:
                _, x, y, z = [float(i) for i in foo]
                a.append([x, y, z])
            elif len(foo) == 3:
                x, y, z = [float(i) for i in foo]
                flag = False
                b.append([x, y, z])
            elif len(foo) == 4:
                _, x, y, z = [float(i) for i in foo]
                c.append([x, y, z])

    C1 = np.array(c)
    B1 = np.array(b)
    A1 = np.array(a)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    fig.tight_layout()

    ax.set_title(f"{plane.name} WAKE")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(A1[:, 0], A1[:, 1], A1[:, 2], color='r', s=5.)
    ax.scatter(B1[:, 0], B1[:, 1], B1[:, 2], color='k', s=5.)
    ax.scatter(C1[:, 0], C1[:, 1], C1[:, 2], color='g', s=5.)

    plane.visAirplane(fig, ax, movement=- plane.CG)
    ax.view_init(50, 150)

    ax.axis('scaled')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)



def ang2case(angle):
    if angle >= 0:
        folder = str(angle)[::-1].zfill(7)[::-1] + "_AoA"
    else:
        folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1] + "_AoA"

    return folder
