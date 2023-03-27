import matplotlib.pyplot as plt
import os
import numpy as np

colors = ['r', 'k', 'b', 'g', 'c', 'm', 'y', 'r', 'k', 'b', 'g']
markers = ['x', 'o', '.', "*"]


def plotAirfoil(data, airfoil, solvers='All', size=(10, 10)):
    fig, axs = plt.subplots(2, 2, figsize=size)

    fig.suptitle(f'NACA {airfoil[4:]} Aero Coefficients', fontsize=16)
    axs[0, 0].set_title('Cm vs AoA')
    axs[0, 0].set_ylabel('Cm')

    axs[0, 1].set_title('Cd vs AoA')
    axs[0, 1].set_xlabel('AoA')
    axs[0, 1].set_ylabel('Cd')

    axs[1, 0].set_title('Cl vs AoA')
    axs[1, 0].set_xlabel('AoA')
    axs[1, 0].set_ylabel('Cl')

    axs[1, 1].set_title('Cl vs Cd')
    axs[1, 1].set_xlabel('Cd')

    if solvers == ['All']:
        solvers = ["Xfoil", "Foil2Wake", "OpenFoam"]

    for i, reynolds in enumerate(list(data[airfoil])):
        for j, solver in enumerate(solvers):
            try:
                polar = data[airfoil][reynolds][solver]

                aoa, cl, cd, cm = polar.T.values
                c = colors[i]
                m = markers[j]
                style = f"{c}{m}-"
                label = f"{airfoil}: {reynolds} - {solver}"
                axs[0, 1].plot(aoa, cd, style, label=label,
                               markersize=3, linewidth=1)
                axs[1, 0].plot(aoa, cl, style, label=label,
                               markersize=3, linewidth=1)
                axs[1, 1].plot(cd, cl, style, label=label,
                               markersize=3, linewidth=1)
                axs[0, 0].plot(aoa, cm, style, label=label,
                               markersize=3, linewidth=1)
            except KeyError as solver:
                print(f"Run Doesn't Exist: {airfoil},{reynolds},{solver}")

    fig.tight_layout()
    if len(solvers) == 3:
        per = -.85
    elif len(solvers) == 2:
        per = -.6
    else:
        per = -0.4
    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()
    axs[0, 0].grid()

    axs[1, 0].legend(bbox_to_anchor=(-0.1, per),  ncol=3,
                     fancybox=True, loc='lower left')


def plotReynolds(data, airfoil, reyn, solvers='All', size=(10, 10)):
    fig, axs = plt.subplots(2, 2, figsize=size)
    fig.suptitle(
        f'NACA {airfoil[4:]}- Reynolds={reyn}\n Aero Coefficients', fontsize=16)
    axs[0, 0].set_title('Cm vs AoA')
    axs[0, 0].set_ylabel('Cm')

    axs[0, 1].set_title('Cd vs AoA')
    axs[0, 1].set_xlabel('AoA')
    axs[0, 1].set_ylabel('Cd')

    axs[1, 0].set_title('Cl vs AoA')
    axs[1, 0].set_xlabel('AoA')
    axs[1, 0].set_ylabel('Cl')

    axs[1, 1].set_title('Cl vs Cd')
    axs[1, 1].set_xlabel('Cd')

    if solvers == ['All']:
        solvers = ["Xfoil", "Foil2Wake", "OpenFoam"]

        for j, solver in enumerate(solvers):
            try:
                polar = data[airfoil][reyn][solver]
                aoa, cl, cd, cm = polar.T.values
                c = colors[j]
                m = markers[j]
                style = f"{c}{m}-"
                label = f"{airfoil}: {reyn} - {solver}"
                axs[0, 1].plot(aoa, cd, style, label=label,
                               markersize=3, linewidth=1)
                axs[1, 0].plot(aoa, cl, style, label=label,
                               markersize=3, linewidth=1)
                axs[1, 1].plot(cd, cl, style, label=label,
                               markersize=3, linewidth=1)
                axs[0, 0].plot(aoa, cm, style, label=label,
                               markersize=3, linewidth=1)
            except KeyError as solver:
                print(f"Run Doesn't Exist: {airfoil},{reyn},{solver}")

    fig.tight_layout()

    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()
    axs[0, 0].grid()

    axs[1, 0].legend(bbox_to_anchor=(-0.1, -0.25),  ncol=3,
                     fancybox=True, loc='lower left')


def plotCP(angle):
    fname = 'COEFPRE.OUT'
    folders = next(os.walk('.'))[1]
    if angle < 0:
        anglef = 'm'+str(angle)[::-1].strip('-').zfill(6)[::-1]
    else:
        anglef = str(angle)[::-1].zfill(7)[::-1]
    fname = f'{anglef}/{fname}'
    data = np.loadtxt(fname).T
    c = data[0]
    p1 = data[1]
    plt.title('Pressure Coefficient')
    plt.xlabel('x/c')
    plt.ylabel('C_p')
    plt.plot(c, p1)
    plt.show()


def plotMultipleCPs(angles):
    fname = 'COEFPRE.OUT'
    folders = next(os.walk('.'))[1]
    for angle in angles:
        print(angle)
        if angle < 0:
            anglef = 'm'+str(angle)[::-1].strip('-').zfill(6)[::-1]
        else:
            anglef = str(angle)[::-1].zfill(7)[::-1]
        floc = f'{anglef}/{fname}'
        data = np.loadtxt(floc).T
        c = data[0]
        p1 = data[1]
        plt.title('Pressure Coefficient')
        plt.xlabel('x/c')
        plt.ylabel('C_p')
        plt.plot(c, p1)
    plt.show()


def plotAirplanes(data, airplanes, solvers=['All'], size=(10, 10)):
    fig, axs = plt.subplots(2, 2, figsize=size)
    if len(airplanes) == 1:
        fig.suptitle(
            f'{airplanes} Aero Coefficients', fontsize=16)
    else:
        fig.suptitle(
            f'Airplanes Aero Coefficients', fontsize=16)
        
    axs[0, 0].set_title('Cm vs AoA')
    axs[0, 0].set_ylabel('Cm')

    axs[0, 1].set_title('Cd vs AoA')
    axs[0, 1].set_xlabel('AoA')
    axs[0, 1].set_ylabel('Cd')

    axs[1, 0].set_title('Cl vs AoA')
    axs[1, 0].set_xlabel('AoA')
    axs[1, 0].set_ylabel('Cl')

    axs[1, 1].set_title('Cl vs Cd')
    axs[1, 1].set_xlabel('Cd')

    if solvers == ['All']:
        solvers = ["Potential", "ONERA", "2D"]
        
    for i,airplane in enumerate(airplanes):
        for j, solver in enumerate(solvers):
            try:
                polar = data[airplane]
                aoa = polar["AoA"]
                cl = polar[f"CL_{solver}"]
                cd = polar[f"CD_{solver}"]
                cm = polar[f"Cm_{solver}"]
                c = colors[j]
                m = markers[i]
                style = f"{c}{m}--"
                label = f"{airplane} - {solver}"
                axs[0, 1].plot(aoa, cd, style, label=label,
                            markersize=3.5, linewidth=1)
                axs[1, 0].plot(aoa, cl, style, label=label,
                            markersize=3.5, linewidth=1)
                axs[1, 1].plot(cd, cl, style, label=label,
                            markersize=3.5, linewidth=1)
                axs[0, 0].plot(aoa, cm, style, label=label,
                            markersize=3.5, linewidth=1)
            except KeyError as solver:
                print(f"Run Doesn't Exist: {airplane},{solver}")

    fig.tight_layout()
    for axR in axs:
        for ax in axR:
            ax.axhline(y=0, color='k')
            ax.axvline(x=0, color='k')
            ax.grid()

    axs[1, 0].legend(bbox_to_anchor=(-0.1, -0.25),  ncol=3,
                     fancybox=True, loc='lower left')
