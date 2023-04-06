import matplotlib.pyplot as plt
import os
import numpy as np

from . import colors, markers


def plotAirfoil(data, airfoil, solvers='All', size=(10, 10), AoA_bounds=None):
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
                if AoA_bounds is not None:
                    # Get data where AoA is in AoA bounds
                    polar = polar.loc[(polar['AoA'] >= AoA_bounds[0]) &
                                      (polar['AoA'] <= AoA_bounds[1])]

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

    for i, airplane in enumerate(airplanes):
        skip = False
        for j, solver in enumerate(solvers):
            try:
                polar = data[airplane]
                aoa = polar["AoA"]
                if airplane == "XFLR":
                    cl = polar[f"CL"]
                    cd = polar[f"CD"]
                    cm = polar[f"Cm"]
                    skip = True
                    c = "m"
                    m = "x"
                    style = f"{c}{m}-"

                    label = f"{airplane}"
                else:
                    cl = polar[f"CL_{solver}"]
                    cd = polar[f"CD_{solver}"]
                    cm = polar[f"Cm_{solver}"]
                    c = colors[i]
                    m = markers[j]
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
            if skip == True:
                break
    fig.tight_layout()
    for axR in axs:
        for ax in axR:
            ax.axhline(y=0, color='k')
            ax.axvline(x=0, color='k')
            ax.grid()

    axs[1, 0].legend()  # (bbox_to_anchor=(-0.1, -0.25),  ncol=3,
    # fancybox=True, loc='lower left')


def plotConvergence(data, plane, angles=["All"], solvers=['All'], plotError=True, size=(10, 10)):
    # Define 3 subplots that will be filled with Fx Fz and My vs Iterations
    fig, axs = plt.subplots(3, 3, figsize=size)
    fig.suptitle(
        f'{plane} Convergence', fontsize=16)

    axs[0, 0].set_title('Fx vs Iterations')
    axs[0, 0].set_ylabel('Fx')

    axs[0, 1].set_title('Fy vs Iterations')
    axs[0, 1].set_ylabel('Fy')

    axs[0, 2].set_title('Fz vs Iterations')
    axs[0, 2].set_ylabel('Fz')

    axs[1, 0].set_title('Mx vs Iterations')
    axs[1, 0].set_ylabel('Mx')
    axs[1, 0].set_xlabel('Iterations')

    axs[1, 1].set_title('My vs Iterations')
    axs[1, 1].set_ylabel('My')
    axs[1, 1].set_xlabel('Iterations')

    axs[1, 2].set_title('Mz vs Iterations')
    axs[1, 2].set_ylabel('Mz')
    axs[1, 2].set_xlabel('Iterations')

    axs[2, 0].set_title('ERROR vs Iterations')
    axs[2, 0].set_ylabel('ERROR')
    axs[2, 0].set_xlabel('Iterations')

    axs[2, 1].set_title('ERRORM vs Iterations')
    axs[2, 1].set_ylabel('ERRORM')
    axs[2, 1].set_xlabel('Iterations')

    fig.delaxes(axs[2, 2])
    # Fill plots with data
    if solvers == ['All']:
        solvers = ["", "2D", "DS2D"]

    cases = data[plane]
    i = -1
    j = -1
    toomuchData = False
    for ang in list(cases.keys()):
        num = - float(
            ''.join(c for c in ang if (c.isdigit() or c == '.')))
        if ang.startswith('m'):
            ang_num = -num
        else:
            ang_num = num

        if ang_num in angles or angles == ["All"]:
            runHist = cases[ang]
            i += 1
            j = 0
        else:
            continue

        for solver in solvers:
            try:
                it = runHist["TTIME"].astype(float)
                it = it/it[0]

                fx = runHist[f"TFORC{solver}(1)"].astype(float)
                fy = runHist[f"TFORC{solver}(2)"].astype(float)
                fz = runHist[f"TFORC{solver}(3)"].astype(float)
                mx = runHist[f"TAMOM{solver}(1)"].astype(float)
                my = runHist[f"TAMOM{solver}(2)"].astype(float)
                mz = runHist[f"TAMOM{solver}(2)"].astype(float)
                error = runHist[f"ERROR"].astype(float)
                errorM = runHist[f"ERRORM"].astype(float)

                if plotError == True:
                    it = it[1:].values
                    fx = np.abs(fx[1:].values - fx[:-1].values)
                    fy = np.abs(fy[1:].values - fy[:-1].values)
                    fz = np.abs(fz[:-1].values - fz[1:].values)
                    mx = np.abs(mx[:-1].values - mx[1:].values)
                    my = np.abs(my[:-1].values - my[1:].values)
                    mz = np.abs(mz[:-1].values - mz[1:].values)

                j += 1
                if (i > len(colors) - 1):
                    toomuchData = True
                    break
                c = colors[i]
                m = markers[j]
                style = f"{c}{m}--"

                label = f"{plane} - {solver} - {ang_num}"
                axs[0, 0].plot(it, fx, style, label=label,
                               markersize=2.0, linewidth=1)
                axs[0, 1].plot(it, fy, style, label=label,
                               markersize=2.0, linewidth=1)
                axs[0, 2].plot(it, fz, style, label=label,
                               markersize=2.0, linewidth=1)

                axs[1, 0].plot(it, mx, style, label=label,
                               markersize=2.0, linewidth=1)
                axs[1, 1].plot(it, my, style, label=label,
                               markersize=2.0, linewidth=1)
                axs[1, 2].plot(it, mz, style, label=label,
                               markersize=2.0, linewidth=1)

                axs[2, 0].plot(it, error, style, label=label,
                               markersize=2.0, linewidth=1)
                axs[2, 1].plot(it, errorM, style, label=label,
                               markersize=2.0, linewidth=1)
            except KeyError as solver:
                print(f"Run Doesn't Exist: {plane},{solver},{ang}")
    if toomuchData == True:
        print(f"Too much data to plot, only plotting {len(colors)} cases")

    fig.tight_layout()
    for axR in axs:
        for ax in axR:
            ax.grid(which='both', axis='both')
            ax.set_yscale('log')

    axs[2, 1].legend(bbox_to_anchor=(1, -0.25),  ncol=2,
                     fancybox=True, loc='lower left')
