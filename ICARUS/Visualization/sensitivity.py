import matplotlib.pyplot as plt

from . import colors
from . import markers


def plotSensitivity(
    data,
    plane,
    trim,
    relative=False,
    vars2s=["All"],
    solvers=["2D"],
    size=(16, 7),
):

    fig, axs = plt.subplots(2, 3, figsize=size)
    fig.suptitle(f"{plane.name} Convergence", fontsize=16)

    axs[0, 0].set_title("Fx vs epsilon")
    axs[0, 0].set_ylabel("Fx")

    axs[0, 1].set_title("Fy vs epsilon")
    axs[0, 1].set_ylabel("Fy")

    axs[0, 2].set_title("Fz vs epsilon")
    axs[0, 2].set_ylabel("Fz")

    axs[1, 0].set_title("Mx vs epsilon")
    axs[1, 0].set_ylabel("Mx")
    axs[1, 0].set_xlabel("Epsilon")

    axs[1, 1].set_title("My vs epsilon")
    axs[1, 1].set_ylabel("My")
    axs[1, 1].set_xlabel("Epsilon")

    axs[1, 2].set_title("Mz vs epsilon")
    axs[1, 2].set_ylabel("Mz")
    axs[1, 2].set_xlabel("Epsilon")

    try:
        cases = data
    except:
        print("No Sensitivity Results")
        return
    i = -1
    j = -1
    toomuchData = False

    fx_trim = trim[f"TFORC{solvers[0]}(1)"].astype(float).values
    fy_trim = trim[f"TFORC{solvers[0]}(2)"].astype(float).values
    fz_trim = trim[f"TFORC{solvers[0]}(3)"].astype(float).values
    mx_trim = trim[f"TAMOM{solvers[0]}(1)"].astype(float).values
    my_trim = trim[f"TAMOM{solvers[0]}(2)"].astype(float).values
    mz_trim = trim[f"TAMOM{solvers[0]}(2)"].astype(float).values
    if not relative:
        axs[0, 0].axhline(
            fx_trim,
            xmin=-1,
            xmax=1,
            color="k",
            label="Trim",
            linewidth=1,
        )
        axs[0, 1].axhline(
            fy_trim,
            xmin=-1,
            xmax=1,
            color="k",
            label="Trim",
            linewidth=1,
        )
        axs[0, 2].axhline(
            fz_trim,
            xmin=-1,
            xmax=1,
            color="k",
            label="Trim",
            linewidth=1,
        )

        axs[1, 0].axhline(
            mx_trim,
            xmin=-1,
            xmax=1,
            color="k",
            label="Trim",
            linewidth=1,
        )
        axs[1, 1].axhline(
            my_trim,
            xmin=-1,
            xmax=1,
            color="k",
            label="Trim",
            linewidth=1,
        )
        axs[1, 2].axhline(
            mz_trim,
            xmin=-1,
            xmax=1,
            color="k",
            label="Trim",
            linewidth=1,
        )

    for var in list(cases.keys()):

        if var in vars2s or vars2s == ["All"]:
            runHist = cases[var]
            i += 1
            j = 0
        else:
            continue

        for solver in solvers:
            try:
                epsilon = runHist["Epsilon"].astype(float)
                fx = runHist[f"TFORC{solver}(1)"].astype(float)
                fy = runHist[f"TFORC{solver}(2)"].astype(float)
                fz = runHist[f"TFORC{solver}(3)"].astype(float)
                mx = runHist[f"TAMOM{solver}(1)"].astype(float)
                my = runHist[f"TAMOM{solver}(2)"].astype(float)
                mz = runHist[f"TAMOM{solver}(2)"].astype(float)

                if relative:
                    fx = fx - fx_trim
                    fy = fy - fy_trim
                    fz = fz - fz_trim
                    mx = mx - mx_trim
                    my = my - my_trim
                    mz = mz - mz_trim

                j += 1
                if i > len(colors) - 1:
                    toomuchData = True
                    break
                c = colors[i]
                m = markers[j]
                style = f"{c}{m}--"

                label = f"{plane.name} - {solver} - {var}"
                axs[0, 0].scatter(epsilon, fx, marker=m, label=label, linewidth=3.0)
                axs[0, 1].scatter(epsilon, fy, marker=m, label=label, linewidth=3.0)
                axs[0, 2].scatter(epsilon, fz, marker=m, label=label, linewidth=3.0)

                axs[1, 0].scatter(epsilon, mx, marker=m, label=label, linewidth=3.0)
                axs[1, 1].scatter(epsilon, my, marker=m, label=label, linewidth=3.0)
                axs[1, 2].scatter(epsilon, mz, marker=m, label=label, linewidth=3.0)

            except KeyError as solver:
                print(f"Run Doesn't Exist: {plane.name},{solver},{var}")
    if toomuchData == True:
        print(f"Too much data to plot, only plotting {len(colors)} cases")

    fig.tight_layout()
    for axR in axs:
        for ax in axR:
            ax.grid(which="both", axis="both")
            ax.set_xscale("log")
            ax.set_yscale("log")

    axs[1, 0].legend()  # (bbox_to_anchor=(-0.1, -0.25),  ncol=3,
    # fancybox=True, loc='lower left')
    plt.show()
