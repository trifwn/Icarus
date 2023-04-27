import numpy as np
import pandas as pd
import os


def forces2polars(CASEDIR, HOMEDIR):
    os.chdir(CASEDIR)

    folders = next(os.walk('.'))[1]
    print('Making Polars')
    pols = []
    for folder in folders:
        os.chdir(os.path.join(CASEDIR,folder))
        files = next(os.walk('.'))[2]
        if "LOADS_aer.dat" in files:
            name = float(
                ''.join(c for c in folder if (c.isdigit() or c == '.')))
            dat = np.loadtxt("LOADS_aer.dat")[-1]
            if folder.startswith("m"):
                a = [- name, *dat]
            else:
                a = [name, *dat]
            pols.append(a)
        os.chdir(f"{CASEDIR}")
    df = pd.DataFrame(pols, columns=cols)
    df.pop('TTIME')
    df.pop("PSIB")

    df = df.sort_values("AoA").reset_index(drop=True)
    df.to_csv('forces.gnvp3', index=False)
    os.chdir(HOMEDIR)
    return df


def forces2pertrubRes(DYNDIR, HOMEDIR):
    os.chdir(DYNDIR)
    folders = next(os.walk('.'))[1]
    print('Logging Pertrubations')
    pols = []
    for folder in folders:
        os.chdir(os.path.join(DYNDIR,folder))
        files = next(os.walk('.'))[2]
        if "LOADS_aer.dat" in files:
            dat = np.loadtxt("LOADS_aer.dat")[-1]
            if folder == "Trim":
                pols.append([0, str(folder), *dat])
                continue

            # RECONSTRUCT NAME
            value = ''
            name = ''
            flag = False
            for c in folder[1:]:
                if (c != '_') and (not flag):
                    value += c
                elif (c == '_'):
                    flag = True
                else:
                    name += c
            value = float(value)
            if folder.startswith("m"):
                value = - value

            pols.append([value, name,  *dat])
            os.chdir(os.path.join(DYNDIR,folder))
        os.chdir(f"{DYNDIR}")
    df = pd.DataFrame(pols, columns=["Epsilon", "Type", *cols[1:]])
    df.pop('TTIME')
    df.pop("PSIB")
    df = df.sort_values("Type").reset_index(drop=True)
    df.to_csv('pertrubations.genu', index=False)
    os.chdir(HOMEDIR)
    return df


def rotateForces(rawpolars, alpha, preferred="2D", save = False):
    Data = pd.DataFrame()
    AoA = alpha * np.pi/180

    for enc, name in zip(["", "2D", "DS2D"], ["Potential", "2D", "ONERA"]):
        Fx = rawpolars[f"TFORC{enc}(1)"]
        Fy = rawpolars[f"TFORC{enc}(2)"]
        Fz = rawpolars[f"TFORC{enc}(3)"]

        Mx = rawpolars[f"TAMOM{enc}(1)"]
        My = rawpolars[f"TAMOM{enc}(2)"]
        Mz = rawpolars[f"TAMOM{enc}(3)"]

        Fx_new = Fx * np.cos(-AoA) - Fz * np.sin(-AoA)
        Fy_new = Fy
        Fz_new = Fx * np.sin(-AoA) + Fz * np.cos(-AoA)

        Mx_new = Mx * np.cos(-AoA) - Mz * np.sin(-AoA)
        My_new = My
        Mz_new = Mx * np.sin(-AoA) + Mz * np.cos(-AoA)

        Data[f"Fx_{name}"] = Fx_new
        Data[f"Fy_{name}"] = Fy_new
        Data[f"Fz_{name}"] = Fz_new
        Data[f"L_{name}"] = Mx_new
        Data[f"M_{name}"] = My_new
        Data[f"N_{name}"] = Mz_new

    Data["AoA"] = alpha
    # print(f"Using {preferred} polars")
    Data["Fx"] = Data[f"Fx_{preferred}"]
    Data["Fy"] = Data[f"Fy_{preferred}"]
    Data["Fz"] = Data[f"Fz_{preferred}"]
    Data["L"] = Data[f"L_{preferred}"]
    Data["M"] = Data[f"M_{preferred}"]
    Data["N"] = Data[f"N_{preferred}"]
    # Reindex the dataframe sort by AoA
    return Data.sort_values(by="AoA").reset_index(drop=True)


cols = ["AoA",
        "TTIME",
        "PSIB",
        "TFORC(1)",
        "TFORC(2)",
        "TFORC(3)",
        "TAMOM(1)",
        "TAMOM(2)",
        "TAMOM(3)",
        "TFORC2D(1)",
        "TFORC2D(2)",
        "TFORC2D(3)",
        "TAMOM2D(1)",
        "TAMOM2D(2)",
        "TAMOM2D(3)",
        "TFORCDS2D(1)",
        "TFORCDS2D(2)",
        "TFORCDS2D(3)",
        "TAMOMDS2D(1)",
        "TAMOMDS2D(2)",
        "TAMOMDS2D(3)"]
