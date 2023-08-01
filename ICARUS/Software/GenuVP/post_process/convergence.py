import pandas as pd
from pandas import DataFrame


# GET THE SCANNING FROM THE DATABASE AND MAKE DF WITH IT
def getLoadsConvergence(file: str) -> DataFrame | None:
    try:
        return pd.read_csv(file, delim_whitespace=True, names=cols)
    except Exception as e:
        # print(f"Cant Load {file} as Loads_Aero.dat!\nGot error {e}")
        return None


def addErrorConvergence2df(file: str, df: DataFrame) -> DataFrame:
    try:
        with open(file, encoding="UTF-8") as f:
            lines: list[str] = f.readlines()
        time: list[int] = []
        error: list[float] = []
        errorm: list[float] = []
        for line in lines:
            if not line.startswith(" STEP="):
                continue

            a: list[str] = line[6:].split()
            time.append(int(a[0]))
            error.append(float(a[2]))
            errorm.append(float(a[6]))
        try:
            foo: int = len(df["TTIME"])
            if foo > len(time):
                df = df.tail(len(time))
            else:
                error = error[-foo:]
                errorm = errorm[-foo:]
            df["ERROR"] = error
            df["ERRORM"] = errorm
        except ValueError as e:
            print(f"Some Run Had Problems!\n{e}")

    except FileNotFoundError:
        print(f"No gnvp.out file found in {file}!")

    finally:
        return df


cols: list[str] = [
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
    "TAMOMDS2D(3)",
]
