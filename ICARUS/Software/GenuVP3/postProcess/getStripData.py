import os
from typing import Any

from pandas import DataFrame

from ICARUS.Database import DB3D
from ICARUS.Vehicle.plane import Airplane


def get_strip_data(
    plane: Airplane,
    case: str,
    NBs: list[int],
) -> tuple[DataFrame, DataFrame]:
    """Function to get strip data from a case simulation.

    Args:
        pln (Airplane): Plane Object
        case (str): String containing the case folder
        NBs (list[int]): list with all lifting surfaces for which to get data

    Returns:
        tuple[DataFrame, DataFrame]: Returns a dataframe with all strip data and a dataframe with all strip data for the NBs
    """
    directory: str = os.path.join(DB3D, plane.CASEDIR, case)
    files: list[str] = os.listdir(directory)
    strip_data: list[list[Any]] = []
    for file in files:
        if file.startswith("strip"):
            filename: str = os.path.join(directory, file)
            with open(filename, encoding="UTF-8") as f:
                data: list[str] = f.readlines()
            data_num: list[float] = [float(item) for item in data[-1].split()]
            file = file[6:]
            body = int(file[:2])
            strip = int(file[3:5])
            strip_data.append([body, strip, *data_num])

    strip_data_df: DataFrame = DataFrame(strip_data, columns=stripColumns).sort_values(
        ["Body", "Strip"],
        ignore_index=True,
    )
    nbs_data: DataFrame = strip_data_df[strip_data_df["Body"].isin(NBs)]

    return strip_data_df, nbs_data


stripColumns: list[str] = [
    "Body",
    "Strip",
    "Time",
    "RNONDIM",  # PSIB / B
    "PSIB",  # AZIMUTHAL ANGLE
    "FALFAM",  # GWNIA PROSPTOSIS
    "FALFAGEM",  # GEOMETRIC PROSPTOSIS XWRIS INDUCED
    "AMACHS(IST)",  # MACH NUMBER ???
    "AMACH0S(IST)",  # MACH NUMBER ???
    "VELAVEL(IST)",  # AVERAGE VELOCITY OF STRIP
    "VELAVELG(IST)",  # AVERAGE VELOCITY OF STRIP DUE TO MOTION OF BODY
    "CLIFTSGN",
    "CDRAGSGN",  # Potential
    "CNTGN",
    "CNTGN",
    "CMOMSGN",
    "CLIFTS2D",
    "CDRAGS2D",  # 2D / Strip area
    "CNT2D",
    "CNT2D",
    "CMOMS2D",
    "CLIFTSDS2D",
    "CDRAGSDS2D",  # ONERA / Strip area
    "CNTDS2D",
    "CNTDS2D",
    "CMOMSDS2D",
    "FSTRGNL(3, IST) / ALSPAN(IST)",  # Potential N/m
    "FSTRGNL(1, IST) / ALSPAN(IST)",
    "AMSTRGNL(IST) / ALSPAN(IST)",
    "FSTR2DL(3, IST) / ALSPAN(IST)",  # 2D N/m
    "FSTR2DL(1, IST) / ALSPAN(IST)",
    "AMSTR2DL(IST) / ALSPAN(IST)",
    "FSTRDS2DL(3, IST) / ALSPAN(IST)",  # ONERA N/m
    "FSTRDS2DL(1, IST) / ALSPAN(IST)",
    "AMSTRDS2DL(IST) / ALSPAN(IST)",
    "Uind",
    "Vind",
    "Wind",
    "FALFA1M",
    "CIRCtmp(IST)",  # CIRCULATION
]
