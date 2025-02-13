import os
from typing import Any

from pandas import DataFrame

from ICARUS.database import Database
from ICARUS.vehicle.plane import Airplane


def get_strip_data(
    plane: Airplane,
    case: str,
    NBs: list[int],
    genuvp_version: int,
) -> tuple[DataFrame, DataFrame]:
    """Function to get strip data from a case simulation.

    Args:
        pln (Airplane): Plane Object
        case (str): String containing the case folder
        NBs (list[int]): list with all lifting surfaces for which to get data

    Returns:
        tuple[DataFrame, DataFrame]: Returns a dataframe with all strip data and a dataframe with all strip data for the NBs

    """
    DB = Database.get_instance()
    directory = DB.vehicles_db.get_case_directory(
        airplane=plane,
        solver=f"GenuVP{genuvp_version}",
        case=case,
    )
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
    try:
        strip_data_df: DataFrame = DataFrame(
            strip_data,
            columns=strip_columns_3,
        ).sort_values(
            ["Body", "Strip"],
            ignore_index=True,
        )
    except ValueError:
        try:
            strip_data_df = DataFrame(strip_data, columns=strip_columns_7).sort_values(
                ["Body", "Strip"],
                ignore_index=True,
            )
        except ValueError:
            raise ValueError("Strip data columns are not in the correct format")
    nbs_data: DataFrame = strip_data_df[strip_data_df["Body"].isin(NBs)]

    return strip_data_df, nbs_data


strip_columns_3: list[str] = [
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


strip_columns_7: list[str] = [
    "Body",
    "Strip",
    "TTIME",
    "PSIB",
    "RLOC",
    "Dspan",
    "Forc_all(nb)%RTIP",
    "FALFAM",
    "FALFAGEM",
    "AMACH",
    "ReNum",
    "CLiftGN",
    "CDragGN",
    "CNormGN(3)",
    "CNormGN(1)",
    "CMomeGN",
    "CLift2D",
    "CDrag2D",
    "CNorm2D(3)",
    "CNorm2D(1)",
    "CMome2D",
    "CLiftDS2D",
    "CDragDS2D",
    "CNormDS2D(3)",
    "CNormDS2D(1)",
    "CMomeDS2D",
    "FSTRGNL(3)/Dspan",
    "FSTRGNL(1)/Dspan",
    "FSTRGNL(2)/Dspan",
    "FSTR2DL(3)/Dspan",
    "FSTR2DL(1)/Dspan",
    "FSTR2DL(2)/Dspan",
    "FSTRDS2DL(3)/Dspan",
    "FSTRDS2DL(1)/Dspan",
    "FSTRDS2DL(2)/Dspan",
    "Uind",  # "VIndLoc(1)",
    "Vind",  # "VIndLoc(2)",
    "Wind",  # "VIndLoc(3)",
    "VelRel",
    "VRel_Ge",
    "TwiPitM",
    "CIRC",
    "FSTRGN(1)/Dspan",
    "FSTRGN(2)/Dspan",
    "FSTRGN(3)/Dspan",
    "FSTR2D(1)/Dspan",
    "FSTR2D(2)/Dspan",
    "FSTR2D(3)/Dspan",
    "FSTRDS2D(1)/Dspan",
    "FSTRDS2D(2)/Dspan",
    "FSTRDS2D(3)/Dspan",
    "MSTRGN(1)/Dspan",
    "MSTRGN(2)/Dspan",
    "MSTRGN(3)/Dspan",
    "MSTR2D(1)/Dspan",
    "MSTR2D(2)/Dspan",
    "MSTR2D(3)/Dspan",
    "MSTRDS2D(1)/Dspan",
    "MSTRDS2D(2)/Dspan",
    "MSTRDS2D(3)/Dspan",
    "FALFA_F",
    "FALFA_U",
    "FALFA_Ci",
    "ALCHORD",
    "FALFA_G",
    "AlfaFL",
]
