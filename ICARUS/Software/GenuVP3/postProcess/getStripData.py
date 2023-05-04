import os

import pandas as pd

from ICARUS.Database import DB3D


def getStripData(pln, case, NBs):
    directory = os.path.join(DB3D, pln.CASEDIR, case)
    files = os.listdir(directory)
    stripDat = []
    for file in files:
        if file.startswith("strip"):
            filename = os.path.join(directory, file)
            with open(filename, encoding="UTF-8") as f:
                data = f.readlines()
            data = [float(item) for item in data[-1].split()]
            file = file[6:]
            body = int(file[:2])
            strip = int(file[3:5])
            stripDat.append([body, strip, *data])

    stripDat = pd.DataFrame(stripDat, columns=stripColumns).sort_values(
        ["Body", "Strip"], ignore_index=True,
    )
    data = stripDat[stripDat["Body"].isin(NBs)]

    return stripDat, data


stripColumns = [
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
    "FSTRGNL(3, IST) / ALSPAN(IST)",  ## Potential N/m
    "FSTRGNL(1, IST) / ALSPAN(IST)",
    "AMSTRGNL(IST) / ALSPAN(IST)",
    "FSTR2DL(3, IST) / ALSPAN(IST)",  ## 2D N/m
    "FSTR2DL(1, IST) / ALSPAN(IST)",
    "AMSTR2DL(IST) / ALSPAN(IST)",
    "FSTRDS2DL(3, IST) / ALSPAN(IST)",  ## ONERA N/m
    "FSTRDS2DL(1, IST) / ALSPAN(IST)",
    "AMSTRDS2DL(IST) / ALSPAN(IST)",
    "Uind",
    "Vind",
    "Wind",
    "FALFA1M",
    "CIRCtmp(IST)",  # CIRCULATION
]
