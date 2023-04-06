
import pandas as pd
import os


def GNVPstrips(pln, CASE, HOMEDIR):
    print(pln.CASEDIR)
    files = os.listdir(f"{pln.CASEDIR}/{CASE}")
    stripDat = []
    for file in files:
        if file.startswith('strip'):
            with open(f"{pln.CASEDIR}/{CASE}/{file}", 'r') as file:
                data = file.readlines()
            data = [float(item) for item in data[-1].split()]
            stripDat.append(data)
    data = pd.DataFrame(stripDat, columns=stripColumns)

    os.chdir(HOMEDIR)
    return data


stripColumns = [
    "TTIME",
    "RNONDIM",
    "PSIB",
    "FALFAM",
    "FALFAGEM",
    "AMACHS(IST)", "AMACH0S(IST)",
    "VELAVEL(IST)", "VELAVELG(IST)",

    "CLIFTSGN(IST)", "CDRAGSGN(IST)",
    "CNTGN(3, IST)", "CNTGN(1, IST)",
    "CMOMSGN(IST)",

    "CLIFTS2D(IST)", "CDRAGS2D(IST)",
    "CNT2D(3, IST)", "CNT2D(1, IST)",
    "CMOMS2D(IST)",

    "CLIFTSDS2D(IST)", "CDRAGSDS2D(IST)",
    "CNTDS2D(3, IST)", "CNTDS2D(1, IST)",
    "CMOMSDS2D(IST)",

    "FSTRGNL(3, IST) / ALSPAN(IST)",
    "FSTRGNL(1, IST) / ALSPAN(IST)",
    "AMSTRGNL(IST) / ALSPAN(IST)",

    "FSTR2DL(3, IST) / ALSPAN(IST)",
    "FSTR2DL(1, IST) / ALSPAN(IST)",
    "AMSTR2DL(IST) / ALSPAN(IST)",

    "FSTRDS2DL(3, IST) / ALSPAN(IST)",
    "FSTRDS2DL(1, IST) / ALSPAN(IST)",
    "AMSTRDS2DL(IST) / ALSPAN(IST)",

    "VELINDL(1, IST)",
    "VELINDL(2, IST)",
    "VELINDL(3, IST)",

    "FALFA1M",
    "CIRCtmp(IST)"
]
