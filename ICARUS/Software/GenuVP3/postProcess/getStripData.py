from ICARUS.Database import DB3D
import pandas as pd
import os


def getStripData(pln, case, NBs):
    directory = os.path.join(DB3D, pln.CASEDIR, case)
    files = os.listdir(directory)
    stripDat = []
    for file in files:
        if file.startswith('strip'):
            filename = os.path.join(directory,file)
            with open(filename, 'r') as f:
                data = f.readlines()
            data = [float(item) for item in data[-1].split()]
            file = file[6:]
            body = int(file[:2])
            strip = int(file[3:5])
            stripDat.append([body, strip, *data])
    
    stripDat = pd.DataFrame(stripDat, columns=stripColumns).sort_values(['Body', 'Strip'],ignore_index = True)
    data = stripDat[stripDat['Body'].isin(NBs)]
    
    return stripDat, data

stripColumns = [
    "Body",
    "Strip",
    "Time",
    "RNONDIM",
    "PSIB",
    "FALFAM",
    "FALFAGEM",
    "AMACHS(IST)", "AMACH0S(IST)",
    "VELAVEL(IST)", "VELAVELG(IST)",

    "CLIFTSGN", "CDRAGSGN",
    "CNTGN", "CNTGN",
    "CMOMSGN",

    "CLIFTS2D", "CDRAGS2D",
    "CNT2D", "CNT2D",
    "CMOMS2D",

    "CLIFTSDS2D", "CDRAGSDS2D",
    "CNTDS2D", "CNTDS2D",
    "CMOMSDS2D",

    "FSTRGNL(3, IST) / ALSPAN(IST)",
    "FSTRGNL(1, IST) / ALSPAN(IST)",
    "AMSTRGNL(IST) / ALSPAN(IST)",

    "FSTR2DL(3, IST) / ALSPAN(IST)",
    "FSTR2DL(1, IST) / ALSPAN(IST)",
    "AMSTR2DL(IST) / ALSPAN(IST)",

    "FSTRDS2DL(3, IST) / ALSPAN(IST)",
    "FSTRDS2DL(1, IST) / ALSPAN(IST)",
    "AMSTRDS2DL(IST) / ALSPAN(IST)",

    "Uind",
    "Vind",
    "Wind",

    "FALFA1M",
    "CIRCtmp(IST)"
]
