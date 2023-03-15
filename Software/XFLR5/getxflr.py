import pandas as pd

def getdf():
    xflr = pd.read_csv('XFLR5/plane-20_0 m_s.txt',delim_whitespace=True,skiprows=7)
    xflr.pop("Beta")
    xflr.pop("CDi")
    xflr.pop("CDv")
    xflr.pop("Cni")
    xflr.pop("QInf")
    xflr.pop("XCP")
    return xflr