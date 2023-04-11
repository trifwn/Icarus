import os
from ICARUS.Database.Database_3D import Database_3D


def airPolars(plot=False):
    print('Testing Airplane Polars...')
    HOMEDIR = os.getcwd()

    db = Database_3D(HOMEDIR)
    planenames = ['bmark']

    db.importXFLRpolar(f"{HOMEDIR}/ICARUS/XFLR5/bmark.txt", 'bmark')
    planenames.append(f"XFLR_bmark")
    if plot:
        from ICARUS.Visualization.airplanePolars import plotAirplanePolars
        plotAirplanePolars(db.Data, planenames, ["2D"], size=(10, 10))
    desired = db.Data['XFLR_bmark']
    actual = db.Data['bmark']
    return desired, actual
