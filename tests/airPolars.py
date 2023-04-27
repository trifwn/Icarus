import os
from ICARUS.Database.db import DB
from ICARUS.Software.XFLR5.polars import readPolars3D
from tests.planes import ap as ap


def airPolars(plot=False):
    print('Testing Airplane Polars...')

    db = DB()
    db.loadData()
    
    db3d = db.vehiclesDB
    planenames = [ap.name]
    BMARKLOC = os.path.join(db.HOMEDIR, 'Data', 'XFLR5', 'bmark.txt')
    readPolars3D(db3d, BMARKLOC, 'bmark')
    planenames.append(f"XFLR_bmark")
    if plot:
        from ICARUS.Visualization.airplanePolars import plotAirplanePolars
        plotAirplanePolars(db3d.Data, planenames, ["All"], size=(10, 10))
    desired = db3d.Data['XFLR_bmark']
    actual = db3d.Data['bmark']
    return desired, actual
