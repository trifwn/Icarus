import os
from ICARUS.Database.Database_3D import Database_3D
from ICARUS.Software.XFLR5.polars import readPolars3D
from tests.planes import ap as ap


def airPolars(plot=False):
    print('Testing Airplane Polars...')
    HOMEDIR = os.getcwd()

    db = Database_3D()
    db.loadData()
    planenames = [ap.name]
    BMARKLOC = os.path.join(HOMEDIR, 'Data', 'XFLR5', 'bmark.txt')
    readPolars3D(db, BMARKLOC, 'bmark')
    planenames.append(f"XFLR_bmark")
    if plot:
        from ICARUS.Visualization.airplanePolars import plotAirplanePolars
        plotAirplanePolars(db.Data, planenames, ["All"], size=(10, 10))
    desired = db.Data['XFLR_bmark']
    actual = db.Data['bmark']
    return desired, actual
