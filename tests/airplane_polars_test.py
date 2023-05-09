import os

from Data.Planes.simple_wing import airplane as airplane
from ICARUS.Database.Database_3D import Database_3D
from ICARUS.Database.db import DB
from ICARUS.Software.XFLR5.polars import readPolars3D


def airPolars(plot: bool | None = False):
    print("Testing Airplane Polars...")

    db = DB()
    db.loadData()

    db3d: Database_3D = db.vehiclesDB
    planenames = [airplane.name]
    BMARKLOC = os.path.join(db.HOMEDIR, "Data", "XFLR5", "bmark.txt")
    readPolars3D(db3d, BMARKLOC, "bmark")
    planenames.append("XFLR_bmark")
    if plot:
        from ICARUS.Visualization.airplanePolars import plot_airplane_polars

        plot_airplane_polars(db3d.data, planenames, ["All"], size=(10, 10))

    desired = db3d.data["XFLR_bmark"]
    actual = db3d.data["bmark"]
    return desired, actual
