import numpy as np
import time
import os 

# # MODULES
from ICARUS.Software.XFLR5.polars import readPolars2D
from ICARUS.Database.db import DB
from ICARUS.Database import XFLRDB

start_time = time.time()
HOMEDIR = os.getcwd()

# # DB CONNECTION
db = DB()
foildb = db.foilsDB
foildb.loadData()
readPolars2D(foildb,XFLRDB)
airfoils = foildb.getAirfoils()


from Data.Planes.hermes import hermes
from ICARUS.Flight_Dynamics.dyn_plane import dyn_Airplane as dp
from ICARUS.Enviroment.definition import EARTH
from ICARUS.Software.GenuVP3.gnvp3 import get_gnvp3

names = ["Hermes_50", "Hermes_100",'Hermes_200',"Hermes_400"]
maxiter = {
    "Hermes_400"    : 400,
    "Hermes_200"    : 200,
    "Hermes_100"    : 100,
    "Hermes_50"     : 50
}
timestep = {
    "Hermes_400"    : 5e-3,
    "Hermes_200"    : 2e-2,
    "Hermes_100"    : 8e-2,
    "Hermes_50"     : 5e-1
}
for name in names:
    # # Get Plane
    ap = hermes(airfoils ,name)

    # # Import Enviroment
    print(EARTH)

    # # Get Solver
    gnvp3 = get_gnvp3(db)

    # ## AoA Run
    analysis = gnvp3.getAvailableAnalyses()[2] # ANGLES PARALLEL
    gnvp3.setAnalysis(analysis)
    options = gnvp3.getOptions(analysis)

    AoAmin = -6
    AoAmax = 8
    NoAoA = (AoAmax - AoAmin) + 1
    angles = np.linspace(AoAmin, AoAmax, NoAoA)
    Uinf = 20
    ap.defineSim(Uinf, EARTH.AirDensity)

    options.plane.value         = ap
    options.environment.value   = EARTH
    options.db.value            = db
    options.solver2D.value      = 'XFLR'
    options.maxiter.value       = maxiter[name]
    options.timestep.value      = timestep[name]
    options.Uinf.value          = Uinf
    options.angles.value        = angles

    gnvp3.printOptions()

    gnvp3.run()
    polars_time = time.time()
    print("Polars took : --- %s seconds --- in Parallel Mode" %
        (time.time() - polars_time))
    polars = gnvp3.getResults()
    ap.save()

    # # Dynamics

    # ### Define and Trim Plane
    try:
        dyn = dp(ap,polars)
    except:
        import sys
        sys.exit("Plane could not be trimmed")

    # ### Pertrubations
    if name == "Hermes_400":
        epsilons = {
            "u": 0.01,
            "w": 0.01,
            "q": 0.001,
            "theta": 0.01 ,
            "v": 0.01,
            "p": 0.001,
            "r": 0.001,
            "phi": 0.001
        }
    else:
        epsilons = None
    dyn.allPerturb("Central",epsilons)
    dyn.get_pertrub()

    # Define Analysis for Pertrubations
    analysis = gnvp3.getAvailableAnalyses()[4] # Pertrubations PARALLEL
    print(f"Selecting Analysis: {analysis}")
    gnvp3.setAnalysis(analysis)
    options = gnvp3.getOptions(analysis)

    # Set Options
    dyn.defineSim(dyn.trim['U'], EARTH.AirDensity)
    options.plane.value         = dyn
    options.environment.value   = EARTH
    options.db.value            = db
    options.solver2D.value      = 'XFLR'
    options.maxiter.value       = maxiter[name]
    options.timestep.value      = timestep[name]
    options.Uinf.value          = dyn.trim['U']
    options.angles.value        = dyn.trim['AoA']

    # Run Analysis
    gnvp3.printOptions()

    pert_time = time.time()
    print("Running Pertrubations")
    gnvp3.run()
    print("Pertrubations took : --- %s seconds ---" %
            (time.time() - pert_time))
    dyn.save()

    # Get Results
    results = gnvp3.getAvailableRetrieval()[1] # Pertr
    print(f"Selecting Retrieval: {results}")
    gnvp3.setRetrivalMethod(results)
    options = gnvp3.getRetrievalOptions(results)


    options.HOMEDIR.value = db.analysesDB.HOMEDIR
    options.DYNDIR.value = os.path.join(db.vehiclesDB.DATADIR, ap.CASEDIR, "Dynamics")
    gnvp3.printRetrievalOptions()
    polars = gnvp3.getResults()

    ## Sensitivity ANALYSIS

    # print time it took
    print("PROGRAM TERMINATED")
    print("Execution took : --- %s seconds ---" % (time.time() - start_time))
