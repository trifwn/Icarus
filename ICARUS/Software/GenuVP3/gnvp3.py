from ICARUS.Workers.solver import Solver
from ICARUS.Workers.analysis import Analysis
from ICARUS.Workers.resRetrival import Results
from ICARUS.Software.GenuVP3.angles import runGNVPanglesParallel, runGNVPangles
from ICARUS.Software.GenuVP3.pertrubations import runGNVPpertrParallel, runGNVPpertr
from ICARUS.Software.GenuVP3.filesInterface import GNVPexe
from ICARUS.Software.GenuVP3.filesInterface import makePolar,pertrResults
from ICARUS.Database.db import DB


def get_gnvp3(db: DB):
    gnvp3 = Solver(name = 'gnvp3',solverType = '3D', fidelity = 2, db = db)

    # # Define GNVP3 Analyses
    options = {
        'HOMEDIR': 'Home Directory',
        "CASEDIR": 'Case Directory',
    }
    rerun = Analysis('gnvp3','rerun', GNVPexe, options)
    makePlanePolar = Results('makePlanePolar','gnvp3', makePolar, options)
        
    options = {
        "plane":        "Plane Object",
        "environment":  "Environment",
        "db":           "Database",
        "solver2D":     "2D Solver",
        "maxiter":      "Max Iterations",
        "timestep":     "Timestep",
        "Uinf":         "Velocity Magnitude",
        "angles":       "Angle to run",
    }

    anglesSerial = Analysis('gnvp3','Angles_Serial', runGNVPangles, options) 
    anglesParallel = anglesSerial << {'name': 'Angles_Parallel', 'execute': runGNVPanglesParallel}
    pertrubationSerial = anglesSerial << {'name': 'Pertrubation_Serial', 'execute': runGNVPpertr}
    pertrubationParallel = anglesSerial << {'name': 'Pertrubation_Parallel', 'execute': runGNVPpertrParallel}

    options = {
        'HOMEDIR':  'Home Directory',
        "DYNDIR":   'Dynamics Directory',
    }
    planePertResults = Results('planePertResults','gnvp3', pertrResults, options)
    
    gnvp3.addAnalyses([
        rerun,
        anglesSerial,
        anglesParallel,
        pertrubationSerial,
        pertrubationParallel
        ])
    
    gnvp3.addResRetrival([makePlanePolar,planePertResults])
    
    return gnvp3


# # EXAMPLE USAGE
# ## Define Solver
# db = DB()
# db.loadData()
# gnvp3 = get_gnvp3(db)
# analysis = gnvp3.getAvailableAnalyses()[0]
# gnvp3.setAnalysis(analysis)
# options = gnvp3.getOptions(analysis)
# from ICARUS.Database.Database_3D import Database_3D, ang2case
# import os 
# HOMEDIR = os.getcwd()
# db = Database_3D(HOMEDIR)
# plane = db.Planes["Hermes"]
# CASEDIR = plane.CASEDIR + "/" + ang2case(0.) + "/"
# options['HOMEDIR'].value = HOMEDIR
# options['CASEDIR'].value = CASEDIR
# gnvp3.run()