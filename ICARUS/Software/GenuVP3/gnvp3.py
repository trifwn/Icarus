from ICARUS.Workers.solver import Solver
from ICARUS.Workers.analysis import Analysis
from ICARUS.Software.GenuVP3.angles import runGNVPanglesParallel, runGNVPangles
from ICARUS.Software.GenuVP3.pertrubations import runGNVPpertrParallel, runGNVPpertr
from ICARUS.Software.GenuVP3.filesInterface import GNVPexe


# ## Define Solver
gnvp3 = Solver(name = 'gnvp3',solverType = '3D', fidelity = 2)

# # Define GNVP3 Analyses
options = {
    'HOMEDIR': 'Home Directory',
    "CASEDIR": 'Case Directory',
}
rerun = Analysis('gnvp3','rerun', GNVPexe, options)  

options = {
    "plane":        "Plane Object",
    "environment":  "Environment",
    "foildb":       "2D Database",
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

gnvp3.addAnalyses([
    rerun,
    anglesSerial,
    anglesParallel,
    pertrubationSerial,
    pertrubationParallel
    ])


# EXAMPLE USAGE
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