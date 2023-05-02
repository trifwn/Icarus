from ICARUS.Workers.solver import Solver
from ICARUS.Workers.analysis import Analysis
from ICARUS.Software.GenuVP3.analyses.angles import runGNVPanglesParallel, runGNVPangles, processGNVPangles
from ICARUS.Software.GenuVP3.analyses.pertrubations import runGNVPpertrParallel, runGNVPpertr, processGNVPpertrubations
from ICARUS.Software.GenuVP3.filesInterface import GNVPexe
from ICARUS.Database.db import DB


def get_gnvp3(db: DB):
    gnvp3 = Solver(name = 'gnvp3',solverType = '3D', fidelity = 2, db = db)

    # # Define GNVP3 Analyses
    options = {
        'HOMEDIR': 'Home Directory',
        "CASEDIR": 'Case Directory',
    }
    rerun = Analysis('gnvp3','rerun', GNVPexe, options)
        
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
    
    secondary_options = {
        'Split_Symmetric_Bodies' : (False, 'Split Symmetric Bodies And Contstruct Them In GNVP3'),
        # 'Integration_Scheme' : (2, 'NMETHT=1 for Euler =2 for Adams Bashford time integrat. scheme'),
        # 'Tip_Emmision' : (0, 'NEMTIP=0,1. The latter means that tip-emission takes place'),
        # 'Tip_Emmision_Begins' : (0, 'NTIMET Time at which tip-emission begins'),
        # 'Leading_Edge_Separation' : (0, '=0(no action), 1(leading-edge separ. takes place)'),
        # 'Leading_Edge_Separation_Begins' : (0, 'NTIMEL Time at which leading-edge separ. begins'),
        # 'Relaxation_Factor' : (0.9, 'RELAXS relaxation factor for the singularity distributions'),
        # 'Pot_Convergence_Tolerence' : (0.01, "EPSDS convergence tolerance of the potential calculations"),
        # 'Movement_Levels': (4, "NLEVELT number of movements levels"),
        # 'Vortex_Particle_Count' : (1., "NNEVP0 Number of vortex particles created within a time step per near-wake element of a thin  wing"),
        # 'Vortex_Particle_Relaxation' : (1.,'RELAXU relaxation factor for the emission velocity'),
        # 'Minimum_Width_Parameter' : (1., 'PARVEC parameter for the minimum width of the near-wake element'),
        # 'NEMIS': (1, '1 or 2 UKNOWN'),
        # 'Bound_Vorticity_Cutoff': (0.003, 'EPSFB  Cut-off length for the bound vorticity'),
        # 'Wake_Vorticity_Cutoff': (0.003, 'EPSFW  Cut-off length for the near-wake vorticity'),
        # 'Cutoff_Length_Sources': (0.003, 'EPSSR  Cut-off length for source distributions'),
        # 'Cutoff_Length_Sources2': (0.003, 'EPSDI  Cut-off length for source distributions'),
        # 'Vortex_Cutoff_Length_f': (0.500, 'EPSVR  Cut-off length for the free vortex particles (final)'),
        # 'Vortex_Cutoff_Length_i': (0.500,  'EPSO   Cut-off length for the free vortex particles (init.) '),
        # 'EPSINT': (0.001,  'EPSINT UKNOWN'),
        # 'Particle_Dissipation_Factor': (0., 'COEF    Factor for the disipation of particles'),
        # 'Upper_Deformation_Rate': (0.001, 'RMETM   Upper bound of the deformation rate'),
        # 'Wake_Deformation_Parameter': (1, 'IDEFW   Parameter for the deformation induced by the near wake '),
        # '': (1000.,  'REFLEN  Length used in VELEF for suppresing far-particle calc.'),
        # 'Particle_Subdivision_Parameter': (0, 'IDIVVRP Parameter for the subdivision of particles'),
        # 'Subdivision_Length_Scale': (1000.,  'FLENSC  Length scale for the subdivision of particles'),
        # 'Wake_Particle_Merging_Parameter': (0, 'NREWAK  Parameter for merging of particles'),
        # 'Particle_Merging_Parameter': (0, 'NMER    Parameter for merging of particles'),
        # 'Merging_Starting_Distance': (0, 'XREWAK  X starting distance of merging'),
        # 'Merging_Radius': (0, 'RADMER  Radius for merging'),
        # 'Elasticity_Solver': (0, 'IYNELST (1=BEAMDYN,2-ALCYONE,3=GAST)'),
    }

    anglesSerial = Analysis('gnvp3','Angles_Serial', runGNVPangles,
                            options,
                            secondary_options= secondary_options,
                            unhook = processGNVPangles) 

    anglesParallel = anglesSerial << {
        'name': 'Angles_Parallel',
        'execute': runGNVPanglesParallel,
        'unhook' : processGNVPangles
        }
    pertrubationSerial = anglesSerial << {
        'name': 'Pertrubation_Serial', 
        'execute': runGNVPpertr,
        'unhook' : processGNVPpertrubations
        }
    pertrubationParallel = anglesSerial << {
        'name': 'Pertrubation_Parallel',
        'execute': runGNVPpertrParallel,
        'unhook' : processGNVPpertrubations
        }
    
    gnvp3.addAnalyses([
        rerun,
        anglesSerial,
        anglesParallel,
        pertrubationSerial,
        pertrubationParallel
        ])
    
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