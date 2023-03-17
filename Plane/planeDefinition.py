import numpy as np
import os


class Airplane():
    def __init__(self, airfoils, WingAngle, HasWing=True, HasElevator=True, HasRudder=True):
        self.Polars = {}
        self.angles = []

        self.bodies = []
        self.airfoils = airfoils
        self.airMovement = self.airMov()

        if HasWing == True:
            self.bodies.append(self.WingL())
            self.bodies.append(self.WingR())
        if HasElevator == True:
            self.bodies.append(self.ElevatorR())
            self.bodies.append(self.ElevatorL())
        if HasRudder == True:
            self.bodies.append(self.Rudder())

        self.params = self.setParams(WingAngle)

        if (HasWing == True) and (HasRudder == False) and (HasRudder == False):
            self.CASENAME = "Wing"

        if (HasWing == False) and (HasRudder == True) and (HasRudder == False):
            self.CASENAME = "Rudder"

        if (HasWing == False) and (HasRudder == False) and (HasRudder == True):
            self.CASENAME = "Elevator"

        if (HasWing == True) and (HasRudder == True) and (HasRudder == True):
            self.CASENAME = "Plane"

    def accessDB(self, HOMEDIR, DBDIR):
        os.chdir(DBDIR)
        CASEDIR = self.CASENAME
        os.system(f"mkdir -p {CASEDIR}")
        os.chdir(CASEDIR)
        self.CASEDIR = os.getcwd()
        self.HOMEDIR = HOMEDIR
        self.DBDIR = DBDIR
        os.chdir(HOMEDIR)

    def visAirplane(self):
        return "Visualization not implemented Yet"

    def angleCASE(self, angle):
        self.currAngle = angle
        self.angles.append(self.currAngle)
        if self.currAngle in self.Polars.keys():
            pass
        else:
            self.Polars[self.currAngle] = {}

        if angle >= 0:
            folder = str(angle)[::-1].zfill(7)[::-1] + "/"
        else:
            folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1] + "/"

        try:
            self.ANGLEDIR = f"{self.CASEDIR}/{folder}"
            os.system(f"mkdir -p {self.ANGLEDIR}")
        except AttributeError:
            print("DATABASE is not initialized!")
            
    def batchangles(self, angles):
        for angle in angles:
            self.angles.append(angle)
            if angle in self.Polars.keys():
                pass
            else:
                self.Polars[angle] = {}

    def runSolver(self, solver, args, kwargs={}):
        solver(*args, **kwargs)

    def setupSolver(self, setupsolver, args, kwargs={}):
        setupsolver(*args, **kwargs)

    def cleanRes(self, cleanFun, args, kwargs={}):
        cleanFun(*args, **kwargs)

    def makePolars(self, makePolFun, solverName, args, kwargs={}):
        polarsdf = makePolFun(*args, **kwargs)
        self.Polars = polarsdf

    def airMov(self):
        airMovement = {
            'alpha_s': 0.,
            'alpha_e': 0.,
            'beta_s': 0.,
            'beta_e': 0.,
            'phi_s': 0.,
            'phi_e': 0.
        }
        return airMovement

    def setParams(self, WingAngle):
        params = {
            "nBods": len(self.bodies),  # len(Surfaces)
            "nBlades": len(self.airfoils),  # len(NACA)
            "maxiter": 50,
            "timestep": 10,
            "Uinf": [20. * np.cos(WingAngle*np.pi/180), 0.0, 20. * np.sin(WingAngle*np.pi/180)],
            "rho": 1.225,
            "visc": 0.0000156,
        }
        return params

    def WingL(self):
        wingL = {
            'NB': 1,
            "NACA": 4415,
            "name": "Lwing",
            'bld': 'Lwing.bld',
            'cld':  '4415.cld',
            'NNB': 25,
            'NCWB': 25,
            'is_right': False,
            "x_0": 0.,
            "z_0": 0.,
            "y_0": 0.,
            "pitch": 2.8,
            "cone": 0.,
            "wngang": 0.,
            "x_end": 0.,
            "z_end": 0.,
            "y_end": 1.130,
            "Root_chord": 0.159,
            "Tip_chord": 0.072
        }
        return wingL

    def WingR(self):
        wingR = {
            'NB': 2,
            "NACA": 4415,
            "name": "Rwing",
            'bld': 'Rwing.bld',
            'cld':  '4415.cld',
            'NNB': 25,
            'NCWB': 25,
            'is_right': True,
            "x_0": 0.,
            "z_0": 0.,
            "y_0": 0.,
            "pitch": 2.8,
            "cone": 0.,
            "wngang": 0.,
            "x_end": 0.,
            "z_end": 0.,
            "y_end": 1.130,
            "Root_chord": 0.159,
            "Tip_chord": 0.072
        }
        return wingR

    def ElevatorR(self):
        elevatorR = {
            'NB': 3,
            "NACA": '0008',
            "name": "Ltail",
            'bld': 'Ltail.bld',
            'cld':  '0008.cld',
            'NNB': 25,
            'NCWB': 25,
            'is_right': False,
            "x_0": 0.54,
            "z_0": 0.,
            "y_0": 0.,
            "pitch": 0.,
            "cone": 0.,
            "wngang": 0.,
            "x_end": 0.,
            "z_end": 0.,
            "y_end": 0.169,
            "Root_chord": 0.130,
            "Tip_chord": 0.03
        }
        return elevatorR

    def ElevatorL(self):
        elevatoL = {
            'NB': 4,
            "NACA": '0008',
            "name": "Rtail",
            'bld': 'Rtail.bld',
            'cld':  '0008.cld',
            'NNB': 25,
            'NCWB': 25,
            'is_right': True,
            "x_0": 0.54,
            "z_0": 0.,
            "y_0": 0.,
            "pitch": 0.,
            "cone": 0.,
            "wngang": 0.,
            "x_end": 0.,
            "z_end": 0.,
            "y_end": 0.169,
            "Root_chord": 0.130,
            "Tip_chord": 0.03
        }
        return elevatoL

    def Rudder(self):
        rudder = {
            'NB': 5,
            "NACA": '0008',
            "name": "rudder",
            'bld': 'rudder.bld',
            'cld':  '0008.cld',
            'NNB': 25,
            'NCWB': 25,
            'is_right': True,
            "x_0": 0.54,
            "z_0": 0.1,
            "y_0": 0.,
            "pitch": 0.,
            "cone": 0.,
            "wngang": 90.,
            "x_end": 0.,
            "z_end": 0.,
            "y_end": 0.169,
            "Root_chord": 0.130,
            "Tip_chord": 0.03
        }
        return rudder
