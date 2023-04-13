import os
import pandas as pd

from ICARUS.Software.GenuVP3.postProcess.forces import rotateForces
from ICARUS.Vehicle.plane import Airplane

from .pertrubations import longitudalPerturb, lateralPerturb
from .disturbances import disturbance as dst
from .trim import trimState


class dyn_Airplane(Airplane):
    def __init__(self, pln, polars3D=None):
        """Class for the dynamic analysis of an airplane.
                The airplane is assumed to be of the airplane class.
                Inputs:
                - pln: Airplane class
                - polars3D: DataFrame with the polars of the airplane
        """
        self.pln = pln
        self.name = f"dyn_{pln.name}"
        if polars3D is None:
            if pln.Polars.empty:
                print("No polars found in the airplane object or Specified")
            else:
                print("Using polars from the airplane object")
                self.polars3D = self.formatPolars(pln.Polars)
        else:
            self.polars3D = polars3D

        # Compute Trim State
        self.trim = trimState(self)
        self.defineSim(self.dens, self.trim['U'])
        self.disturbances = []
        self.sensitivity = {}
        self.sensResults = {}

    def get_polars3D(self):
        return self.polars3D

    def change_polars3D(self, polars3D):
        self.polars3D = polars3D
        self.trim = trimState(self)

    def formatPolars(self, rawPolars):
        forces = rotateForces(rawPolars, rawPolars["AoA"])
        return self.makeAeroCoeffs(forces)

    def makeAeroCoeffs(self, Forces):
        Data = pd.DataFrame()

        Data[f"CL"] = Forces[f"Fz"] / (self.pln.Q*self.pln.S)
        Data[f"CD"] = Forces[f"Fx"] / (self.pln.Q*self.pln.S)
        Data[f"Cm"] = Forces[f"M"] / (self.pln.Q*self.pln.S*self.pln.MAC)
        Data[f"Cn"] = Forces[f"N"] / (self.pln.Q*self.pln.S*self.pln.MAC)
        Data[f"Cl"] = Forces[f"L"] / (self.pln.Q*self.pln.S*self.pln.MAC)
        Data["AoA"] = Forces["AoA"]
        return Data

    def allPerturb(self, scheme, epsilon=None):
        """Function to add a perturbations to the airplane for dynamic analysis
        Inputs:
        - scheme: "Central", "Forward", "Backward"
        - epsilon: Disturbance Magnitudes
        """
        self.scheme = scheme
        self.epsilons = {}

        self.disturbances = [*longitudalPerturb(self, scheme, epsilon),
                             *lateralPerturb(self, scheme, epsilon)]
        self.disturbances.append(dst(None, 0))

    def accessDynamics(self, HOMEDIR):
        self.HOMEDIR = HOMEDIR
        self.DBDIR = self.pln.DBDIR
        self.CASEDIR = self.pln.CASEDIR

        os.chdir(self.CASEDIR)
        os.system(f"mkdir -p Dynamics")
        os.chdir("Dynamics")
        self.DynDir = os.getcwd()
        os.chdir(HOMEDIR)

    def sensitivityAnalysis(self, var, space):
        self.sensitivity[var] = []
        for e in space:
            self.sensitivity[var].append(dst(var, e))

    def get_pertrub(self):
        for dst in self.disturbances:
            print(dst)

    def setPertResults(self, makePolFun, args, kwargs={}):
        petrubdf = makePolFun(*args, **kwargs)
        self.pertubResults = petrubdf

    def __str__(self):
        str = f"Dynamic AirPlane Object for {self.pln.name}\n"
        str += f"Trimmed at: {self.trim['U']} m/s, {self.trim['AoA']} deg\n"
        str += f"Surfaces:\n"
        for surfaces in self.pln.surfaces:
            str += f"\n\t{surfaces.name} with Area: {surfaces.S}, Inertia: {surfaces.I}, Mass: {surfaces.M}\n"
        return str

    def __getattr__(self, name):
        """Function to return an attribute of the airplane object (self.pln)
        if its name is not in the dynamic plane object (self)
        """
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            flag = False
            try:
                return self.pln.__dict__[name]
            except KeyError:
                flag = True
            finally:
                if flag == True:
                    raise AttributeError(
                        f"'dyn_plane' or 'plane' object has no attribute '{name}'")
