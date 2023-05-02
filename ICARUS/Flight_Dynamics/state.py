import pandas as pd

from ICARUS.Software.GenuVP3.postProcess.forces import rotateForces
from ICARUS.Enviroment.definition import Environment
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Core.struct import Struct

from .pertrubations import longitudalPerturb, lateralPerturb
from .dyn_plane import StabilityDerivativesDS
from .Stability.longitudalFD import longitudalStability
from .Stability.lateralFD import lateralStability
from .disturbances import disturbance as dst
from .trim import trimState

class State():
    """Class for the state of a vehicle.
    """
    def __init__(self, name: str, pln: Airplane, forces, env: Environment):
        self.vehicle = pln
        self.S = pln.S
        self.MAC = pln.MAC
        
        self.name = name
        self.polars = self.formatPolars(forces)
        
        self.trim = trimState(self)
        self.Q = 0.5* env.AirKinematicViscosity * self.trim['U']**2      
        self.disturbances = []
        self.sensitivity = Struct()
        self.sensResults = Struct()
        
    def formatPolars(self, forces):
        forces = rotateForces(forces, forces["AoA"])
        return self.makeAeroCoeffs(forces)
    
    def makeAeroCoeffs(self, Forces):
        Data = pd.DataFrame()
        
        Data[f"CL"] = Forces[f"Fz"] / (self.Q*self.S)
        Data[f"CD"] = Forces[f"Fx"] / (self.Q*self.S)
        Data[f"Cm"] = Forces[f"M"] / (self.Q*self.S*self.MAC)
        Data[f"Cn"] = Forces[f"N"] / (self.Q*self.S*self.MAC)
        Data[f"Cl"] = Forces[f"L"] / (self.Q*self.S*self.MAC)
        Data["AoA"] = Forces["AoA"]
        return Data
    
    def all_Pertrubations(self, scheme, epsilon=None):
        """Function to add a perturbations to the airplane for 
        dynamic analysis
        Inputs:
        - scheme: "Central", "Forward", "Backward"
        - epsilon: Disturbance Magnitudes
        """
        self.scheme = scheme
        self.epsilons = {}

        self.disturbances = [*longitudalPerturb(self, scheme, epsilon),
                             *lateralPerturb(self, scheme, epsilon)]
        self.disturbances.append(dst(None, 0))
    
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

    def stabilityFD(self, scheme='Central'):
        self.scheme = scheme
        X, Z, M = longitudalStability(self, '2D')
        Y, L, N = lateralStability(self, 'Potential')
        self.SBderivativesDS = StabilityDerivativesDS(X, Y, Z, L, M, N)
        
    def __str__(self):
        str = f"Dynamic AirPlane {self.name}"
        # str += f"\nTrimmed at: {self.trim['U']} m/s, {self.trim['AoA']} deg\n"
        # str += f"Surfaces:\n"
        # for surfaces in self.surfaces:
        #     str += f"\n\t{surfaces.name} with Area: {surfaces.S}, Inertia: {surfaces.I}, Mass: {surfaces.M}\n"
        return str