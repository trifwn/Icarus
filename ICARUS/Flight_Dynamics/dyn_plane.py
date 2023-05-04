import pandas as pd

from .disturbances import disturbance as dst
from .pertrubations import lateralPerturb
from .pertrubations import longitudalPerturb
from .Stability.lateralFD import lateralStability
from .Stability.longitudalFD import longitudalStability
from .trim import trim_state
from ICARUS.Core.struct import Struct
from ICARUS.Software.GenuVP3.postProcess.forces import rotateForces
from ICARUS.Vehicle.plane import Airplane


class dyn_Airplane(Airplane):
    """Class for the dynamic analysis of an airplane.
    The airplane is assumed to be of the airplane class.
    Inputs:
    - pln: Airplane class
    - polars3D: DataFrame with the polars of the airplane
    """

    def __init__(self, pln, polars3D=None):
        self.__dict__.update(pln.__dict__)
        self.name = f"dyn_{pln.name}"
        if polars3D is None:
            if pln.Polars.empty:
                print("No polars found in the airplane object or Specified")
            else:
                print("Using polars from the airplane object")
                polars3D = self.formatPolars(pln.Polars)
        else:
            self.polars3D = self.formatPolars(polars3D)

        # Compute Trim State
        self.trim = trim_state(self)
        self.defineSim(self.dens, self.trim["U"])
        self.disturbances = []
        self.sensitivity = {}
        self.sensResults = {}

    def get_polars3D(self):
        return self.polars3D

    def change_polars3D(self, polars3D):
        self.polars3D = polars3D
        self.trim = trim_state(self)

    def formatPolars(self, rawPolars):
        forces = rotateForces(rawPolars, rawPolars["AoA"])
        return self.makeAeroCoeffs(forces)

    def makeAeroCoeffs(self, Forces):
        Data = pd.DataFrame()

        Data[f"CL"] = Forces[f"Fz"] / (self.Q * self.S)
        Data[f"CD"] = Forces[f"Fx"] / (self.Q * self.S)
        Data[f"Cm"] = Forces[f"M"] / (self.Q * self.S * self.MAC)
        Data[f"Cn"] = Forces[f"N"] / (self.Q * self.S * self.MAC)
        Data[f"Cl"] = Forces[f"L"] / (self.Q * self.S * self.MAC)
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

        self.disturbances = [
            *longitudalPerturb(self, scheme, epsilon),
            *lateralPerturb(self, scheme, epsilon),
        ]
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

    def stabilityFD(self, scheme="Central"):
        self.scheme = scheme
        X, Z, M = longitudalStability(self, "2D")
        Y, L, N = lateralStability(self, "Potential")
        self.SBderivativesDS = StabilityDerivativesDS(X, Y, Z, L, M, N)

    def __str__(self):
        str = f"Dynamic AirPlane {self.name}"
        # str += f"\nTrimmed at: {self.trim['U']} m/s, {self.trim['AoA']} deg\n"
        # str += f"Surfaces:\n"
        # for surfaces in self.surfaces:
        #     str += f"\n\t{surfaces.name} with Area: {surfaces.S}, Inertia: {surfaces.I}, Mass: {surfaces.M}\n"
        return str


class StabilityDerivativesDS(Struct):
    def __init__(self, X, Y, Z, L, M, N):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.L = L
        self.M = M
        self.N = N

    def __str__(self):
        string = f"Dimensional Stability Derivatives:\n"
        string += "\nLongitudal Derivatives\n"
        string += f"Xu=\t{self.X['u']}\n"
        string += f"Xw=\t{self.X['w']}\n"
        string += f"Zu=\t{self.Z['u']}\n"
        string += f"Zw=\t{self.Z['w']}\n"
        string += f"Zq=\t{self.Z['q']}\n"
        string += f"Mu=\t{self.M['u']}\n"
        string += f"Mw=\t{self.M['w']}\n"
        string += f"Mq=\t{self.M['q']}\n"

        string += "\nLateral Derivatives\n"
        string += f"Yv=\t{self.Y['v']}\n"
        string += f"Yp=\t{self.Y['p']}\n"
        string += f"Yr=\t{self.Y['r']}\n"
        string += f"Lv=\t{self.L['v']}\n"
        string += f"Lp=\t{self.L['p']}\n"
        string += f"Lr=\t{self.L['r']}\n"
        string += f"Nv=\t{self.N['v']}\n"
        string += f"Np=\t{self.N['p']}\n"
        string += f"Nr=\t{self.N['r']}"

        return string
