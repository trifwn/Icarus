import numpy as np
from .disturbances import disturbance as dst
import pandas as pd
import os
import jsonpickle


class dyn_plane():
    def __init__(self, pln, polars2D, polars3D=None):
        """Class for the dynamic analysis of an airplane.
                The airplane is assumed to be of the airplane class.
                Inputs:
                - pln: Airplane class
                - polars2D: DataFrame with the polars of the airfoils
                - polars3D: DataFrame with the polars of the airplane
        """

        self.pln = pln
        self.name = f"dyn_{pln.name}"
        if polars3D is None:
            if pln.Polars.empty:
                print("No polars found in the airplane object or Specified")
            else:
                self.rawpolars = pln.Polars
                self.polars3D = self.formatPolars()
        else:
            self.polars3D = polars3D

        # Compute Trim State
        self.trim = self.trimState()
        self.defineSim(self.dens)
        self.disturbances = []

    def get_polars3D(self):
        return self.polars3D

    def change_polars3D(self, polars3D):
        self.polars3D = polars3D
        self.trim = self.trimState()

    def formatPolars(self, preferred="2D"):
        beta = 0
        Data = pd.DataFrame()

        Data["AoA"] = self.rawpolars["AoA"]
        AoA = Data["AoA"] * np.pi/180
        for enc, name in zip(["", "2D", "DS2D"], ["Potential", "2D", "ONERA"]):
            Fx = self.rawpolars[f"TFORC{enc}(1)"]
            Fy = self.rawpolars[f"TFORC{enc}(2)"]
            Fz = self.rawpolars[f"TFORC{enc}(3)"]

            Mx = self.rawpolars[f"TAMOM{enc}(1)"]
            My = self.rawpolars[f"TAMOM{enc}(2)"]
            Mz = self.rawpolars[f"TAMOM{enc}(3)"]

            Fx_new = Fx * np.cos(-AoA) - Fz * np.sin(-AoA)
            Fy_new = Fy
            Fz_new = Fx * np.sin(-AoA) + Fz * np.cos(-AoA)

            My_new = My - \
                Fx_new * self.pln.CG[2] + \
                Fy_new * self.pln.CG[1] + \
                Fz_new * self.pln.CG[0]

            Data[f"CL_{name}"] = Fz_new / (self.pln.Q*self.pln.S)
            Data[f"CD_{name}"] = Fx_new / (self.pln.Q*self.pln.S)
            Data[f"Cm_{name}"] = My_new / (self.pln.Q*self.pln.S*self.pln.MAC)
        print(f"Using {preferred} polar for dynamic analysis")
        Data[f"CL"] = Data[f"CL_{preferred}"]
        Data[f"CD"] = Data[f"CD_{preferred}"]
        Data[f"Cm"] = Data[f"Cm_{preferred}"]

        # Reindex the dataframe sort by AoA
        return Data.sort_values(by="AoA").reset_index(drop=True)

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

    def trimState(self):
        """This function returns the trim conditions of the airplane
        It is assumed that the airplane is trimmed at a constant altitude
        The trim conditions are:
        - Velocity
        - Angle of attack
        - Angle of sideslip         ! NOT IMPLEMENTED YET
        - Elevator deflection       ! NOT IMPLEMENTED YET
        - Aileron deflection        ! NOT IMPLEMENTED YET
        - Rudder deflection         ! NOT IMPLEMENTED YET
        - Throttle setting          ! NOT IMPLEMENTED YET
        - Engine torque             ! NOT IMPLEMENTED YET
        - Engine power              ! NOT IMPLEMENTED YET
        - Engine thrust             ! NOT IMPLEMENTED YET
        - Engine fuel flow          ! NOT IMPLEMENTED YET
        - Engine fuel consumption   ! NOT IMPLEMENTED YET
        - Engine fuel remaining     ! NOT IMPLEMENTED YET
        """

        # Index of interest in the Polar Dataframe
        trimLoc1 = np.argmin(np.abs(self.polars3D["Cm"]))

        # Find the polar that is closest to the trim but positive
        trimLoc2 = trimLoc1
        if self.polars3D["Cm"][trimLoc1] < 0:
            while self.polars3D["Cm"][trimLoc2] < 0:
                trimLoc2 += 1
        else:
            while self.polars3D["Cm"][trimLoc2] > 0:
                trimLoc2 -= 1

        # from trimLoc1 and trimLoc2, interpolate the angle where Cm = 0
        dCm = self.polars3D["Cm"][trimLoc2] - self.polars3D["Cm"][trimLoc1]
        dAoA = self.polars3D["AoA"][trimLoc2] - self.polars3D["AoA"][trimLoc1]

        AoA_trim = self.polars3D["AoA"][trimLoc1] - \
            self.polars3D["Cm"][trimLoc1] * dAoA / dCm

        Cm_trim = self.polars3D["Cm"][trimLoc1] + \
            (self.polars3D["Cm"][trimLoc2] - self.polars3D["Cm"][trimLoc1]) * \
            (AoA_trim - self.polars3D["AoA"][trimLoc1]) / \
            (self.polars3D["AoA"][trimLoc2] - self.polars3D["AoA"][trimLoc1])

        CL_trim = self.polars3D["CL"][trimLoc1] + \
            (self.polars3D["CL"][trimLoc2] - self.polars3D["CL"][trimLoc1]) * \
            (AoA_trim - self.polars3D["AoA"][trimLoc1]) / \
            (self.polars3D["AoA"][trimLoc2] - self.polars3D["AoA"][trimLoc1])

        # Print How accurate is the trim
        print(
            f"Cm is {self.polars3D['Cm'][trimLoc1]} instead of 0 at AoA = {self.polars3D['AoA'][trimLoc1]}")
        print(
            f"Interpolated values are: AoA = {AoA_trim} , Cm = {Cm_trim}, Cl = {CL_trim}")

        # Find the trim velocity
        S = self.pln.S
        dens = self.pln.dens
        W = self.pln.M * 9.81
        U_cruise = np.sqrt(W / (0.5 * dens * CL_trim * S))

        trim = {
            "U": U_cruise,
            "AoA": AoA_trim,
        }
        return trim

    def perturb(self, variable, amplitude):
        """Function to add a perturbation to the airplane
        Inputs:
        - variable: string with the variable to perturb
        - amplitude: amplitude of the perturbation
        """
        self.disturbances[variable] = dst(variable, amplitude)

    def allPerturb(self, epsilon, scheme):
        self.disturbances = [*self.longitudalPerturb(epsilon, scheme),
                             *self.lateralPerturb(epsilon, scheme)]
        self.disturbances.append(dst(None, 0))

    def longitudalPerturb(self, epsilon, scheme):
        """Function to add all longitudinal perturbations
        needed to compute the aero derivatives
        Inputs:
        - variable: string with the variable to perturb
        - amplitude: amplitude of the perturbation
        """
        disturbances = []
        for var in ["u", "w", "q", "theta"]:
            if scheme == "Central":
                disturbances.append(dst(var, epsilon))
                disturbances.append(dst(var, -epsilon))
            elif scheme == "Forward":
                disturbances.append(dst(var, epsilon))
            elif scheme == "Backward":
                disturbances.append(dst(var, -epsilon))
            else:
                raise ValueError(
                    "Scheme must be 'Central', 'Forward' or 'Backward'")
        return disturbances

    def lateralPerturb(self, epsilon, scheme):
        """Function to add all lateral perturbations 
        needed to compute the aero derivatives
        Inputs:
        - variable: string with the variable to perturb
        - amplitude: amplitude of the perturbation
        """
        disturbances = []
        for var in ["v", "p", "r", "phi"]:
            if scheme == "Central":
                disturbances.append(dst(var, epsilon))
                disturbances.append(dst(var, -epsilon))
            elif scheme == "Forward":
                disturbances.append(dst(var, epsilon))
            elif scheme == "Backward":
                disturbances.append(dst(var, -epsilon))
            else:
                raise ValueError(
                    "Scheme must be 'Central', 'Forward' or 'Backward'")
        return disturbances

    def get_pertrub(self):
        for dst in self.disturbances:
            print(dst)

    def defineSim(self, dens):
        self.U = self.trim["U"]
        self.dens = dens
        self.Q = 0.5 * dens * self.U ** 2

    def accessDB(self, HOMEDIR):
        self.HOMEDIR = HOMEDIR
        self.DBDIR = self.pln.DBDIR
        self.CASEDIR = self.pln.CASEDIR

        os.chdir(self.CASEDIR)
        os.system(f"mkdir -p Dynamics")
        os.chdir("Dynamics")
        self.DynDir = os.getcwd()
        os.chdir(HOMEDIR)

    def toJSON(self):
        return jsonpickle.encode(self)

    def save(self):
        os.chdir(self.CASEDIR)
        with open(f'{self.name}.json', 'w') as f:
            f.write(self.toJSON())
        os.chdir(self.HOMEDIR)

    def __str__(self):
        str = f"Dynamic Plane Object for {self.pln.name}\n"
        str += f"Trimmed at: {self.trim['U']} m/s, {self.trim['AoA']} deg\n"
        str += f"Surfaces:\n"
        for surfaces in self.pln.surfaces:
            str += f"\n\t{surfaces.name} with Area: {surfaces.S}, Inertia: {surfaces.I}, Mass: {surfaces.M}\n"
        return str

    def runAnalysis(self, solver, args, kwargs={}):
        solver(*args, **kwargs)

    # def approxAeroDerivatives(self):
    #     """Finds the Stability Derivatives using approximations
    #     GENERATED BY COPILOT
    #     """
    #     S = self.pln.S
    #     dens = self.pln.dens
    #     W = self.pln.M * 9.81
    #     U = self.trim["U"]
    #     AoA = self.trim["AoA"]
    #     CL = self.polars3D["CL"][np.argmin(np.abs(self.polars3D["AoA"] - AoA))]
    #     CD = self.polars3D["CD"][np.argmin(np.abs(self.polars3D["AoA"] - AoA))]
    #     Cm = self.polars3D["Cm"][np.argmin(np.abs(self.polars3D["AoA"] - AoA))]
    #     CY = self.polars3D["CY"][np.argmin(np.abs(self.polars3D["AoA"] - AoA))]
    #     Cl = self.polars3D["Cl"][np.argmin(np.abs(self.polars3D["AoA"] - AoA))]
    #     Cn = self.polars3D["Cn"][np.argmin(np.abs(self.polars3D["AoA"] - AoA))]

    #     # Static Stability Derivatives
    #     X_u = -dens * U * S * CD / W
    #     X_w = -dens * U * S * CL / W
    #     X_q = 0
    #     X_de = 0

    #     Z_u = -dens * U * S * CL / W
    #     Z_w = -dens * U * S * CD / W
    #     Z_q = 0
    #     Z_de = 0

    #     M_u = -dens * U * S * self.pln.c * Cm / W
    #     M_w = 0
    #     M_q = -dens * U * S * self.pln.c * Cm / W
    #     M_de = 0

    #     Y_v = 0
    #     Y_p = 0
    #     Y_r = 0
    #     Y_da = 0
    #     Y_dr = 0

    #     L_v = 0
    #     L_p = -dens * U * S * self.pln.b * Cl / W
    #     L_r = 0
    #     L_da = 0
    #     L_dr = 0

    #     N_v = 0
    #     N_p = 0
    #     N_r = -dens * U * S * self.pln.b * Cn / W
    #     N_da = 0
    #     N_dr = 0
