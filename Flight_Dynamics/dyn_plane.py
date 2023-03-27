import numpy as np
from .disturbances import disturbance as dst
import os
import jsonpickle


class dyn_plane():
    def __init__(self, pln, polars):
        """Class for the dynamic analysis of an airplane.
                The airplane is assumed to be of the airplane class.
                Inputs:
                - pln: Airplane class
                - polars: DataFrame with the polars of the airplane
        """

        self.pln = pln
        self.polars = polars

        # Compute Trim State
        self.trim = self.trimState()
        self.disturbances = []

    def get_polars(self):
        return self.polars

    def change_polars(self, polars):
        self.polars = polars
        self.trim = self.trimState()

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
        trim_ind = np.argmin(np.abs(self.polars["Cm"]))

        # Trim - related Aerodynamic Parameters of interest
        AoA_trim = self.polars["AoA"][trim_ind]
        CL_trim = self.polars["CL"][trim_ind]

        # How accurate is the trim
        print(f"Closest trim is at: {AoA_trim} deg")
        if np.abs(self.polars["Cm"][trim_ind]) > 1e-5:
            print(f"Cm is {self.polars['Cm'][trim_ind]} instead of 0")

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

    def approxAeroDerivatives(self):
        """Finds the Stability Derivatives using approximations"""
        S = self.pln.S
        dens = self.pln.dens
        W = self.pln.M * 9.81
        U = self.trim["U"]
        AoA = self.trim["AoA"]
        CL = self.polars["CL"][np.argmin(np.abs(self.polars["AoA"] - AoA))]
        CD = self.polars["CD"][np.argmin(np.abs(self.polars["AoA"] - AoA))]
        Cm = self.polars["Cm"][np.argmin(np.abs(self.polars["AoA"] - AoA))]
        CY = self.polars["CY"][np.argmin(np.abs(self.polars["AoA"] - AoA))]
        Cl = self.polars["Cl"][np.argmin(np.abs(self.polars["AoA"] - AoA))]
        Cn = self.polars["Cn"][np.argmin(np.abs(self.polars["AoA"] - AoA))]

        # Static Stability Derivatives
        X_u = -dens * U * S * CD / W
        X_w = -dens * U * S * CL / W
        X_q = 0
        X_de = 0

        Z_u = -dens * U * S * CL / W
        Z_w = -dens * U * S * CD / W
        Z_q = 0
        Z_de = 0

        M_u = -dens * U * S * self.pln.c * Cm / W
        M_w = 0
        M_q = -dens * U * S * self.pln.c * Cm / W
        M_de = 0

        Y_v = 0
        Y_p = 0
        Y_r = 0
        Y_da = 0
        Y_dr = 0

        L_v = 0
        L_p = -dens * U * S * self.pln.b * Cl / W
        L_r = 0
        L_da = 0
        L_dr = 0

        N_v = 0
        N_p = 0
        N_r = -dens * U * S * self.pln.b * Cn / W
        N_da = 0
        N_dr = 0

        # Gradient of the Static Stability Derivatives
    def get_pertrub(self):
        for dst in self.disturbances:
            print(dst)

    def __str__(self):
        str = f"Dynamic Plane Object for {self.pln.name}\n"
        str += f"Trimmed at: {self.trim['U']} m/s, {self.trim['AoA']} deg\n"
        str += f"Surfaces:\n"
        for surfaces in self.pln.surfaces:
            str += f"\n\t{surfaces.name} with Area: {surfaces.S}, Inertia: {surfaces.I}, Mass: {surfaces.M}\n"
        return str

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
        self.DynDir = os.getcwd()
        os.chdir(HOMEDIR)

    def runSolver(self, solver, args, kwargs={}):
        solver(*args, **kwargs)

    def setupSolver(self, setupsolver, args, kwargs={}):
        setupsolver(*args, **kwargs)

    def toJSON(self):
        return jsonpickle.encode(self)

    def save(self):
        os.chdir(self.CASEDIR)
        with open(f'{self.name}.json', 'w') as f:
            f.write(self.toJSON())
        os.chdir(self.HOMEDIR)
