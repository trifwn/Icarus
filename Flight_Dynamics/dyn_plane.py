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
                self.polars3D.plot("AoA", "Cm")
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

            My_new = My  # - \
            # Fx_new * self.pln.CG[2] + \
            # Fy_new * self.pln.CG[1] + \
            # Fz_new * self.pln.CG[0]

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
        self.scheme = scheme
        self.epsilons = {}
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
            self.epsilons[var] = epsilon
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
            self.epsilons[var] = epsilon
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

    def logResults(self, makePolFun, args, kwargs={}):
        petrubdf = makePolFun(*args, **kwargs)
        self.pertubResults = petrubdf

    def longitudalStability(self):
        """This Function Requires the results from perturbation analysis
        For the Longitudinal Motion, in addition to the state space variables an analysis with respect to the derivative of w perturbation is needed.
        These derivatives are in this function are added externally and called Xw_dot,Zw_dot,Mw_dot. Depending on the Aerodynamics Solver, 
        these w_dot derivatives can either be computed like the rest derivatives, or require an approximation concerning the downwash velocity 
        that the main wing induces on the tail wing
        """
        pertr = self.pertubResults
        eps = self.epsilons
        Mass = self.pln.M
        U = self.trim["U"]   # TRIM
        theta = self.trim["AoA"] * np.pi / 180   # TRIM
        G = - 9.81
        Ix, Iy, Iz = self.pln.I
        X = {}
        Z = {}
        M = {}
        pertr = pertr.sort_values(by=["Epsilon"]).reset_index(drop=True)
        trimState = pertr[pertr["Type"] == "Trim"]
        mode = "2D"
        for var in ["u", "w", "q", "theta"]:
            if self.scheme == "Central":
                front = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] > 0)]
                back = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] < 0)]
            elif self.scheme == "Forward":
                front = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] > 0)]
                back = trimState
            elif self.scheme == "Backward":
                front = trimState
                back = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] < 0)]
            Xf = float(front[f"TFORC{mode}(1)"].to_numpy())
            Xb = float(back[f"TFORC{mode}(1)"].to_numpy())
            X[var] = (Xf - Xb)/(2*eps[var])

            Zf = float(front[f"TFORC{mode}(3)"].to_numpy())
            Zb = float(back[f"TFORC{mode}(3)"].to_numpy())
            Z[var] = (Zf - Zb)/(2*eps[var])

            Mf = float(front[f"TAMOM{mode}(2)"].to_numpy())
            Mb = float(back[f"TAMOM{mode}(2)"].to_numpy())
            M[var] = (Mf - Mb)/(2*eps[var])

        X["w_dot"] = 0
        Z["w_dot"] = 0

        xu = X["u"]/Mass + (X["w_dot"] * Z["u"])/(Mass*(Mass-Z["w_dot"]))
        xw = X["w"]/Mass + (X["w_dot"] * Z["w"])/(Mass*(Mass-Z["w_dot"]))
        # X["q"]/Mass + (X["w_dot"] * (Z["q"]+Mass*U*np.sin(theta))/(Mass*(Mass-Z["w_dot"]))
        xq = 0
        xth = -G*np.cos(theta) - (X["w_dot"]*G *
                                  np.sin(theta))/((Mass-Z["w_dot"]))

        zu = Z['u']/(Mass-Z["w_dot"])
        zw = Z['w']/(Mass-Z["w_dot"])
        zq = (Z['q']+Mass*U*np.sin(theta))/(Mass-Z["w_dot"])
        zth = -(Mass*G*np.sin(theta))/(Mass-Z["w_dot"])

        mu = M['u']/Iy + Z['u']*M["w_dot"]/(Iy*(Mass-Z["w_dot"]))
        mw = M['w']/Iy + Z['w']*M["w_dot"]/(Iy*(Mass-Z["w_dot"]))
        mq = M['q']/Iy + ((Z['q']+Mass*U*np.sin(theta)) *
                          M["w_dot"])/(Iy*(Mass-Z["w_dot"]))
        mth = (Mass*G*np.sin(theta)*M["w_dot"])/(Iy*(Mass-Z["w_dot"]))

        self.AstarLong = np.array([[X["u"], X["w"], X["q"], X["theta"]],
                                   [Z['u'], Z['w'], Z['q'], Z['theta']],
                                   [M['u'], M['w'], M['q'], M['theta']],
                                   [0, 1, 0, 0]])

        self.Along = np.array([[xu, xw, xq, xth],
                               [zu, zw, zq, zth],
                               [mu, mw, mq, mth],
                               [0, 1, 0, 0]])

    def lateralStability(self):
        """This Function Requires the results from perturbation analysis
        """
        pertr = self.pertubResults
        eps = self.epsilons
        Mass = self.pln.M
        U = self.trim["U"]
        theta = self.trim["AoA"] * np.pi / 180
        G = - 9.81
        Ix, Iy, Iz = self.pln.I
        Ixz = self.pln.Ixz  # self.pln.Ixz
        Y = {}
        L = {}
        N = {}
        pertr = pertr.sort_values(by=["Epsilon"]).reset_index(drop=True)
        trimState = pertr[pertr["Type"] == "Trim"]
        mode = "2D"
        for var in ["v", "p", "r", "phi"]:
            if self.scheme == "Central":
                back = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] < 0)]
                front = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] > 0)]
                de = 2 * eps[var]
            elif self.scheme == "Forward":
                back = trimState
                front = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] > 0)]
                de = eps[var]
            elif self.scheme == "Backward":
                back = pertr[(pertr["Type"] == var) & (pertr["Epsilon"] < 0)]
                front = trimState
                de = eps[var]
            Yf = float(front[f"TFORC{mode}(2)"].to_numpy())
            Yb = float(back[f"TFORC{mode}(2)"].to_numpy())
            Y[var] = (Yf - Yb)/de

            Lf = float(front[f"TAMOM{mode}(1)"].to_numpy())
            Lb = float(back[f"TAMOM{mode}(1)"].to_numpy())
            L[var] = (Lf - Lb)/de

            Nf = float(front[f"TAMOM{mode}(3)"].to_numpy())
            Nb = float(back[f"TAMOM{mode}(3)"].to_numpy())
            N[var] = (Nf - Nb)/de

        yv = Y['v']/Mass
        yp = (Y['p'] + Mass*U * np.sin(theta))/Mass
        yr = (Y['r'] - Mass*U * np.cos(theta))/Mass
        yphi = -G*np.cos(theta)

        lv = (Iz*L['v']+Ixz*N['v'])/(Ix*Iz-Ixz**2)
        lp = (Iz*L['p']+Ixz*N['p'])/(Ix*Iz-Ixz**2)
        lr = (Iz*L['r']+Ixz*N['r'])/(Ix*Iz-Ixz**2)
        lphi = 0

        nv = (Ix*N['v']+Ixz*L['v'])/(Ix*Iz-Ixz**2)
        n_p = (Ix*N['p']+Ixz*L['p'])/(Ix*Iz-Ixz**2)
        nr = (Ix*N['r']+Ixz*L['r'])/(Ix*Iz-Ixz**2)
        nph = 0

        self.AstarLat = np.array([[Y['v'], Y['p'], Y['r'], Y['phi']],
                                  [L['v'], L['p'], L['r'], L['phi']],
                                  [N['v'], N['p'], N['r'], N['phi']],
                                  [0, 1, 0, 0]])

        self.Alat = np.array([[yv, yp, yr, yphi],
                              [lv, lp, lr, lphi],
                              [nv, n_p, nr, nph],
                              [0, 1, 0, 0]])

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
