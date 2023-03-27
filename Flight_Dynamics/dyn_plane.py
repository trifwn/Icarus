import numpy as np


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
            try:
                return self.pln.__dict__[name]
            except KeyError:
                pass
            finally:
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
