import numpy as np


class disturbance():
    def __init__(self, variable, amplitude):
        self.amplitude = amplitude
        if amplitude > 0:
            self.isPositive = True
        else:
            self.isPositive = False

        if variable == "u":
            self.axis = 1
            self.type = "Derivative"  # Translational Only get Derivative
            self.isRotational = False
        elif variable == "w":
            self.axis = 3
            self.type = "Derivative"  # Translational Only get Derivative
            self.isRotational = False
        elif variable == "q":
            self.axis = 2
            self.type = "Derivative"  # Translational Only get Derivative
            self.isRotational = True
        elif variable == "theta":
            self.axis = 2
            self.type = "Value"  # Rotational
            self.isRotational = True

        elif variable == "v":
            self.axis = 2
            self.type = "Derivative"  # Translational Only get Derivative
            self.isRotational = False
        elif variable == "p":
            self.axis = 1
            self.type = "Derivative"  # Translational Only get Derivative
            self.isRotational = True
        elif variable == "r":
            self.axis = 3
            self.type = "Derivative"  # Rotational Only get Derivative
            self.isRotational = True
        elif variable == "phi":
            self.axis = 1
            self.type = "Value"  # Rotational
            self.isRotational = True
        elif variable == None:
            self.axis = None
            self.type = None
            self.amplitude = None
            self.name = "Trim"
            self.var = "Trim"
        else:
            raise ValueError("Invalid disturbance variable")
        if variable is not None:
            self.name = f"{variable} disturbance"
            self.var = variable

    def __str__(self):
        return f"{self.name}:\tType:\t{self.type} and \tAmplitude:\t{self.amplitude}."
