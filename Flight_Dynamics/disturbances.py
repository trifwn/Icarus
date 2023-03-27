import numpy as np


class disturbance():
    def __init__(self, variable, amplitude):
        self.amplitude = amplitude
        if variable == "u":
            self.axis = 1
            self.type = "Derivative"  # Linear Only get Derivative
            self.isRotational = False
        elif variable == "w":
            self.axis = 3
            self.type = "Derivative"  # Linear Only get Derivative
            self.isRotational = False
        elif variable == "q":
            self.axis = 2
            self.type = "Derivatice"  # Linear Only get Derivative
            self.isRotational = True
        elif variable == "theta":
            self.axis = 2
            self.type = "Value"  # LINEAR
            self.isRotational = True

        elif variable == "v":
            self.axis = 2
            self.type = "Derivative"  # Linear Only get Derivative
            self.isRotational = False
        elif variable == "p":
            self.axis = 1
            self.type = "Derivative"  # Linear Only get Derivative
            self.isRotational = True
        elif variable == "r":
            self.axis = 3
            self.type = "Derivative"  # Rotational Only get Derivative
            self.isRotational = True
        elif variable == "phi":
            self.axis = 1
            self.type = "Value"  # LINEAR
            self.isRotational = True
        else:
            raise ValueError("Invalid disturbance variable")

        self.name = f"{variable} disturbance"

    def __str__(self):
        return f"Disturbance {self.name} of type {self.type} and amplitude {self.amplitude}."
