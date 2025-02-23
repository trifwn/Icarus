class Disturbance:
    """Class for representing disturbances"""

    def __init__(self, variable: str | None, amplitude_value: float) -> None:
        if amplitude_value > 0:
            self.is_positive: bool = True
            self.amplitude: float | None = amplitude_value
        elif amplitude_value < 0:
            self.amplitude = amplitude_value
            self.is_positive = False
        elif amplitude_value == 0:
            self.amplitude = None
            self.is_positive = False

        if variable == "u":
            self.axis: int | None = 1
            self.type: str | None = "Derivative"  # Translational Only get Derivative
            self.isRotational: bool = False
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
        elif variable is None:
            self.axis = None
            self.type = None
            self.amplitude = None
            self.name: str = "Trim"
            self.var: str = "Trim"
        else:
            raise ValueError("Invalid disturbance variable")
        if variable is not None:
            self.name = f"{variable} disturbance"
            self.var = variable

    def __str__(self) -> str:
        return f"{self.name}:\tType:\t{self.type} and \tAmplitude:\t{self.amplitude} and \tAxis:\t{self.axis}"
