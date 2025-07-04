class Disturbance:
    """Class for representing disturbances"""

    def __init__(self, variable: str | None, amplitude_value: float) -> None:
        if amplitude_value > 0:
            self.amplitude: float | None = amplitude_value
        elif amplitude_value < 0:
            self.amplitude = amplitude_value
        elif amplitude_value == 0:
            self.amplitude = None

        if variable == "u":
            self.axis: int | None = 1
            self.type: str | None = "Derivative"  # Translational Only get Derivative
        elif variable == "w":
            self.axis = 3
            self.type = "Derivative"  # Translational Only get Derivative
        elif variable == "q":
            self.axis = 2
            self.type = "Derivative"  # Translational Only get Derivative
        elif variable == "theta":
            self.axis = 2
            self.type = "Value"  # Rotational

        elif variable == "v":
            self.axis = 2
            self.type = "Derivative"  # Translational Only get Derivative
        elif variable == "p":
            self.axis = 1
            self.type = "Derivative"  # Translational Only get Derivative
        elif variable == "r":
            self.axis = 3
            self.type = "Derivative"  # Rotational Only get Derivative
        elif variable == "phi":
            self.axis = 1
            self.type = "Value"  # Rotational
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

    @property
    def is_rotational(self) -> bool:
        """Check if the disturbance is rotational"""
        translation_only = ["u", "w", "v", "Trim"]
        return self.var not in translation_only and self.var is not None

    @property
    def is_positive(self) -> bool:
        """Check if the disturbance is positive"""
        return self.amplitude is not None and self.amplitude > 0

    def __str__(self) -> str:
        return f"{self.name}:\tType:\t{self.type} and \tAmplitude:\t{self.amplitude} and \tAxis:\t{self.axis}"
