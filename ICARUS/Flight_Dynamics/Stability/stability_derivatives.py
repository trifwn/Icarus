from ICARUS.Core.struct import Struct
from ICARUS.Flight_Dynamics.Stability.state_space import LateralStateSpace
from ICARUS.Flight_Dynamics.Stability.state_space import LongitudalStateSpace


class StabilityDerivativesDS(Struct):
    """
    Class to store stability derivatives of Dynamic Analysis.
    """

    def __init__(
        self,
        longitudal_state_space: LongitudalStateSpace,
        lateral_state_space: LateralStateSpace,
    ) -> None:
        """
        Initialize Stability Derivatives

        Args:
            longitudal_state_space (LongitudalStateSpace): Longitudal State Space Object
            lateral_state_space (LateralStateSpace): Lateral Space Object
        """
        self.X: dict[str, float] = longitudal_state_space.X
        self.Y: dict[str, float] = lateral_state_space.Y
        self.Z: dict[str, float] = longitudal_state_space.M
        self.L: dict[str, float] = lateral_state_space.L
        self.M: dict[str, float] = longitudal_state_space.M
        self.N: dict[str, float] = lateral_state_space.N

    def __str__(self) -> str:
        """
        String Representation of Stability Derivatives

        Returns:
            str: String Representation of Stability Derivatives
        """
        string = "Dimensional Stability Derivatives:\n"
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
