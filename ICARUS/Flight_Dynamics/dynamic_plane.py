from ICARUS.Core.struct import Struct


class StabilityDerivativesDS(Struct):
    """
    Class to store stability derivatives of Dynamic Analysis.
    """

    def __init__(
        self,
        X: dict[str, float],
        Y: dict[str, float],
        Z: dict[str, float],
        L: dict[str, float],
        M: dict[str, float],
        N: dict[str, float],
    ) -> None:
        """
        Initialize Stability Derivatives
        Args:
            X (dict[str, float]): Derivatives based on X
            Y (dict[str, float]): Derivatives based on Y
            Z (dict[str, float]): Derivatives based on Z
            L (dict[str, float]): Derivatives based on L
            M (dict[str, float]): Derivatives based on M
            N (dict[str, float]): Derivatives based on N
        """
        self.X: dict[str, float] = X
        self.Y: dict[str, float] = Y
        self.Z: dict[str, float] = Z
        self.L: dict[str, float] = L
        self.M: dict[str, float] = M
        self.N: dict[str, float] = N

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
