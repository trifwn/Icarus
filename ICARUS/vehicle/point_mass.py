import numpy as np
from scipy.integrate import quad
from ICARUS.core.types import FloatArray

class PointMass:
    """
    A simple point mass model for a vehicle.
    """

    def __init__(self, name: str, position: FloatArray, mass: float, inertia: FloatArray = np.zeros(6)) -> None:
        """
        Initialize the point mass model.

        :param name: Name of the vehicle.
        :param position: Position of the vehicle (m).
        :param mass: Mass of the vehicle (kg).
        :param inertia: Inertia of the vehicle (kg*m^2).
        """
        self.name = name
        self._mass = mass
        self._inertia: FloatArray = inertia
        self._position: FloatArray = position

    @property
    def position(self) -> FloatArray:
        """
        Get the position of the vehicle.

        :return: Position of the vehicle (m).
        """
        return self._position

    @position.setter
    def position(self, position: FloatArray) -> None:
        """
        Set the position of the vehicle.

        :param position: Position of the vehicle (m).
        """
        self._position = position

    @property
    def position_x(self) -> float:
        """
        Get the x position of the vehicle.

        :return: X position of the vehicle (m).
        """
        return float(self._position[0])

    @property
    def position_y(self) -> float:
        """
        Get the y position of the vehicle.

        :return: Y position of the vehicle (m).
        """
        return float(self._position[1])

    @property
    def position_z(self) -> float:
        """
        Get the z position of the vehicle.

        :return: Z position of the vehicle (m).
        """
        return float(self._position[2])

    @position_x.setter
    def position_x(self, position_x: float) -> None:
        """
        Set the x position of the vehicle.

        :param position_x: X position of the vehicle (m).
        """
        self._position[0] = position_x

    @position_y.setter
    def position_y(self, position_y: float) -> None:
        """
        Set the y position of the vehicle.

        :param position_y: Y position of the vehicle (m).
        """
        self._position[1] = position_y

    @position_z.setter
    def position_z(self, position_z: float) -> None:
        """
        Set the z position of the vehicle.

        :param position_z: Z position of the vehicle (m).
        """
        self._position[2] = position_z

    @property
    def mass(self) -> float:
        """
        Get the mass of the vehicle.

        :return: Mass of the vehicle (kg).
        """
        return self._mass

    @mass.setter
    def mass(self, mass: float) -> None:
        """
        Set the mass of the vehicle.

        :param mass: Mass of the vehicle (kg).
        """
        self._mass = mass

    @property
    def inertia(self) -> FloatArray:
        """
        Get the inertia of the vehicle.

        :return: Inertia of the vehicle (kg*m^2).
        """
        return self._inertia

    @inertia.setter
    def inertia(self, inertia: FloatArray) -> None:
        """
        Set the inertia of the vehicle.

        :param inertia: Inertia of the vehicle (kg*m^2).
        """
        # Assume inertia is a 6D vector
        if len(inertia) != 6:
            raise ValueError("Inertia must be a 6D vector.")
        self._inertia = inertia

    @property
    def I_xx(self) -> float:
        """
        Get the xx inertia of the vehicle.

        :return: XX inertia of the vehicle (kg*m^2).
        """
        return float(self._inertia[0])

    @property
    def I_yy(self) -> float:
        """
        Get the yy inertia of the vehicle.

        Returns:
            float: YY inertia of the vehicle (kg*m^2).
        """
        return float(self._inertia[1])

    @property
    def I_zz(self) -> float:
        """
        Get the zz inertia of the vehicle.

        Returns:
            float: ZZ inertia of the vehicle (kg*m^2).
        """
        return float(self._inertia[2])

    @property
    def I_xy(self) -> float:
        """
        Get the xy inertia of the vehicle.

        Returns:
            float: XY inertia of the vehicle (kg*m^2).
        """
        return float(self._inertia[3])

    @property
    def I_xz(self) -> float:
        """
        Get the xz inertia of the vehicle.

        Returns:
            float: XZ inertia of the vehicle (kg*m^2).
        """
        return float(self._inertia[4])

    @property
    def I_yz(self) -> float:
        """
        Get the yz inertia of the vehicle.

        Returns:
            float: YZ inertia of the vehicle (kg*m^2).
        """
        return float(self._inertia[5])

    @I_xx.setter
    def I_xx(self, I_xx: float) -> None:
        """
        Set the xx inertia of the vehicle.

        :param I_xx: XX inertia of the vehicle (kg*m^2).
        """
        self._inertia[0] = I_xx

    @I_yy.setter
    def I_yy(self, I_yy: float) -> None:
        """
        Set the yy inertia of the vehicle.

        :param I_yy: YY inertia of the vehicle (kg*m^2).
        """
        self._inertia[1] = I_yy

    @I_zz.setter
    def I_zz(self, I_zz: float) -> None:
        """
        Set the zz inertia of the vehicle.

        :param I_zz: ZZ inertia of the vehicle (kg*m^2).
        """
        self._inertia[2] = I_zz

    @I_xy.setter
    def I_xy(self, I_xy: float) -> None:
        """
        Set the xy inertia of the vehicle.

        :param I_xy: XY inertia of the vehicle (kg*m^2).
        """
        self._inertia[3] = I_xy

    @I_xz.setter
    def I_xz(self, I_xz: float) -> None:
        """
        Set the xz inertia of the vehicle.

        :param I_xz: XZ inertia of the vehicle (kg*m^2).
        """
        self._inertia[4] = I_xz

    @I_yz.setter
    def I_yz(self, I_yz: float) -> None:
        """
        Set the yz inertia of the vehicle.

        :param I_yz: YZ inertia of the vehicle (kg*m^2).
        """
        self._inertia[5] = I_yz

    @property
    def inertia_matrix(self) -> FloatArray:
        """
        Get the inertia matrix of the vehicle.

        :return: Inertia matrix of the vehicle (kg*m^2).
        """
        return np.array(
            [
                [self.I_xx, -self.I_xy, -self.I_xz],
                [-self.I_xy, self.I_yy, -self.I_yz],
                [-self.I_xz, -self.I_yz, self.I_zz],
            ],
        )

    def __repr__(self) -> str:
        """
        Get the string representation of the point mass model.

        :return: String representation of the point mass model.
        """
        return f"PointMass(name={self.name}, position={self.position}, mass={self.mass}, inertia={self.inertia})"

    def __str__(self) -> str:
        """
        Get the string representation of the point mass model.

        :return: String representation of the point mass model.
        """
        return self.__repr__()

    def copy(self) -> "PointMass":
        """
        Create a copy of the point mass model.

        :return: Copy of the point mass model.
        """
        return PointMass(
            name=self.name,
            position=self.position.copy(),
            mass=self.mass,
            inertia=self.inertia.copy(),
        )

    @classmethod
    def from_distribution(
        cls,
        name: str,
        position: FloatArray,
        mass: float,
        distribution,
    ) -> "PointMass":
        """
        Create a point mass from a distribution.

        :param position: Position of the vehicle (m).
        :param mass: Mass of the vehicle (kg).
        :param distribution: Function to calculate the moments of inertia. F(x,y,z) = mass_fraction at (x,y,z)
        """
        inertia = np.zeros(6)

        # Integrate the distribution to get the moments of inertia
        # XX intergral of the distribution
        inertia[0] = quad(
            lambda x: quad(
                lambda y: quad(
                    lambda z: distribution(x, y, z) * (y**2 + z**2),
                    -np.inf,
                    np.inf,
                )[0],
                -np.inf,
                np.inf,
            )[0],
            -np.inf,
            np.inf,
        )[0]

        # YY intergral of the distribution
        inertia[1] = quad(
            lambda x: quad(
                lambda y: quad(
                    lambda z: distribution(x, y, z) * (x**2 + z**2),
                    -np.inf,
                    np.inf,
                )[0],
                -np.inf,
                np.inf,
            )[0],
            -np.inf,
            np.inf,
        )[0]

        # ZZ intergral of the distribution
        inertia[2] = quad(
            lambda x: quad(
                lambda y: quad(
                    lambda z: distribution(x, y, z) * (x**2 + y**2),
                    -np.inf,
                    np.inf,
                )[0],
                -np.inf,
                np.inf,
            )[0],
            -np.inf,
            np.inf,
        )[0]

        # XY intergral of the distribution
        inertia[3] = quad(
            lambda x: quad(
                lambda y: quad(
                    lambda z: distribution(x, y, z) * (x * y),
                    -np.inf,
                    np.inf,
                )[0],
                -np.inf,
                np.inf,
            )[0],
            -np.inf,
            np.inf,
        )[0]

        # XZ intergral of the distribution
        inertia[4] = quad(
            lambda x: quad(
                lambda y: quad(
                    lambda z: distribution(x, y, z) * (x * z),
                    -np.inf,
                    np.inf,
                )[0],
                -np.inf,
                np.inf,
            )[0],
            -np.inf,
            np.inf,
        )[0]

        # YZ intergral of the distribution
        inertia[5] = quad(
            lambda x: quad(
                lambda y: quad(
                    lambda z: distribution(x, y, z) * (y * z),
                    -np.inf,
                    np.inf,
                )[0],
                -np.inf,
                np.inf,
            )[0],
            -np.inf,
            np.inf,
        )[0]

        inertia = inertia * mass
        return cls(name, position, mass, inertia)
