from ICARUS.core.types import FloatArray
from scipy.integrate import quad
import numpy as np

class PointMass(object):
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
        self._inertia = inertia
        self._position = position

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
        return self._position[0]
    
    @property
    def position_y(self) -> float:
        """
        Get the y position of the vehicle.

        :return: Y position of the vehicle (m).
        """
        return self._position[1]

    @property
    def position_z(self) -> float:
        """
        Get the z position of the vehicle.

        :return: Z position of the vehicle (m).
        """
        return self._position[2]
    
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
        self._inertia = inertia

    @classmethod
    def from_distribution(
        cls,
        name : str,
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
