import numpy as np

from ICARUS.core.types import FloatArray

from .engine import ConceptualEngine


class ConceptualPlane:
    def __init__(
        self,
        S: float,
        AR: float,
        a0: float,
        m: float,
        engine: ConceptualEngine,
    ) -> None:
        """Constructor for the Conceptual Plane Class

        Args:
            S (float): Area of the wing
            AR (float): Aspect Ratio of the wing
            a0 (float): Zero lift angle of attack
            m (float): Mass of the aircraft
            engine (ConceptualEngine): Conceptual Engine


        """
        self.S: float = S
        self.AR: float = AR
        self.a0: float = a0
        self.m: float = m

        self.engine: ConceptualEngine = engine

    def cl(
        self,
        angle_of_attack: float | FloatArray,
    ) -> FloatArray | float:
        """Calculates the 3D lift coefficient based on the 2D lift coefficient and the aspect ratio

        Args:
            angle_of_attack (float): Angle of attack of the wing

        """
        cl_3d = self.cl_2d(angle_of_attack) / (1 + (2 / self.AR))

        # cl = self.main_wing.cl_3d * (
        #     1
        #     - self.fueselage.k
        #     * (self.main_wing.cl_0 / self.main_wing.cl_3d)
        #     * (self.fueselage.area / self.main_wing.area)
        # )
        return cl_3d

    def cl_2d(self, angle_of_attack: float | FloatArray) -> float | FloatArray:
        """Calculates the 2D lift coefficient based on the angle of attack

        Args:
            angle_of_attack (float): Angle of attack of the wing

        """
        cl = 2 * np.pi * (angle_of_attack - self.a0)

        return cl

    def cd0(self) -> float:
        return 0.05

    def cd(
        self,
        angle_of_attack: float | FloatArray,
        velocity: float,
    ) -> FloatArray | float:
        """Returns the value of cd based on the aoa

        Args:
            angle_of_attack (float): AOA
            velocity (float): Velocity

        """
        cd = self.cl(angle_of_attack) * velocity**2 + self.cd0()
        return cd

    def trust(self, velocity: float, aoa: float, amperes: float) -> None:
        """Calculates the trust of the plane based on the the engine model

        Args:
            velocity (float): _description_
            aoa (float): _description_
            amperes (float): _description_

        """
