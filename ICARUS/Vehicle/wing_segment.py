import numpy as np

from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Core.types import FloatArray
from ICARUS.Vehicle.lifting_surface import Lifting_Surface
from ICARUS.Vehicle.utils import define_linear_chord
from ICARUS.Vehicle.utils import define_linear_span
from ICARUS.Vehicle.utils import define_linear_twist
from ICARUS.Vehicle.utils import DiscretizationType
from ICARUS.Vehicle.utils import DistributionType


class Wing_Segment(Lifting_Surface):
    def __init__(
        self,
        name: str,
        airfoil: str | Airfoil,
        origin: FloatArray,
        orientation: FloatArray,
        span: float,
        sweep_offset: float,
        root_chord: float,
        tip_chord: float,
        twist_root: float,
        twist_tip: float,
        root_dihedral_angle: float,
        tip_dihedral_angle: float,
        is_symmetric: bool = False,
        # Geometry generation
        spanwise_chord_distribution: DistributionType = DistributionType.LINEAR,
        spanwise_dihedral_distibution: DistributionType = DistributionType.LINEAR,
        spanwise_twist_distribution: DistributionType = DistributionType.LINEAR,
        # Geometry discretization
        span_spacing: DiscretizationType = DiscretizationType.EQUAL,
        chord_spacing: DiscretizationType = DiscretizationType.EQUAL,
        N: int = 15,
        M: int = 5,
        mass: float = 1.0,
    ):
        """Creates a wing segment. A wing segment is a lifting surface with a finite span. The wing segment
        is discretized into a number of panels in the spanwise and chordwise directions. The wing segment
        is convrted into a lifting surface object.

        Args:
            name (str): Name of the wing segment
            airfoil (str): Name of the airfoil to be used or Airfoil object.
            origin (FloatArray): Origin of the wing segment
            orientation (FloatArray):  Orientation of the wing segment
            span (float): Span of the wing segment
            root_chord (float): Root chord of the wing segment
            tip_chord (float): Tip chord of the wing segment
            twist_root (float): Twist at the root of the wing segment
            twist_tip (float): Twist at the tip of the wing segment
            root_dihedral_angle (float): Dihedral angle at the root of the wing segment
            tip_dihedral_angle (float):  Dihedral angle at the tip of the wing segment
            is_symmetric (bool, optional): Whethere the wing is symmetric along the y axis. Defaults to False.
            spanwise_chord_distribution (DistributionType, optional): Spanwise chord distribution. Defaults to DistributionType.LINEAR.
            spanwise_dihedral_distibution (DistributionType, optional): Spanwise dihedral distribution. Defaults to DistributionType.LINEAR.
            spanwise_twist_distribution (DistributionType, optional): Spanwise twist distribution. Defaults to DistributionType.LINEAR.
            span_spacing (DiscretizationType, optional): Discretization type for the spanwise direction. Defaults to DiscretizationType.EQUAL.
            chord_spacing (DiscretizationType, optional): Discretization type for the chordwise direction. Defaults to DiscretizationType.EQUAL.
            N (int, optional): Number of panels for the span . Defaults to 15.
            M (int, optional): Number of panels for the chord. Defaults to 5.
        """

        # Define chord function based on distribution type
        if spanwise_chord_distribution == DistributionType.LINEAR:
            chord_fun = define_linear_chord
        else:
            raise NotImplementedError(f"Spanwise chord distribution type {spanwise_chord_distribution} not implemented")

        # Define span function based on distribution type
        if spanwise_dihedral_distibution == DistributionType.LINEAR:
            span_fun = define_linear_span
        else:
            raise NotImplementedError(
                f"Spanwise dihedral distribution type {spanwise_dihedral_distibution} not implemented",
            )

        # Define twist function based on distribution type
        if spanwise_twist_distribution == DistributionType.LINEAR:
            twist_fun = define_linear_twist
        else:
            raise NotImplementedError(f"Spanwise twist distribution type {spanwise_twist_distribution} not implemented")

        super().__init__(
            name=name,
            airfoil=airfoil,
            origin=origin,
            orientation=orientation,
            is_symmetric=is_symmetric,
            span=span,
            sweep_offset=0,
            dih_angle=0,
            chord_fun=chord_fun,
            chord=np.array([root_chord, tip_chord]),
            span_fun=span_fun,
            N=N,
            M=M,
            mass=mass,
            twist=np.array([twist_root, twist_tip]),
            twist_fun=twist_fun,
        )
