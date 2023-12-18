from functools import partial

import numpy as np

from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Core.types import FloatArray
from ICARUS.Vehicle.lifting_surface import Lifting_Surface
from ICARUS.Vehicle.utils import DiscretizationType
from ICARUS.Vehicle.utils import DistributionType
from ICARUS.Vehicle.utils import equal_spacing_function
from ICARUS.Vehicle.utils import linear_distribution_function_factory
from ICARUS.Vehicle.utils import SymmetryAxes


class Wing_Segment(Lifting_Surface):
    def __init__(
        self,
        name: str,
        root_airfoil: str | Airfoil,
        origin: FloatArray,
        orientation: FloatArray,
        span: float,
        root_chord: float,
        tip_chord: float,
        sweepback_angle: float = 0.0,
        sweep_offset: float = 0.0,
        twist_root: float = 0.0,
        twist_tip: float = 0.0,
        root_dihedral_angle: float = 0.0,
        tip_dihedral_angle: float = 0.0,
        tip_airfoil: str | Airfoil | None = None,
        symmetries: list[SymmetryAxes] | SymmetryAxes = SymmetryAxes.NONE,
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
        """
        Creates a wing segment. A wing segment is a lifting surface with a finite span. The wing segment
        is discretized into a number of panels in the spanwise and chordwise directions. The wing segment
        is basically a constructor of a Lifting_Surface.

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
            spanwise_chord_distribution (DistributionType, optional): Spanwise chord distribution. Defaults to DistributionType.LINEAR.
            symmetries (list[SymmetryAxes] | SymmetryAxes, optional): Symmetries of the wing segment. Defaults to SymmetryAxes.NONE.
            spanwise_dihedral_distibution (DistributionType, optional): Spanwise dihedral distribution. Defaults to DistributionType.LINEAR.
            spanwise_twist_distribution (DistributionType, optional): Spanwise twist distribution. Defaults to DistributionType.LINEAR.
            span_spacing (DiscretizationType, optional): Discretization type for the spanwise direction. Defaults to DiscretizationType.EQUAL.
            chord_spacing (DiscretizationType, optional): Discretization type for the chordwise direction. Defaults to DiscretizationType.EQUAL.
            N (int, optional): Number of panels for the span . Defaults to 15.
            M (int, optional): Number of panels for the chord. Defaults to 5.
        """
        # Define Symmetries
        if not isinstance(symmetries, list):
            symmetries = [symmetries]

        # Define Sweepback
        if SymmetryAxes.Y in symmetries:
            span = span / 2

        # Define chord function based on distribution type
        if spanwise_chord_distribution == DistributionType.LINEAR:
            chord_fun = linear_distribution_function_factory(
                x0=0,
                x1=1,
                y0=root_chord,
                y1=tip_chord,
            )
        else:
            raise NotImplementedError(f"Spanwise chord distribution type {spanwise_chord_distribution} not implemented")

        # Define span function based on distribution type
        if spanwise_dihedral_distibution == DistributionType.LINEAR:
            # Define Dihedral Angle
            # Convert to dehidral angle to radians and then to a function of span
            dehidral_fun = linear_distribution_function_factory(
                x0=0,
                x1=1,
                y0=root_dihedral_angle * np.pi / 180,
                y1=tip_dihedral_angle * np.pi / 180,
            )
        else:
            raise NotImplementedError(
                f"Spanwise dihedral distribution type {spanwise_dihedral_distibution} not implemented",
            )

        # Define twist function based on distribution type
        if spanwise_twist_distribution == DistributionType.LINEAR:
            twist_fun = linear_distribution_function_factory(
                x0=0,
                x1=1,
                y0=twist_root,
                y1=twist_tip,
            )
        else:
            raise NotImplementedError(f"Spanwise twist distribution type {spanwise_twist_distribution} not implemented")

        # Define X Offset
        # We can either define the sweepback angle or the sweep offset
        # If the sweep offset is defined, we can calculate the sweepback angle
        if sweep_offset != 0:
            sweepback_angle = np.arctan(sweep_offset / span) * 180 / np.pi
        else:
            sweep_offset = np.tan(sweepback_angle * np.pi / 180) * span

        x_offset_fun = linear_distribution_function_factory(
            x0=0,
            x1=1,
            y0=0,
            y1=sweep_offset,
        )

        #### DESCRITIZATIONS ####
        # Define spanwise discretization
        if span_spacing == DiscretizationType.EQUAL:
            # Define the spanwise discretization function
            span_disc_fun = partial(equal_spacing_function, N=N, stretching=1.0)
            # equal_spacing_function_factory(N)
        else:
            raise NotImplementedError(f"Spanwise discretization type {span_spacing} not implemented")

        # Define chordwise discretization
        if chord_spacing == DiscretizationType.EQUAL:
            # Define the chordwise discretization function
            chord_disc_fun = partial(equal_spacing_function, N=M, stretching=1.0)
            # equal_spacing_function_factory(M)
        else:
            raise NotImplementedError(f"Chordwise discretization type {chord_spacing} not implemented")

        if tip_airfoil is None:
            tip_airfoil = root_airfoil

        # Create lifting surface object from the super().from_span_percentage_function constructor
        instance = super().from_span_percentage_functions(
            name=name,
            origin=origin,
            orientation=orientation,
            symmetries=symmetries,
            root_airfoil=root_airfoil,
            tip_airfoil=tip_airfoil,
            span=span,
            # Discretization
            span_discretization_function=span_disc_fun,
            chord_discretization_function=chord_disc_fun,
            # Geometry
            chord_as_a_function_of_span_percentage=chord_fun,
            x_offset_as_a_function_of_span_percentage=x_offset_fun,
            dihedral_as_a_function_of_span_percentage=dehidral_fun,
            twist_as_a_function_of_span_percentage=twist_fun,
            N=N,
            M=M,
            mass=mass,
        )

        self.__dict__ = instance.__dict__
        self.span_spacing = span_spacing
        self.chord_spacing = chord_spacing
        
