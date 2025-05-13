from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable

import numpy as np

from ICARUS.airfoils.airfoil import Airfoil
from ICARUS.core.types import FloatArray
from ICARUS.vehicle.control_surface import ControlSurface
from ICARUS.vehicle.control_surface import NoControl
from ICARUS.vehicle.surface import WingSurface
from ICARUS.vehicle.utils import DiscretizationType
from ICARUS.vehicle.utils import DistributionType
from ICARUS.vehicle.utils import SymmetryAxes
from ICARUS.vehicle.utils import cosine_spacing_function
from ICARUS.vehicle.utils import cspacer
from ICARUS.vehicle.utils import equal_spacing_function
from ICARUS.vehicle.utils import linear_distribution_function_factory
from ICARUS.vehicle.utils import sine_spacing_function


class WingSegment(WingSurface):
    def __init__(
        self,
        name: str,
        origin: FloatArray,
        orientation: FloatArray,
        span: float,
        tip_chord: float,
        root_chord: float,
        root_airfoil: str | Airfoil,
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
        span_spacing: DiscretizationType | str = DiscretizationType.LINEAR,
        chord_spacing: DiscretizationType | str = DiscretizationType.COSINE,
        N: int = 15,
        M: int = 5,
        mass: float = 1.0,
        controls: list[ControlSurface] = [NoControl],
        is_lifting: bool = True,
    ):
        """Creates a wing segment. A wing segment is a lifting surface with a finite span. The wing segment
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
        # Set the distributions of the wing segment
        self._dihedral_distribution = spanwise_dihedral_distibution
        self._twist_distribution = spanwise_twist_distribution
        self._chord_distribution = spanwise_chord_distribution

        self._spanwise_chord_distribution = spanwise_chord_distribution
        self._spanwise_dihedral_distibution = spanwise_dihedral_distibution
        self._spanwise_twist_distribution = spanwise_twist_distribution

        self._span_spacing = (
            span_spacing if isinstance(span_spacing, DiscretizationType) else DiscretizationType[span_spacing]
        )
        self._chord_spacing = (
            chord_spacing if isinstance(chord_spacing, DiscretizationType) else DiscretizationType[chord_spacing]
        )

        # Set the airfoils of the wing segment
        if isinstance(root_airfoil, str):
            root_airfoil = Airfoil.naca(root_airfoil)
        self._root_airfoil = root_airfoil
        if tip_airfoil is not None:
            if isinstance(tip_airfoil, str):
                tip_airfoil = Airfoil.naca(tip_airfoil)
        else:
            tip_airfoil = root_airfoil
        self._tip_airfoil = tip_airfoil

        # Set the symmetries of the wing segment
        if isinstance(symmetries, SymmetryAxes):
            symmetries = [symmetries]
        self.symmetries = symmetries

        # Set the wing segment main characteristics
        self._span = span
        if SymmetryAxes.Y in self.symmetries:
            span = span / 2
        self._root_chord = root_chord
        self._tip_chord = tip_chord

        # Set the twist and dihedral angles
        self._twist_root = twist_root
        self._twist_tip = twist_tip
        self._root_dihedral_angle = root_dihedral_angle
        self._tip_dihedral_angle = tip_dihedral_angle

        # Define X Offset
        # We can either define the sweepback angle or the sweep offset
        # If the sweep offset is defined, we can calculate the sweepback angle
        if sweep_offset != 0:
            self._sweepback_angle: float = np.arctan(sweep_offset / span) * 180 / np.pi
            self._sweep_offset: float = sweep_offset
        else:
            self._sweep_offset = np.tan(sweepback_angle * np.pi / 180) * span
            self._sweepback_angle = sweepback_angle

        # Set the wing segment discretization
        self.N = N
        self.M = M

        # Set the name
        self.name = name

        # Set the origin and orientation
        self._origin = origin
        self._orientation = orientation

        # Set the mass
        self._mass = mass

        # Wether the surface has lift
        self.is_lifting = is_lifting

        # set the moving segments
        self.controls = controls

        # Create the wing segment
        self._recalculate()

    def _recalculate(self) -> None:
        # Define chord function based on distribution type
        if self.spanwise_chord_distribution == DistributionType.LINEAR:
            chord_fun = linear_distribution_function_factory(
                x0=0,
                x1=1,
                y0=self._root_chord,
                y1=self._tip_chord,
            )
        else:
            raise NotImplementedError(
                f"Spanwise chord distribution type {self.spanwise_chord_distribution} not implemented",
            )

        # Define span function based on distribution type
        if self.spanwise_dihedral_distibution == DistributionType.LINEAR:
            # Define Dihedral Angle
            # Convert to dehidral angle to radians and then to a function of span
            dehidral_fun = linear_distribution_function_factory(
                x0=0,
                x1=1,
                y0=self._root_dihedral_angle * np.pi / 180,
                y1=self._tip_dihedral_angle * np.pi / 180,
            )
        else:
            raise NotImplementedError(
                f"Spanwise dihedral distribution type {self.spanwise_dihedral_distibution} not implemented",
            )

        # Define twist function based on distribution type
        if self.spanwise_twist_distribution == DistributionType.LINEAR:
            twist_fun = linear_distribution_function_factory(
                x0=0,
                x1=1,
                y0=self._twist_root * np.pi / 180,
                y1=self._twist_tip * np.pi / 180,
            )
        else:
            raise NotImplementedError(
                f"Spanwise twist distribution type {self.spanwise_twist_distribution} not implemented",
            )

        x_offset_fun = linear_distribution_function_factory(
            x0=0,
            x1=1,
            y0=0,
            y1=self._sweep_offset,
        )

        #### DESCRITIZATIONS ####
        # Define spanwise discretization
        if self._span_spacing == DiscretizationType.LINEAR:
            # Define the spanwise discretization function
            span_disc_fun = partial(equal_spacing_function, N=self.N, stretching=1.0)
            # equal_spacing_function_factory(N)
        elif self._span_spacing == DiscretizationType.SINE:
            span_disc_fun = partial(
                sine_spacing_function,
                N=-self.N,
                stretching=1.0,
                factor=0.0,
            )
        elif self._span_spacing == DiscretizationType.COSINE:
            if SymmetryAxes.Y in self.symmetries:
                span_disc_fun = partial(
                    sine_spacing_function,
                    N=self.N,
                    stretching=1.0,
                    factor=0.2,
                )
            else:
                span_disc_fun = partial(
                    cosine_spacing_function,
                    N=self.N,
                    stretching=1.0,
                )
        elif self._span_spacing == DiscretizationType.INV_COSINE:
            span_disc_fun = partial(
                cosine_spacing_function,
                N=-self.N,
                stretching=1.0,
            )
        else:
            raise NotImplementedError(
                f"Spanwise discretization type {self.span_spacing} not implemented",
            )

        # Define chordwise discretization
        if self._chord_spacing == DiscretizationType.LINEAR:
            # Define the chordwise discretization function
            chord_disc_fun: Callable[[int], float] = partial(
                equal_spacing_function,
                N=self.M,
                stretching=1.0,
            )
            # equal_spacing_function_factory(M)
        elif self._chord_spacing == DiscretizationType.COSINE:
            chord_disc_fun = partial(
                cosine_spacing_function,
                N=self.M,
                stretching=1.0,
            )
            # chord_disc_fun = partial(cspacer, N=self.M, cspace=1.0)
        elif self._chord_spacing == DiscretizationType.SINE:
            # chord_disc_fun = partial(
            #     sine_spacing_function,
            #     N=-self.M,
            #     stretching=1.0,
            #     factor=0.5,
            # )
            chord_disc_fun = partial(cspacer, N=self.M, cspace=2.0)
        elif self._chord_spacing == DiscretizationType.USER_DEFINED:
            try:
                chord_disc_fun = self.chord_discretization_function
            except AttributeError:
                raise AttributeError("Chord discretization function not defined")
        else:
            raise NotImplementedError(
                f"Chordwise discretization type {self.chord_spacing} not implemented",
            )

        # Create lifting surface object from the super().from_span_percentage_function constructor
        instance = super().from_span_percentage_functions(
            name=self.name,
            origin=self._origin,
            orientation=self._orientation,
            symmetries=self.symmetries,
            root_airfoil=self.root_airfoil,
            tip_airfoil=self.tip_airfoil,
            span=self.span,
            # Discretization
            span_discretization_function=span_disc_fun,
            chord_discretization_function=chord_disc_fun,
            # Geometry
            chord_as_a_function_of_span_percentage=chord_fun,
            x_offset_as_a_function_of_span_percentage=x_offset_fun,
            dihedral_as_a_function_of_span_percentage=dehidral_fun,
            twist_as_a_function_of_span_percentage=twist_fun,
            N=self.N,
            M=self.M,
            mass=self._mass,
            controls=self.controls,
            is_lifting=self.is_lifting,
        )

        # Get the properties of the wing segment instance and overwrite the properties of the wing segment
        # with the new values
        self.__dict__.update(instance.__dict__)

    # Create a decorator to recalculate the wing segment when some of the properties are changed
    # Specifically, when the setter of the span, root chord or tip chord is called we want to
    # recall the constructor of the wing segment (from_span_percentage_functions) to recalculate
    # the wing segment and store the new values
    @staticmethod
    def recalculation(func: Callable[..., None]) -> Callable[..., None]:
        def wrapper(self: WingSegment, *args: Any, **kwargs: Any) -> None:
            # Call the function
            func(self, *args, **kwargs)
            # Recalculate the wing segment
            self._recalculate()

        return wrapper

    #################### GETTERS AND SETTERS #####################################
    # Properties that are defined and get recalculated when the setter is called
    # are defined here
    ##############################################################################

    @property
    def root_chord(self) -> float:
        return self._root_chord

    @root_chord.setter
    @recalculation
    def root_chord(self, value: float) -> None:
        self._root_chord = value

    @property
    def tip_chord(self) -> float:
        return self._tip_chord

    @tip_chord.setter
    @recalculation
    def tip_chord(self, value: float) -> None:
        self._tip_chord = value

    # Define the rest of the properties
    @property
    def span(self) -> float:
        return self._span

    @span.setter
    @recalculation
    def span(self, value: float) -> None:
        if SymmetryAxes.Y in self.symmetries:
            value = value / 2
        self._span = value

    @property
    def twist_root(self) -> float:
        return self._twist_root

    @twist_root.setter
    @recalculation
    def twist_root(self, value: float) -> None:
        self._twist_root = value

    @property
    def twist_tip(self) -> float:
        return self._twist_tip

    @twist_tip.setter
    @recalculation
    def twist_tip(self, value: float) -> None:
        self._twist_tip = value

    @property
    def root_dihedral_angle(self) -> float:
        return self._root_dihedral_angle

    @root_dihedral_angle.setter
    @recalculation
    def root_dihedral_angle(self, value: float) -> None:
        self._root_dihedral_angle = value

    @property
    def tip_dihedral_angle(self) -> float:
        return self._tip_dihedral_angle

    @tip_dihedral_angle.setter
    @recalculation
    def tip_dihedral_angle(self, value: float) -> None:
        self._tip_dihedral_angle = value

    @property
    def sweepback_angle(self) -> float:
        return self._sweepback_angle

    @sweepback_angle.setter
    @recalculation
    def sweepback_angle(self, value: float) -> None:
        self._sweepback_angle = value

    @property
    def sweep_offset(self) -> float:
        return self._sweep_offset

    @sweep_offset.setter
    @recalculation
    def sweep_offset(self, value: float) -> None:
        self._sweep_offset = value

    @property
    def spanwise_chord_distribution(self) -> DistributionType:
        return self._spanwise_chord_distribution

    @spanwise_chord_distribution.setter
    @recalculation
    def spanwise_chord_distribution(self, value: DistributionType) -> None:
        self._spanwise_chord_distribution = value

    @property
    def spanwise_dihedral_distibution(self) -> DistributionType:
        return self._spanwise_dihedral_distibution

    @spanwise_dihedral_distibution.setter
    @recalculation
    def spanwise_dihedral_distibution(self, value: DistributionType) -> None:
        self._spanwise_dihedral_distibution = value

    @property
    def spanwise_twist_distribution(self) -> DistributionType:
        return self._spanwise_twist_distribution

    @spanwise_twist_distribution.setter
    @recalculation
    def spanwise_twist_distribution(self, value: DistributionType) -> None:
        self._spanwise_twist_distribution = value

    @property
    def span_spacing(self) -> DiscretizationType:
        if isinstance(self._span_spacing, str):
            return DiscretizationType[self._span_spacing]
        return self._span_spacing

    @span_spacing.setter
    @recalculation
    def span_spacing(self, value: DiscretizationType) -> None:
        self._span_spacing = value

    @property
    def chord_spacing(self) -> DiscretizationType:
        return self._chord_spacing

    @chord_spacing.setter
    @recalculation
    def chord_spacing(self, value: DiscretizationType) -> None:
        self._chord_spacing = value

    @property
    def root_airfoil(self) -> Airfoil:
        return self._root_airfoil

    @root_airfoil.setter
    @recalculation
    def root_airfoil(self, value: Airfoil) -> None:
        self._root_airfoil = value

    @property
    def tip_airfoil(self) -> Airfoil:
        return self._tip_airfoil

    @tip_airfoil.setter
    @recalculation
    def tip_airfoil(self, value: Airfoil) -> None:
        self._tip_airfoil = value

    def __repr__(self) -> str:
        return f"{self.name} (Wing Segment): S={self.area:.2f} m^2, Span={self.span:.2f} m, MAC={self.mean_aerodynamic_chord:.2f} m"

    def __str__(self) -> str:
        return f"{self.name} (Wing Segment): S={self.area:.2f} m^2, Span={self.span:.2f} m, MAC={self.mean_aerodynamic_chord:.2f} m"

    def __setstate__(self, state: dict[str, Any]) -> None:
        WingSegment.__init__(
            self,
            name=state["name"],
            origin=state["origin"],
            orientation=state["orientation"],
            span=state["span"],
            tip_chord=state["tip_chord"],
            root_chord=state["root_chord"],
            root_airfoil=state["root_airfoil"],
            sweepback_angle=state["sweepback_angle"],
            sweep_offset=state["sweep_offset"],
            twist_root=state["twist_root"],
            twist_tip=state["twist_tip"],
            root_dihedral_angle=state["root_dihedral_angle"],
            tip_dihedral_angle=state["tip_dihedral_angle"],
            tip_airfoil=state["tip_airfoil"],
            symmetries=state["symmetries"],
            spanwise_chord_distribution=state["spanwise_chord_distribution"],
            spanwise_dihedral_distibution=state["spanwise_dihedral_distibution"],
            spanwise_twist_distribution=state["spanwise_twist_distribution"],
            span_spacing=state["span_spacing"],
            chord_spacing=state["chord_spacing"],
            N=state["N"],
            M=state["M"],
            mass=state["mass"],
            controls=state["controls"],
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "origin": self.origin,
            "orientation": self.orientation,
            "span": self.span,
            "tip_chord": self.tip_chord,
            "root_chord": self.root_chord,
            "root_airfoil": self.root_airfoil,
            "sweepback_angle": self.sweepback_angle,
            "sweep_offset": self.sweep_offset,
            "twist_root": self.twist_root,
            "twist_tip": self.twist_tip,
            "root_dihedral_angle": self.root_dihedral_angle,
            "tip_dihedral_angle": self.tip_dihedral_angle,
            "tip_airfoil": self.tip_airfoil,
            "symmetries": self.symmetries,
            "spanwise_chord_distribution": self.spanwise_chord_distribution,
            "spanwise_dihedral_distibution": self.spanwise_dihedral_distibution,
            "spanwise_twist_distribution": self.spanwise_twist_distribution,
            "span_spacing": self.span_spacing,
            "chord_spacing": self.chord_spacing,
            "N": self.N,
            "M": self.M,
            "mass": self.mass,
            "controls": self.controls,
        }
