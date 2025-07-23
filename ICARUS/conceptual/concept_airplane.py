from typing import Any
from typing import Callable

import numpy as np

from ICARUS.core.types import FloatArray
from ICARUS.core.units import si_to_imperial

from .criteria.FAR import drag_coeff_skin
from .criteria.FAR.get_all_criteria import get_all_far_criteria


class ConceptAirplane:
    """Conceptual AIRPLANE class to calculate all FAR criteria
    and produce Match Point Diagrams.
    """

    def __init__(
        self,
        ASPECT_RATIO: float,
        AREA: float,
        MTOW: float,
        MAX_LANDING_WEIGHT: float,
        TAKEOFF_DIST: float,
        LANDING_DIST: float,
        NO_OF_ENGINES: float,
        THRUST_PER_ENGINE: float,
        CRUISE_SPEED: float,
        CRUISE_MACH: float,
        CRUISE_MAX_MACH: float,
        CRUISE_ALTITUDE_MAX: float,
        MAX_RANGE: float,
        WEIGHT_RATIO: float,
        SFC: float,
        PAYLOAD_WEIGHT: float,
        CL_MAX: float,
        CD_0: float,
        VAT: float,
        OSWALD_LANDING: float,
        OSWALD_CLIMB: float,
        OSWALD_CRUISE: float,
        UNITS: str = "Imperial",
    ) -> None:
        if UNITS == "Metric":
            AREA = AREA * si_to_imperial["m2"]
            MTOW = MTOW * si_to_imperial["kg"]
            TAKEOFF_DIST = TAKEOFF_DIST * si_to_imperial["m"]
            MTOW = MTOW * si_to_imperial["kg"]
            TAKEOFF_DIST = TAKEOFF_DIST * si_to_imperial["m"]
            CRUISE_ALTITUDE_MAX = CRUISE_ALTITUDE_MAX * si_to_imperial["m"]
            MAX_LANDING_WEIGHT = MAX_LANDING_WEIGHT * si_to_imperial["kg"]
            LANDING_DIST = LANDING_DIST * si_to_imperial["m"]
            PAYLOAD_WEIGHT = PAYLOAD_WEIGHT * si_to_imperial["kg"]

        ## GET ALL PARAMETERS
        # DIMENSIONAL PARAMETERS
        self.ASPECT_RATIO: float = ASPECT_RATIO
        self.AREA: float = AREA

        # TAKEOFF PARAMETERS
        self.MTOW: float = MTOW

        # CRUISE PARAMETERS
        self.TAKEOFF_DIST: float = TAKEOFF_DIST
        self.CRUISE_SPEED: float = CRUISE_SPEED
        self.CRUISE_MACH: float = CRUISE_MACH
        self.CRUISE_MAX_MACH: float = CRUISE_MAX_MACH
        self.CRUISE_ALTITUDE_MAX: float = CRUISE_ALTITUDE_MAX
        self.MAX_RANGE: float = MAX_RANGE

        # LANDING PARAMETERS
        self.VAT: float = VAT
        self.MLW = MAX_LANDING_WEIGHT
        self.LANDING_DIST = LANDING_DIST

        # ENGINE PARAMETERS
        self.SFC: float = SFC
        self.THRUST_PER_ENGINE: float = THRUST_PER_ENGINE
        self.NO_OF_ENGINES: float = NO_OF_ENGINES

        # WEIGHT PARAMETERS
        self.WEIGHT_RATIO: float = WEIGHT_RATIO
        self.PAYLOAD_WEIGHT: float = PAYLOAD_WEIGHT

        # AERODYNAMIC PARAMETERS
        self.CL_MAX: float = CL_MAX  # with flaps
        self.CD_0: float = CD_0
        self.OSWALD_LANDING: float = OSWALD_LANDING
        self.OSWALD_CLIMB: float = OSWALD_CLIMB
        self.OSWALD_CRUISE: float = OSWALD_CRUISE
        self._sigma = 1

        self.property_dict = {
            "ASPECT_RATIO",
            "AREA",
            "MTOW",
            "TAKEOFF_DIST",
            "CRUISE_SPEED",
            "CRUISE_MACH",
            "CRUISE_MAX_MACH",
            "CRUISE_ALTITUDE_MAX",
            # "MAX_RANGE",
            "VAT",
            "MLW",
            "LANDING_DIST",
            "SFC",
            "THRUST_PER_ENGINE",
            "NO_OF_ENGINES",
            "WEIGHT_RATIO",
            # "PAYLOAD_WEIGHT",
            "CL_MAX",
            "CD_0",
            "OSWALD_LANDING",
            "OSWALD_CLIMB",
            "OSWALD_CRUISE",
        }
        self.missing_vals: list[str] = self.get_missing_parameters()

    @property
    def THRUST(self) -> float | None:
        if self.THRUST_PER_ENGINE is None or self.NO_OF_ENGINES is None:
            return None
        return self.THRUST_PER_ENGINE * self.NO_OF_ENGINES

    @property
    def FAR_LANDING_DIST(self) -> float | None:
        if self.LANDING_DIST is None:
            return None
        return self.LANDING_DIST / 0.6

    @property
    def FAR_TAKEOFF_DIST(self) -> float | None:
        if self.TAKEOFF_DIST is None:
            return None
        return self.TAKEOFF_DIST  # 1.15

    # LIFT COEFFICIENTS
    @property
    def CL_APP(self) -> float | None:
        if self.CL_MAX is None:
            return None
        return self.CL_MAX / 1.699

    @property
    def CL_TAKEOFF(self) -> float | None:
        if self.CL_MAX is None:
            return None
        return self.CL_MAX / 1.21

    @property
    def CL_CLIMB(self) -> float | None:
        if self.CL_TAKEOFF is None:
            return None
        return self.CL_MAX / 1.44

    @property
    def CL_CRUISE(self) -> float | None:
        if self.CD_0 is None:
            return None
        if self.OSWALD_CRUISE is None:
            return None
        if self.ASPECT_RATIO is None:
            return None
        return float(
            np.sqrt(
                self.CD_0 * (np.pi * self.OSWALD_CRUISE * self.ASPECT_RATIO),
            ),  # / np.sqrt(1 - self.CRUISE_MACH**2)
        )

    # DRAG COEFFICIENTS
    @property
    def CD_LANDING(self) -> float | None:
        if self.CD_0 is None:
            return None
        return drag_coeff_skin(
            cd_0=self.CD_0,
            flap_extension=30,
            landing_gear_cd=0.015,
        )  # degrees

    @property
    def CD_CLIMB(self) -> float | None:
        if self.CD_0 is None:
            return None
        return drag_coeff_skin(
            cd_0=self.CD_0,
            flap_extension=30,
            landing_gear_cd=0,
        )  # degrees

    @property
    def CD_CRUISE(self) -> float | None:
        if self.CD_0 is None:
            return None
        return 2 * self.CD_0

    @property
    def L_OVER_D(self) -> float | None:
        cl = self.CL_CRUISE
        cd = self.CD_CRUISE
        if cl is None or cd is None:
            return None
        return cl / cd

    @property
    def RANGE(self) -> float:
        if self.MAX_RANGE is None:
            # return -12527.1851 * self.WEIGHT_RATIO +11696.4201
            return -0.1189 * self.PAYLOAD + 4383
        return self.MAX_RANGE

    @property
    def PAYLOAD(self) -> float:
        if self.PAYLOAD_WEIGHT is None:
            return self.WEIGHT_RATIO * 105359 - 61509
        return self.PAYLOAD_WEIGHT

    @property
    def sigma(self) -> float:
        if self._sigma is not None:
            return self._sigma
        return 1.0

    def get_missing_parameters(self) -> list[str]:
        missing_vals: list[str] = []
        for item in self.property_dict:
            if callable(self.__dict__[item]):
                if item is None:
                    missing_vals.append(item)
            elif getattr(self, item) is None:
                missing_vals.append(item)
        return missing_vals

    def set_parameters(self, missing_val_dict: dict[str, Any]) -> dict[str, Any]:
        rest_of_args: dict[str, Any] = {}
        for name, value in missing_val_dict.items():
            if name in self.property_dict:
                setattr(self, name, value)
            else:
                rest_of_args[name] = value
        return rest_of_args

    def supress_parameters(self, values_to_suppress: list[str]) -> None:
        for value in values_to_suppress:
            if value not in self.property_dict:
                print(f"{value} is not a valid parameter")
            setattr(self, value, None)
        # print(self.get_missing_parameters())

    def far_criteria(
        self,
        **missing_vals: dict[str, bool],
    ) -> tuple[
        tuple[FloatArray, FloatArray],
        tuple[FloatArray, FloatArray],
        tuple[FloatArray, FloatArray],
        tuple[FloatArray, FloatArray],
        tuple[FloatArray, FloatArray],
        float,
        float,
        float,
        float,
        float,
    ]:
        kwargs = self.set_parameters(missing_vals)
        if not self.get_missing_parameters():
            res = get_all_far_criteria(
                ASPECT_RATIO=self.ASPECT_RATIO,
                AREA=self.AREA,
                MTOW=self.MTOW,
                FAR_TAKEOFF_DISTANCE=self.FAR_TAKEOFF_DIST,  # type: ignore[arg-type]  # noqa
                FAR_LANDING_DISTANCE=self.FAR_LANDING_DIST,  # type: ignore[arg-type]  # noqa
                NO_OF_ENGINES=self.NO_OF_ENGINES,  # type: ignore[arg-type]  # noqa
                THRUST=self.THRUST,  # type: ignore[arg-type]  # noqa
                CD_0=self.CD_0,
                CD_LANDING=self.CD_LANDING,  # type: ignore[arg-type]  # noqa
                CD_CLIMB=self.CD_CLIMB,  # type: ignore[arg-type]  # noqa
                OSWALD_LANDING=self.OSWALD_LANDING,
                OSWALD_CLIMB=self.OSWALD_CLIMB,
                OSWALD_CRUISE=self.OSWALD_CRUISE,
                CL_APP=self.CL_APP,  # type: ignore[arg-type]  # noqa
                CL_CRUISE=self.CL_CRUISE,  # type: ignore[arg-type]  # noqa
                CL_TAKEOFF=self.CL_TAKEOFF,  # type: ignore[arg-type]  # noqa
                CL_CLIMB=self.CL_CLIMB,  # type: ignore[arg-type]  # noqa
                CRUISE_ALTITUDE=self.CRUISE_ALTITUDE_MAX,
                CRUISE_MACH=self.CRUISE_MACH,
                L_OVER_D=self.L_OVER_D,  # type: ignore[arg-type]  # noqa
                WEIGHT_RATIO=self.WEIGHT_RATIO,
                RANGE=self.RANGE,
                SFC=self.SFC,
                PAYLOAD_WEIGHT=self.PAYLOAD,
                sigma=self.sigma,
                **kwargs,
            )
            return res
        raise (ValueError(f"Missing values: {self.missing_vals}"))

    def partial_fun_factory(
        self,
        objective_function: Callable[..., Any],
        args_to_suppress: list[str] = [],
        **kwargs: dict[str, Any],
    ) -> Callable[..., Any]:
        not_defined: list[str] = [
            arg for arg in args_to_suppress or self.get_missing_parameters()
        ]
        print(
            f"Function will try to evaluate with the following params as inputs: {not_defined}",
        )
        for arg in not_defined:
            if arg not in self.property_dict:
                print(f"{arg} is not a valid parameter")

        def optimize_fun(
            new_args: list[Any] | Any,
            **new_kwargs: dict[str, Any],
        ) -> Any:
            if not isinstance(new_args, list):
                new_args = new_args.tolist()

            kwargs.update(new_kwargs)
            self.supress_parameters(args_to_suppress)
            args_to_set = {}

            if len(not_defined) > 1:
                args_to_set.update({k: v for k, v in zip(not_defined, new_args)})
            elif len(not_defined) == 1:
                args_to_set.update({not_defined[0]: new_args[0]})

            self.set_parameters(args_to_set)
            (
                landing_curve,
                failed_approach_curve,
                takeoff_curve,
                climb_curve,
                cruise_curve,
                est_weight,
                est_S,
                est_thrust,
                op_thrust_loading,
                op_wing_loading,
            ) = self.far_criteria(**kwargs)

            return objective_function(
                self,
                landing_curve,
                failed_approach_curve,
                takeoff_curve,
                climb_curve,
                cruise_curve,
                est_weight,
                est_S,
                est_thrust,
                op_thrust_loading,
                op_wing_loading,
            )

        return optimize_fun
