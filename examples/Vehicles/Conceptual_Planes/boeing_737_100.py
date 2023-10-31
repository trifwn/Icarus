import numpy as np

from ICARUS.Conceptual.concept_airplane import ConceptAirplane
from ICARUS.Conceptual.Criteria.FAR.helper_functions import drag_coeff_skin

UNITS_IMPERIAL: dict[str, float] = {
    "m": 3.28084,
    "m2": 10.7639,
    "kg": 2.20462,
}


ASPECT_RATIO = 8.5  # span_m**2 / Area_m

# CONVERT TO STUPIDITY UNITS
AREA: float = 1097  # ft2

## TAKEOFF
MTOW: float = 110000  # lb^2
TAKEOFF_DIST: float = 4600  # ft

## CRUISE
CRUISE_SPEED = 420  # kts
CRUISE_MACH = 0.73  # M
CRUISE_MAX_MACH = 0.82  # M
CRUISE_ALTITUDE_MAX = 37000  # ft
MAX_RANGE = 1400  # nmi

## LANDING
MLW: float = 89699
LANDING_DIST: float = 5305  # ft
VAT: float = np.sqrt(LANDING_DIST / 0.3)  # 140.2974
FAR_LANDING_DIST: float = LANDING_DIST  # / 0.6 # FAR 25.125

## ENGINE PARAMETERS
SFC = 0.796  # lb/lbf/hr
THRUST_PER_ENGINE = 14000  # lbf
NO_OF_ENGINES = 2
THRUST = THRUST_PER_ENGINE * NO_OF_ENGINES

# WEIGHT RATIO
WEIGHT_RATIO = 0.90
PAYLOAD_WEIGHT = 78000 - 57170  # lb


# CONSTANTS
# wing_loading_at_landing: float = MLW/AREA
# CL_MAX: float = far_inverse_landing_criterion_cl_max(
# V_app= VAT,
# wing_loading= wing_loading_at_landing,
# )
CL_MAX = 2.6
CD_0 = 0.02


# LANDING
OSWALD_LANDING = 0.7
CL_APP: float = CL_MAX / 1.69
CD_LANDING: float = drag_coeff_skin(cd_0=CD_0, flap_extension=30, landing_gear_cd=0.015)  # degrees

# TAKEOFF
CL_TAKEOFF: float = CL_MAX / 1.21

# CLIMB
OSWALD_CLIMB = 0.7
CL_CLIMB: float = CL_MAX / 1.44
CD_CLIMB: float = drag_coeff_skin(cd_0=CD_0, flap_extension=30, landing_gear_cd=0)  # degrees

# CRUISE
## MAX L/D
OSWALD_CRUISE = 0.85
CL_CRUISE = np.sqrt(CD_0 * (np.pi * OSWALD_CRUISE * ASPECT_RATIO))  # / np.sqrt(1 - CRUISE_MACH**2)
CD_CRUISE: float = 2 * CD_0
L_OVER_D = CL_CRUISE / CD_CRUISE

boeing_737_100: ConceptAirplane = ConceptAirplane(
    ASPECT_RATIO=ASPECT_RATIO,
    AREA=AREA,
    MTOW=MTOW,
    MAX_LANDING_WEIGHT=MLW,
    TAKEOFF_DIST=TAKEOFF_DIST,
    CRUISE_SPEED=CRUISE_SPEED,
    CRUISE_MACH=CRUISE_MACH,
    CRUISE_MAX_MACH=CRUISE_MAX_MACH,
    CRUISE_ALTITUDE_MAX=CRUISE_ALTITUDE_MAX,
    MAX_RANGE=MAX_RANGE,
    VAT=VAT,
    LANDING_DIST=LANDING_DIST,
    THRUST_PER_ENGINE=THRUST_PER_ENGINE,
    NO_OF_ENGINES=NO_OF_ENGINES,
    WEIGHT_RATIO=WEIGHT_RATIO,
    PAYLOAD_WEIGHT=PAYLOAD_WEIGHT,
    SFC=SFC,
    CL_MAX=CL_MAX,
    CD_0=CD_0,
    OSWALD_LANDING=OSWALD_LANDING,
    OSWALD_CLIMB=OSWALD_CLIMB,
    OSWALD_CRUISE=OSWALD_CRUISE,
)
