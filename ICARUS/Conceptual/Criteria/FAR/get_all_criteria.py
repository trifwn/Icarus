import numpy as np
import matplotlib.pyplot as plt
from .landing_criterion import far_1_landing
from .failed_approach_criterion  import far_2_failed_approach
from .takeoff_criterion import far_3_takeoff
from .climb_criterion import far_4_climb
from .cruise_criterion import far_5_cruise_speed
from .range import range_criterion
from .usefull_load import usefull_load_criterion

def get_all_far_criteria(
    ASPECT_RATIO: float,
    AREA: float,
    MTOW: float,
    FAR_TAKEOFF_DISTANCE: float,
    FAR_LANDING_DISTANCE: float,
    NO_OF_ENGINES: int,
    THRUST: float,
    CD_0: float,
    CD_LANDING: float,
    CD_CLIMB: float,
    OSWALD_LANDING: float,
    OSWALD_CLIMB: float,
    OSWALD_CRUISE: float,
    CL_APP: float,
    CL_CRUISE: float,
    CL_TAKEOFF: float,
    CL_CLIMB: float,
    CRUISE_ALTITUDE: float,
    CRUISE_MACH: float,
    L_OVER_D: float,
    WEIGHT_RATIO: float,
    RANGE: float,
    SFC: float,
    PAYLOAD_WEIGHT: float,
    sigma: float = 1,
    plot: bool = False,
    start_plot: bool = False,
    show_plot: bool = False,
    clip = [5,400]
):
    print(sigma)
    thrust_loading = np.linspace(0,clip[0],500) # thrust loading
    wing_loading = np.linspace(5,clip[1],500) # wing loading
    far_1 = far_1_landing(
        l_landing= FAR_LANDING_DISTANCE,
        cl_approach= CL_APP,
        thrust_loading= thrust_loading,
        sigma= sigma,
    )

    far_2 = far_2_failed_approach(
        no_of_engines= NO_OF_ENGINES,
        cl_app= CL_APP,
        cd = CD_LANDING,
        AR = ASPECT_RATIO,
        e = OSWALD_LANDING,
        wing_loading= wing_loading, 
    )

    far_3 = far_3_takeoff(
        cl= CL_TAKEOFF,
        l_t = FAR_TAKEOFF_DISTANCE,
        wing_loading= wing_loading,
        sigma= sigma,
    )

    far_4 = far_4_climb(
        no_of_engines= NO_OF_ENGINES,
        cl_2= CL_CLIMB, 
        cd= CD_CLIMB,
        AR= ASPECT_RATIO,
        e= OSWALD_CLIMB, 
        wing_loading= wing_loading,
    )

    far_5 = far_5_cruise_speed(
        altitude= CRUISE_ALTITUDE, 
        MACH= CRUISE_MACH, 
        cd_0= CD_0,
        AR= ASPECT_RATIO,
        cl= CL_CRUISE,
        e= OSWALD_CRUISE,
        wing_loading= wing_loading,
    )

    wing_loading_far_1 , thrust_loading_far_1 = far_1
    wing_loading_far_1 = wing_loading_far_1 / WEIGHT_RATIO
    far_1 = (wing_loading_far_1 , thrust_loading_far_1)
    
    wing_loading_far_2 ,thrust_loading_far_2 = far_2
    thrust_loading_far_2 = thrust_loading_far_2 * WEIGHT_RATIO
    far_2 = (wing_loading_far_2 ,thrust_loading_far_2)

    wing_loading_far_3 ,thrust_loading_far_3 = far_3

    wing_loading_far_4 ,thrust_loading_far_4 = far_4

    wing_loading_far_5 ,thrust_loading_far_5 = far_5

    # # METHOD 1
    idx = np.argwhere(np.diff(np.sign(wing_loading_far_3 - wing_loading_far_1))).flatten()
    OP_thrust_loading = thrust_loading_far_3[idx]
    OP_wing_loading = wing_loading_far_3[idx]

    # # METHOD 2
    # if thrust_loading_far_2[0] > thrust_loading_far_4[0]:
    #     idx = np.argmin(np.abs(thrust_loading_far_3 - thrust_loading_far_2))
    #     OP_thrust_loading = thrust_loading_far_3[idx]
    #     OP_wing_loading = wing_loading_far_3[idx]
    # else:
    #     idx = np.argmin(np.abs(thrust_loading_far_3 - thrust_loading_far_4))
    #     OP_thrust_loading = thrust_loading_far_3[idx]
    #     OP_wing_loading = wing_loading_far_3[idx]
    
    # if OP_wing_loading > wing_loading_far_1[5]:
    #     OP_wing_loading = wing_loading_far_1[5]
    
    fuel_frac = range_criterion(
        range= RANGE, 
        mach= CRUISE_MACH,
        l_over_d= L_OVER_D, 
        sfc= SFC,
    )

    W_G = usefull_load_criterion(
        thrust_loading= OP_thrust_loading,
        wf_wg= fuel_frac,
        w_p= PAYLOAD_WEIGHT,
    )
    S = W_G /  OP_wing_loading 
    thrust_c = OP_thrust_loading * W_G 
    if plot:
        if start_plot:
            plt.plot(wing_loading_far_1,thrust_loading_far_1, color = 'k', label="Landing")
            plt.plot(wing_loading_far_2,thrust_loading_far_2, color = 'm', label="Failed Approach")
            plt.plot(wing_loading_far_3,thrust_loading_far_3, color = 'r', label="Takeoff")
            plt.plot(wing_loading_far_4,thrust_loading_far_4, color = 'b', label="Climb")
            plt.plot(wing_loading_far_5,thrust_loading_far_5, color = 'g', label="Cruise")
            plt.scatter(OP_wing_loading, OP_thrust_loading, label="Operating Point", s= 75, marker= 'x') # type: ignore
            plt.scatter(MTOW/AREA,THRUST/MTOW, label="EMBRAER 170", s= 75, marker= 'x') # type: ignore
        else:
            # plt.plot(wing_loading_far_1,thrust_loading_far_1, color = 'k' )
            # plt.plot(wing_loading_far_2,thrust_loading_far_2, color = 'm')
            # plt.plot(wing_loading_far_3,thrust_loading_far_3, color = 'r' )
            # plt.plot(wing_loading_far_4,thrust_loading_far_4, color = 'b' )
            # plt.plot(wing_loading_far_5,thrust_loading_far_5, color = 'g' )
            plt.scatter(OP_wing_loading, OP_thrust_loading, marker= 'x') # type: ignore

        if show_plot:
            plt.legend()
            plt.xlabel("Wing Loading (lb/ft^2)")
            plt.ylabel("Thrust Loading (lbf/lb)")
            plt.grid()
            plt.show()
    return far_1, far_2, far_3, far_4, far_5, W_G, S, thrust_c, OP_thrust_loading, OP_wing_loading
