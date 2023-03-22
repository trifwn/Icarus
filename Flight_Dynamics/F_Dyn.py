import numpy as np
import pandas as pd


def rad2deg(x):
    return(x*180/np.pi)


def deg2rad(x):
    return(x*np.pi/180)


def Trim(Plane, rho, Wing_Area, Cm_curve, CL_curve, Aoa_vector, Weight):
    # Index of interest in the Polar Dataframe
    Trim_ind = np.argmin(np.abs(Cm_curve))

    # How accurate is the trim
    print(f"Cm thst should be zero is {Cm_curve[Trim_ind] }")

    # Trim - related Aerodynamic Parameters of interest
    AoA_Trim = Aoa_vector[Trim_ind]
    CL_Trim = CL_curve[Trim_ind]
    U_cruise = Weight/(0.5*rho*Wing_Area*CL_Trim)

    return U_cruise, AoA_Trim


def Perturbations_Analysis(Plane, U_cruise, AoA_Trim, Velocity_incr, AngVelocity_incr):
    # Axis with Capital Letter means body-fitted, axis with lower case means Stream-Wise
    # The Axis notation will help during the integration of dynamic aanalysis woth GenuVP or any 3D Aerodynamics Solver

    # Dictionary of Longitudinal Perturbations
    Long_dict = {"Body_Movements": ["q", AngVelocity_incr, -AngVelocity_incr, " Angular Velocity, Y axis"],
                 "Wind_Movements": [['u', U_cruise+Velocity_incr, U_cruise-Velocity_incr, " Linear Velocity, x axis"],
                                    ['w', Velocity_incr, -Velocity_incr, " Linear Velocity, z axis"]]
                 }

    # Dictionary of Lateral Perturbations
    Lat_dict = {"Body_Movements": [["p", AngVelocity_incr, -AngVelocity_incr, " Angular Velocity, X axis"],
                                   ["r", AngVelocity_incr, -AngVelocity_incr, " Angular Velocity, Z axis"]],
                "Wind_Movements": ['v', Velocity_incr, -Velocity_incr, " Linear Velocity, y axis"]
                }
    return Long_dict, Lat_dict


def State_Space_Generation(AoA_trim, U_cruise, Mass, Ix, Iy, Iz, Ixz, Long_res_dict, Lat_res_dict, Velocity_incr, AngVelocity_incr, Xw_dot, Zw_dot, Mw_dot, Trim, theta_value):
    # Trim position
    if Trim == True:
        theta = 0
    else:
        theta = theta_value

    # This Function Requires the results from perturbation analysis stored in a dictionary with the structure:
    # /Type of coordinate system/Perturbed Variable/Force or moment/[Positive increment,Negative Increment]

    # For the Longitudinal Motion, in addition to the state space variables an analysis with respect to the derivative of w perturbation is needed. These derivatives are in this function are added externally and called Xw_dot,Zw_dot,Mw_dot.
    # Depending on the Aerodynamics Solver, these w_dot derivatives can either be computed like the rest derivatives, or require an approximation concerning the downwash velocity that the main wing induces on the tail wing

    # Longitudinal Motion
    Xu = (Long_res_dict["Wind Movements"]["u"]["Force X"][1] -
          Long_res_dict["Wind Movements"]["u"]["Force X"][0])/(2*Velocity_incr)
    Zu = (Long_res_dict["Wind Movements"]["u"]["Force Z"][1] -
          Long_res_dict["Wind Movements"]["u"]["Force Z"][0])/(2*Velocity_incr)
    Mu = (Long_res_dict["Wind Movements"]["u"]["Moment Y"][1] -
          Long_res_dict["Wind Movements"]["u"]["Moment Y"][0])/(2*Velocity_incr)

    Xw = (Long_res_dict["Wind Movements"]["w"]["Force X"][1]-Xw -
          Long_res_dict["Wind Movements"]["w"]["Force X"][0])/(2*Velocity_incr)
    Zw = (Long_res_dict["Wind Movements"]["w"]["Force Z"][1] -
          Long_res_dict["Wind Movements"]["w"]["Force Z"][0])/(2*Velocity_incr)
    Mw = (Long_res_dict["Wind Movements"]["w"]["Moment Y"][1] -
          Long_res_dict["Wind Movements"]["w"]["Moment Y"][0])/(2*Velocity_incr)

    Xq = (Long_res_dict["Body Movements"]["q"]["Force X"][1] -
          Long_res_dict["Body Movements"]["q"]["Force X"][0])/(2*AngVelocity_incr)
    Zq = (Long_res_dict["Body Movements"]["q"]["Force Z"][1] -
          Long_res_dict["Body Movements"]["q"]["Force Z"][0])/(2*AngVelocity_incr)
    Mq = (Long_res_dict["Body Movements"]["q"]["Moment Y"][1] -
          Long_res_dict["Body Movements"]["q"]["Moment Y"][0])/(2*AngVelocity_incr)

    Long_Mat = np.zeros((4, 4))

    Long_Mat[0, 0] = Xu/Mass + (Xw_dot*Zu)/(Mass*(Mass-Zw_dot))
    Long_Mat[0, 1] = Xw/Mass + (Xw_dot*Zw)/(Mass*(Mass-Zw_dot))
    Long_Mat[0, 2] = Xq/Mass + (Xw_dot*(Zq+Mass*U_cruise))/(Mass*(Mass-Zw_dot))
    Long_Mat[0, 3] = -9.81*np.cos(theta) - \
        (Xw_dot*9.81*np.sin(theta))/((Mass-Zw_dot))

    Long_Mat[1, 0] = Zu/(Mass-Zw_dot)
    Long_Mat[1, 1] = Zw/(Mass-Zw_dot)
    Long_Mat[1, 2] = (Zq+Mass*U_cruise)/(Mass-Zw_dot)
    Long_Mat[1, 3] = -(Mass*9.81*np.sin(theta))/(Mass-Zw_dot)

    Long_Mat[2, 0] = Mu/Iy + Zu*Mw_dot/(Iy*(Mass-Zw_dot))
    Long_Mat[2, 1] = Mw/Iy + Zw*Mw_dot/(Iy*(Mass-Zw_dot))
    Long_Mat[2, 2] = Mq/Iy + ((Zq+Mass*U_cruise)*Mw_dot)/(Iy*(Mass-Zw_dot))
    Long_Mat[2, 3] = (Mass*9.81*np.sin(theta)*Mw_dot)/(Iy*(Mass-Zw_dot))

    Long_Mat[3, 2] = 1

    # Lateral Motion

    Yv = (Lat_res_dict["Wind Movements"]["v"]["Force Y"][1] -
          Lat_res_dict["Wind Movements"]["v"]["Force Y"][0])/(2*Velocity_incr)
    Lv = (Lat_res_dict["Wind Movements"]["v"]["Moment X"][1] -
          Lat_res_dict["Wind Movements"]["v"]["Moment X"][0])/(2*Velocity_incr)
    Nv = (Lat_res_dict["Wind Movements"]["v"]["Moment Z"][1] -
          Lat_res_dict["Wind Movements"]["v"]["Moment Z"][0])/(2*Velocity_incr)

    Yp = (Lat_res_dict["Body Movements"]["p"]["Force Y"][1] -
          Lat_res_dict["Body Movements"]["p"]["Force Y"][0])/(2*AngVelocity_incr)
    Lp = (Lat_res_dict["Body Movements"]["p"]["Moment X"][1] -
          Lat_res_dict["Body Movements"]["p"]["Moment X"][0])/(2*AngVelocity_incr)
    Np = (Lat_res_dict["Body Movements"]["p"]["Moment Z"][1] -
          Lat_res_dict["Body Movements"]["p"]["Moment Z"][0])/(2*AngVelocity_incr)

    Yr = (Lat_res_dict["Body Movements"]["r"]["Force Y"][1] -
          Lat_res_dict["Body Movements"]["r"]["Force Y"][0])/(2*AngVelocity_incr)
    Lr = (Lat_res_dict["Body Movements"]["r"]["Moment X"][1] -
          Lat_res_dict["Body Movements"]["r"]["Moment X"][0])/(2*AngVelocity_incr)
    Nr = (Lat_res_dict["Body Movements"]["r"]["Moment Z"][1] -
          Lat_res_dict["Body Movements"]["r"]["Moment Z"][0])/(2*AngVelocity_incr)

    Lat_Mat = np.zeros((4, 4))

    Lat_Mat[0, 0] = Yv/Mass
    Lat_Mat[0, 1] = Yp/Mass
    Lat_Mat[0, 2] = (Yr-Mass*U_cruise)/(Mass)
    Lat_Mat[0, 3] = 9.81*np.cos(theta)

    Lat_Mat[1, 0] = (Iz*Lv+Ixz*Nv)/(Ix*Iz-Ixz**2)
    Lat_Mat[1, 1] = (Iz*Lp+Ixz*Np)/(Ix*Iz-Ixz**2)
    Lat_Mat[1, 2] = (Iz*Lr+Ixz*Nr)/(Ix*Iz-Ixz**2)

    Lat_Mat[2, 0] = (Ix*Nv+Ixz*Lv)/(Ix*Iz-Ixz**2)
    Lat_Mat[2, 1] = (Iz*Np+Ixz*Lp)/(Ix*Iz-Ixz**2)
    Lat_Mat[2, 2] = (Iz*Nr+Ixz*Lr)/(Ix*Iz-Ixz**2)

    Lat_Mat[3, 1] = 1

    return Long_Mat, Lat_Mat
