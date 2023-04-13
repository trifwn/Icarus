from .disturbances import disturbance as dst


def longitudalPerturb(plane, scheme, epsilon):
    """Function to add all longitudinal perturbations
    needed to compute the aero derivatives
    Inputs:
    - variable: string with the variable to perturb
    - amplitude: amplitude of the perturbation
    """
    disturbances = []
    if epsilon is None:
        del (epsilon)
        epsilon = {"u": 0.01,
                   "w": 0.01,
                   "q": 0.25,
                   "theta": 0.01  # /plane.trim["U"]
                   }

    for var in ["u", "w", "q", "theta"]:
        plane.epsilons[var] = epsilon[var]
        if scheme == "Central":
            disturbances.append(dst(var, epsilon[var]))
            disturbances.append(dst(var, -epsilon[var]))
        elif scheme == "Forward":
            disturbances.append(dst(var, epsilon[var]))
        elif scheme == "Backward":
            disturbances.append(dst(var, -epsilon[var]))
        else:
            raise ValueError(
                "Scheme must be 'Central', 'Forward' or 'Backward'")
    return disturbances


def lateralPerturb(plane, scheme, epsilon):
    """Function to add all lateral perturbations
    needed to compute the aero derivatives
    Inputs:
    - variable: string with the variable to perturb
    - amplitude: amplitude of the perturbation
    """
    disturbances = []
    if epsilon is None:
        del (epsilon)
        epsilon = {"v": 0.01,
                   "p": 0.25,
                   "r": 0.25,
                   "phi": 0.01
                   }

    for var in ["v", "p", "r", "phi"]:
        plane.epsilons[var] = epsilon[var]
        if scheme == "Central":
            disturbances.append(dst(var, epsilon[var]))
            disturbances.append(dst(var, -epsilon[var]))
        elif scheme == "Forward":
            disturbances.append(dst(var, epsilon[var]))
        elif scheme == "Backward":
            disturbances.append(dst(var, -epsilon[var]))
        else:
            raise ValueError(
                "Scheme must be 'Central', 'Forward' or 'Backward'")
    return disturbances
