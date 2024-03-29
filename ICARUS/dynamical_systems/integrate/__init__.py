from .backward_euler import BackwardEulerIntegrator
from .base_integrator import Integrator
from .crank_nicolson import CrankNicolsonIntegrator
from .forward_euler import ForwardEulerIntegrator
from .gauss_legendre import GaussLegendreIntegrator
from .newmark import NewmarkIntegrator
from .rk4 import RK4Integrator
from .rk45 import RK45Integrator

__all__ = [
    "Integrator",
    "BackwardEulerIntegrator",
    "ForwardEulerIntegrator",
    "RK4Integrator",
    "RK45Integrator",
    "CrankNicolsonIntegrator",
    "GaussLegendreIntegrator",
    "NewmarkIntegrator",
]

integrators = {
    "BackwardEuler": BackwardEulerIntegrator,
    "ForwardEuler": ForwardEulerIntegrator,
    "RK4": RK4Integrator,
    "RK45": RK45Integrator,
    "CrankNicolson": CrankNicolsonIntegrator,
    "GaussLegendre": GaussLegendreIntegrator,
    "Newmark": NewmarkIntegrator,
}
