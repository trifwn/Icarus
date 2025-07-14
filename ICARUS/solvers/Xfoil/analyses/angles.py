from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from xfoil import XFoil
from xfoil.model import Airfoil as XFAirfoil

from ICARUS.airfoils import Airfoil
from ICARUS.airfoils.metrics.aerodynamic_dataclasses import AirfoilOperatingConditions
from ICARUS.airfoils.metrics.aerodynamic_dataclasses import AirfoilOperatingPointMetrics
from ICARUS.airfoils.metrics.aerodynamic_dataclasses import AirfoilPressure
from ICARUS.airfoils.metrics.polars import AirfoilPolar
from ICARUS.core.types import FloatArray

if TYPE_CHECKING:
    from ICARUS.solvers.Xfoil.xfoil import XfoilSolverParameters


def xfoil_aseq(
    reynolds: float,
    mach: float,
    min_aoa: float,
    max_aoa: float,
    aoa_step: float,
    airfoil: Airfoil,
    solver_parameters: XfoilSolverParameters,
) -> AirfoilPolar:
    mach = 0

    xf = XFoil()
    xf.print = solver_parameters.print
    xf.Re = reynolds
    xf.M = mach

    pts = airfoil.to_selig()
    xpts = pts[0]
    ypts = pts[1]
    xf_airf_obj = XFAirfoil(x=xpts, y=ypts)
    xf.airfoil = xf_airf_obj

    params_dict = asdict(solver_parameters)
    for key, value in params_dict.items():
        if key == "repanel_n":
            if value > 0:
                xf.repanel(value)
        elif key == "print":
            continue
        else:
            setattr(xf, key, value)

    aXF, clXF, cdXF, cmXF, cpXF = xf.aseq(min_aoa, max_aoa, aoa_step)

    metrics = []
    for angle, cl, cd, cm, cp in zip(aXF, clXF, cdXF, cmXF, cpXF):
        op = AirfoilOperatingConditions(
            aoa=angle,
            reynolds_number=reynolds,
            mach_number=mach,
        )
        metric = AirfoilOperatingPointMetrics(
            operating_conditions=op,
            Cl=cl,
            Cd=cd,
            Cm=cm,
            Cp_min=cp,
        )
        metrics.append(metric)
    return AirfoilPolar.from_airfoil_metrics(metrics)


def xfoil_aseq_reset_bl(
    reynolds: float,
    mach: float,
    angles: list[float] | FloatArray,
    airfoil: Airfoil,
    solver_parameters: XfoilSolverParameters,
) -> AirfoilPolar:
    xf = XFoil()
    xf.Re = reynolds
    xf.M = 0.0

    xpts, ypts = airfoil.to_selig()
    airfoil_obj = XFAirfoil(x=xpts, y=ypts)
    xf.airfoil = airfoil_obj

    params_dict = asdict(solver_parameters)
    for key, value in params_dict.items():
        if key == "repanel_n":
            if value > 0:
                xf.repanel(value)
        elif key == "print":
            xf.print = value
        else:
            setattr(xf, key, value)

    # xf.filter()

    metrics = []
    for angle in angles:
        op = AirfoilOperatingConditions(
            aoa=angle,
            reynolds_number=reynolds,
            mach_number=mach,
        )

        cl, cd, cm, cp = xf.a(angle)
        x, y, cp_distribution = xf.get_cp_distribution()

        cp_distribution = AirfoilPressure(x=x, y=y, cp=cp_distribution)

        metric = AirfoilOperatingPointMetrics(
            operating_conditions=op,
            Cl=cl,
            Cd=cd,
            Cm=cm,
            Cp_min=cp,
            Cp_distribution=cp_distribution,
        )
        metrics.append(metric)

        # Reset the boundary layer state
        xf.reset_bls()

    return AirfoilPolar.from_airfoil_metrics(metrics)
