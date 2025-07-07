from __future__ import annotations

import os
from functools import lru_cache
from typing import Any
from typing import Self

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pandas import DataFrame
from pandas import Index

from ICARUS.airfoils.metrics.aerodynamic_dataclasses import AirfoilOperatingPointMetrics
from ICARUS.core.interpolation.pandas_series import get_linear_series
from ICARUS.core.interpolation.pandas_series import interpolate_series
from ICARUS.core.interpolation.pandas_series import interpolate_series_index
from ICARUS.core.interpolation.pandas_series import interpolate_series_value


class PolarNotAccurate(Exception):
    """Exception Raised when the Polar is not accurate"""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ReynoldsNotIncluded(Exception):
    """Exception Raised when the Polar is not included"""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class AirfoilPolar:
    """Airfoil Polar Class for a single Reynolds number."""

    @classmethod
    def from_airfoil_metrics(
        cls,
        list_of_metrics: list[AirfoilOperatingPointMetrics],
    ) -> Self:
        """Create an AirfoilPolar from a list of AirfoilOperatingPointMetrics."""
        if not list_of_metrics:
            raise ValueError("List of metrics is empty.")

        # Extract Reynolds number from the first metric
        reynolds = list_of_metrics[0].operating_conditions.reynolds_number
        if not all(
            metric.operating_conditions.reynolds_number == reynolds
            for metric in list_of_metrics
        ):
            raise ValueError("All metrics must have the same Reynolds number.")

        data = {
            "AoA": [metric.operating_conditions.aoa for metric in list_of_metrics],
            f"CL_{reynolds}": [metric.Cl for metric in list_of_metrics],
            f"CD_{reynolds}": [metric.Cd for metric in list_of_metrics],
            f"Cm_{reynolds}": [metric.Cm for metric in list_of_metrics],
        }
        df = DataFrame(data)
        # Remove NaN values
        df.dropna(inplace=True, axis=0, how="any")
        df.sort_values(by="AoA", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return cls(reynolds, df)

    def __init__(self, reynolds: float, df: DataFrame) -> None:
        self.reynolds = reynolds

        self.df = df.rename(
            columns={
                f"CL_{reynolds}": "CL",
                f"CD_{reynolds}": "CD",
                f"Cm_{reynolds}": "Cm",
            },
        )

        self.angles: pd.Series[float] = self.df["AoA"]

        self.cl_curve: pd.Series[float] = self.df["CL"]
        self.cl_curve.index = Index(self.df["AoA"].astype("float64"))

        self.cd_curve: pd.Series[float] = self.df["CD"]
        self.cd_curve.index = Index(self.df["AoA"].astype("float64"))

        self.cm_curve: pd.Series[float] = self.df["Cm"]
        self.cm_curve.index = Index(self.df["AoA"].astype("float64"))

    def is_empty(self) -> bool:
        """Check if the polar is empty."""
        return self.df.empty or len(self.angles) == 0 or self.df["CL"].empty

    @property
    def reynolds_string(self) -> str:
        """Get Reynolds number as a string."""
        return np.format_float_scientific(
            self.reynolds,
            sign=False,
            precision=3,
            min_digits=3,
        ).replace("+", "")

    def get_aero_coefficients(self, aoa: float) -> tuple[float, float, float]:
        """Get Aero Coefficients"""
        cl = interpolate_series_value(aoa, self.cl_curve)
        cd = interpolate_series_value(aoa, self.cd_curve)
        cm = interpolate_series_value(aoa, self.cm_curve)
        return cl, cd, cm

    def examine(self) -> tuple[bool, str]:
        """Examine the polar for issues."""
        return self.examine_run(self.cl_curve, self.cd_curve, self.zero_lift_angle)

    @property
    @lru_cache()
    def zero_lift_angle(self) -> float:
        """Get Zero Lift Angle from Cl Curve"""
        return interpolate_series_index(0.0, self.cl_curve)

    @property
    @lru_cache()
    def zero_lift_cm(self) -> float:
        """Get Zero Lift Cm from Cm Curve"""
        return interpolate_series_value(self.zero_lift_angle, self.cm_curve)

    @property
    @lru_cache()
    def cl_slope(self) -> float:
        """Get Slope of Cl Curve"""
        cl_linear: pd.Series[float] = get_linear_series(self.cl_curve)
        return float(cl_linear.diff().mean())

        # cls = self.cl_curve.to_numpy()
        # aoas = self.angles

        # zero_lift_angle = self.get_zero_lift_angle()
        # max_angle = zero_lift_angle + 3
        # min_angle = zero_lift_angle - 3
        # min_index = np.argmin(np.abs(aoas - min_angle))
        # max_index = np.argmin(np.abs(aoas - max_angle))

        # cl_slope = float(
        #     np.poly1d(
        #         np.polyfit(
        #             aoas[min_index:max_index],
        #             cls[min_index:max_index],
        #             1,
        #         ),
        #     )[1],
        # )
        # return cl_slope / 180 * (np.pi)  # Convert to per radian slope

    def get_positive_stall(self) -> tuple[float, float, float]:
        """Get Positive Stall Angle"""
        curve = self.cl_curve / self.cd_curve
        pos_stall_aoa = curve.idxmax()

        aoa = float(pos_stall_aoa)
        cl = self.cl_curve.loc[pos_stall_aoa]
        cd = self.cd_curve.loc[pos_stall_aoa]
        return (aoa, cl, cd)

    def get_negative_stall(self) -> tuple[float, float, float]:
        """Get Negative Stall Angle"""
        curve = self.cl_curve / self.cd_curve
        neg_stall_aoa = curve.idxmin()

        aoa = float(neg_stall_aoa)
        cl = self.cl_curve.loc[neg_stall_aoa]
        cd = self.cd_curve.loc[neg_stall_aoa]
        return (aoa, cl, cd)

    def get_minimum_cl_cd(self) -> tuple[float, float, float]:
        """Get Minimum CL/CD Ratio"""
        curve = self.cd_curve / self.cl_curve
        min_cl_cd_aoa = curve.idxmin()

        aoa = float(min_cl_cd_aoa)
        cl = self.cl_curve.loc[min_cl_cd_aoa]
        cd = self.cd_curve.loc[min_cl_cd_aoa]
        return (aoa, cl, cd)

    def plot_cl(
        self,
        ax: Axes | None = None,
        add_zero_lift: bool = True,
        color: tuple[float, float, float] | str | None = None,
    ) -> None:
        """Plot Cl Curve"""
        if not isinstance(ax, Axes):
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(
            self.angles,
            self.cl_curve,
            label=f"Reynolds {self.reynolds:,.0f}",
            color=color,
        )
        ax.set_xlabel("Angle of Attack [deg]")
        ax.set_ylabel("Lift Coefficient")
        ax.set_title(f"Reynolds {self.reynolds:,.0f} Cl Curve")
        ax.legend()
        ax.grid(True)
        ax.axhline(0.0, color="k", linestyle="--")
        if add_zero_lift:
            ax.axvline(self.zero_lift_angle, color="r", linestyle="--")

    def plot_cd(
        self,
        ax: Axes | None = None,
        color: tuple[float, float, float] | str | None = None,
    ) -> None:
        """Plot Cd Curve"""
        if not isinstance(ax, Axes):
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(
            self.angles,
            self.cd_curve,
            label=f"Reynolds {self.reynolds:,.0f}",
            color=color,
        )
        ax.set_xlabel("Angle of Attack [deg]")
        ax.set_ylabel("Drag Coefficient")
        ax.set_title(f"Reynolds {self.reynolds:,.0f} Cd Curve")
        ax.legend()
        ax.grid(True)
        ax.axvline(self.zero_lift_angle, color="r", linestyle="--")
        ax.axhline(0.0, color="k", linestyle="--")

    def plot_cl_over_cd(
        self,
        ax: Axes | None = None,
        color: tuple[float, float, float] | str | None = None,
    ) -> None:
        """Plot Cl Over Cd Curve"""
        cl_over_cd = self.cl_curve / self.cd_curve
        cl_over_cd.index = Index(self.df["AoA"].astype("float32"))

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(
            self.angles,
            cl_over_cd,
            label=f"Reynolds {self.reynolds:,.0f}",
            color=color,
        )
        ax.set_xlabel("Angle of Attack [deg]")
        ax.set_ylabel("Lift to Drag Ratio")
        ax.set_title(f"Reynolds {self.reynolds:,.0f} Cl/Cd Curve")
        ax.legend()
        ax.grid(True)
        ax.axvline(self.zero_lift_angle, color="r", linestyle="--")
        ax.axhline(0.0, color="k", linestyle="--")

    def __getstate__(self) -> dict[str, Any]:
        state = {}
        state["reynolds"] = self.reynolds
        state["df"] = self.df.copy()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        reynolds: float = state["reynolds"]
        df: DataFrame = state["df"]
        AirfoilPolar.__init__(self, reynolds, df.astype("float64"))

    def to_json(self) -> str:
        """Pickle the state object to a json string.

        Returns:
            str: Json String

        """
        encoded: str = str(jsonpickle.encode(self))
        return encoded

    def save(self, directory: str, filename: str | None = None) -> None:
        """Save the state object to a json file.

        Args:
            directory (str): Directory to save the state to.

        """
        # Check if directory exists
        os.makedirs(directory, exist_ok=True)
        if filename is None:
            filename = "polar.json"

        fname: str = os.path.join(directory, filename)
        with open(fname, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @staticmethod
    def interpolate_polar(
        polar1: AirfoilPolar,
        polar2: AirfoilPolar,
        reynolds: float,
    ) -> AirfoilPolar:
        """Interpolate between two polars at a given Reynolds number."""
        if polar1.reynolds == polar2.reynolds:
            raise ReynoldsNotIncluded(
                "Reynolds numbers must be different for interpolation.",
            )

        # Interpolate angles
        angles = np.unique(np.concatenate((polar1.angles, polar2.angles)))

        cl_curve1 = interpolate_series(angles, polar1.cl_curve)
        cd_curve1 = interpolate_series(angles, polar1.cd_curve)
        cm_curve1 = interpolate_series(angles, polar1.cm_curve)

        cl_curve2 = interpolate_series(angles, polar2.cl_curve)
        cd_curve2 = interpolate_series(angles, polar2.cd_curve)
        cm_curve2 = interpolate_series(angles, polar2.cm_curve)

        # Interpolate coefficients
        cl_curve = np.interp(
            reynolds,
            [polar1.reynolds, polar2.reynolds],
            [cl_curve1, cl_curve2],
        )
        cd_curve = np.interp(
            reynolds,
            [polar1.reynolds, polar2.reynolds],
            [cd_curve1, cd_curve2],
        )
        cm_curve = np.interp(
            reynolds,
            [polar1.reynolds, polar2.reynolds],
            [cm_curve1, cm_curve2],
        )

        df = DataFrame(
            {
                "AoA": angles,
                "CL": cl_curve,
                "CD": cd_curve,
                "Cm": cm_curve,
            },
        )

        return AirfoilPolar(reynolds, df)

    @staticmethod
    def examine_run(
        cl_curve: pd.Series[float],
        cd_curve: pd.Series[float],
        zero_lift_angle: float,
    ) -> tuple[bool, str]:
        cl_array = cl_curve.to_numpy()
        cd_array = cd_curve.to_numpy()
        aoa_array = np.deg2rad(cl_curve.index.to_numpy())
        cl_slopes = np.gradient(cl_array, aoa_array)
        cd_slopes = np.gradient(cd_array, aoa_array)
        max_angle = zero_lift_angle + 4
        min_angle = zero_lift_angle
        min_index = np.argmin(np.abs(aoa_array - np.deg2rad(min_angle)))
        max_index = np.argmin(np.abs(aoa_array - np.deg2rad(max_angle)))
        max_positive_CL_slope = 3 * (2 * np.pi)
        min_negative_CL_slope = -0.2 * (2 * np.pi)
        max_positive_CD_slope = 0.9
        min_negative_CD_slope = -0.9

        CL_cond = np.any(
            np.greater(cl_slopes[min_index:max_index], max_positive_CL_slope),
        ) or np.any(
            np.less(cl_slopes[min_index:max_index], min_negative_CL_slope),
        )
        CD_cond = np.any(
            np.greater(cd_slopes[min_index:max_index], max_positive_CD_slope),
        ) or np.any(
            np.less(cd_slopes[min_index:max_index], min_negative_CD_slope),
        )
        if CL_cond and not CD_cond:
            return True, "CL problem"
        elif CD_cond and not CL_cond:
            return True, "CD problem"
        elif CD_cond and CL_cond:
            return True, "CL and CD problem"
        return False, "No problem"
